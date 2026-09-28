"""How long a sequence each candidate model can train on, on one 16 GB card.

    python training/memory_probe.py --model Qwen/Qwen3-4B-Instruct-2507 --device 0

One LoRA training step -- forward, backward, optimizer -- at each length, in
fp16 with gradient checkpointing, the way training/train.py runs it. Each
step runs in its own process, so an out-of-memory leaves nothing behind for
the next. Two losses are tried: the model's own (logits at every position)
and training/losses.py's (logits only where the plan is). A length that runs
out of memory ends that loss's climb.

gfx906 has no flash or memory-efficient attention: SDPA falls back to its
math kernel, which holds the whole score matrix, heads x length^2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LENGTHS = (4096, 6144, 8192, 12288, 16384)
#: The plan's share of a sequence is 11-13% (training/prepare.py); probe above it.
PLAN_SHARE = 0.15


def one(model_name: str, length: int, loss: str, attention: str, block: int,
        offload: bool = False) -> dict:
    import contextlib

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(Path(__file__).parent))
    from losses import sliced_loss

    if attention == "chunked":
        import chunked_attention

        chunked_attention.register(block)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, attn_implementation=attention).cuda()
    model.config.use_cache = False
    model.train()   # gradient checkpointing only applies in training mode
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
        task_type="CAUSAL_LM"))
    # LoRA weights train in fp32 under fp16 autocast, as the Trainer runs them.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, length), device="cuda")
    labels = input_ids.clone()
    labels[:, : int(length * (1 - PLAN_SHARE))] = -100
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    seconds = []
    for _ in range(2):     # the second step is the one timed
        torch.cuda.synchronize()
        started = time.time()
        # Offload: what backward keeps (each layer's checkpointed input) waits in host RAM.
        saved = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else contextlib.nullcontext()
        with saved, torch.autocast("cuda", dtype=torch.float16):
            if loss == "sliced":
                value = sliced_loss(model, input_ids, labels)
            else:
                value = model(input_ids=input_ids, labels=labels).loss
        scaler.scale(value).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        seconds.append(time.time() - started)
    return {
        "status": "ok", "loss_value": float(value),
        "weights_gib": round(weights / 2**30, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "step_seconds": round(seconds[-1], 1),
        "tokens_per_second": round(length / seconds[-1]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="0", help="which card (HIP_VISIBLE_DEVICES)")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--losses", nargs="+", default=["full", "sliced"])
    parser.add_argument("--attention", default="sdpa", choices=("sdpa", "chunked"))
    parser.add_argument("--block", type=int, default=1024, help="chunked attention's query block")
    parser.add_argument("--offload", action="store_true",
                        help="keep backward's saved tensors in host RAM")
    parser.add_argument("--out", type=Path, default=Path("/scratch/runs/probe.jsonl"))
    parser.add_argument("--one", nargs=2, metavar=("LENGTH", "LOSS"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.one:
        import torch

        length, loss = int(args.one[0]), args.one[1]
        try:
            result = one(args.model, length, loss, args.attention, args.block, args.offload)
        except torch.OutOfMemoryError as exc:
            result = {"status": "oom", "error": str(exc).splitlines()[0][:200]}
        print("RESULT " + json.dumps(result), flush=True)
        return 0

    env = {**os.environ, "HIP_VISIBLE_DEVICES": args.device,
           "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for loss in args.losses:
        for length in args.lengths:
            started = time.time()
            process = subprocess.run(
                [sys.executable, __file__, "--model", args.model, "--one", str(length), loss,
                 "--attention", args.attention, "--block", str(args.block)]
                + (["--offload"] if args.offload else []),
                env=env, capture_output=True, text=True)
            found = [line for line in process.stdout.splitlines() if line.startswith("RESULT ")]
            if found:
                result = json.loads(found[-1][len("RESULT "):])
            else:
                tail = (process.stderr or "").strip().splitlines()[-3:]
                result = {"status": "crash", "error": " | ".join(tail)[:300]}
            result = {"model": args.model, "length": length, "loss": loss,
                      "attention": args.attention,
                      "block": args.block if args.attention == "chunked" else None,
                      "offload": args.offload, **result,
                      "wall_seconds": round(time.time() - started)}
            with args.out.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            if result["status"] != "ok":
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
