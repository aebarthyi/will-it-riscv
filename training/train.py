"""LoRA fine-tuning on data/prepared, a model copy per card or one sharded across them.

    torchrun --nproc_per_node=2 training/train.py training/configs/qwen3-4b-12k.yaml

With ``fsdp: true`` the model is sharded across the cards instead (FSDP2):
each decoder layer is its own unit, and the loss is the root, so the 8 GB of
fp16 weights cost each card half. That half is what 16k needs. The cards
then run every forward in lockstep, and the checkpoints are sharded
(torch.distributed.checkpoint), with a full adapter beside them for export.

A plain loop rather than a Trainer, for three things it has to control:
the loss is training/losses.py's (it calls the model's layers directly, past
any DDP wrapper, so the LoRA gradients are averaged across cards by hand at
each optimizer step); what backward keeps can wait in host RAM
(``offload``, which is what fits 4B at 12k on a 16 GB card); and the cards
overheat, so between sequences a hot card pauses until it cools, and a run a
watchdog kills resumes from its last checkpoint.

The same loop runs on CUDA and ROCm cards, on Gaudi (``device: hpu``: HCCL,
lazy mode's mark_step, FusedSDPA), and on the CPU for a smoke test (gloo).
Gaudi compiles a graph for every shape it meets, so there each sequence is
padded to one of a few lengths (``bucket``) and the loss is ``static``.
"""

from __future__ import annotations

import contextlib
import datetime
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from losses import IGNORE, sliced_loss, static_loss  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "data": "data/prepared/12288",
    "out": None,                  # default: $WILL_IT_RISCV_RUNS (else /scratch/runs)/<config>
    "init_adapter": None,         # a checkpoint directory to start the LoRA from (same rank)
    "lora_r": 16, "lora_alpha": 32,
    "lr": 1e-4, "warmup": 0.03, "epochs": 3, "accumulation": 8, "clip": 1.0,
    "attention_block": 512, "offload": True,
    # gfx906 has no bf16 and no flash attention: float16 and the chunked kernel.
    # A datacentre card takes "bfloat16" and "sdpa" (or "flash_attention_2");
    # a Gaudi, "bfloat16" and "fused" (hpu_attention.py).
    "dtype": "float16", "attention": "chunked",
    "device": "auto",             # cuda (CUDA or ROCm), hpu (Gaudi), cpu (a smoke test)
    # "sliced": logits only where the plan is; "static": every position, in
    # chunks of loss_chunk, so shapes follow the length alone (for Gaudi).
    "loss": "sliced", "loss_chunk": 4096,
    "bucket": 0,                  # pad lengths to a few sizes, the smallest this (0: no padding)
    # Qwen3-8B: {enable_thinking: false}, so the plan follows an empty think block.
    "chat_template_kwargs": {},
    "fsdp": False,                # shard the model across the cards (FSDP2) instead of copying it
    "longest_first": False,       # a smoke test: take the longest sequences first
    "eval_every": 100, "save_every": 25,   # optimizer steps
    "max_steps": None,            # stop early (a smoke test)
    "seed": 0,
    # Between sequences, a card this hot pauses until it is back under `resume_below`.
    "pause_edge": 80, "pause_hotspot": 90, "resume_below": 70,
}


def log(*parts) -> None:
    if dist.get_rank() == 0:
        print(*parts, flush=True)


def tokenize(tokenizer, example: dict, template: dict) -> dict:
    """input_ids, and labels that are IGNORE everywhere but the plan."""
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False,
                                           add_generation_prompt=True, **template)
    full = tokenizer.apply_chat_template(example["prompt"] + example["completion"],
                                         tokenize=False, **template)
    assert full.startswith(prompt), example["meta"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    assert ids[:len(prompt_ids)] == prompt_ids, example["meta"]
    labels = [IGNORE] * len(prompt_ids) + ids[len(prompt_ids):]
    return {"input_ids": ids, "labels": labels, "key": example["meta"].get("key")}


def load(tokenizer, path: Path, template: dict) -> list[dict]:
    with path.open() as handle:
        return [tokenize(tokenizer, json.loads(line), template) for line in handle]


def padded(length: int, bucket: int) -> int:
    """The bucket a length goes in: four sizes an octave, none under ``bucket``.

    So 2,800 tokens train as 3,072 and 20,000 as 20,480: at most a quarter
    more, about an eighth on average, and two dozen sizes up to 32k.
    """
    if not bucket:
        return length
    step = max(bucket, (1 << (length - 1).bit_length()) // 8)
    return -(-length // step) * step


class Device:
    """What differs between a CUDA or ROCm card, a Gaudi, and the CPU."""

    def __init__(self, kind: str, local_rank: int):
        if kind == "auto":
            try:
                import habana_frameworks.torch  # noqa: F401
                kind = "hpu"
            except ImportError:
                kind = "cuda" if torch.cuda.is_available() else "cpu"
        self.kind = kind
        if kind == "hpu":
            import habana_frameworks.torch.core as htcore
            import habana_frameworks.torch.distributed.hccl  # noqa: F401  (registers "hccl")

            self._mark_step = htcore.mark_step
            self.backend, self.torch = "hccl", torch.device("hpu")
        elif kind == "cuda":
            torch.cuda.set_device(local_rank)
            self.backend, self.torch = "nccl", torch.device("cuda", local_rank)
        else:
            self.backend, self.torch = "gloo", torch.device("cpu")

    def step(self) -> None:
        """Lazy mode runs what has queued up so far; elsewhere, nothing to do."""
        if self.kind == "hpu":
            self._mark_step()

    def peak_gib(self) -> float | None:
        if self.kind == "cuda":
            return round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if self.kind == "hpu":
            import habana_frameworks.torch.hpu as hthpu

            return round(hthpu.max_memory_allocated() / 2**30, 2)
        return None

    def tensors(self, example: dict, bucket: int, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The example's ids and labels, padded at the end to its bucket.

        Causal attention never lets a token see what follows it, and the pads'
        labels are IGNORE, so the padding changes no loss and no gradient.
        """
        ids, labels = example["input_ids"], example["labels"]
        extra = padded(len(ids), bucket) - len(ids)
        return (torch.tensor([ids + [pad] * extra], device=self.torch),
                torch.tensor([labels + [IGNORE] * extra], device=self.torch))


def hwmon(device: int) -> Path | None:
    """The card's hwmon directory in sysfs, found by its PCI bus."""
    bus = torch.cuda.get_device_properties(device).pci_bus_id
    found = glob.glob(f"/sys/bus/pci/devices/0000:{bus:02x}:00.0/hwmon/hwmon*")
    return Path(found[0]) if found else None


def temperatures(sensor: Path | None) -> tuple[int, int] | None:
    """(edge, hotspot) in degrees C."""
    if sensor is None:
        return None
    try:
        return tuple(int((sensor / f"temp{i}_input").read_text()) // 1000 for i in (1, 2))
    except (OSError, ValueError):
        return None


def cool_down(sensor: Path | None, config: dict) -> float:
    """Wait while this card is too hot; the seconds waited."""
    t = temperatures(sensor)
    if t is None or (t[0] <= config["pause_edge"] and t[1] <= config["pause_hotspot"]):
        return 0.0
    started = time.time()
    print(f"[rank {dist.get_rank()}] pausing at edge {t[0]} hotspot {t[1]}", flush=True)
    while t is not None and max(t) >= config["resume_below"]:
        time.sleep(5)
        t = temperatures(sensor)
    return time.time() - started


class PlanLoss(torch.nn.Module):
    """The loss as a module: FSDP's root, so its forward gathers what the loss uses."""

    def __init__(self, model, kind: str = "sliced", chunk: int = 4096):
        super().__init__()
        self.model = model
        self.kind, self.chunk = kind, chunk

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.kind == "static":
            return static_loss(self.model, input_ids, labels, self.chunk)
        return sliced_loss(self.model, input_ids, labels)


def shard(model, kind: str, chunk: int) -> PlanLoss:
    """Every decoder layer its own unit; the root holds the tied embedding and head."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from losses import causal_lm

    policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    for layer in causal_lm(model).model.layers:
        fully_shard(layer, mp_policy=policy)
    root = PlanLoss(model, kind, chunk)
    fully_shard(root, mp_policy=policy)
    return root


def sharded_norm(params: list[torch.Tensor]) -> torch.Tensor:
    """The gradients' global L2 norm, from each card's shard."""
    local = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)
    for p in params:
        g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
        local += g.float().pow(2).sum()
    dist.all_reduce(local)
    return local.sqrt()


def average_gradients(params: list[torch.Tensor], world: int) -> None:
    grads = [p.grad for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for g in grads:
        g.copy_(flat[offset:offset + g.numel()].view_as(g))
        offset += g.numel()


def evaluate(loss_fn, examples: list[dict], device: Device, sensor: Path | None, config: dict,
             pad: int) -> float:
    """Mean plan loss over the validation set, split across the cards."""
    rank, world = dist.get_rank(), dist.get_world_size()
    total = torch.zeros(2, device=device.torch)
    loss_fn.eval()
    # Sharded, every card must run as many forwards as the others: the one
    # short of an example runs the first one again and doesn't count it.
    turns = math.ceil(len(examples) / world)
    with torch.no_grad(), torch.autocast(device.kind, dtype=getattr(torch, config["dtype"])):
        for turn in range(turns):
            index = turn * world + rank
            example = examples[index] if index < len(examples) else examples[0]
            cool_down(sensor, config)
            ids, labels = device.tensors(example, config["bucket"], pad)
            loss = loss_fn(ids, labels).float()
            if index < len(examples):
                total[0] += loss
                total[1] += 1
            device.step()
    loss_fn.train()
    dist.all_reduce(total)
    return (total[0] / total[1]).item()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    config_path = Path(argv[0])
    config = {**DEFAULTS, **(yaml.safe_load(config_path.read_text()) or {})}
    runs = Path(os.environ.get("WILL_IT_RISCV_RUNS", "/scratch/runs"))
    out = Path(config["out"] or runs / config_path.stem)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = Device(config["device"], local_rank)
    # A hot card pauses between sequences while the other waits in a collective.
    dist.init_process_group(device.backend, timeout=datetime.timedelta(hours=1))
    rank, world = dist.get_rank(), dist.get_world_size()
    sensor = hwmon(local_rank) if device.kind == "cuda" else None

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    import chunked_attention

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data = REPO / config["data"]
    template = config["chat_template_kwargs"] or {}
    train = load(tokenizer, data / "train.jsonl", template)
    valid = load(tokenizer, data / "valid.jsonl", template)

    dtype = getattr(torch, config["dtype"])
    if config["attention"] == "chunked":
        chunked_attention.register(config["attention_block"])
    elif config["attention"] == "fused":
        import hpu_attention

        hpu_attention.register()
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], dtype=dtype, attn_implementation=config["attention"]).to(device.torch)
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    torch.manual_seed(config["seed"])
    model = get_peft_model(model, LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.0,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    if config["init_adapter"]:
        # Training on from a finished run's adapter (rejection-sampling SFT); a resume
        # below replaces these weights with its own.
        from safetensors.torch import load_file

        weights = load_file(str(Path(config["init_adapter"]) / "adapter_model.safetensors"))
        loaded = set_peft_model_state_dict(model, weights)
        # Keys that match nothing would leave the LoRA at its initialisation, silently.
        assert not loaded.unexpected_keys, loaded.unexpected_keys[:5]
        log(f"adapter from {config['init_adapter']}: {len(weights)} tensors")
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:              # LoRA weights train in fp32 under fp16 autocast
        p.data = p.data.float()
        dist.broadcast(p.data, 0)
    if config["fsdp"]:
        loss_fn = shard(model, config["loss"], config["loss_chunk"])
        params = [p for p in loss_fn.parameters() if p.requires_grad]
    else:
        loss_fn = PlanLoss(model, config["loss"], config["loss_chunk"])

    # Every card takes the same number of sequences an epoch; the remainder waits.
    per_rank = len(train) // world
    steps_per_epoch = math.ceil(per_rank / config["accumulation"])
    total_steps = steps_per_epoch * config["epochs"]
    if config["max_steps"]:
        total_steps = min(total_steps, config["max_steps"])
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=0.0)
    schedule = get_cosine_schedule_with_warmup(
        optimizer, math.ceil(config["warmup"] * total_steps), total_steps)
    # Sharded gradients are DTensors, which GradScaler doesn't take: the same
    # dynamic loss scale by hand, the overflow check agreed across the cards.
    # bf16 has float32's range: nothing to scale, but the same checkpoint shape.
    scaler = None if config["fsdp"] else torch.amp.GradScaler(
        device.kind, enabled=config["dtype"] == "float16")
    scale = {"value": 2.0 ** 16, "good": 0}

    step = 0
    state_path = out / "last" / "state.pt"
    if config["fsdp"] and (out / "last" / "extra.json").exists():
        step = load_sharded(out / "last", loss_fn, optimizer, schedule, scale)
        log(f"resumed at step {step}")
    elif not config["fsdp"] and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, state["adapter"])
        for p in params:
            p.data = p.data.float()
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        log(f"resumed at step {step}")
    params = [p for p in model.parameters() if p.requires_grad]

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"{len(train)} train, {len(valid)} valid; {world} {device.kind} devices,"
        f" {steps_per_epoch} steps an epoch, {total_steps} steps; out {out}")

    def save(name: str) -> None:
        if config["fsdp"]:
            save_sharded(out / name, model, loss_fn, optimizer, schedule, scale, step)
            return
        if rank != 0:
            return
        from peft import get_peft_model_state_dict

        target = out / name
        model.save_pretrained(target)
        torch.save({"adapter": get_peft_model_state_dict(model),
                    "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                    "scaler": scaler.state_dict(), "step": step}, target / "state.pt")

    def record(entry: dict) -> None:
        if rank == 0:
            with (out / "log.jsonl").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)

    offload = (lambda: torch.autograd.graph.save_on_cpu(pin_memory=True)) if config["offload"] \
        else contextlib.nullcontext
    while step < total_steps:
        epoch, position = divmod(step, steps_per_epoch)
        order = list(range(len(train)))
        random.Random(config["seed"] + epoch).shuffle(order)
        if config["longest_first"]:
            order.sort(key=lambda i: -len(train[i]["input_ids"]))
        mine = order[rank * per_rank:(rank + 1) * per_rank]
        start = position * config["accumulation"]
        batch = mine[start:start + config["accumulation"]]

        started, tokens, paused, losses = time.time(), 0, 0.0, []
        for index in batch:
            paused += cool_down(sensor, config)
            example = train[index]
            ids, labels = device.tensors(example, config["bucket"], pad)
            with offload(), torch.autocast(device.kind, dtype=dtype):
                loss = loss_fn(ids, labels)
            if scaler is not None:
                scaler.scale(loss / len(batch)).backward()
            else:
                (loss * (scale["value"] / len(batch))).backward()
            device.step()
            # Kept on the device: an item() here would wait for the card every sequence.
            losses.append(loss.detach().float())
            tokens += len(example["input_ids"])
        if scaler is not None:
            average_gradients(params, world)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, config["clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            norm = sharded_norm(params) / scale["value"]   # reduce-scatter already averaged
            if torch.isfinite(norm):
                factor = min(1.0, config["clip"] / (norm.item() + 1e-6)) / scale["value"]
                for p in params:
                    p.grad.mul_(factor)
                optimizer.step()
                scale["good"] += 1
                if scale["good"] % 200 == 0:
                    scale["value"] *= 2
            else:
                scale["value"] /= 2
                scale["good"] = 0
                log(f"overflow at step {step + 1}; loss scale now {scale['value']:g}")
        optimizer.zero_grad(set_to_none=True)
        device.step()
        schedule.step()
        step += 1

        # float32, not 64: a Gaudi has no float64.
        summary = torch.stack([torch.stack(losses).sum(),
                               torch.tensor(float(len(losses)), device=device.torch),
                               torch.tensor(float(tokens), device=device.torch)])
        dist.all_reduce(summary)
        # The cards wait on each other, so the step lost the longest pause of any.
        longest = torch.tensor([paused], device=device.torch)
        dist.all_reduce(longest, op=dist.ReduceOp.MAX)
        paused = longest.item()
        seconds = time.time() - started
        t = temperatures(sensor)
        record({"step": step, "epoch": round(step / steps_per_epoch, 3),
                "loss": round((summary[0] / summary[1]).item(), 4),
                "grad_norm": round(norm.item(), 3), "lr": schedule.get_last_lr()[0],
                "tokens_per_second": round(summary[2].item() / max(seconds - paused, 1e-6)),
                "seconds": round(seconds), "paused": round(paused),
                "peak_gib": device.peak_gib(),
                "temps": t, "time": time.strftime("%H:%M:%S")})
        if step % config["save_every"] == 0 or step == total_steps:
            save("last")
        if step % config["eval_every"] == 0 or step % steps_per_epoch == 0 or step == total_steps:
            record({"step": step, "valid_loss": round(
                evaluate(loss_fn, valid, device, sensor, config, pad), 4)})
            save(f"checkpoint-{step}")
        dist.barrier()

    dist.destroy_process_group()
    return 0


def save_sharded(target: Path, model, loss_fn, optimizer, schedule, scale: dict, step: int) -> None:
    """Sharded state to resume from, and the whole adapter for export.py."""
    import torch.distributed.checkpoint as dcp
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_state_dict)

    model_state, optim_state = get_state_dict(
        loss_fn, optimizer, options=StateDictOptions(ignore_frozen_params=True))
    dcp.save({"model": model_state, "optim": optim_state}, checkpoint_id=str(target / "dcp"))
    full = get_model_state_dict(loss_fn, options=StateDictOptions(
        full_state_dict=True, cpu_offload=True, ignore_frozen_params=True))
    if dist.get_rank() == 0:
        full = {k.removeprefix("model."): v for k, v in full.items()}
        adapter = get_peft_model_state_dict(model, state_dict=full)
        save_file({k: v.contiguous() for k, v in adapter.items()},
                  str(target / "adapter_model.safetensors"))
        model.peft_config["default"].save_pretrained(str(target))
        (target / "extra.json").write_text(json.dumps(
            {"step": step, "schedule": schedule.state_dict(), "scale": scale}) + "\n")
    dist.barrier()


def load_sharded(source: Path, loss_fn, optimizer, schedule, scale: dict) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_state_dict, set_state_dict)

    options = StateDictOptions(ignore_frozen_params=True)
    model_state, optim_state = get_state_dict(loss_fn, optimizer, options=options)
    state = {"model": model_state, "optim": optim_state}
    dcp.load(state, checkpoint_id=str(source / "dcp"))
    # The frozen base weights aren't in the checkpoint; every trainable one must be.
    result = set_state_dict(
        loss_fn, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"],
        options=StateDictOptions(ignore_frozen_params=True, strict=False))
    trainable = {n for n, p in loss_fn.named_parameters() if p.requires_grad}
    missing = trainable & set(result.missing_keys)
    assert not missing and not result.unexpected_keys, (sorted(missing)[:5], result.unexpected_keys[:5])
    extra = json.loads((source / "extra.json").read_text())
    schedule.load_state_dict(extra["schedule"])
    scale.update(extra["scale"])
    return extra["step"]


if __name__ == "__main__":
    raise SystemExit(main())
