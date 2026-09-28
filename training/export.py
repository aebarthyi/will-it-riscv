"""A trained LoRA -> one GGUF file that llama-server serves.

    python training/export.py /scratch/runs/qwen3-4b-12k/checkpoint-54
    python training/export.py --base Qwen/Qwen3-4B-Instruct-2507    # the base, for the zero-shot row

The adapter is merged into its base model (``merge_and_unload``) in fp32 on
the CPU -- the LoRA trained in fp32, and merging into fp16 weights would round
each delta before it is added -- then saved in fp16 with the base's tokenizer
and chat template. llama.cpp's ``convert_hf_to_gguf.py`` writes the GGUF,
quantised to Q8_0 directly. It runs under this venv, with llama.cpp's own
gguf-py: its requirements file pins a CPU torch that would replace ours.

The merged checkpoint is deleted once the GGUF is written, unless --keep-merged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_CPP = Path(os.environ.get("LLAMA_CPP", "/scratch/llama.cpp"))
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def merge(adapter: Path | None, base: str, out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32, device_map="cpu")
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model = model.to(torch.float16)
    model.save_pretrained(out, safe_serialization=True)
    # The base's own tokenizer files: transformers re-saves only tokenizer.json,
    # and the converter's BPE path reads vocab.json and merges.txt.
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(base, allow_patterns=TOKENIZER_FILES))
    for name in TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copy(source / name, out / name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adapter", type=Path, nargs="?",
                        help="a checkpoint directory from training/train.py; none exports the base")
    parser.add_argument("--base", help="default: the adapter's base_model_name_or_path")
    parser.add_argument("--out", type=Path,
                        help="the .gguf to write; default: <adapter>/<name>-q8_0.gguf,"
                             " or /scratch/runs/base/<model>-q8_0.gguf")
    parser.add_argument("--outtype", default="q8_0", choices=("q8_0", "f16"))
    parser.add_argument("--keep-merged", action="store_true")
    args = parser.parse_args(argv)

    if args.adapter is None and args.base is None:
        parser.error("give an adapter, or --base to export the base model alone")
    base = args.base
    if base is None:
        base = json.loads((args.adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    if args.out is None:
        if args.adapter is not None:
            name = f"{args.adapter.resolve().parent.name}-{args.adapter.name}"
            args.out = args.adapter / f"{name}-{args.outtype}.gguf"
        else:
            args.out = Path("/scratch/runs/base") / f"{base.split('/')[-1]}-{args.outtype}.gguf"
    converter = LLAMA_CPP / "convert_hf_to_gguf.py"
    if not converter.exists():
        parser.error(f"no {converter}; set LLAMA_CPP to a llama.cpp checkout")

    merged = args.out.with_suffix(".merged")
    print(f"merging {args.adapter or '(no adapter)'} into {base} -> {merged}", flush=True)
    merge(args.adapter, base, merged)
    print(f"converting -> {args.out}", flush=True)
    subprocess.run([sys.executable, str(converter), str(merged), "--outtype", args.outtype,
                    "--outfile", str(args.out)], check=True)
    if not args.keep_merged:
        shutil.rmtree(merged)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
