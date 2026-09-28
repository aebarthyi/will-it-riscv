"""data/sft -> TRL's conversational prompt-completion format, at one maximum length.

    python training/prepare.py --max-tokens 12288

The prompt is the system prompt and the evidence pack, the completion is the
plan, so the loss falls on the plan only. An example longer than the maximum
is dropped, never truncated: a plan with its end cut off teaches a model to
stop mid-plan. One in 16 training examples of each kind is carved off for
validation -- by a hash of its key, ranked over the whole training set, so
the split holds whatever the maximum and the few script-driven repositories
are represented too; test is kept
exactly as exported, and its long examples are counted, not dropped -- the
evaluation sees every one of them.

Lengths are counted by rendering the chat template to text and tokenizing
that. ``apply_chat_template(tokenize=True)`` returns a dict in current
transformers, and its len() is the number of keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from will_it_riscv import evaluate

REPO = Path(__file__).resolve().parents[1]
VALID_EVERY = 16


def kind(example: dict) -> str:
    """A script-driven repository (git), a PyPI package (sdist, or its repo), or
    a quiz question (will-it-riscv-quiz export), so each keeps its share of valid."""
    if str(example["meta"].get("key", "")).startswith("quiz:"):
        return "quiz"
    return evaluate.kind(example["meta"])


def to_prompt_completion(example: dict) -> dict:
    messages = example["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"], example["meta"]
    return {"prompt": messages[:2], "completion": messages[2:], "meta": example["meta"]}


def measure(tokenizer, row: dict) -> tuple[int, int]:
    """Tokens in the whole sequence, and in the completion the loss falls on."""
    prompt = tokenizer.apply_chat_template(
        row["prompt"], tokenize=False, add_generation_prompt=True)
    whole = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
    assert whole.startswith(prompt), "the template renders the prompt differently alone"
    total = len(tokenizer(whole, add_special_tokens=False)["input_ids"])
    return total, total - len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def validation_keys(rows: list[dict]) -> set[str]:
    """Every 16th of each kind, in the order of a hash of its key."""
    held: set[str] = set()
    for k in ("repo", "pypi"):
        keys = sorted((r["meta"]["key"] for r in rows if r["meta"]["kind"] == k),
                      key=lambda key: hashlib.sha256(key.encode()).hexdigest())
        held.update(keys[::VALID_EVERY])
    return held


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sft", type=Path, default=REPO / "data" / "sft")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="whose tokenizer and chat template to measure with")
    parser.add_argument("--out", type=Path,
                        help="default: data/prepared/<max-tokens>")
    args = parser.parse_args(argv)
    out = args.out or REPO / "data" / "prepared" / str(args.max_tokens)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    stats: dict = {"model": args.model, "max_tokens": args.max_tokens, "dropped": [],
                   "test_over": []}
    exported: dict[str, list[dict]] = {}
    for name in ("train", "test"):
        exported[name] = []
        for line in (args.sft / f"{name}.jsonl").read_text().splitlines():
            row = to_prompt_completion(json.loads(line))
            total, completion = measure(tokenizer, row)
            row["meta"] = {**row["meta"], "kind": kind(row), "tokens": total,
                           "completion_tokens": completion}
            exported[name].append(row)
    held = validation_keys(exported["train"])
    for row in exported["train"]:
        if row["meta"]["tokens"] > args.max_tokens:
            stats["dropped"].append({"key": row["meta"]["key"], "tokens": row["meta"]["tokens"]})
        else:
            splits["valid" if row["meta"]["key"] in held else "train"].append(row)
    for row in exported["test"]:
        if row["meta"]["tokens"] > args.max_tokens:
            stats["test_over"].append(row["meta"]["key"])
        splits["test"].append(row)

    for split, rows in splits.items():
        with (out / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        counts = {k: sum(r["meta"]["kind"] == k for r in rows) for k in ("repo", "pypi")}
        tokens = sorted(r["meta"]["tokens"] for r in rows)
        stats[split] = {
            "examples": len(rows), **counts,
            "tokens": sum(tokens), "median": tokens[len(tokens) // 2] if tokens else 0,
            "max": tokens[-1] if tokens else 0,
            "completion_tokens": sum(r["meta"]["completion_tokens"] for r in rows),
        }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    for split in splits:
        s = stats[split]
        print(f"{split:5s} {s['examples']:4d} examples ({s['repo']} repo, {s['pypi']} pypi)"
              f"  {s['tokens']:>9,} tokens, median {s['median']:,}, max {s['max']:,}"
              f"  plan share {s['completion_tokens'] / max(s['tokens'], 1):.0%}")
    print(f"dropped from train: {len(stats['dropped'])}; "
          f"test over {args.max_tokens}: {len(stats['test_over'])}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
