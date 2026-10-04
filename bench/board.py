"""The scoreboard: planners side by side, from will-it-riscv-eval --json files.

    python bench/board.py RUN.json[,RERUN.json] [RUN2.json[,...]] ... [--against PLANNER]

Each argument is one eval's scores; a comma joins files of one eval, later
ones replacing an example's earlier scores (a rerun of what failed). Every
planner in them gets a row per kind (all, repo, pypi): accuracy -- verdict,
required overlap, F1 -- and what it spent per example: thousands of tokens,
dollars, unblocker sessions. Only examples every planner scored are counted,
so the rows compare like with like; an unparsed plan counts as wrong.

--against PLANNER adds, for every other planner, the paired difference from
it on each measure, with a 95% bootstrap interval over examples.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

MEASURES = ("verdict", "required", "f1", "tokens", "usd", "unblock_sessions")


def load(arguments: list[str]) -> dict:
    """planner -> key -> score."""
    table: dict = defaultdict(dict)
    for argument in arguments:
        for path in argument.split(","):
            for score in json.load(open(path))["scores"]:
                table[score["planner"]][score["key"]] = score
    return table


def value(score: dict, measure: str) -> float:
    if measure == "tokens":
        return score.get("tokens", 0) / 1000
    got = score.get(measure)
    if got is None:
        return 0.0
    return float(got)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def interval(diffs: list[float], rounds: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(diffs) for _ in diffs]) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--against")
    args = parser.parse_args()
    table = load(args.runs)
    shared = set.intersection(*(set(scores) for scores in table.values()))
    kinds = {key: next(iter(table.values()))[key].get("kind", "") for key in shared}
    print(f"{len(shared)} examples scored by all of {', '.join(table)}")
    header = f"{'planner':<22}{'kind':<6}{'n':>4}" + "".join(f"{m[:9]:>10}" for m in MEASURES)
    print(header)
    for planner, scores in table.items():
        for kind in ("all", "repo", "pypi"):
            keys = sorted(k for k in shared if kind == "all" or kinds[k] == kind)
            if not keys:
                continue
            row = [mean([value(scores[k], m) for k in keys]) for m in MEASURES]
            print(f"{planner:<22}{kind:<6}{len(keys):>4}" + "".join(f"{v:>10.3f}" for v in row))
    if args.against:
        base = table[args.against]
        print(f"\npaired, minus {args.against} (95% bootstrap):")
        for planner, scores in table.items():
            if planner == args.against:
                continue
            for kind in ("all", "repo"):
                keys = sorted(k for k in shared if kind == "all" or kinds[k] == kind)
                parts = []
                for measure in MEASURES:
                    diffs = [value(scores[k], measure) - value(base[k], measure) for k in keys]
                    low, high = interval(diffs)
                    parts.append(f"{measure} {mean(diffs):+.3f} [{low:+.3f}, {high:+.3f}]")
                print(f"  {planner} {kind}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
