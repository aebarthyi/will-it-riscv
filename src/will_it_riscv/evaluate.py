"""How good a planner is, against the verified plans of held-out repositories.

Every planner gets the same thing a trained one would: the evidence pack a
repository's example was labelled from. Its plan is scored against the
verified one.

  parsed      the answer is a plan at all
  cited       of its citations, how many hold (the lines exist and say the quote)
  grounded    of its citations, how many are lines the pack showed
  precision   of its steps, how much matches a step of the reference
  recall      of the reference's steps, how much it found
  optional    of the steps both have, how many agree on default versus optional
  order       of the orderings the reference insists on, how many it keeps
  verdict     with --execute: run it, and does it reach the same answer
  required    with --execute: overlap of what its run and the reference's require

Steps match softly. Two system-packages steps match as far as their package
lists overlap; two configures of the same directory match more the more of
their -D flags agree; a python-install matches on its manifest and section.

Baselines: ``autoplan`` (the build-file reader recursion uses today) and
``teacher-first`` (the teacher's plan before any of it was sent back).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from . import modelplan
from .evidence import EvidencePack
from .plan import (
    CMAKE_CONFIGURE,
    MESON_SETUP,
    PYTHON_INSTALL,
    PYTHON_RUN,
    SYSTEM_PACKAGES,
    Plan,
    Step,
    check_evidence,
    load_plan,
    package_spec,
)

Planner = Callable[[Path, EvidencePack, dict], Optional[Plan]]


@dataclass
class Score:
    key: str
    planner: str
    parsed: bool = False
    citations: int = 0
    holding: int = 0
    grounded: int = 0
    precision: float = 0.0
    recall: float = 0.0
    optional: Optional[float] = None
    order: Optional[float] = None
    verdict: Optional[bool] = None
    required: Optional[float] = None

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def cited(self) -> Optional[float]:
        return self.holding / self.citations if self.citations else None

    @property
    def grounding(self) -> Optional[float]:
        return self.grounded / self.citations if self.citations else None


# ------------------------------------------------------------------ matching


def _names(packages: list[str]) -> set:
    return {(package_spec(p) or (p, None, None))[0] for p in packages}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _directory(path: str) -> str:
    path = path.strip().rstrip("/")
    return "." if path in ("", ".", "./") else path.removeprefix("./")


def similarity(a: Step, b: Step) -> float:
    """How far two steps are the same step, from 0 to 1."""
    if a.kind != b.kind:
        return 0.0
    if a.kind == PYTHON_INSTALL:
        if a.manifest is not None and b.manifest is not None:
            if _directory(a.manifest) != _directory(b.manifest):
                return 0.0
            return 1.0 if a.section == b.section else 0.5
        return _jaccard(_names(a.packages), _names(b.packages)) if a.packages or b.packages else 0.0
    if a.kind == SYSTEM_PACKAGES:
        return _jaccard(_names(a.packages), _names(b.packages))
    if a.kind in (CMAKE_CONFIGURE, MESON_SETUP):
        if _directory(a.source) != _directory(b.source):
            return 0.0
        flags = _jaccard(set(a.defines.items()), set(b.defines.items()))
        return 0.5 + 0.5 * flags
    if a.kind == PYTHON_RUN:
        if _directory(a.script or "") != _directory(b.script or ""):
            return 0.0
        return 0.5 + 0.5 * _jaccard(set(a.args), set(b.args))
    return 0.0


def match(candidate: list[Step], reference: list[Step]) -> list[tuple[int, int, float]]:
    """The best one-to-one pairing, greedily, most similar first."""
    pairs = sorted(
        (
            (similarity(c, r), i, j)
            for i, c in enumerate(candidate) for j, r in enumerate(reference)
        ),
        reverse=True,
    )
    used_c: set = set()
    used_r: set = set()
    matched = []
    for value, i, j in pairs:
        if value <= 0 or i in used_c or j in used_r:
            continue
        used_c.add(i)
        used_r.add(j)
        matched.append((i, j, value))
    return matched


def _before(plan: Plan) -> set:
    """Every (a, b) where the plan says a must run before b, transitively."""
    direct = {s.id: set(s.after) for s in plan.steps}
    closure: set = set()
    for step in plan.steps:
        pending = list(direct.get(step.id, ()))
        seen: set = set()
        while pending:
            earlier = pending.pop()
            if earlier in seen:
                continue
            seen.add(earlier)
            closure.add((earlier, step.id))
            pending += list(direct.get(earlier, ()))
    return closure


def score(
    candidate: Optional[Plan], reference: Plan, root: Path, pack: EvidencePack,
    key: str = "", planner: str = "",
) -> Score:
    result = Score(key=key, planner=planner)
    if candidate is None:
        return result
    result.parsed = True
    cited = [e for s in candidate.steps for e in s.evidence]
    result.citations = len(cited)
    problems = check_evidence(Plan(repo=candidate.repo, steps=candidate.steps), root)
    result.holding = max(0, len(cited) - len(problems))
    result.grounded = sum(1 for e in cited if pack.shows(e.path, e.start, e.end))

    pairs = match(candidate.steps, reference.steps)
    total = sum(value for _, _, value in pairs)
    result.precision = total / len(candidate.steps) if candidate.steps else 0.0
    result.recall = total / len(reference.steps) if reference.steps else 0.0
    if pairs:
        agree = sum(
            1 for i, j, _ in pairs if candidate.steps[i].optional == reference.steps[j].optional
        )
        result.optional = agree / len(pairs)
        to_candidate = {reference.steps[j].id: candidate.steps[i].id for i, j, _ in pairs}
        wanted = [(a, b) for a, b in _before(reference) if a in to_candidate and b in to_candidate]
        if wanted:
            order = [s.id for s in candidate.ordered()]
            position = {step_id: index for index, step_id in enumerate(order)}
            kept = sum(
                1 for a, b in wanted
                if position[to_candidate[a]] < position[to_candidate[b]]
            )
            result.order = kept / len(wanted)
    return result


# ------------------------------------------------------------------ planners


def autoplan_planner(root: Path, pack: EvidencePack, meta: dict) -> Optional[Plan]:
    from .autoplan import auto_plan

    return auto_plan(root, meta["entry"]["name"])


def teacher_first(directory: Path) -> Planner:
    """The teacher's first answer, before anything was sent back."""

    def planner(root: Path, pack: EvidencePack, meta: dict) -> Optional[Plan]:
        transcript = json.loads((directory / "transcript.json").read_text())
        if not transcript:
            return None
        plan, _, _ = modelplan.check(transcript[0]["text"], root, pack)
        return plan

    return planner


def model_planner(backend: modelplan.Backend, rounds: int = 1) -> Planner:
    def planner(root: Path, pack: EvidencePack, meta: dict) -> Optional[Plan]:
        outcome = modelplan.plan_repository(root, backend, pack=pack, rounds=rounds)
        return outcome.plan

    return planner


# ------------------------------------------------------------------ running


def evaluate(
    out: Path, planners: dict[str, Planner], split: str = "test", execute: bool = False,
    python: str = "3.12", progress=print,
) -> list[Score]:
    from . import dataset

    env = dataset.environment(python) if execute else None
    scores: list[Score] = []
    for directory in sorted(p for p in out.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("tier") != dataset.VERIFIED or meta.get("split") != split:
            continue
        root = Path(meta["root"])
        if not root.is_dir():
            progress(f"{directory.name}: its tree is gone from {root}; skipped")
            continue
        reference = load_plan(directory / "plan.json")
        pack = EvidencePack.from_dict(json.loads((directory / "pack.json").read_text()))
        reference_run = None
        if (directory / "run.json").is_file():
            reference_run = json.loads((directory / "run.json").read_text())
        for name, planner in {**planners, "teacher-first": teacher_first(directory)}.items():
            try:
                candidate = planner(root, pack, meta)
            except Exception as exc:  # noqa: BLE001 - a planner failing scores zero
                progress(f"{directory.name}: {name} failed: {exc}")
                candidate = None
            result = score(candidate, reference, root, pack, directory.name, name)
            if env is not None and candidate is not None and reference_run is not None:
                ran = dataset.summary(dataset.run_plan(candidate, root, env))
                result.verdict = ran["verdict"] == reference_run["verdict"]
                result.required = _overlap(set(ran["required"]), set(reference_run["required"]))
            scores.append(result)
    return scores


def _overlap(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def table(scores: list[Score]) -> list[dict]:
    """One row per planner: every metric, averaged over the repositories."""
    rows = []
    for planner in dict.fromkeys(s.planner for s in scores):
        mine = [s for s in scores if s.planner == planner]

        def mean(values: list) -> Optional[float]:
            present = [v for v in values if v is not None]
            return round(sum(present) / len(present), 3) if present else None

        rows.append({
            "planner": planner,
            "repos": len(mine),
            "parsed": mean([1.0 if s.parsed else 0.0 for s in mine]),
            "cited": mean([s.cited for s in mine]),
            "grounded": mean([s.grounding for s in mine]),
            "precision": mean([s.precision for s in mine]),
            "recall": mean([s.recall for s in mine]),
            "f1": mean([s.f1 for s in mine]),
            "optional": mean([s.optional for s in mine]),
            "order": mean([s.order for s in mine]),
            "verdict": mean([None if s.verdict is None else float(s.verdict) for s in mine]),
            "required": mean([s.required for s in mine]),
        })
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="will-it-riscv-eval",
        description="Score planners against the verified plans of held-out repositories.",
    )
    parser.add_argument("--out", required=True, type=Path, help="a labelled dataset directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--planner", choices=("claude-cli", "anthropic", "openai"),
                        help="a model to score beside the baselines")
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-url")
    parser.add_argument("--rounds", type=int, default=1,
                        help="let the model revise this many times (default: 1, none)")
    parser.add_argument("--execute", action="store_true",
                        help="also run every plan, and compare what the runs require")
    parser.add_argument("--json", type=Path, help="write every score here")
    parser.add_argument("--python", default="3.12")
    args = parser.parse_args(argv)

    planners: dict[str, Planner] = {"autoplan": autoplan_planner}
    if args.planner:
        backend = modelplan.backend(args.planner, args.planner_model, args.planner_url)
        planners[backend.name] = model_planner(backend, args.rounds)
    scores = evaluate(
        args.out, planners, args.split, args.execute, args.python,
        progress=lambda m: print(m, file=sys.stderr),
    )
    rows = table(scores)
    columns = ["planner", "repos", "parsed", "cited", "grounded", "precision", "recall", "f1",
               "optional", "order", "verdict", "required"]
    print("  ".join(f"{c:>9}" for c in columns))
    for row in rows:
        print("  ".join(f"{'-' if row[c] is None else row[c]!s:>9}" for c in columns))
    if args.json:
        args.json.write_text(json.dumps(
            {"summary": rows, "scores": [asdict(s) | {"f1": s.f1} for s in scores]}, indent=2,
        ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
