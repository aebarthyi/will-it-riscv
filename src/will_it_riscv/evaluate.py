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
  required    with --execute: overlap of what its run and the reference's are
              shown to need -- by a configure that ran to the end, or the
              project's own manifest -- each dependency once; not scored where
              the reference's run shows nothing (planrun.proven), unless its
              default configures all ran to the end: then nothing is the answer
  req_all     the same over everything required, the plan's own package lists
              included: which apt lines were copied as much as what is needed
  tokens      what the example cost in model tokens, in and out: the planner's,
              and with --unblock the unblocker agent's (in thousands, per repo)
  fresh       the same less what was read back from a prompt cache: the tokens
              a model had to take in for the first time, and write
  usd         the same in dollars, where the backend says

Each score also keeps how long its planner and run took (``seconds``), which
is what orders a later --jobs run slowest first.

Steps match softly. Two system-packages steps match as far as their package
lists overlap; two configures of the same directory match more the more of
their -D flags agree; a python-install matches on its manifest and section.

Baselines: ``autoplan`` (the build-file reader recursion uses today) and
``teacher-first`` (the teacher's plan before any of it was sent back).

The table gives every planner a row over all of them, then one for each
kind: script-driven repositories (``repo``) and PyPI packages (``pypi``).
There are far more of the second, and the first is where a model has to
beat reading build files, so an average over both hides what matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Collection, Optional

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
    kind: str = ""
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
    required_all: Optional[float] = None
    plan_check: list = field(default_factory=list)
    """With --execute: what checking its run against the plan found -- an
    option moved off its default, a configure the driver never ran."""
    seconds: Optional[float] = None
    planner_tokens_in: int = 0
    planner_tokens_out: int = 0
    planner_usd: Optional[float] = None
    planner_tokens_cached: int = 0
    unblock_sessions: int = 0
    unblock_replays: int = 0
    unblock_tokens_in: int = 0
    unblock_tokens_out: int = 0
    unblock_usd: float = 0.0
    unblock_tokens_cached: int = 0

    @property
    def tokens(self) -> int:
        return (self.planner_tokens_in + self.planner_tokens_out
                + self.unblock_tokens_in + self.unblock_tokens_out)

    @property
    def fresh(self) -> int:
        """Tokens less those read from a prompt cache: an agent that takes
        many turns reads its context again on each, at a tenth of the price."""
        return self.tokens - self.planner_tokens_cached - self.unblock_tokens_cached

    @property
    def usd(self) -> Optional[float]:
        if self.planner_usd is None and not self.unblock_usd:
            return None
        return (self.planner_usd or 0.0) + self.unblock_usd

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


def kind(meta: dict) -> str:
    """A script-driven repository (git) or a PyPI package (sdist, or its repo)."""
    return "repo" if (meta.get("origin") or "").startswith("git ") else "pypi"


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


def model_planner(
    backend: modelplan.Backend, rounds: int = 1, transcripts: Optional[Path] = None,
    env=None,
) -> Planner:
    """``transcripts``: keep each answer and what was sent back, one file a key.

    ``env``: run each plan that holds, and send back what running it showed
    was the plan's fault, as labelling does and as the tool itself would. The
    last run is left in ``meta["run"]``, as ``(plan, result)``, so it is not
    run twice.
    """

    def planner(root: Path, pack: EvidencePack, meta: dict) -> Optional[Plan]:
        verify = None
        if env is not None:
            from . import dataset

            def verify(plan: Plan) -> list[str]:
                result = dataset.run_plan(plan, root, env)
                meta["run"] = (plan, result)
                return dataset.plan_faults(result)

        outcome = modelplan.plan_repository(
            root, backend, pack=pack, rounds=rounds, verify=verify)
        meta["usage"] = (*outcome.tokens, outcome.cost_usd, outcome.cached_tokens)
        if transcripts is not None:
            transcripts.mkdir(parents=True, exist_ok=True)
            (transcripts / f"{meta['key']}.json").write_text(json.dumps({
                "backend": outcome.backend,
                "error": outcome.error,
                "attempts": [
                    {"text": a.text, "problems": a.problems, "moved": a.moved,
                     "output_tokens": a.completion.output_tokens if a.completion else None}
                    for a in outcome.attempts
                ],
            }, indent=2))
        return outcome.plan

    return planner


def saved_planner(directories: list[Path]) -> Planner:
    """The last answer a model gave, kept by an earlier run's --transcripts:
    scoring it again after the executor or the measures change costs no model
    call. With several directories, a later one's answer for a key wins."""

    def planner(root: Path, pack: EvidencePack, meta: dict) -> Optional[Plan]:
        for directory in reversed(directories):
            kept = directory / f"{meta['key']}.json"
            if kept.is_file():
                # As plan_repository kept it: the last answer that parsed.
                for attempt in reversed(json.loads(kept.read_text()).get("attempts") or []):
                    plan, _, _ = modelplan.check(attempt["text"], root, pack)
                    if plan is not None:
                        return plan
                return None
        return None

    return planner


def agent_planner(agent, env) -> Planner:
    """The agent (:mod:`agent`): it plans, runs, fixes and stops on its own;
    what it spent goes in ``meta["usage"]``. Its fixes at stuck points are
    kept for replay, so scoring its plan again (``--unblock replay``) is free."""

    def planner(root: Path, pack: EvidencePack, meta: dict) -> Optional[Plan]:
        name = (meta.get("entry") or {}).get("name") or meta["key"]
        outcome = agent(root, pack, name, arch=env.target.arch, distro=env.distro.spec.id,
                        python=".".join(map(str, env.target.python_version)),
                        distro_label=env.distro.spec.label)
        meta["usage"] = (outcome.tokens_in, outcome.tokens_out, outcome.cost,
                         outcome.tokens_cached)
        return outcome.plan

    return planner


# ------------------------------------------------------------------ running


def evaluate(
    out: Path, planners: dict[str, Planner], split: str = "test", execute: bool = False,
    python: str = "3.12", progress=print, keys: Optional[Collection[str]] = None,
    env=None, baselines: bool = True, jobs: int = 1,
) -> list[Score]:
    """``keys``: only these examples; with ``jobs`` above 1 they start in this
    order, so put the slow ones first and the last few don't run alone.

    ``jobs``: examples at once, each in its own forked process: a run is
    mostly CMake and parsing its trace, and one slow repository then holds
    one worker, not the whole evaluation.
    """
    from . import dataset

    if execute and env is None:
        env = dataset.environment(python)
    if not execute:
        env = None
    wanted = set(keys) if keys is not None else None
    directories = []
    for directory in sorted(p for p in out.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("tier") != dataset.VERIFIED or meta.get("split") != split:
            continue
        if wanted is not None and directory.name not in wanted:
            continue
        directories.append(directory)
    if jobs <= 1:
        return [s for d in directories for s in _example(d, planners, env, baselines, progress)]

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    global _WORK
    if keys is not None:
        start = {key: i for i, key in enumerate(keys)}
        directories.sort(key=lambda d: start[d.name])
    if env is not None:
        _warm(env)
    _WORK = (planners, env, baselines, progress)
    results: dict[str, list[Score]] = {}
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork"),
                             initializer=_fresh_clients) as pool:
        running = {pool.submit(_work, d): d.name for d in directories}
        for done, future in enumerate(as_completed(running), 1):
            key = running[future]
            try:
                results[key] = future.result()
            except Exception as exc:  # noqa: BLE001 - a worker dying loses one example
                progress(f"{key}: failed: {exc}")
                results[key] = []
            progress(f"{done}/{len(directories)} done: {key}")
    return [s for d in sorted(directories) for s in results[d.name]]


_WORK: tuple = ()


def _work(directory: Path) -> list[Score]:
    return _example(directory, *_WORK)


def _fresh_clients() -> None:
    """A worker's own HTTP connections: forked ones are the parent's sockets."""
    import httpx

    env = _WORK[1]
    if env is None:
        return
    client = httpx.Client(timeout=60, follow_redirects=True)
    env.index.client = env.distro.client = client
    reference = getattr(env.distro, "_reference_index", None)
    if reference is not None:
        reference.client = client


def _warm(env) -> None:
    """Load the archive's lists before forking, so the workers share one copy."""
    distro = env.distro
    distro.names()
    distro.version("")
    distro.contents()
    distro.named("")


def _example(directory: Path, planners: dict[str, Planner], env, baselines: bool,
             progress) -> list[Score]:
    import time

    from . import dataset, unblock

    meta = json.loads((directory / "meta.json").read_text())
    found = dataset.tree(meta)
    if found.root is None:
        progress(f"{directory.name}: its tree is {found.error}; skipped "
                 f"(will-it-riscv-dataset check --fetch brings it back)")
        return []
    root = found.root
    reference = load_plan(directory / "plan.json")
    pack = EvidencePack.from_dict(json.loads((directory / "pack.json").read_text()))
    reference_run = None
    recorded = dataset.run_file(directory, env.distro.spec.id if env else dataset.LABEL_DISTRO)
    if recorded.is_file():
        reference_run = json.loads(recorded.read_text())
    elif env is not None:
        progress(f"{directory.name}: no reference run in {recorded.name}; "
                 "verdict and required are not scored")
    everyone = dict(planners)
    if baselines:
        everyone["teacher-first"] = teacher_first(directory)
    scores = []
    for name, planner in everyone.items():
        began = time.monotonic()
        unblock.reset_spent()
        context = {**meta, "key": directory.name}
        try:
            candidate = planner(root, pack, context)
        except Exception as exc:  # noqa: BLE001 - a planner failing scores zero
            progress(f"{directory.name}: {name} failed: {exc}")
            candidate = None
        result = score(candidate, reference, root, pack, directory.name, name)
        result.kind = kind(meta)
        if env is not None and candidate is not None and reference_run is not None:
            done = context.get("run")
            if done is not None and done[0] is candidate:
                ran = dataset.summary(done[1])
            else:
                ran = dataset.summary(dataset.run_plan(candidate, root, env))
            result.verdict = ran["verdict"] == reference_run["verdict"]
            result.required, result.required_all = required_overlap(
                ran, reference_run, shown=configured(reference, reference_run))
            result.plan_check = ran.get("plan_check", [])
            if "proven" not in reference_run:
                progress(f"{directory.name}: its reference run predates proven requirements; "
                         "required is not scored (will-it-riscv-dataset rerun)")
        if "usage" in context:
            (result.planner_tokens_in, result.planner_tokens_out, result.planner_usd,
             result.planner_tokens_cached) = context["usage"]
        agent = unblock.spent()
        result.unblock_sessions, result.unblock_replays = agent["sessions"], agent["replays"]
        result.unblock_tokens_in = agent["tokens_in"]
        result.unblock_tokens_out = agent["tokens_out"]
        result.unblock_usd = agent["cost"]
        result.unblock_tokens_cached = agent["tokens_cached"]
        result.seconds = round(time.monotonic() - began, 1)
        scores.append(result)
    return scores


def required_overlap(
    ran: dict, reference: dict, shown: bool = False,
) -> tuple[Optional[float], float]:
    """``(proven, all)``: how far two runs agree on what the default build is
    shown to need, and on everything they require. The first is None where
    the reference's run was recorded before runs said, or shows nothing --
    unless ``shown``: its configures ran to the end and needed nothing, as
    amqp-cpp's do, and claiming something is then wrong."""
    everything = _overlap(set(ran["required"]), set(reference["required"]))
    if "proven" not in reference or not (reference["proven"] or shown):
        return None, everything
    return _overlap(set(ran.get("proven", ())), set(reference["proven"])), everything


def configured(plan: Plan, run: dict) -> bool:
    """Whether a run configured the plan's default build to the end: it has
    a default configure, and every one completed."""
    status = {s["id"]: s["status"] for s in run.get("steps", ())}
    configures = [s for s in plan.steps
                  if not s.optional and s.kind in (CMAKE_CONFIGURE, MESON_SETUP)]
    return bool(configures) and all(status.get(s.id) == "completed" for s in configures)


def _overlap(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def table(scores: list[Score]) -> list[dict]:
    """Per planner, a row over every repository and one for each kind."""
    rows = []
    groups = [
        (planner, group)
        for planner in dict.fromkeys(s.planner for s in scores)
        for group in ("all", *sorted({s.kind for s in scores}))
    ]
    for planner, group in groups:
        mine = [s for s in scores if s.planner == planner and group in ("all", s.kind)]
        if not mine:
            continue

        def mean(values: list) -> Optional[float]:
            present = [v for v in values if v is not None]
            return round(sum(present) / len(present), 3) if present else None

        rows.append({
            "planner": planner,
            "kind": group,
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
            "req_all": mean([s.required_all for s in mine]),
            "ktokens": mean([s.tokens / 1000 for s in mine]),
            "kfresh": mean([s.fresh / 1000 for s in mine]),
            "usd": mean([s.usd for s in mine]),
            "sessions": mean([float(s.unblock_sessions) for s in mine]),
        })
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="will-it-riscv-eval",
        description="Score planners against the verified plans of held-out repositories.",
    )
    parser.add_argument("--out", required=True, type=Path, help="a labelled dataset directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--planner",
                        choices=("claude-cli", "anthropic", "openai", "agent", "saved"),
                        help="a model to score beside the baselines; agent: Claude (via claude "
                             "-p, --planner-model) planning with tools, needs --execute; saved: "
                             "the last answers kept in --saved")
    parser.add_argument("--saved", metavar="DIR[,DIR...]",
                        help="for --planner saved: --transcripts directories of earlier runs; "
                             "a later one's answer for a key wins")
    parser.add_argument("--planner-model")
    parser.add_argument("--lean-pack", action="store_true",
                        help="for agent: name what CI shows (workflows, Dockerfiles, the scripts "
                             "they call) instead of showing it; it reads what it needs")
    parser.add_argument("--planner-url",
                        help="for openai: the server, or several of the same model, comma-separated")
    parser.add_argument("--planner-slots", type=int, default=1,
                        help="requests each server takes at once (llama-server's -np)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="let the model revise this many times (default: 1, none)")
    parser.add_argument("--execute", action="store_true",
                        help="also run every plan, and compare what the runs require")
    parser.add_argument("--run-feedback", action="store_true",
                        help="with --execute: run each plan the model writes, and send back "
                             "what the run shows is wrong with it, as labelling does")
    parser.add_argument("--no-baselines", action="store_true",
                        help="score only --planner, not autoplan and teacher-first")
    parser.add_argument("--json", type=Path, help="write every score here")
    parser.add_argument("--transcripts", type=Path,
                        help="keep what the model wrote, one file per repository, here")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--distro", default="debian:trixie",
                        help="with --execute: the archive plans run against; the references "
                             "must have been run against it too (will-it-riscv-dataset rerun "
                             "--distro)")
    parser.add_argument("--keys", type=Path,
                        help="score only these examples of the split: a file, one key a line; "
                             "with --jobs they start in its order, so list the slow ones first")
    parser.add_argument("--jobs", type=int, default=1,
                        help="examples to score at once, each in its own process")
    parser.add_argument("--unblock", metavar="claude[:MODEL]",
                        help="with --execute: hand the configures the executor's rules cannot get "
                             "past to an agent (see will_it_riscv.unblock), and count what it "
                             "spends; the references should have been run the same way")
    args = parser.parse_args(argv)

    if args.run_feedback and not args.execute:
        parser.error("--run-feedback needs --execute")
    if args.no_baselines and not args.planner:
        parser.error("--no-baselines needs --planner")
    if (args.planner == "saved") != bool(args.saved):
        parser.error("--planner saved and --saved go together")
    if args.unblock:
        if not args.execute:
            parser.error("--unblock needs --execute")
        import os

        # Read by every configure, in this process and the workers it forks.
        os.environ["WILL_IT_RISCV_UNBLOCK"] = args.unblock
    env = None
    if args.execute:
        from . import dataset

        env = dataset.environment(args.python, distro=args.distro)
    planners: dict[str, Planner] = {} if args.no_baselines else {"autoplan": autoplan_planner}
    if args.planner == "agent":
        if env is None:
            parser.error("--planner agent needs --execute")
        from .agent import ClaudeAgent

        model = args.planner_model or "sonnet"
        name = f"agent-lean:{model}" if args.lean_pack else f"agent:{model}"
        planners[name] = agent_planner(ClaudeAgent(model=model, lean=args.lean_pack), env)
    elif args.planner == "saved":
        saved = [Path(d) for d in args.saved.split(",")]
        planners[f"saved:{saved[0].name}"] = saved_planner(saved)
    elif args.planner:
        backend = modelplan.backend(
            args.planner, args.planner_model, args.planner_url, args.planner_slots)
        planners[backend.name] = model_planner(
            backend, args.rounds, args.transcripts, env if args.run_feedback else None)
    keys = list(dict.fromkeys(args.keys.read_text().split())) if args.keys else None
    scores = evaluate(
        args.out, planners, args.split, args.execute, args.python,
        progress=lambda m: print(m, file=sys.stderr), keys=keys, env=env,
        baselines=not args.no_baselines, jobs=args.jobs,
    )
    rows = table(scores)
    columns = ["planner", "kind", "repos", "parsed", "cited", "grounded", "precision", "recall", "f1",
               "optional", "order", "verdict", "required", "req_all", "ktokens", "kfresh", "usd",
               "sessions"]
    print("  ".join(f"{c:>9}" for c in columns))
    for row in rows:
        print("  ".join(f"{'-' if row[c] is None else row[c]!s:>9}" for c in columns))
    if args.json:
        args.json.write_text(json.dumps(
            {"summary": rows, "scores": [
                asdict(s) | {"f1": s.f1, "tokens": s.tokens, "fresh": s.fresh, "usd": s.usd}
                for s in scores]},
            indent=2,
        ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
