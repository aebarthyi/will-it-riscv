"""A dataset of verified plans, for teaching a small model to write them.

Labels are cheap here because the executor is the verifier. A teacher
writes a plan from a repository's evidence pack; the plan is checked --
does it parse, does every citation hold, does it cite only what the pack
showed -- and then run in the pretend environment. What went wrong because
of the plan (a manifest that is not there, a configure pointed at a
directory without a CMakeLists.txt, a driver that configured something the
plan does not list) goes back to the teacher, once more. What survives is
a training example: the pack in, the plan out.

  label     fetch each repository in a corpus, label it, verify it
  export    write the verified examples as chat-format JSONL for SFT
  status    how far a labelling run has got, and what it cost
  seed-pypi a corpus of compiled PyPI packages with no riscv64 wheel

A corpus is JSONL, one repository a line:

  {"name": "mfc", "git": "https://github.com/MFlowCode/MFC"}
  {"name": "gromacs", "git": "https://github.com/gromacs/gromacs", "ref": "v2025.3"}
  {"name": "numpy", "pypi": "numpy", "version": "2.5.3"}
  {"name": "mine", "path": "/src/mine", "split": "test"}

Every example lands in its own directory -- pack, plan, transcript, what
running it showed -- so a run can stop and pick up where it left off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from . import evidence, modelplan
from .plan import Plan, load_plan, plan_to_dict

#: One in this many repositories is held out for evaluation, by name.
TEST_EVERY = 10

VERIFIED = "verified"
PARTIAL = "partial"
REJECTED = "rejected"
TEACHER_ERROR = "error"
"""The teacher could not answer -- a usage limit, a network failure. Not a
label: the next run tries it again."""
#: Teacher failures in a row that stop a run: past this, it is not one repository.
MAX_TEACHER_ERRORS = 2


@dataclass
class Entry:
    name: str
    git: Optional[str] = None
    ref: Optional[str] = None
    pypi: Optional[str] = None
    version: Optional[str] = None
    path: Optional[str] = None
    split: Optional[str] = None

    @property
    def key(self) -> str:
        """A directory name: the name, and the version or ref when there is one."""
        tail = self.version or self.ref
        raw = f"{self.name}-{tail}" if tail else self.name
        return re.sub(r"[^A-Za-z0-9_.+-]", "_", raw)

    @property
    def held_out(self) -> str:
        if self.split:
            return self.split
        digest = int(hashlib.sha256(self.name.lower().encode()).hexdigest(), 16)
        return "test" if digest % TEST_EVERY == 0 else "train"


def load_corpus(path: Path) -> list[Entry]:
    entries = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
            entry = Entry(**raw)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"{path}:{number}: not a corpus entry: {exc}") from exc
        if sum(bool(x) for x in (entry.git, entry.pypi, entry.path)) != 1:
            raise SystemExit(f"{path}:{number}: an entry names one of git, pypi or path")
        entries.append(entry)
    return entries


# ------------------------------------------------------------------ fetching


@dataclass
class Fetched:
    root: Optional[Path]
    origin: str = ""
    error: Optional[str] = None


def fetch(entry: Entry, cache_root: Path, index=None) -> Fetched:
    """The repository's tree, from git, from the index, or from disk."""
    if entry.path:
        root = Path(entry.path).expanduser()
        return Fetched(root, str(root)) if root.is_dir() else Fetched(None, error="no such dir")
    if entry.git:
        return _git(entry, cache_root / "git")
    from .sources import fetch as fetch_source

    assert entry.pypi
    if index is None:
        return Fetched(None, error="a pypi entry needs the package index")
    version = entry.version or _latest(index, entry.pypi)
    if version is None:
        return Fetched(None, error=f"{entry.pypi} is not on the index")
    tree = fetch_source(index, entry.pypi, version, cache_root / "sources")
    if not tree.ok:
        return Fetched(None, error=tree.error)
    entry.version = version
    return Fetched(tree.path, f"{tree.kind} {tree.origin}")


def _latest(index, name: str) -> Optional[str]:
    project = index.project(name)
    if project is None or not project.releases:
        return None
    stable = [v for v in project.releases if not v.is_prerelease]
    return str(max(stable or project.releases))


def _git(entry: Entry, cache: Path) -> Fetched:
    assert entry.git
    destination = cache / entry.key
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if not (destination / ".git").is_dir():
        command = ["git", "clone", "--quiet", "--depth", "1"]
        if entry.ref:
            command += ["--branch", entry.ref]
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.run(
                [*command, entry.git, str(destination)], capture_output=True, text=True,
                timeout=900, env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return Fetched(None, error=f"git clone: {exc}")
        if process.returncode != 0:
            tail = (process.stderr or "").strip().splitlines()
            return Fetched(None, error=f"git clone {entry.git}: {tail[-1] if tail else 'failed'}")
    commit = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return Fetched(destination, f"git {entry.git}@{commit or entry.ref or 'HEAD'}")


# ------------------------------------------------------------------ labelling


@dataclass
class Environment:
    """What running a plan needs: the index, the archive, the target."""

    index: Any
    distro: Any
    target: Any
    pip_cache: Path
    timeout: int = 600


def environment(python: str = "3.12", arch: str = "riscv64", timeout: int = 600) -> Environment:
    import httpx

    from .cache import Cache
    from .distro import DistroIndex, resolve_spec
    from .index import PackageIndex
    from .target import Target

    cache = Cache()
    client = httpx.Client(timeout=60, follow_redirects=True)
    major, minor = (int(p) for p in python.split(".")[:2])
    target = Target.parse(arch, (major, minor))
    distro = DistroIndex(client, cache, resolve_spec("debian:trixie"), target.arch)
    distro.names()
    return Environment(PackageIndex(client, cache), distro, target, cache.root / "pip", timeout)


def run_plan(plan: Plan, root: Path, env: Environment):
    from . import planrun

    return planrun.execute(
        plan, root, index=env.index, target=env.target, distro=env.distro,
        timeout=env.timeout, pip_cache=env.pip_cache,
    )


def plan_faults(result) -> list[str]:
    """What went wrong in running a plan because of the plan itself.

    A configure that stops is the build's business, and so is a package
    with nothing for the target: those are answers. A manifest that is not
    there, a configure pointed at a directory with nothing to configure, a
    driver that configured what the plan does not list -- those are the
    plan's, and go back to whoever wrote it.
    """
    problems = list(result.evidence_problems)
    for outcome in result.steps:
        detail = outcome.detail or ""
        if outcome.status == "failed" and (
            detail.startswith("no ") or re.match(r"^[^:]+: \[Errno", detail)
            or "not a" in detail
        ):
            problems.append(f"step {outcome.step.id!r} could not run: {detail}")
        for line in getattr(outcome, "plan_check", None) or []:
            if "which no plan step does" in line or "which the driver never configured" in line:
                problems.append(f"step {outcome.step.id!r}: {line}")
    return problems


def summary(result) -> dict:
    """What running a plan showed, for the record."""
    answer = result.answer
    return {
        "steps": [
            {"id": o.step.id, "kind": o.step.kind, "status": o.status, "detail": o.detail}
            for o in result.steps
        ],
        "plan_check": [
            line for o in result.steps for line in (getattr(o, "plan_check", None) or [])
        ],
        "verdict": answer.verdict if answer else None,
        "headline": answer.headline if answer else None,
        "required": sorted(n.id for n in result.nodes.values() if n.required),
    }


def label(
    entry: Entry, root: Path, planner: modelplan.Backend, out: Path,
    env: Optional[Environment] = None, rounds: int = modelplan.ROUNDS,
) -> dict:
    """Plan one repository with the teacher, verify it, and write it all down."""
    started = time.monotonic()
    directory = out / entry.key
    directory.mkdir(parents=True, exist_ok=True)
    pack = evidence.build(root, entry.name)
    (directory / "pack.txt").write_text(pack.render())
    (directory / "pack.json").write_text(json.dumps(pack.to_dict()))
    executed: dict = {}

    def verify(plan: Plan) -> list[str]:
        if env is None:
            return []
        result = run_plan(plan, root, env)
        executed["result"] = result
        return plan_faults(result)

    outcome = modelplan.plan_repository(root, planner, pack=pack, rounds=rounds, verify=verify)
    tier = REJECTED
    if outcome.plan is not None:
        tier = VERIFIED if outcome.ok else PARTIAL
        (directory / "plan.json").write_text(
            json.dumps(plan_to_dict(outcome.plan), indent=2) + "\n"
        )
    if outcome.error:
        # It stopped answering partway: whatever it wrote is not its answer.
        tier = TEACHER_ERROR
    if "result" in executed:
        (directory / "run.json").write_text(json.dumps(summary(executed["result"]), indent=2))
    transcript = [
        {
            "text": a.text, "problems": a.problems,
            "input_tokens": a.completion.input_tokens if a.completion else 0,
            "output_tokens": a.completion.output_tokens if a.completion else 0,
            "cost_usd": a.completion.cost_usd if a.completion else None,
        }
        for a in outcome.attempts
    ]
    (directory / "transcript.json").write_text(json.dumps(transcript, indent=2))
    meta = {
        "entry": asdict(entry), "split": entry.held_out, "root": str(root),
        "backend": outcome.backend, "tier": tier, "rounds": len(outcome.attempts),
        "error": outcome.error, "pack_digest": pack.digest(),
        "pack_chars": len(pack.render()), "cost_usd": outcome.cost_usd,
        "executed": env is not None, "seconds": round(time.monotonic() - started, 1),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _say(message: str) -> None:
    print(message, flush=True)


def label_corpus(
    entries: list[Entry], out: Path, planner: modelplan.Backend, *,
    execute: bool = True, jobs: int = 1, redo: bool = False, python: str = "3.12",
    progress=_say,
) -> list[dict]:
    """Label every entry not labelled yet, until done or the teacher stops answering.

    Two teacher failures in a row -- a usage limit, most likely -- stop the
    run; nothing after is attempted, and nothing it failed on is recorded as
    done. Running the same command again picks up where this one stopped.
    """
    import threading

    out.mkdir(parents=True, exist_ok=True)
    from .cache import Cache

    cache_root = Cache().root / "dataset"
    env = environment(python) if execute else None
    index = env.index if env else None
    if index is None and any(e.pypi for e in entries):
        index = environment(python).index
    stop = threading.Event()
    lock = threading.Lock()
    state = {"errors": 0, "spent": 0.0, "done": 0}

    def one(entry: Entry) -> Optional[dict]:
        done = out / entry.key / "meta.json"
        if done.exists() and not redo:
            meta = json.loads(done.read_text())
            if meta.get("tier") != TEACHER_ERROR:
                return meta
        if stop.is_set():
            return None
        fetched = fetch(entry, cache_root, index)
        if fetched.root is None:
            meta = {"entry": asdict(entry), "tier": REJECTED, "error": fetched.error}
            (out / entry.key).mkdir(parents=True, exist_ok=True)
            done.write_text(json.dumps(meta, indent=2))
            progress(f"{entry.key}: not fetched: {fetched.error}")
            return meta
        meta = label(entry, fetched.root, planner, out, env)
        meta["origin"] = fetched.origin
        done.write_text(json.dumps(meta, indent=2))
        with lock:
            state["spent"] += meta.get("cost_usd") or 0.0
            if meta["tier"] == TEACHER_ERROR:
                state["errors"] += 1
                if state["errors"] >= MAX_TEACHER_ERRORS and not stop.is_set():
                    stop.set()
                    progress(
                        f"stopping: the teacher has failed {state['errors']} times in a row "
                        f"({meta.get('error')}). Run the same command again to carry on."
                    )
            else:
                state["errors"] = 0
                state["done"] += 1
            spent = state["spent"]
        cost = f", ${meta['cost_usd']:.3f}" if meta.get("cost_usd") else ""
        progress(
            f"{entry.key}: {meta['tier']} in {meta['rounds']} round(s){cost} "
            f"(${spent:.2f} this run)"
        )
        return meta

    if jobs <= 1:
        results = [one(e) for e in entries]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(one, entries))
    return [r for r in results if r is not None]


# ------------------------------------------------------------------ export


def examples(out: Path, tiers: tuple[str, ...] = (VERIFIED,)) -> list[dict]:
    """Every labelled example of these tiers: the pack in, the plan out."""
    found = []
    for directory in sorted(p for p in out.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("tier") not in tiers or not (directory / "plan.json").is_file():
            continue
        plan = load_plan(directory / "plan.json")
        pack = (directory / "pack.txt").read_text()
        found.append({
            "messages": [
                {"role": "system", "content": modelplan.SYSTEM_PROMPT},
                {"role": "user", "content": pack},
                {"role": "assistant", "content": json.dumps(modelplan.to_wire(plan))},
            ],
            "meta": {
                "name": meta["entry"]["name"], "key": directory.name, "split": meta["split"],
                "tier": meta["tier"], "origin": meta.get("origin"),
                "pack_digest": meta.get("pack_digest"),
            },
        })
    return found


def export(out: Path, destination: Path, tiers: tuple[str, ...] = (VERIFIED,)) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    handles = {}
    try:
        for example in examples(out, tiers):
            split = example["meta"]["split"]
            if split not in handles:
                handles[split] = (destination / f"{split}.jsonl").open("w")
            handles[split].write(json.dumps(example) + "\n")
            counts[split] = counts.get(split, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


# ------------------------------------------------------------------ seeding


def needs_building(index, target, name: str) -> Optional[str]:
    """The latest version, if it compiles something and has no wheel for the target.

    Platform wheels say it compiles; an sdist says it can be built; no wheel
    the target accepts says it has to be. One index page, nothing resolved.
    """
    project = index.project(name)
    if project is None or not project.releases:
        return None
    stable = [v for v in project.releases if not v.is_prerelease]
    release = project.releases[max(stable or project.releases)]
    if not release.has_sdist or not release.wheels:
        return None
    if all(tag == "any" for tag in release.platform_tags()):
        return None
    if release.matching_wheels(target.tags()):
        return None
    return str(release.version)


def seed_pypi(
    names: list[str], env: Environment, want: int, progress=print, workers: int = 8
) -> list[Entry]:
    """Compiled packages with an sdist and no riscv64 wheel: what needs a plan."""
    def check(name: str) -> Optional[Entry]:
        try:
            version = needs_building(env.index, env.target, name)
        except Exception as exc:  # noqa: BLE001 - one bad page is not the corpus
            progress(f"{name}: skipped ({exc})")
            return None
        return Entry(name=name, pypi=name, version=version) if version else None

    entries: list[Entry] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for entry in pool.map(check, names):
            if entry is not None and len(entries) < want:
                entries.append(entry)
                progress(f"{entry.name} {entry.version}: needs building for {env.target.arch}")
    return entries


_TOP_PYPI = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"


def top_pypi(limit: int) -> list[str]:
    """The most downloaded projects on PyPI, most first."""
    import httpx

    data = httpx.get(_TOP_PYPI, timeout=60, follow_redirects=True).json()
    return [row["project"] for row in data.get("rows", [])[:limit]]


# ------------------------------------------------------------------ cli


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="will-it-riscv-dataset",
        description="Label repositories with verified build plans, for fine-tuning a planner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    labelling = sub.add_parser("label", help="plan, verify and record every corpus entry")
    labelling.add_argument("--corpus", required=True, type=Path)
    labelling.add_argument("--out", required=True, type=Path)
    labelling.add_argument("--planner", default="claude-cli",
                           choices=("claude-cli", "anthropic", "openai"))
    labelling.add_argument("--planner-model")
    labelling.add_argument("--planner-url")
    labelling.add_argument("--only", action="append", default=[], metavar="NAME")
    labelling.add_argument("--limit", type=int)
    labelling.add_argument("--jobs", type=int, default=1)
    labelling.add_argument("--redo", action="store_true")
    labelling.add_argument("--no-execute", action="store_true",
                           help="check plans against the tree, but do not run them")
    labelling.add_argument("--python", default="3.12")

    exporting = sub.add_parser("export", help="write verified examples as chat JSONL")
    exporting.add_argument("--out", required=True, type=Path)
    exporting.add_argument("--to", required=True, type=Path)
    exporting.add_argument("--include-partial", action="store_true")

    status = sub.add_parser("status", help="summarise a labelling run")
    status.add_argument("--out", required=True, type=Path)

    seeding = sub.add_parser("seed-pypi", help="a corpus of compiled PyPI packages")
    seeding.add_argument("--top", type=int, default=2000,
                         help="look through this many of the most downloaded projects")
    seeding.add_argument("--want", type=int, default=300)
    seeding.add_argument("--to", required=True, type=Path)
    seeding.add_argument("--python", default="3.12")

    args = parser.parse_args(argv)
    if args.command == "label":
        entries = load_corpus(args.corpus)
        if args.only:
            entries = [e for e in entries if e.name in args.only]
        if args.limit:
            entries = entries[: args.limit]
        planner = modelplan.backend(args.planner, args.planner_model, args.planner_url)
        results = label_corpus(
            entries, args.out, planner, execute=not args.no_execute, jobs=args.jobs,
            redo=args.redo, python=args.python,
        )
        _print_status(results)
        return 0
    if args.command == "export":
        tiers = (VERIFIED, PARTIAL) if args.include_partial else (VERIFIED,)
        counts = export(args.out, args.to, tiers)
        written = ", ".join(f"{split}: {n}" for split, n in sorted(counts.items()))
        print(written or "nothing to export")
        return 0
    if args.command == "status":
        metas = [
            json.loads((p / "meta.json").read_text())
            for p in sorted(args.out.iterdir()) if (p / "meta.json").is_file()
        ]
        _print_status(metas)
        return 0
    assert args.command == "seed-pypi"
    env = environment(args.python)
    names = top_pypi(args.top)
    entries = seed_pypi(names, env, args.want, progress=lambda m: print(m, file=sys.stderr))
    with args.to.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps({k: v for k, v in asdict(entry).items() if v}) + "\n")
    print(f"{len(entries)} packages written to {args.to}")
    return 0


def _print_status(metas: list[dict]) -> None:
    tiers: dict[str, int] = {}
    cost = 0.0
    for meta in metas:
        tiers[meta.get("tier", "?")] = tiers.get(meta.get("tier", "?"), 0) + 1
        cost += meta.get("cost_usd") or 0.0
    parts = [f"{tier}: {n}" for tier, n in sorted(tiers.items())]
    print(f"{len(metas)} entries — " + ", ".join(parts) + (f"; ${cost:.2f} spent" if cost else ""))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
