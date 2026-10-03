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
  check     find each example's tree on this machine, rebuild its pack
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


def _git(entry: Entry, cache: Path, commit: Optional[str] = None) -> Fetched:
    """A shallow clone at ``entry.ref``, or at ``commit`` when one is pinned."""
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
    if commit and _head(destination) != commit:
        # A shallow clone has only its tip: fetch the one commit, then stand on it.
        for command in (["fetch", "--quiet", "--depth", "1", "origin", commit],
                        ["checkout", "--quiet", "--detach", commit]):
            process = subprocess.run(
                ["git", "-C", str(destination), *command], capture_output=True, text=True,
                timeout=900, env=env,
            )
            if process.returncode != 0:
                tail = (process.stderr or "").strip().splitlines()
                return Fetched(None, error=f"git {command[0]} {commit}: "
                                           f"{tail[-1] if tail else 'failed'}")
    from .sources import init_submodules

    init_submodules(destination)
    head = _head(destination)
    return Fetched(destination, f"git {entry.git}@{head or entry.ref or 'HEAD'}")


def _head(destination: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


# ------------------------------------------------------------------ finding a tree again


def tree(
    meta: dict, cache_root: Optional[Path] = None, index=None, fetch_missing: bool = False,
) -> Fetched:
    """Where an example's source tree is on this machine.

    ``meta.json`` records the absolute root it was labelled from, which names
    another machine's cache once the dataset moves. So the root is tried as
    recorded, then the same place in this machine's cache -- everything
    after the cache's ``dataset/`` directory is the same wherever the cache
    is. Failing both, and when asked to, the tree is fetched again: git at
    the commit ``origin`` recorded, PyPI at the recorded version.
    """
    if not meta.get("root"):
        return Fetched(None, error="no tree was recorded")
    recorded = Path(meta["root"])
    if recorded.is_dir():
        return Fetched(recorded, meta.get("origin", ""))
    if cache_root is None:
        from .cache import Cache

        cache_root = Cache().root / "dataset"
    relocated = _in_cache(recorded, cache_root)
    if relocated is not None and relocated.is_dir():
        return Fetched(relocated, meta.get("origin", ""))
    if not fetch_missing:
        return Fetched(None, error=f"not at {relocated or recorded}")
    entry = Entry(**meta["entry"])
    if entry.git:
        origin = meta.get("origin") or ""
        commit = origin.rsplit("@", 1)[1] if origin.startswith("git ") and "@" in origin else None
        return _git(entry, cache_root / "git", commit)
    return fetch(entry, cache_root, index)


def _in_cache(recorded: Path, cache_root: Path) -> Optional[Path]:
    """The recorded root, moved into ``cache_root``: the part below ``dataset/``."""
    parts = recorded.parts
    for i, part in enumerate(parts[:-1]):
        if part == "dataset" and parts[i + 1] in ("git", "sources"):
            return cache_root.joinpath(*parts[i + 1:])
    return None


def check(
    out: Path, fetch_missing: bool = False, index=None, progress=print,
) -> dict[str, list[str]]:
    """Every verified example's tree found, and its pack rebuilt to the same digest.

    A pack that rebuilds differently means the tree is not the one the label
    was written from: the plan's citations may no longer hold against it.
    """
    outcomes: dict[str, list[str]] = {}
    for directory in sorted(p for p in out.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("tier") != VERIFIED:
            continue
        found = tree(meta, index=index, fetch_missing=fetch_missing)
        if found.root is None:
            outcome = "missing"
            progress(f"{directory.name}: missing: {found.error}")
        elif evidence.build(found.root, meta["entry"]["name"]).digest() != meta["pack_digest"]:
            outcome = "differs"
            progress(f"{directory.name}: the pack rebuilt from {found.root} differs")
        else:
            outcome = "ok"
        outcomes.setdefault(outcome, []).append(directory.name)
    return outcomes


# ------------------------------------------------------------------ labelling


@dataclass
class Environment:
    """What running a plan needs: the index, the archive, the target."""

    index: Any
    distro: Any
    target: Any
    pip_cache: Path
    timeout: int = 600


#: What labels were made and checked against; ``--distro`` elsewhere.
LABEL_DISTRO = "debian:trixie"


def environment(
    python: str = "3.12", arch: str = "riscv64", timeout: int = 600, distro: str = LABEL_DISTRO,
) -> Environment:
    import httpx

    from .cache import Cache
    from .distro import DistroIndex, resolve_spec
    from .index import PackageIndex
    from .target import Target

    cache = Cache()
    client = httpx.Client(timeout=60, follow_redirects=True)
    major, minor = (int(p) for p in python.split(".")[:2])
    target = Target.parse(arch, (major, minor))
    distro = DistroIndex(client, cache, resolve_spec(distro), target.arch)
    if distro.names() is None:
        raise RuntimeError(f"{distro.spec.label}: {distro.error}")
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
        if outcome.status in ("failed", "stopped") and "COMPILER" in detail:
            for name, value in outcome.step.defines.items():
                if name.endswith("_COMPILER") and _PLACEHOLDER.search(value):
                    problems.append(
                        f"step {outcome.step.id!r} sets {name} to {value!r}, "
                        "which is filled in by whoever runs the build, and the configure "
                        f"stopped on it: {detail}; leave the compiler to CMake"
                    )
    for step_id, name, near, distro in getattr(result, "unknown_names", []):
        problems.append(
            f"step {step_id!r}: {distro} has no package named {name!r} on any architecture"
            + (f" (it has {', '.join(near)})" if near else "")
            + "; name the Debian package the build needs, or leave it out"
        )
    return problems


#: A value someone fills in: ${FC}, $CC, <C compiler>, @CXX@.
_PLACEHOLDER = re.compile(r"\$\{?\w+\}?|<[^<>]+>|@\w+@")


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


def revise(
    directory: Path, planner: modelplan.Backend, env: Environment,
    rounds: int = modelplan.ROUNDS,
) -> Optional[dict]:
    """Send a verified label back to the teacher with what running it now shows.

    For labels made before the executor could tell a plan's mistake from an
    answer: a package name Debian has nowhere read as "nothing for riscv64".
    The teacher sees its plan and the problems, as it would have in a later
    round, and its answer is run and checked like any other. What was there
    before goes to ``revised-from/``; nothing changes when the plan holds.
    Returns the revision record, or None when there was nothing to revise.
    """
    meta = json.loads((directory / "meta.json").read_text())
    found = tree(meta)
    if found.root is None:
        raise RuntimeError(f"{directory.name}: its tree is {found.error}")
    root = found.root
    pack = evidence.EvidencePack.from_dict(json.loads((directory / "pack.json").read_text()))
    before = load_plan(directory / "plan.json")
    problems = plan_faults(run_plan(before, root, env))
    if not problems:
        return None
    executed: dict = {}

    def verify(plan: Plan) -> list[str]:
        result = run_plan(plan, root, env)
        executed["result"] = result
        return plan_faults(result)

    history = [
        {"role": "assistant", "content": json.dumps(modelplan.to_wire(before))},
        {"role": "user", "content": modelplan.feedback(problems)},
    ]
    outcome = modelplan.plan_repository(
        root, planner, pack=pack, rounds=rounds, verify=verify, history=history)
    record = {
        "problems": problems, "backend": outcome.backend, "error": outcome.error,
        "ok": outcome.ok, "cost_usd": outcome.cost_usd,
        "attempts": [{"text": a.text, "problems": a.problems} for a in outcome.attempts],
    }
    if outcome.ok and "result" in executed:
        kept = directory / "revised-from"
        kept.mkdir(exist_ok=True)
        for name in ("plan.json", "run.json", "plan.relabelled.json", "relabel.json"):
            if (directory / name).is_file():
                (directory / name).replace(kept / name)
        (directory / "plan.json").write_text(
            json.dumps(plan_to_dict(outcome.plan), indent=2) + "\n")
        (directory / "run.json").write_text(json.dumps(summary(executed["result"]), indent=2))
        meta["revised"] = True
        meta["cost_usd"] = (meta.get("cost_usd") or 0) + (outcome.cost_usd or 0)
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    (directory / "revision.json").write_text(json.dumps(record, indent=2))
    return record


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


def examples(
    out: Path, tiers: tuple[str, ...] = (VERIFIED,), relabelled: bool = False,
    corrections: bool = False, transcripts: tuple[Path, ...] = (),
) -> list[dict]:
    """Every labelled example of these tiers: the pack in, the plan out.

    ``relabelled``: the plan as the executor re-marked it, where there is one
    (``will-it-riscv-dataset relabel``), rather than as the teacher wrote it.

    ``corrections``: also, for every answer that was sent back, an example
    that continues from it -- the pack, that answer, what was wrong with it
    -- to the verified plan, so a model learns to act on what it is told.
    The answers are the teacher's own earlier rounds, the plan a revision
    replaced (``revise``), and a model's, from the ``--transcripts`` of an
    eval run with ``--run-feedback``, a directory of one file a key.
    """
    found = []
    for directory in sorted(p for p in out.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("tier") not in tiers or not (directory / "plan.json").is_file():
            continue
        chosen = directory / "plan.relabelled.json"
        plan = load_plan(chosen if relabelled and chosen.is_file() else directory / "plan.json")
        pack = (directory / "pack.txt").read_text()
        answer = {"role": "assistant", "content": json.dumps(modelplan.to_wire(plan))}
        start = [
            {"role": "system", "content": modelplan.SYSTEM_PROMPT},
            {"role": "user", "content": pack},
        ]
        base = {
            "name": meta["entry"]["name"], "key": directory.name, "split": meta["split"],
            "tier": meta["tier"], "origin": meta.get("origin"),
            "pack_digest": meta.get("pack_digest"),
        }
        found.append({"messages": [*start, answer], "meta": base})
        if not corrections or meta["split"] != "train":
            continue    # the eval asks for a first answer; a test key's corrections go nowhere
        for source, (text, problems) in _sent_back(directory, transcripts):
            found.append({
                "messages": [
                    *start, {"role": "assistant", "content": text},
                    {"role": "user", "content": modelplan.feedback(problems)}, answer,
                ],
                "meta": {**base, "key": f"{directory.name}#{source}", "of": directory.name},
            })
    return found


def _sent_back(directory: Path, transcripts: tuple[Path, ...]):
    """``(source, (answer, problems))`` for every answer about this key that went back."""
    tried = directory / "transcript.json"
    if tried.is_file():
        for i, attempt in enumerate(json.loads(tried.read_text())):
            if attempt.get("problems"):
                yield f"teacher-{i}", (attempt["text"], attempt["problems"])
    revision = directory / "revision.json"
    before = directory / "revised-from" / "plan.json"
    if revision.is_file() and before.is_file():
        record = json.loads(revision.read_text())
        if record.get("ok"):
            text = json.dumps(modelplan.to_wire(load_plan(before)))
            yield "revised", (text, record["problems"])
    for n, where in enumerate(transcripts):
        mined = where / f"{directory.name}.json"
        if not mined.is_file():
            continue
        for i, attempt in enumerate(json.loads(mined.read_text()).get("attempts", [])):
            if attempt.get("problems"):
                yield f"model{n}-{i}", (attempt["text"], attempt["problems"])


def export(
    out: Path, destination: Path, tiers: tuple[str, ...] = (VERIFIED,), relabelled: bool = False,
    corrections: bool = False, transcripts: tuple[Path, ...] = (),
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    handles = {}
    try:
        for example in examples(out, tiers, relabelled, corrections, transcripts):
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


_RELABEL_ENV: dict = {}


def _relabel_one(directory: Path) -> tuple[str, Optional[dict], Optional[str]]:
    """One example, in a worker that keeps its distro index between examples."""
    from . import quiz, relabel
    from .cache import Cache

    try:
        if not _RELABEL_ENV:
            _RELABEL_ENV["distro"] = environment().distro
            traces = Cache().root / "quiz" / quiz.TRACES
            traces.mkdir(parents=True, exist_ok=True)
            _RELABEL_ENV["trace"] = lambda key, root, defines: quiz._trace(
                key, root, defines, traces)
        report = relabel.relabel_example(directory, _RELABEL_ENV["distro"], _RELABEL_ENV["trace"])
        return directory.name, report, None
    except Exception as exc:  # noqa: BLE001 - one example's trouble is its own
        return directory.name, None, str(exc)[:200]


def _relabel_all(out: Path, only: list[str], jobs: int) -> int:
    directories = [
        p for p in sorted(out.iterdir())
        if (p / "meta.json").is_file() and (p / "plan.json").is_file()
        and json.loads((p / "meta.json").read_text()).get("tier") == VERIFIED
        and (not only or p.name in only)
    ]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_relabel_one, directories))
    else:
        results = [_relabel_one(d) for d in directories]
    moved = {"optional": 0, "required": 0}
    for key, report, error in results:
        if error:
            print(f"{key}: {error}", file=sys.stderr)
        elif report:
            for change in report["changes"]:
                moved[change["to"]] += 1
            if report["changes"]:
                moves = ", ".join(f"{c['package']} -> {c['to']}" for c in report["changes"])
                print(f"{key}: {moves}")
    ran = sum(1 for _, r, _ in results if r)
    print(f"{ran} of {len(results)} examples had a default configure; "
          f"{moved['optional']} packages now optional, {moved['required']} now required")
    return 0


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
    exporting.add_argument("--relabelled", action="store_true",
                           help="use the executor's marks where relabel wrote them")
    exporting.add_argument("--corrections", action="store_true",
                           help="also write, for every answer sent back, an example going "
                                "from it and its problems to the verified plan")
    exporting.add_argument("--transcripts", type=Path, action="append", default=[],
                           help="with --corrections: a model's answers too, from an eval "
                                "run's --transcripts directory")

    relabelling = sub.add_parser(
        "relabel", help="let the default configures mark each stated package required or not")
    relabelling.add_argument("--out", required=True, type=Path)
    relabelling.add_argument("--only", action="append", default=[], metavar="KEY")
    relabelling.add_argument("--jobs", type=int, default=1)

    revising = sub.add_parser(
        "revise", help="send verified labels back to the teacher with what running them shows")
    revising.add_argument("--out", required=True, type=Path)
    revising.add_argument("--only", action="append", default=[], metavar="KEY")
    revising.add_argument("--planner", default="claude-cli",
                          choices=("claude-cli", "anthropic", "openai"))
    revising.add_argument("--planner-model")
    revising.add_argument("--planner-url")
    revising.add_argument("--jobs", type=int, default=1)

    rerunning = sub.add_parser(
        "rerun", help="run verified plans again on today's executor and record the runs")
    rerunning.add_argument("--out", required=True, type=Path)
    rerunning.add_argument("--split", help="only this split (default: every example)")
    rerunning.add_argument("--only", action="append", default=[], metavar="KEY")
    rerunning.add_argument("--jobs", type=int, default=1)
    rerunning.add_argument("--distro", default=LABEL_DISTRO,
                           help="run against this archive instead (default: %(default)s); "
                                "the run goes to its own file, beside run.json")

    checking = sub.add_parser(
        "check", help="find every verified example's tree and rebuild its pack")
    checking.add_argument("--out", required=True, type=Path)
    checking.add_argument("--fetch", action="store_true",
                          help="fetch missing trees again, at the recorded commit or version")

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
        counts = export(args.out, args.to, tiers, relabelled=args.relabelled,
                        corrections=args.corrections, transcripts=tuple(args.transcripts))
        written = ", ".join(f"{split}: {n}" for split, n in sorted(counts.items()))
        print(written or "nothing to export")
        return 0
    if args.command == "relabel":
        return _relabel_all(args.out, args.only, args.jobs)
    if args.command == "rerun":
        return _rerun_all(args)
    if args.command == "revise":
        return _revise_all(args)
    if args.command == "check":
        index = environment().index if args.fetch else None
        outcomes = check(args.out, fetch_missing=args.fetch, index=index)
        print(", ".join(f"{k}: {len(v)}" for k, v in sorted(outcomes.items())) or "nothing")
        return 0 if set(outcomes) <= {"ok"} else 1
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


def run_file(directory: Path, distro: str = LABEL_DISTRO) -> Path:
    """Where the verified plan's run against this distro is kept: ``run.json``
    for the one labels were made against, ``run.ubuntu-resolute.json`` and so on."""
    if distro == LABEL_DISTRO:
        return directory / "run.json"
    return directory / f"run.{distro.replace(':', '-')}.json"


def rerun(directory: Path, env: Environment) -> tuple[Optional[str], Optional[str]]:
    """Run a verified plan again on today's executor, and record what it shows.

    The eval compares a planner's run with the reference's; recorded by an
    older executor, that compares the planner against the executor's past.
    Against another distro, it is that distro's reference (``run_file``).
    The run it replaces is kept as ``run.before.json`` (and so on). Returns
    the verdicts, before and after.
    """
    meta = json.loads((directory / "meta.json").read_text())
    found = tree(meta)
    if found.root is None:
        raise RuntimeError(f"{directory.name}: its tree is {found.error}")
    ran = summary(run_plan(load_plan(directory / "plan.json"), found.root, env))
    recorded = run_file(directory, env.distro.spec.id)
    before = json.loads(recorded.read_text())["verdict"] if recorded.is_file() else None
    if recorded.is_file():
        recorded.replace(recorded.with_name(recorded.name.replace("run.", "run.before.", 1)))
    recorded.write_text(json.dumps(ran, indent=2))
    return before, ran["verdict"]


def _rerun_all(args) -> int:
    env = environment(distro=args.distro)
    keys = args.only or sorted(
        p.name for p in args.out.iterdir()
        if (p / "meta.json").is_file()
        and (meta := json.loads((p / "meta.json").read_text())).get("tier") == VERIFIED
        and (args.split is None or meta.get("split") == args.split)
    )

    def one(key: str):
        try:
            return key, rerun(args.out / key, env)
        except Exception as exc:  # noqa: BLE001 - one bad example does not stop the rest
            return key, exc

    changed = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for key, got in pool.map(one, keys):
            if isinstance(got, Exception):
                print(f"{key}: {got}", flush=True)
                continue
            before, after = got
            changed += before is not None and before != after
            print(f"{key}: {before} -> {after}" if before != after else f"{key}: {after}",
                  flush=True)
    print(f"{changed} of {len(keys)} verdicts changed")
    return 0


def _revise_all(args) -> int:
    planner = modelplan.backend(args.planner, args.planner_model, args.planner_url)
    env = environment()
    keys = args.only or sorted(
        p.name for p in args.out.iterdir()
        if (p / "meta.json").is_file()
        and json.loads((p / "meta.json").read_text()).get("tier") == VERIFIED
    )
    spent = 0.0

    def one(key: str) -> tuple[str, Optional[dict]]:
        try:
            return key, revise(args.out / key, planner, env)
        except Exception as exc:  # noqa: BLE001 - one bad example does not stop the rest
            return key, {"ok": False, "error": str(exc), "problems": []}

    with ThreadPoolExecutor(args.jobs) as pool:
        for key, record in pool.map(one, keys):
            if record is None:
                print(f"{key}: holds", flush=True)
                continue
            spent += record.get("cost_usd") or 0.0
            why = record.get("error") or "still wrong"
            state = "revised" if record["ok"] else f"not revised ({why})"
            print(f"{key}: {len(record['problems'])} problems, {state}; ${spent:.2f} so far",
                  flush=True)
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
