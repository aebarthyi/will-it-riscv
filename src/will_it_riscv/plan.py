"""A build plan: everything a repository's build runs, in order.

The plan is what a model is for. Reading a README, a CI workflow and a
bootstrap script, and saying "./mfc.sh build installs toolchain/ with uv,
then configures three CMake targets with these flags", is reading
comprehension. It needs no knowledge of riscv64 at all. Everything after the
plan -- running each step in the pretend environment, joining what they ask
for into one graph, checking the archive -- is deterministic, and is what
makes the answer worth trusting whatever wrote the plan.

So a plan is data, not shell: typed steps that the executor hands to
machinery that already exists, each citing the lines it came from. A cited
line that does not say what the plan claims is caught before anything runs,
which is the cheapest check there is on a model making things up.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PLAN_VERSION = 1

PYTHON_INSTALL = "python-install"
SYSTEM_PACKAGES = "system-packages"
CMAKE_CONFIGURE = "cmake-configure"
PYTHON_RUN = "python-run"
STEP_KINDS = (PYTHON_INSTALL, SYSTEM_PACKAGES, CMAKE_CONFIGURE, PYTHON_RUN)

_STEP_FIELDS = {
    "id", "kind", "evidence", "after", "note", "provides", "optional", "enabled_by",
    "manifest", "extras", "packages", "source", "defines", "script", "args",
}
_PLAN_FIELDS = {"version", "repo", "entry", "steps", "unsure"}
_EVIDENCE = re.compile(r"^(?P<path>[^:]+):(?P<start>\d+)(?:-(?P<end>\d+))?$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PlanError(ValueError):
    """A plan that cannot be run. Every problem is listed, not just the first."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Evidence:
    """Where a step came from: ``path:line`` or ``path:first-last``."""

    path: str
    start: int
    end: int
    quote: Optional[str] = None
    """Text those lines must contain, whitespace aside."""

    def __str__(self) -> str:
        where = f"{self.start}" if self.start == self.end else f"{self.start}-{self.end}"
        return f"{self.path}:{where}"


@dataclass
class Step:
    id: str
    kind: str
    evidence: list[Evidence] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    note: Optional[str] = None
    provides: list[str] = field(default_factory=list)
    """What this step makes exist for the steps after it: MFC's dependency
    targets build FFTW, HDF5, SILO and LAPACK from source."""
    optional: bool = False
    """Not part of the default build. The default build is the minimal spec:
    everything it needs is required. A step that only runs with an extra
    flag, or on a choice left to the user, is optional -- and so is anything
    only it needs."""
    enabled_by: Optional[str] = None
    """What turns an optional step on: ``./mfc.sh build --gpu acc``."""
    manifest: Optional[str] = None
    """python-install: the pyproject.toml or requirements file installed."""
    extras: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    """system-packages: distro package names."""
    source: str = "."
    """cmake-configure: the source directory, relative to the repository."""
    defines: dict[str, str] = field(default_factory=dict)
    """cmake-configure: the -D flags the build itself passes."""
    script: Optional[str] = None
    """python-run: the project's own build driver, relative to the repository."""
    args: list[str] = field(default_factory=list)
    """python-run: what it is run with -- MFC's ``build -j 1``."""


@dataclass
class Plan:
    repo: str
    steps: list[Step]
    entry: Optional[str] = None
    """What a person runs: ``./mfc.sh build``."""
    entry_evidence: list[Evidence] = field(default_factory=list)
    unsure: list[str] = field(default_factory=list)
    """What the planner could not tell. Worth saying: the executor can test it."""

    def step(self, step_id: str) -> Step:
        return next(s for s in self.steps if s.id == step_id)

    def ordered(self) -> list[Step]:
        """Steps in an order that respects ``after``, otherwise as listed."""
        done: list[Step] = []
        placed: set = set()
        pending = list(self.steps)
        while pending:
            for step in pending:
                if all(a in placed for a in step.after):
                    done.append(step)
                    placed.add(step.id)
                    pending.remove(step)
                    break
            else:  # pragma: no cover - parse_plan rejects cycles
                raise PlanError(["steps depend on each other in a cycle"])
        return done


# ------------------------------------------------------------------ reading


def load_plan(path: Path) -> Plan:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PlanError([f"{path}: {exc}"]) from exc
    return parse_plan(data)


def parse_plan(data: Any) -> Plan:
    problems: list[str] = []
    if not isinstance(data, dict):
        raise PlanError(["a plan is a JSON object"])
    for key in sorted(set(data) - _PLAN_FIELDS):
        problems.append(f"unknown field {key!r}")
    if data.get("version", PLAN_VERSION) != PLAN_VERSION:
        problems.append(f"version {data.get('version')!r}; this reads version {PLAN_VERSION}")
    repo = data.get("repo")
    if not isinstance(repo, str) or not repo:
        problems.append("'repo' must name the repository")

    entry = data.get("entry")
    entry_command: Optional[str] = None
    entry_evidence: list[Evidence] = []
    if isinstance(entry, dict):
        entry_command = entry.get("command") if isinstance(entry.get("command"), str) else None
        entry_evidence = _evidence(entry.get("evidence", []), "entry", problems)
    elif entry is not None:
        problems.append("'entry' must be an object with a 'command'")

    raw_steps = data.get("steps")
    steps: list[Step] = []
    if not isinstance(raw_steps, list) or not raw_steps:
        problems.append("'steps' must be a non-empty list")
        raw_steps = []
    for index, raw in enumerate(raw_steps):
        step = _step(raw, index, problems)
        if step is not None:
            steps.append(step)

    ids = [s.id for s in steps]
    for duplicate in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"step id {duplicate!r} is used twice")
    optional = {s.id for s in steps if s.optional}
    for step in steps:
        for other in step.after:
            if other not in ids:
                problems.append(f"step {step.id!r}: 'after' names unknown step {other!r}")
            elif other in optional and not step.optional:
                # The default build cannot wait on something it does not do.
                problems.append(
                    f"step {step.id!r} is part of the default build, but comes after "
                    f"optional step {other!r}"
                )
    if not problems and _has_cycle(steps):
        problems.append("steps depend on each other in a cycle")

    unsure = data.get("unsure", [])
    if not isinstance(unsure, list) or not all(isinstance(u, str) for u in unsure):
        problems.append("'unsure' must be a list of strings")
        unsure = []

    if problems:
        raise PlanError(problems)
    assert isinstance(repo, str)
    return Plan(
        repo=repo, steps=steps, entry=entry_command, entry_evidence=entry_evidence,
        unsure=list(unsure),
    )


def _step(raw: Any, index: int, problems: list[str]) -> Optional[Step]:
    where = f"step {index + 1}"
    if not isinstance(raw, dict):
        problems.append(f"{where} must be an object")
        return None
    step_id = raw.get("id")
    if not isinstance(step_id, str) or not _ID.match(step_id):
        problems.append(f"{where}: 'id' must be a short name like 'configure-simulation'")
        return None
    where = f"step {step_id!r}"
    for key in sorted(set(raw) - _STEP_FIELDS):
        problems.append(f"{where}: unknown field {key!r}")
    kind = raw.get("kind")
    if kind not in STEP_KINDS:
        problems.append(f"{where}: 'kind' must be one of {', '.join(STEP_KINDS)}")
        return None

    step = Step(
        id=step_id,
        kind=kind,
        evidence=_evidence(raw.get("evidence", []), where, problems),
        after=_strings(raw.get("after", []), f"{where}: 'after'", problems),
        note=raw.get("note") if isinstance(raw.get("note"), str) else None,
        provides=_strings(raw.get("provides", []), f"{where}: 'provides'", problems),
    )
    optional = raw.get("optional", False)
    if not isinstance(optional, bool):
        problems.append(f"{where}: 'optional' is true or false")
    elif optional:
        step.optional = True
        enabled_by = raw.get("enabled_by")
        if not isinstance(enabled_by, str) or not enabled_by.strip():
            # Optional means someone has to decide to turn it on. Say how.
            problems.append(f"{where}: an optional step says what turns it on ('enabled_by')")
        else:
            step.enabled_by = enabled_by
    elif "enabled_by" in raw:
        problems.append(f"{where}: 'enabled_by' only means something on an optional step")
    if kind == PYTHON_INSTALL:
        manifest = raw.get("manifest")
        if not isinstance(manifest, str) or not manifest:
            problems.append(f"{where}: a python-install names its 'manifest'")
        else:
            step.manifest = manifest
        step.extras = _strings(raw.get("extras", []), f"{where}: 'extras'", problems)
    elif kind == SYSTEM_PACKAGES:
        step.packages = _strings(raw.get("packages", []), f"{where}: 'packages'", problems)
        if not step.packages:
            problems.append(f"{where}: a system-packages step lists its 'packages'")
    elif kind == PYTHON_RUN:
        script = raw.get("script")
        if not isinstance(script, str) or not script:
            problems.append(f"{where}: a python-run names its 'script'")
        else:
            step.script = script
        step.args = _strings(raw.get("args", []), f"{where}: 'args'", problems)
    else:
        source = raw.get("source", ".")
        if not isinstance(source, str):
            problems.append(f"{where}: 'source' must be a directory")
        else:
            step.source = source
        step.defines = _defines(raw.get("defines", {}), where, problems)
    return step


def _strings(value: Any, where: str, problems: list[str]) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        problems.append(f"{where} must be a list of strings")
        return []
    return list(value)


def _defines(value: Any, where: str, problems: list[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        problems.append(f"{where}: 'defines' must map variable names to values")
        return {}
    defines: dict[str, str] = {}
    for name, raw in value.items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(name)):
            problems.append(f"{where}: {name!r} is not a CMake variable name")
            continue
        if isinstance(raw, bool):
            defines[name] = "ON" if raw else "OFF"
        elif isinstance(raw, (str, int, float)):
            defines[name] = str(raw)
        else:
            problems.append(f"{where}: -D{name} needs a string, number or boolean")
    return defines


def _evidence(value: Any, where: str, problems: list[str]) -> list[Evidence]:
    if not isinstance(value, list):
        problems.append(f"{where}: 'evidence' must be a list")
        return []
    found: list[Evidence] = []
    for item in value:
        quote = None
        at = item
        if isinstance(item, dict):
            at, quote = item.get("at"), item.get("quote")
            if quote is not None and not isinstance(quote, str):
                problems.append(f"{where}: an evidence 'quote' must be a string")
                quote = None
        match = _EVIDENCE.match(at) if isinstance(at, str) else None
        if not match:
            problems.append(f"{where}: evidence {at!r} must look like 'path:12' or 'path:12-18'")
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start or start < 1:
            problems.append(f"{where}: evidence {at!r} has its lines backwards")
            continue
        found.append(Evidence(match.group("path"), start, end, quote))
    return found


def _has_cycle(steps: list[Step]) -> bool:
    graph = {s.id: s.after for s in steps}
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        if any(visit(n) for n in graph.get(node, [])):
            return True
        state[node] = 2
        return False

    return any(visit(s.id) for s in steps)


# ---------------------------------------------------------------- checking


def check_evidence(plan: Plan, root: Path) -> list[str]:
    """Every citation that does not hold, as a sentence saying why.

    A file that is not there, a line past its end, or a quote those lines do
    not contain. Whitespace is ignored; nothing else is.
    """
    root = Path(root).resolve()
    problems: list[str] = []
    cited = [("entry", e) for e in plan.entry_evidence] + [
        (f"step {s.id!r}", e) for s in plan.steps for e in s.evidence
    ]
    for owner, evidence in cited:
        path = (root / evidence.path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            problems.append(f"{owner}: {evidence} is outside the repository")
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            problems.append(f"{owner}: {evidence.path} does not exist")
            continue
        if evidence.end > len(lines):
            problems.append(f"{owner}: {evidence} is past the end ({len(lines)} lines)")
            continue
        if evidence.quote:
            text = " ".join(lines[evidence.start - 1:evidence.end])
            if _squash(evidence.quote) not in _squash(text):
                problems.append(f"{owner}: {evidence} does not say {evidence.quote!r}")
    return problems


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


#: The shape of a plan, for constraining whatever writes one. The executor
#: validates against parse_plan either way; this is so a model's structured
#: output cannot be anything else in the first place.
PLAN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["repo", "steps"],
    "properties": {
        "version": {"const": PLAN_VERSION},
        "repo": {"type": "string"},
        "entry": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command"],
            "properties": {
                "command": {"type": "string"},
                "evidence": {"$ref": "#/$defs/evidence"},
            },
        },
        "unsure": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind", "evidence"],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"enum": list(STEP_KINDS)},
                    "evidence": {"$ref": "#/$defs/evidence"},
                    "after": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                    "provides": {"type": "array", "items": {"type": "string"}},
                    "manifest": {"type": "string"},
                    "extras": {"type": "array", "items": {"type": "string"}},
                    "packages": {"type": "array", "items": {"type": "string"}},
                    "source": {"type": "string"},
                    "defines": {
                        "type": "object",
                        "additionalProperties": {"type": ["string", "boolean", "number"]},
                    },
                    "script": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                    "optional": {"type": "boolean"},
                    "enabled_by": {"type": "string"},
                },
            },
        },
    },
    "$defs": {
        "evidence": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["at"],
                        "properties": {"at": {"type": "string"}, "quote": {"type": "string"}},
                    },
                ]
            },
        }
    },
}
