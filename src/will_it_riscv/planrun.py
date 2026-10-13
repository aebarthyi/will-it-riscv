"""Run a plan in the pretend environment, and join what its steps need.

Nothing here builds anything, and nothing emulates riscv64. Each step is
answered the cheapest way that is still honest:

  python-install    resolved against the index for the target's wheel tags
  system-packages   looked up in the distro's riscv64 archive
  cmake-configure   configured for real, confined to an empty sysroot that
                    grows a stub for whatever the configure insists on
  meson-setup       the same, for Meson: set up as a linux/riscv64 cross
                    build, confined the same way
  python-run        the project's own build driver, run on the host with its
                    build tools shimmed, to see which packages it imports

Every step's asks land in one graph, keyed by ecosystem so that PyPI's numpy
and Debian's python3-numpy stay two things. A requirement that an earlier
step provides -- MFC's CMake wants fypp, which its Python toolchain installs;
it wants FFTW, which its own dependency targets build from source -- is
resolved to that step rather than looked for in the archive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from . import drive
from . import graph as depgraph
from .analyze import Analyzer
from .inputs import RootRequirements, load
from .models import Verdict
from .plan import (
    CMAKE_CONFIGURE,
    MESON_SETUP,
    PYTHON_INSTALL,
    PYTHON_RUN,
    SYSTEM_PACKAGES,
    Plan,
    Step,
    check_evidence,
    package_spec,
)
from .pseudobuild import PseudoBuild
from .pseudobuild import run as run_pseudobuild
from .pseudomeson import run as run_pseudomeson

if TYPE_CHECKING:  # pragma: no cover
    from .distro import DistroIndex
    from .index import PackageIndex
    from .models import Analysis
    from .target import Target

# How each dependency can be had on the target, best first.
BINARY = "binary"
"""Already built for the target: a matching or pure-Python wheel, a distro package."""
TOOLCHAIN = "toolchain"
"""Comes with the compiler."""
PROVIDED = "provided"
"""An earlier step of this plan makes it exist."""
SOURCE = "source"
"""Nothing built for the target, but there is source to build: an sdist."""
UNKNOWN = "unknown"
"""Not settled: a guessed name, an archive not checked, a configure that stopped."""
NONE = "none"
"""Nothing for the target in the index or archive that was checked."""

#: Provided beats everything: when the plan builds FFTW itself, the build does
#: not use the distro's, whatever the archive has.
#: When the archive lacks a tool, or has it too old: where else it can be had
#: for the target. Curated, and kept short: checked September 2026.
#:
#: The Rust project's own releases include riscv64 toolchains (tier 2 with
#: host tools), installed by rustup -- so Debian 13's rustc 1.85 being too
#: old for a crate that wants 1.95 is not the end of it.
UPSTREAM_BINARIES = {
    "rustc": ("rustup", {"riscv64": "riscv64gc-unknown-linux-gnu",
                         "aarch64": "aarch64-unknown-linux-gnu",
                         "x86_64": "x86_64-unknown-linux-gnu"}),
    "cargo": ("rustup", {"riscv64": "riscv64gc-unknown-linux-gnu",
                         "aarch64": "aarch64-unknown-linux-gnu",
                         "x86_64": "x86_64-unknown-linux-gnu"}),
}
# Tools whose source is public but which publish no binaries for riscv64 --
# they would have to be built from source first -- are in upstream.UPSTREAM.

_RANK = {PROVIDED: -1, BINARY: 0, TOOLCHAIN: 0, SOURCE: 1, UNKNOWN: 2, NONE: 3}

# How a requirement is known, strongest first.
BY_CONFIGURE = "configure"
"""A default configure that ran to the end demanded it."""
BY_MANIFEST = "manifest"
"""The project's own manifest -- a pyproject.toml, a requirements file --
declares it, and a default step installs from that. A requirements file of
CI's own is a package list (``BY_LIST``): NEURON's ci/requirements.txt, and
grass's under .github/."""
BY_STOPPED = "stopped"
"""Only a default configure that stopped demanded it. Until a configure
completes, an optional package a stub made present can look required:
octopus's PSolver, and the atlab its Find module then asks for."""
BY_LIST = "listed"
"""Only a package list the plan wrote names it: apt lines out of a README or
a CI script, which carry runtime libraries and tooling along with the build's
needs."""
PROVEN = (BY_CONFIGURE, BY_MANIFEST)

#: Directories whose requirements files are CI's: what CI installs carries
#: its test runners and tooling beside the build's needs.
_CI_DIRS = {"ci", ".ci", ".github", ".gitlab", ".circleci", ".azure-pipelines", ".buildkite",
            ".travis", ".ci_support", "continuous_integration"}


def _ci_manifest(manifest: str) -> bool:
    """Whether a requirements file is CI's: under a CI directory, or named
    for it, as requirements-ci.txt."""
    path = Path(manifest)
    return (any(part.lower() in _CI_DIRS for part in path.parts[:-1])
            or re.search(r"(?:^|[-_.])ci(?:[-_.]|$)", path.stem.lower()) is not None)
_BASIS_RANK = {BY_CONFIGURE: 0, BY_MANIFEST: 1, BY_STOPPED: 2, BY_LIST: 3}


@dataclass
class DepNode:
    id: str
    """``pypi:jaxlib``, ``debian:libfftw3-dev``, ``lib:fftw3``."""
    ecosystem: str
    name: str
    tier: str = UNKNOWN
    detail: str = ""
    version: Optional[str] = None
    required: bool = False
    """Needed by the default build -- the minimal spec. Everything the default
    install brings in counts, whether or not the build step imports it."""
    optional_via: list[str] = field(default_factory=list)
    """When not required: what turns on the optional steps that need it."""
    carried_on: bool = False
    """A default configure looked for it, did not find it, and completed."""
    steps: list[str] = field(default_factory=list)
    """Every step that asked for it."""
    package: Optional[str] = None
    """The distro package that provides it for the target, when one does."""
    provided_by: Optional[str] = None
    """The id of the step that makes it exist, when the plan does."""
    usage: Optional[str] = None
    """For a Python package, once the build driver has run: ``imported`` by
    the build step, ``declared`` by something it imports but never loaded
    itself, or ``unused``. Information for a port, never a reason to drop a
    requirement: MFC's build step never loads jaxlib, and its flamelet
    example cannot run without it. None when nothing ran that could tell."""
    declared_by: list[str] = field(default_factory=list)
    """The imported packages that bring it in, when its usage is declared."""
    wants: Optional[str] = None
    """For a distro package asked for at a version: the version the build
    wants -- 8.7.0, for jax's Bazel. What recursing into its source fetches."""
    basis: Optional[str] = None
    """When required: how that is known, the strongest of every default step
    that needs it (``BY_CONFIGURE`` ... ``BY_LIST``)."""


@dataclass
class StepResult:
    step: Step
    status: str
    """``done``, ``completed`` (a configure that ran to the end), ``stopped``,
    or ``failed`` (could not be run at all)."""
    detail: str = ""
    analysis: Optional[Analysis] = None
    pseudobuild: Optional[PseudoBuild] = None
    graph: Optional[depgraph.Graph] = None
    trace: Optional[drive.DriverTrace] = None
    plan_check: list[str] = field(default_factory=list)
    """For a python-run: how the configures it ran compare with the plan's."""
    driven: list[str] = field(default_factory=list)
    """For a python-run: the plan's configures it was seen to run itself."""


@dataclass
class PlanAnswer:
    verdict: str
    """``yes``, ``yes-after-source-builds``, ``probably``, ``no`` or ``unknown``."""
    headline: str
    blockers: list[str] = field(default_factory=list)
    """Required by the default build, and nothing for the target."""
    from_source: list[str] = field(default_factory=list)
    """Required by the default build, and only source for the target."""
    install: list[str] = field(default_factory=list)
    optional: dict[str, list[str]] = field(default_factory=dict)
    """What turns it on -> the blockers and source builds only it brings."""
    optional_stopped: list[str] = field(default_factory=list)
    """Optional steps that could not be run to the end."""


@dataclass
class PlanResult:
    plan: Plan
    root: Path
    steps: list[StepResult] = field(default_factory=list)
    nodes: dict[str, DepNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    """``(from, to)``: from a step (``step:<id>``) or a node, to a node."""
    evidence_problems: list[str] = field(default_factory=list)
    answer: Optional[PlanAnswer] = None
    used_as_provided: set = field(default_factory=set)
    """Python packages a later step used for what they provide: fypp."""
    unknown_names: list[tuple[str, str, list[str], str]] = field(default_factory=list)
    """``(step, name, near names, distro)``: system packages the distro has no package called."""
    arch: str = "riscv64"
    """The target, for an error that refuses it by name."""
    unconfigured: list[str] = field(default_factory=list)
    """Build files at the project's root the executor can configure --
    CMakeLists.txt, meson.build -- when no default step builds the project:
    then its package lists were looked up and nothing more. qemu's plan of
    apt lines alone answered yes."""

    def why(self, node_id: str) -> list[str]:
        """The shortest chain from a step to this node: step, ..., node."""
        parents: dict[str, list[str]] = {}
        for source, target in self.edges:
            parents.setdefault(target, []).append(source)
        frontier = [[node_id]]
        seen = {node_id}
        while frontier:
            path = frontier.pop(0)
            if path[0].startswith("step:"):
                return path
            for parent in parents.get(path[0], []):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append([parent, *path])
        return [node_id]


def _normal(name: str) -> str:
    name = canonicalize_name(name)
    return name[3:] if name.startswith("lib") and len(name) > 3 else name


def execute(
    plan: Plan,
    root: Path,
    *,
    index: PackageIndex,
    target: Target,
    distro: Optional[DistroIndex] = None,
    timeout: int = 1200,
    progress: Optional[Callable[[str], None]] = None,
    pip_cache: Optional[Path] = None,
) -> PlanResult:
    """Run every step, in order, and join what they need into one graph."""
    root = Path(root)
    result = PlanResult(plan=plan, root=root, evidence_problems=check_evidence(plan, root),
                        arch=target.arch)
    provided: dict[str, str] = {}   # normalised name -> id of the step that provides it

    for step in plan.ordered():
        if progress is not None:
            progress(f"{step.kind} {step.id}")
        if step.kind == PYTHON_INSTALL:
            outcome = _python_install(step, root, index, target, distro, result)
        elif step.kind == SYSTEM_PACKAGES:
            outcome = _system_packages(step, distro, result)
        elif step.kind == PYTHON_RUN:
            outcome = _python_run(step, root, result, timeout, pip_cache, progress)
        else:
            assert step.kind in (CMAKE_CONFIGURE, MESON_SETUP)
            outcome = _configure(
                step, plan, root, target, distro, timeout, provided, result, pip_cache
            )
        result.steps.append(outcome)
        for name in step.provides:
            provided.setdefault(_normal(name), step.id)
        if outcome.analysis is not None:
            for name in outcome.analysis.packages:
                provided.setdefault(_normal(name), step.id)

    _plan_check(result)
    _usage(result)
    result.unconfigured = _unconfigured(plan, root)
    result.answer = _answer(result)
    return result


#: What the executor can configure, by the file at a project's root.
_CONFIGURABLE = {"CMakeLists.txt": CMAKE_CONFIGURE, "meson.build": MESON_SETUP}


def _unconfigured(plan: Plan, root: Path) -> list[str]:
    """The root's configurable build files, when no default step builds the
    project: no configure, no driver of its own, and no install of its own
    pyproject.toml -- pybcj's CMakeLists.txt is for its developers, and pip
    builds it with setuptools."""
    defaults = [s for s in plan.steps if not s.optional]
    if any(s.kind in (CMAKE_CONFIGURE, MESON_SETUP, PYTHON_RUN) for s in defaults):
        return []
    if any(s.kind == PYTHON_INSTALL and s.manifest is not None
           and Path(s.manifest) == Path("pyproject.toml") for s in defaults):
        return []
    return [name for name in _CONFIGURABLE if (root / name).is_file()]


def _node(result: PlanResult, node_id: str, ecosystem: str, name: str) -> DepNode:
    node = result.nodes.get(node_id)
    if node is None:
        node = result.nodes[node_id] = DepNode(id=node_id, ecosystem=ecosystem, name=name)
    return node


def _edge(result: PlanResult, source: str, target: str) -> None:
    if source != target and (source, target) not in result.edges:
        result.edges.append((source, target))


def _need(node: DepNode, step: Step, basis: str) -> None:
    """Record that this step needs the node: required, or optional behind a
    flag. ``basis``: how the step knows it (``BY_CONFIGURE`` ... ``BY_LIST``)."""
    if step.optional:
        flag = step.enabled_by or step.id
        if flag not in node.optional_via:
            node.optional_via.append(flag)
    else:
        node.required = True
        if node.basis is None or _BASIS_RANK[basis] < _BASIS_RANK[node.basis]:
            node.basis = basis


def proven(result: PlanResult) -> list[str]:
    """What the default build is shown to need: by a configure that ran to the
    end, or by the project's own manifest. A dependency counts once -- as the
    configure's node, whatever apt line also installs it."""
    return sorted(n.id for n in result.nodes.values() if n.required and n.basis in PROVEN)


def _settle(
    node: DepNode, tier: str, detail: str, provided_by: Optional[str] = None
) -> None:
    """Keep the best way any step found to have it."""
    if node.detail == "" or _RANK[tier] < _RANK[node.tier]:
        node.tier, node.detail = tier, detail
        if tier == PROVIDED:
            node.package = None   # built by the plan: nothing to install
            node.provided_by = provided_by


# -------------------------------------------------------------------- steps


_PYPI_TIER = {
    Verdict.PURE_PYTHON: BINARY,
    Verdict.WHEEL_AVAILABLE: BINARY,
    Verdict.NEEDS_BUILD: SOURCE,
    Verdict.NEEDS_BUILD_BLOCKED: SOURCE,
    Verdict.NO_DISTRIBUTION: NONE,
    Verdict.UNRESOLVED: UNKNOWN,
}


def _python_install(
    step: Step, root: Path, index: PackageIndex, target: Target,
    distro: Optional[DistroIndex], result: PlanResult,
) -> StepResult:
    """Resolved against the index for the target, never installed."""
    if step.manifest is None:
        try:
            requirements = [Requirement(spec) for spec in step.packages]
        except InvalidRequirement as exc:
            return StepResult(step, "failed", f"{exc}")
        roots = RootRequirements(source=f"step {step.id}", runtime=requirements)
    else:
        try:
            roots = load(root / step.manifest, tuple(step.extras))
        except (OSError, ValueError) as exc:
            return StepResult(step, "failed", f"{step.manifest}: {exc}")
    if step.section == "build-system":
        # What building it needs: its [build-system].requires, installed
        # the way pip's build isolation would.
        roots = RootRequirements(
            source=roots.source, project_name=roots.project_name,
            runtime=list(roots.build), warnings=roots.warnings,
        )
        if not roots.runtime:
            return StepResult(step, "done", "declares no build requirements")
    analysis = Analyzer(index, target).run(roots)
    here = f"step:{step.id}"
    for report in analysis.packages.values():
        node = _node(result, f"pypi:{report.name}", "pypi", report.name)
        node.version = node.version or report.version
        _need(node, step, BY_LIST if step.manifest is None or _ci_manifest(step.manifest)
              else BY_MANIFEST)
        if step.id not in node.steps:
            node.steps.append(step.id)
        detail = report.reasons[0] if report.reasons else report.verdict.value
        tier = _PYPI_TIER[report.verdict]
        if tier != BINARY and distro is not None and distro.available:
            packaged = distro.python_package(report.name)
            if packaged:
                detail += f"; {distro.spec.label} ships {packaged} for {distro.arch}"
        _settle(node, tier, detail)
        for parent in report.required_by:
            # "(project)", or "jax 0.11.2": the requirement's name and version.
            if parent == "(project)":
                _edge(result, here, node.id)
            else:
                _edge(result, f"pypi:{canonicalize_name(parent.split()[0])}", node.id)
    blocked = len([r for r in analysis.packages.values() if _PYPI_TIER[r.verdict] == NONE])
    detail = f"{len(analysis.packages)} packages resolved" + (
        f", {blocked} with nothing for {target.arch}" if blocked else ""
    )
    return StepResult(step, "done", detail, analysis=analysis)


def _system_packages(
    step: Step, distro: Optional[DistroIndex], result: PlanResult
) -> StepResult:
    checked = distro is not None and distro.available
    missing = 0
    for spec in step.packages:
        parsed = package_spec(spec)
        assert parsed is not None   # parse_plan checked
        package, operator, wanted = parsed
        node = _node(result, f"debian:{package}", "debian", package)
        if wanted and (node.wants is None or _newer(wanted, node.wants)):
            node.wants = wanted
        _need(node, step, BY_LIST)
        node.steps.append(step.id)
        _edge(result, f"step:{step.id}", node.id)
        if not checked or distro is None:
            _settle(node, UNKNOWN, "archive not checked")
            continue
        where = f"{distro.spec.label} for {distro.arch}"
        if not distro.has(package) and _unnamed(distro, package):
            # Not an answer about the target: nobody ships anything by that name.
            near = distro.similar(package)
            _settle(node, UNKNOWN, f"{distro.spec.label} has no package named {package}"
                    + (f"; it has {', '.join(near)}" if near else ""))
            result.unknown_names.append((step.id, package, near, distro.spec.label))
            continue
        have = distro.version(package) if wanted and distro.has(package) else None
        shortfall = None
        if not distro.has(package):
            shortfall = f"not in {where}"
        elif wanted and not _satisfies(have, operator, wanted):
            # The name is there; the version the build pins is not.
            shortfall = (
                f"{where} has {have or 'an unknown version'}; the build wants "
                f"{operator} {wanted}"
            )
        if shortfall is not None:
            tier, detail = _elsewhere(package, distro.arch, shortfall)
            if tier == NONE:
                missing += 1
            _settle(node, tier, detail)
            continue
        _settle(node, BINARY, f"in {where}" + (f", {have}" if have and wanted else ""))
        node.package = package
    count = len(step.packages)
    detail = f"{count} package{'s' if count != 1 else ''}" + (
        f", {missing} missing" if missing else ""
    )
    return StepResult(step, "done", detail)


def _python_run(
    step: Step, root: Path, result: PlanResult, timeout: int,
    pip_cache: Optional[Path], progress: Optional[Callable[[str], None]],
) -> StepResult:
    """The project's own build driver, run on the host with its tools shimmed."""
    assert step.script is not None
    if not (root / step.script).is_file():
        return StepResult(step, "failed", f"no {step.script} in the repository")
    dists = {n.name: n.version for n in result.nodes.values() if n.ecosystem == "pypi"}
    trace = drive.trace(
        root, step.script, step.args, dists, timeout=timeout, pip_cache=pip_cache,
        progress=progress,
    )
    here = f"step:{step.id}"
    known = {canonicalize_name(name) for name in dists}
    for dist in sorted(trace.imported & known):
        _edge(result, here, f"pypi:{dist}")
    status = "stopped" if trace.error else "done"
    detail = (
        f"imported {len(trace.imported & known)} of the {len(known)} packages the plan "
        f"installs; called build tools {len(trace.commands)} times"
    )
    if trace.error:
        detail += f"; {trace.error}"
    elif trace.returncode not in (0, None):
        detail += f"; the driver exited with {trace.returncode}"
    return StepResult(step, status, detail, trace=trace)


def _traced(outcome: StepResult) -> bool:
    """A driver run that went to the end: only then is "never imported" true."""
    trace = outcome.trace
    return trace is not None and not trace.error and trace.returncode == 0


def _usage(result: PlanResult) -> None:
    """Which Python packages the build uses, from what its driver imported.

    Imported is used. Declared by something imported, and never loaded, is
    installed as written but not needed by the build: pyrometheus brings
    jaxlib, and MFC's build never touches it. Anything else is unused -- but
    only once a driver run went to the end; a run that stopped early proves
    nothing about what it would have imported next.
    """
    runs = [s for s in result.steps if s.trace is not None]
    if not runs:
        return
    imported = {f"pypi:{d}" for s in runs if s.trace is not None for d in s.trace.imported}
    imported |= {
        n.id for n in result.nodes.values()
        if n.ecosystem == "pypi" and n.id in result.used_as_provided
    }
    children: dict[str, list[str]] = {}
    for source, target in result.edges:
        if source.startswith("pypi:") and target.startswith("pypi:"):
            children.setdefault(source, []).append(target)
    declared: dict[str, list[str]] = {}
    for origin in sorted(imported):
        frontier = list(children.get(origin, []))
        seen: set = set()
        while frontier:
            node_id = frontier.pop()
            if node_id in seen or node_id in imported:
                continue
            seen.add(node_id)
            declared.setdefault(node_id, [])
            if origin not in declared[node_id]:
                declared[node_id].append(origin)
            frontier += children.get(node_id, [])
    complete = all(_traced(s) for s in runs)
    for node in result.nodes.values():
        if node.ecosystem != "pypi":
            continue
        if node.id in imported:
            node.usage = "imported"
        elif node.id in declared:
            node.usage = "declared"
            node.declared_by = [
                result.nodes[d].name for d in declared[node.id] if d in result.nodes
            ]
        elif complete:
            node.usage = "unused"


def _plan_check(result: PlanResult) -> None:
    """Hold the configures the driver actually ran up against the plan's.

    The plan says MFC's post_process target is configured with
    -DMFC_POST_PROCESS=ON; the driver either did that or it did not.
    """
    for outcome in result.steps:
        trace = outcome.trace
        if trace is None:
            continue
        # A default driver run is held against the default configures only:
        # the plan's --gpu variant is not something ./mfc.sh build runs.
        planned = [
            s for s in result.plan.steps
            if s.kind == CMAKE_CONFIGURE and s.optional == outcome.step.optional
        ]
        observed = [_configure_of(argv) for tool, argv in trace.commands if tool == "cmake"]
        configures = [c for c in observed if c is not None]
        matched: set = set()
        unplanned: list[str] = []
        for source, defines in configures:
            hits = [
                s.id for s in planned
                if _same_source(s.source, source)
                and all(defines.get(k) == v for k, v in s.defines.items())
            ]
            if hits:
                matched.update(hits)
            else:
                flags = " ".join(f"-D{k}={v}" for k, v in sorted(defines.items()) if "MFC_" in k
                                 or k == "CMAKE_BUILD_TYPE")
                unplanned.append(f"{source} {flags}".strip())
        outcome.driven = sorted(matched)
        checks = [f"ran {len(configures)} configure{'s' if len(configures) != 1 else ''}; "
                  f"{len(matched)} of the plan's {len(planned)} cmake steps match"]
        checks += [f"the plan has {s.id!r}, which the driver never configured"
                   for s in planned if s.id not in matched]
        checks += [f"the driver configured {u}, which no plan step does" for u in unplanned]
        outcome.plan_check = checks


def _configure_of(argv: list[str]) -> Optional[tuple[str, dict]]:
    """``(source, -D flags)`` for a cmake configure; None for --build and friends."""
    if not argv or argv[0].startswith(("--build", "--install", "-E", "--version", "-P")):
        return None
    source = None
    defines: dict[str, str] = {}
    for index, arg in enumerate(argv):
        if arg == "-S" and index + 1 < len(argv):
            source = argv[index + 1]
        elif arg.startswith("-S") and len(arg) > 2:
            source = arg[2:]
        elif arg.startswith("-D") and "=" in arg:
            name, value = arg[2:].split("=", 1)
            defines[name.split(":", 1)[0]] = value
    if source is None:
        return None
    return source, defines


def _same_source(planned: str, observed: str) -> bool:
    def norm(path: str) -> str:
        path = path.strip().rstrip("/")
        return "." if path in ("", ".", "./") else path.removeprefix("./")
    return norm(planned) == norm(observed)


def _unnamed(distro: DistroIndex, package: str) -> bool:
    """No package by this name on any arch: a wrong name, not a missing package."""
    named = getattr(distro, "named", None)
    return named is not None and named(package) is False


def _elsewhere(package: str, arch: str, shortfall: str) -> tuple[str, str]:
    """Where a tool the archive cannot supply can be had instead, if anywhere."""
    upstream = UPSTREAM_BINARIES.get(package)
    if upstream is not None and arch in upstream[1]:
        installer, triples = upstream
        return BINARY, f"{shortfall}; {installer} ships {triples[arch]} toolchains"
    from .upstream import UPSTREAM

    if package in UPSTREAM:
        return SOURCE, f"{shortfall}; {UPSTREAM[package].note}"
    return NONE, shortfall


def _newer(a: str, b: str) -> bool:
    from packaging.version import InvalidVersion, Version

    try:
        return Version(a) > Version(b)
    except InvalidVersion:
        return False


def _satisfies(have: Optional[str], operator: Optional[str], wanted: str) -> bool:
    """Whether a Debian version meets ``>= wanted`` (or ``== wanted``)."""
    from packaging.version import InvalidVersion, Version

    from .distro import upstream_version

    upstream = upstream_version(have) if have else None
    if upstream is None:
        return False
    try:
        mine, theirs = Version(upstream), Version(upstream_version(wanted) or wanted)
    except InvalidVersion:
        return False
    return mine == theirs if operator == "==" else mine >= theirs


def _configure(
    step: Step, plan: Plan, root: Path, target: Target, distro: Optional[DistroIndex],
    timeout: int, provided: dict[str, str], result: PlanResult,
    pip_cache: Optional[Path] = None,
) -> StepResult:
    """A CMake configure or a Meson setup, and what it asked for."""
    source = root / step.source
    # What earlier steps installed is what the configure's interpreter can
    # import, and what gives it its build tools: numpy, Cython, meson.
    dists = {n.name: n.version for n in result.nodes.values() if n.ecosystem == "pypi"}
    if step.kind == MESON_SETUP:
        outcome = run_pseudomeson(
            source, timeout=timeout, arch=target.arch, options=step.defines,
            python_dists=dists, pip_cache=pip_cache, meson=step.meson,
        )
        missing = "meson.build"
    else:
        outcome = run_pseudobuild(
            source, timeout=timeout, arch=target.arch, defines=step.defines,
            python_dists=dists, pip_cache=pip_cache, prune=True,
        )
        missing = "CMakeLists.txt"
    if outcome is None:
        return StepResult(step, "failed", _no_project(root, step.source, missing))
    checks = [] if step.optional else _moved_defaults(step, outcome, root)
    if outcome.error and not outcome.probes and not outcome.blockers and not outcome.completed:
        # Nothing asked for and nothing blamed. A default build that turns on
        # CUDA dies at project(), before any find_package, with CUDA to blame.
        return StepResult(step, "failed", outcome.error, pseudobuild=outcome, plan_check=checks)
    graph = depgraph.build(outcome, f"{plan.repo}/{step.id}", distro)
    here = f"step:{step.id}"
    ids = {key: f"{node.kind}:{key}" for key, node in graph.nodes.items()}
    for key, gnode in graph.nodes.items():
        node = _node(result, ids[key], gnode.kind, gnode.name)
        needed = gnode.status == depgraph.REQUIRED or gnode.asked_required
        if needed:
            _need(node, step, BY_CONFIGURE if outcome.completed else BY_STOPPED)
        elif gnode.status == depgraph.OPTIONAL and not step.optional:
            node.carried_on = True
        if step.id not in node.steps:
            node.steps.append(step.id)
        tier, detail, by = _cmake_tier(gnode, provided, target.arch)
        _settle(node, tier, detail, by)
        if tier == PROVIDED and needed:
            for name in [gnode.key, gnode.name, *gnode.aliases]:
                if f"pypi:{canonicalize_name(name)}" in result.nodes:
                    result.used_as_provided.add(f"pypi:{canonicalize_name(name)}")
        if node.tier == BINARY:
            node.package = node.package or gnode.available
    for parent, child in graph.edges:
        _edge(result, here if parent == depgraph.ROOT else ids[parent], ids[child])
    status = "completed" if outcome.completed else "stopped"
    chain = " → ".join(graph.nodes[k].name for k in graph.order) or "nothing required"
    detail = f"{outcome.rounds} round{'s' if outcome.rounds != 1 else ''}: {chain}"
    if outcome.python_installed:
        detail += "; installed for the host: " + ", ".join(outcome.python_installed)
    if not outcome.completed and outcome.error:
        detail += f"; stopped at {outcome.error}"
    return StepResult(step, status, detail, pseudobuild=outcome, graph=graph, plan_check=checks)


def _moved_defaults(step: Step, outcome: PseudoBuild, root: Path) -> list[str]:
    """The -D flags of a default configure that move one of the project's
    options off its default, where nothing the step cites sets it.

    The default build is the minimal spec. NEURON's MPI is ON unless someone
    says otherwise: a default step that turns it OFF reports a build without
    what the default one needs. dd4hep's Geant4 is OFF, and only the
    bootstrap script it cites turns it ON.

    A GPU backend is optional whatever the step cites: spfft's README builds
    with -DSPFFT_GPU_BACKEND=CUDA, and its default is OFF.
    """
    options = getattr(outcome, "options", None) or {}
    cited = _cited(step, root)
    moved = []
    for name, value in step.defines.items():
        if name not in options:
            continue
        default, site = options[name]
        if _gpu_on(name, value) and not _gpu_on(name, default):
            moved.append(
                f"-D{name}={value} moves the project's option from its default {default} "
                f"({site}) to a GPU backend: a default step keeps the defaults, whatever "
                "the instructions it cites build, so move it to an optional step")
        elif _truthy(value) != _truthy(default) and not _sets(name, cited):
            moved.append(
                f"-D{name}={value} moves the project's option from its default {default} "
                f"({site}), and nothing the step cites sets it: a default step keeps the "
                "defaults, so drop it, cite the install instructions that set it, or move "
                "it to an optional step")
    return moved


#: A GPU toolkit by value -- SPLA_GPU_BACKEND=CUDA, dbcsr's USE_ACCEL=cuda --
#: or a switch by name: EXCHCXX_ENABLE_HIP, KOKKOS_ENABLE_CUDA, USE_GPU. Not
#: one that turns it off: opensubdiv's NO_CUDA=1.
_GPU_VALUE = re.compile(r"^(CUDA|HIP|ROCM|SYCL)$", re.IGNORECASE)
_GPU_NAME = re.compile(r"(^|_)(CUDA|HIP|ROCM|SYCL|GPU)(_|$)", re.IGNORECASE)
_NEGATED = re.compile(r"(^|_)(NO|DISABLE|WITHOUT|SKIP)(_|$)", re.IGNORECASE)


def _gpu_on(name: str, value: str) -> bool:
    """Whether -DNAME=VALUE turns on a GPU backend."""
    if _GPU_VALUE.match(str(value).strip()):
        return True
    return bool(_GPU_NAME.search(name) and not _NEGATED.search(name) and _truthy(value))


def _no_project(root: Path, source: str, name: str) -> str:
    """Why a configure had nothing to configure, and where the project's
    CMakeLists.txt is: branson's is in src, and a plan for fckit pointed a
    configure at a .cmake script. Told only what was not there, a model sent
    the same step back three times."""
    what = (f"{source} is a file, not a directory with a {name}"
            if (root / source).is_file() else f"no {name} in {source}")
    found = _shallowest(root, name)
    if found == ["."]:
        return f"{what}; the project's {name} is at its root, source \".\""
    if found:
        return (f"{what}; the project's shallowest {name} "
                f"{'is' if len(found) == 1 else 'are'} in {', '.join(found)}: the step's "
                f"source is the directory whose {name} builds the project")
    return what


def _shallowest(root: Path, name: str, depth: int = 3) -> list[str]:
    """The directories nearest the root that hold a file by this name, at most
    four of them, none hidden: ["src"] for branson."""
    level = [root] if root.is_dir() else []
    for _ in range(depth + 1):
        found = sorted(d for d in level if (d / name).is_file())
        if found:
            return [d.relative_to(root).as_posix() or "." for d in found][:4]
        below = []
        for directory in level:
            try:
                below += sorted(child for child in directory.iterdir()
                                if child.is_dir() and not child.name.startswith(".")
                                and not child.is_symlink())
            except OSError:
                continue
        level = below
    return []


def _cited(step: Step, root: Path) -> str:
    """What the step's evidence could set a -D with: its quotes, the lines it
    cites, and the whole of a cited file that is not CMake's own -- a script,
    a README, a preset -- whose command's flags may sit a few lines from the
    line cited: uamqp's setup.py. A CMake file names its options to declare
    and test them -- NEURON's "MUSIC requires -DNRN_ENABLE_MPI=ON" -- so only
    the lines cited count there."""
    said = []
    inside = root.resolve()
    for evidence in step.evidence:
        said.append(evidence.quote or "")
        path = (root / evidence.path).resolve()
        if not path.is_relative_to(inside):
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except (OSError, ValueError):
            continue
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
            said += lines[max(0, evidence.start - 1):min(evidence.end, evidence.start + 200)]
        else:
            said += lines[:5000]
    return "\n".join(said)


def _sets(name: str, text: str) -> bool:
    """Whether the text sets this cache variable: ``-DNAME=``, ``-D NAME:BOOL=``,
    a preset's ``"NAME":``, or a cache file's ``set(NAME``."""
    name = re.escape(name)
    return re.search(rf'-D\s*{name}(?::\w+)?=|"{name}"\s*:|set\(\s*{name}\s', text) is not None


def _truthy(value: str) -> bool:
    """CMake's reading of an option's value: ON, YES, TRUE, Y and non-zero
    numbers are true; OFF, an empty value and anything else are false."""
    upper = str(value).strip().upper()
    if upper in ("ON", "YES", "TRUE", "Y"):
        return True
    try:
        return float(upper) != 0
    except ValueError:
        return False


#: What a vendor ships for some architectures only: NVIDIA's CUDA for x86-64,
#: arm64 and ppc64le. On any other target, a default build that will not
#: configure without it is a "no". (ROCm is left out: its libraries are
#: found, not demanded, and stubs make an auto-detected backend look present.)
VENDOR_ONLY = {
    "cuda": ("NVIDIA's CUDA", {"x86_64", "aarch64", "ppc64le"}),
    "cudatoolkit": ("NVIDIA's CUDA", {"x86_64", "aarch64", "ppc64le"}),
    "nvcc": ("NVIDIA's CUDA compiler", {"x86_64", "aarch64", "ppc64le"}),
}


def _vendor_shortfall(key: str, arch: str) -> Optional[str]:
    """Why a vendor's package cannot be had for this arch, if it cannot."""
    if key not in VENDOR_ONLY:
        return None
    what, arches = VENDOR_ONLY[key]
    return None if arch in arches else f"{what} has no {arch} build"


def _cmake_tier(
    node: depgraph.Node, provided: dict[str, str], arch: str = "riscv64"
) -> tuple[str, str, Optional[str]]:
    for name in [node.key, node.name, *node.aliases]:
        step_id = provided.get(_normal(name))
        if step_id:
            return PROVIDED, f"provided by step {step_id!r}", step_id
    shortfall = _vendor_shortfall(node.key, arch)
    if shortfall and (node.status == depgraph.REQUIRED or node.asked_required):
        return NONE, shortfall, None
    if node.provided_by:
        return TOOLCHAIN, f"comes with {node.provided_by}", None
    if node.available:
        return BINARY, f"{node.available} ✓", None
    if node.missing:
        return NONE, f"{node.debian[0]}: no package for the target", None
    if node.status != depgraph.REQUIRED and not node.asked_required:
        return UNKNOWN, node.proof, None
    unsettled = "no package name the archive knows" if node.guessed else "archive not checked"
    return UNKNOWN, unsettled, None


# ------------------------------------------------------------------- answer


def _answer(result: PlanResult) -> PlanAnswer:
    """The default build is the minimal spec; optional steps are reported apart."""
    required = [n for n in result.nodes.values() if n.required]
    blockers = sorted(n.id for n in required if n.tier == NONE)
    from_source = sorted(n.id for n in required if n.tier == SOURCE)
    unknown = sorted(n.id for n in required if n.tier == UNKNOWN)
    install = sorted({n.package for n in required if n.tier == BINARY and n.package})
    optional: dict[str, list[str]] = {}
    for node in sorted(result.nodes.values(), key=lambda n: n.id):
        if node.required or node.tier not in (NONE, SOURCE):
            continue
        for flag in node.optional_via:
            optional.setdefault(flag, []).append(node.id)
    optional_stopped = [
        s.step.id for s in result.steps
        if s.step.optional and s.status in ("stopped", "failed")
    ]
    stopped = [
        s for s in result.steps if not s.step.optional and s.status in ("stopped", "failed")
    ]
    lists: dict = {
        "blockers": blockers, "from_source": from_source, "install": install,
        "optional": optional, "optional_stopped": optional_stopped,
    }
    if blockers:
        names = ", ".join(result.nodes[b].name for b in blockers[:5])
        more = f" and {len(blockers) - 5} more" if len(blockers) > 5 else ""
        return PlanAnswer(
            "no",
            f"{names}{more}: required by the default build, and nothing for the target "
            "in the index or archive checked",
            **lists,
        )
    for outcome in stopped:
        refusal = _refuses_target(outcome, result.arch)
        if refusal:
            return PlanAnswer(
                "no", f"step {outcome.step.id!r}: the default build refuses {result.arch}: "
                f"{refusal}", **lists,
            )
    if stopped:
        first = stopped[0]
        return PlanAnswer(
            "unknown", f"step {first.step.id!r} {first.status}: {first.detail}", **lists
        )
    if result.unconfigured:
        # Read, not configured, as recursion says of such a node: the
        # packages exist, and nothing shows the project builds with them.
        once = (f", and {len(from_source)} would be built from source" if from_source
                else "")
        return PlanAnswer(
            "probably",
            f"no default step configures the {' or '.join(result.unconfigured)} at the root: "
            f"the packages listed exist for the target{once}, and that is all the run shows",
            **lists,
        )
    if from_source:
        return PlanAnswer(
            "yes-after-source-builds",
            f"every default step completes, once {len(from_source)} "
            f"package{'s' if len(from_source) != 1 else ''} are built from source",
            **lists,
        )
    if unknown:
        return PlanAnswer(
            "probably",
            f"every default step completes; {len(unknown)} requirement"
            f"{'s' if len(unknown) != 1 else ''} could not be checked",
            **lists,
        )
    return PlanAnswer(
        "yes",
        "every default step completes, and everything it needs exists for the target",
        **lists,
    )


#: The names an error might call the target by.
_ARCH_NAMES = {
    "riscv64": r"riscv64|riscv|risc-v", "aarch64": r"aarch64|arm64",
    "x86_64": r"x86_64|x86-64|amd64", "ppc64le": r"ppc64le|ppc64|powerpc64",
    "loongarch64": r"loongarch64|loongarch", "s390x": r"s390x|s390",
}
_REFUSES = re.compile(
    r"unsupported|not supported|isn'?t supported|not a supported|does not support"
    r"|unknown|unrecogni[sz]ed|no support", re.IGNORECASE,
)


def _refuses_target(outcome: StepResult, arch: str) -> Optional[str]:
    """The configure's own error, when it refuses the target by name: acts's
    "Unsupported architecture: riscv64". That is an answer, not a stop."""
    error = getattr(outcome.pseudobuild, "error", None) or ""
    message = re.sub(r"^CMake Error at \S+ \(\w+\):\s*", "", error.strip())
    names = re.compile(rf"(?<![a-z0-9])(?:{_ARCH_NAMES.get(arch, re.escape(arch))})(?![a-z0-9])",
                       re.IGNORECASE)
    named = names.search(message)
    refused = _REFUSES.search(message)
    if named and refused and abs(named.start() - refused.start()) < 80:
        return message[:200]
    return None


# ------------------------------------------------------------------- output

_TIER_STYLE = {
    BINARY: "green", TOOLCHAIN: "green", PROVIDED: "cyan",
    SOURCE: "yellow", UNKNOWN: "magenta", NONE: "bold red",
}
_VERDICT_STYLE = {
    "yes": "bold green", "yes-after-source-builds": "bold yellow",
    "probably": "bold yellow", "no": "bold red",
    "unknown": "bold magenta",
}


def render_text(result: PlanResult, console) -> None:
    from rich.text import Text

    plan = result.plan
    console.print()
    console.print(Text(f"will-it-riscv  ·  plan for {plan.repo}", style="bold"))
    if plan.entry:
        console.print(f"  entry    {plan.entry}", highlight=False)
    console.print()
    console.print(Text("Steps", style="bold"))
    for outcome in result.steps:
        mark = {"done": "✓", "completed": "✓", "stopped": "✗", "failed": "✗"}[outcome.status]
        style = "green" if mark == "✓" else ("yellow" if outcome.step.optional else "red")
        line = Text(f"  {mark} ", style=style)
        line.append(f"{outcome.step.id:<24}", style="bold")
        line.append(f" {outcome.step.kind:<16} ")
        line.append(outcome.detail)
        if outcome.step.optional:
            line.append(f"   (optional: {outcome.step.enabled_by})", style="dim")
        console.print(line, highlight=False)
    for outcome in result.steps:
        for check in outcome.plan_check:
            console.print(f"    plan check: {check}", style="dim", highlight=False)
        trace = outcome.trace
        if trace is not None and trace.not_in_plan:
            console.print(
                "    imported, and nothing in the plan provides it (stubbed): "
                + ", ".join(trace.not_in_plan),
                style="dim yellow",
                highlight=False,
            )
    for problem in result.evidence_problems:
        console.print(f"  evidence: {problem}", style="yellow", highlight=False)
    for name in result.unconfigured:
        console.print(f"  plan: no default step configures the {name} at the root, so the "
                      "answer only looked its packages up", style="yellow", highlight=False)
    console.print()

    answer = result.answer
    assert answer is not None
    line = Text("  Will it riscv?  ")
    line.append(answer.verdict.upper(), style=_VERDICT_STYLE.get(answer.verdict, "bold"))
    line.append(f" — {answer.headline}")
    console.print(line, highlight=False)
    console.print()

    def show(title: str, node_ids: list[str], style: str) -> None:
        if not node_ids:
            return
        console.print(Text(f"  {title} ({len(node_ids)})", style=style))
        for node_id in node_ids:
            node = result.nodes[node_id]
            chain = " → ".join(_label(result, n) for n in result.why(node_id))
            console.print(f"    • {node.id:<28} {node.detail}", highlight=False)
            console.print(f"      {chain}{_usage_note(node)}{_basis_note(node)}", style="dim",
                          highlight=False)

    show("required, and nothing public for the target", answer.blockers, "bold red")
    show("required, and only source for the target", answer.from_source, "bold yellow")
    imported = [n for n in answer.blockers + answer.from_source
                if result.nodes[n].usage == "imported"]
    if any(result.nodes[n].usage for n in answer.blockers + answer.from_source):
        console.print(
            f"    of these, the build step itself imports {len(imported)}"
            + (f": {', '.join(result.nodes[n].name for n in imported)}" if imported else "")
            + " — the rest are installed by default all the same",
            style="dim",
            highlight=False,
        )
    unknown = sorted(n.id for n in result.nodes.values() if n.required and n.tier == UNKNOWN)
    show("not settled", unknown, "bold magenta")
    provided = sorted(n.id for n in result.nodes.values() if n.required and n.tier == PROVIDED)
    if provided:
        console.print(Text(f"  provided by the plan itself ({len(provided)})", style="bold cyan"))
        for node_id in provided:
            node = result.nodes[node_id]
            console.print(f"    • {node.name:<20} {node.detail}", highlight=False)
    if answer.install:
        console.print(f"  sudo apt install {' '.join(answer.install)}", highlight=False)
    known = [n for n in result.nodes.values() if n.required]
    counts = {basis: sum(1 for n in known if n.basis == basis) for basis in _BASIS_RANK}
    if known:
        console.print(
            f"  how the {len(known)} requirements are known: {counts[BY_CONFIGURE]} shown by "
            f"a configure, {counts[BY_MANIFEST]} declared by the project, "
            f"{counts[BY_STOPPED]} asked for by a configure that stopped, "
            f"{counts[BY_LIST]} only named by the plan's package lists",
            style="dim", highlight=False,
        )
    if answer.optional or answer.optional_stopped:
        console.print(Text("  note — optional: only with an extra flag or choice",
                           style="bold blue"))
        crossed = any(
            result.nodes[n].tier == NONE for ids in answer.optional.values() for n in ids
        )
        for flag, node_ids in answer.optional.items():
            names = ", ".join(
                result.nodes[n].name + (" ✗" if result.nodes[n].tier == NONE else "")
                for n in node_ids
            )
            console.print(f"    {flag}:  {names}", highlight=False)
        for step_id in answer.optional_stopped:
            outcome = next(s for s in result.steps if s.step.id == step_id)
            console.print(
                f"    {outcome.step.enabled_by}:  step {step_id!r} {outcome.status}: "
                f"{outcome.detail}",
                highlight=False,
            )
        if crossed:
            console.print("    ✗ nothing public for the target", style="dim")
    carried_on = sorted(
        n.name for n in result.nodes.values()
        if n.carried_on and not n.required and not n.optional_via
    )
    if carried_on:
        console.print(
            f"  note — optional in the configures: absent, and they carried on "
            f"({len(carried_on)}): "
            + ", ".join(carried_on),
            style="dim",
            highlight=False,
        )
    console.print(
        f"  graph: {len(result.nodes)} nodes, {len(result.edges)} edges — "
        "--graph deps.html to explore it",
        style="dim",
    )
    console.print()


def _basis_note(node: DepNode) -> str:
    """How a requirement is known, when not by a configure or a manifest."""
    if node.basis == BY_STOPPED:
        return "   · asked for by a configure that stopped: not proven"
    if node.basis == BY_LIST:
        return "   · only named by the plan's package lists"
    return ""


def _usage_note(node: DepNode) -> str:
    """What the build step's trace showed. Information, never a demotion."""
    if node.usage == "imported":
        return "   · the build step imports it"
    if node.usage == "declared":
        return (
            f"   · installed by default with {', '.join(node.declared_by)}; "
            "the build step itself never loads it"
        )
    if node.usage == "unused":
        return "   · installed by default; the build step itself never loads it"
    return ""


def _label(result: PlanResult, node_id: str) -> str:
    if node_id.startswith("step:"):
        return node_id[len("step:"):]
    node = result.nodes.get(node_id)
    return node.name if node else node_id


def to_dict(result: PlanResult) -> dict:
    answer = result.answer
    return {
        "repo": result.plan.repo,
        "entry": result.plan.entry,
        "answer": (
            {
                "verdict": answer.verdict,
                "headline": answer.headline,
                "blockers": answer.blockers,
                "from_source": answer.from_source,
                "install": answer.install,
                "optional": answer.optional,
                "optional_stopped": answer.optional_stopped,
            }
            if answer else None
        ),
        "evidence_problems": result.evidence_problems,
        "unconfigured": result.unconfigured,
        "steps": [
            {
                "id": s.step.id,
                "kind": s.step.kind,
                "optional": s.step.optional,
                "enabled_by": s.step.enabled_by,
                "status": s.status,
                "detail": s.detail,
                "hard_requirements": (
                    [s.graph.nodes[k].name for k in s.graph.order] if s.graph else []
                ),
                "plan_check": s.plan_check,
                "trace": (
                    {
                        "imported": sorted(s.trace.imported),
                        "installed_for_the_host": s.trace.installed,
                        "not_in_plan": s.trace.not_in_plan,
                        "stubbed": s.trace.stubbed,
                        "made": s.trace.made,
                        "rounds": s.trace.rounds,
                        "returncode": s.trace.returncode,
                        "commands": [{"tool": t, "argv": a} for t, a in s.trace.commands],
                    }
                    if s.trace else None
                ),
            }
            for s in result.steps
        ],
        "nodes": [
            {
                "id": n.id,
                "ecosystem": n.ecosystem,
                "name": n.name,
                "version": n.version,
                "tier": n.tier,
                "detail": n.detail,
                "required": n.required,
                "basis": n.basis,
                "optional_via": n.optional_via,
                "carried_on": n.carried_on,
                "steps": n.steps,
                "package": n.package,
                "provided_by": n.provided_by,
                "usage": n.usage,
                "declared_by": n.declared_by,
                "why": result.why(n.id),
            }
            for n in sorted(result.nodes.values(), key=lambda n: n.id)
        ],
        "edges": [{"from": a, "to": b} for a, b in result.edges],
    }


_FILL = {
    BINARY: "#dcefdc", TOOLCHAIN: "#dcefdc", PROVIDED: "#d9eef2",
    SOURCE: "#fbefc4", UNKNOWN: "#eeeeee", NONE: "#a3122a",
}


def to_dot(result: PlanResult, required_only: bool = True) -> str:
    """One cluster per step, one node per dependency, coloured by tier."""
    from .graph import _quote

    answer = result.answer.verdict if result.answer else "?"
    lines = [
        f"digraph {_quote(result.plan.repo)} {{",
        f"  graph [rankdir=LR, fontname=Helvetica, labelloc=t, "
        f"label={_quote(f'{result.plan.repo} — plan: {answer}')}];",
        '  node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11];',
        '  edge [color="#9a9a9a", arrowsize=0.6];',
    ]
    keep = {
        n.id for n in result.nodes.values() if n.required or n.optional_via or not required_only
    }
    for outcome in result.steps:
        step = outcome.step
        fill = "#e2e2e2" if outcome.status in ("done", "completed") else "#f6c9c9"
        label = f"{step.id}\n{step.kind}"
        style = "filled,bold"
        if step.optional:
            label += f"\noptional: {step.enabled_by}"
            style = "filled,dashed"
        lines.append(
            f"  {_quote('step:' + step.id)} [label={_quote(label)}, shape=box, "
            f'style="{style}", fillcolor="{fill}"];'
        )
    for node in sorted(result.nodes.values(), key=lambda n: n.id):
        if node.id not in keep:
            continue
        label = f"{node.name}\n{node.ecosystem}" + (f" {node.version}" if node.version else "")
        font = ', fontcolor="white"' if node.tier == NONE else ""
        faded = (
            ', style="rounded,filled,dashed", color="#6f8fbf"'
            if not node.required and node.optional_via else ""
        )
        tip = f"{node.tier}: {node.detail}" + (f" · {node.usage}" if node.usage else "")
        lines.append(
            f"  {_quote(node.id)} [label={_quote(label)}, fillcolor=\"{_FILL[node.tier]}\""
            f"{font}{faded}, tooltip={_quote(tip)}];"
        )
    for step in result.plan.steps:
        for before in step.after:
            lines.append(
                f"  {_quote('step:' + before)} -> {_quote('step:' + step.id)} "
                '[style=dotted, color="#555555"];'
            )
    for source, target in result.edges:
        if target in keep and (source.startswith("step:") or source in keep):
            lines.append(f"  {_quote(source)} -> {_quote(target)};")
    for node in result.nodes.values():
        if node.provided_by and node.id in keep:
            lines.append(
                f"  {_quote('step:' + node.provided_by)} -> {_quote(node.id)} "
                '[style=dashed, color="#2a8aa0", label="provides", fontsize=9];'
            )
    lines.append("}")
    return "\n".join(lines)


def exit_code(result: PlanResult) -> int:
    verdict = result.answer.verdict if result.answer else "unknown"
    return {"yes": 0, "yes-after-source-builds": 1, "probably": 1}.get(verdict, 2)

