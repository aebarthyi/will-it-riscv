"""Run a plan in the pretend environment, and join what its steps need.

Nothing here builds anything, and nothing emulates riscv64. Each step is
answered the cheapest way that is still honest:

  python-install    resolved against the index for the target's wheel tags
  system-packages   looked up in the distro's riscv64 archive
  cmake-configure   configured for real, confined to an empty sysroot that
                    grows a stub for whatever the configure insists on
  python-run        the project's own build driver, run on the host with its
                    build tools shimmed, to see which packages it imports

Every step's asks land in one graph, keyed by ecosystem so that PyPI's numpy
and Debian's python3-numpy stay two things. A requirement that an earlier
step provides -- MFC's CMake wants fypp, which its Python toolchain installs;
it wants FFTW, which its own dependency targets build from source -- is
resolved to that step rather than looked for in the archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from packaging.utils import canonicalize_name

from . import drive
from . import graph as depgraph
from .analyze import Analyzer
from .inputs import load
from .models import Verdict
from .plan import (
    CMAKE_CONFIGURE,
    PYTHON_INSTALL,
    PYTHON_RUN,
    SYSTEM_PACKAGES,
    Plan,
    Step,
    check_evidence,
)
from .pseudobuild import PseudoBuild
from .pseudobuild import run as run_pseudobuild

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
_RANK = {PROVIDED: -1, BINARY: 0, TOOLCHAIN: 0, SOURCE: 1, UNKNOWN: 2, NONE: 3}


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
    timeout: int = 600,
    progress: Optional[Callable[[str], None]] = None,
    pip_cache: Optional[Path] = None,
) -> PlanResult:
    """Run every step, in order, and join what they need into one graph."""
    root = Path(root)
    result = PlanResult(plan=plan, root=root, evidence_problems=check_evidence(plan, root))
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
            assert step.kind == CMAKE_CONFIGURE
            outcome = _cmake_configure(
                step, plan, root, target, distro, timeout, provided, result
            )
        result.steps.append(outcome)
        for name in step.provides:
            provided.setdefault(_normal(name), step.id)
        if outcome.analysis is not None:
            for name in outcome.analysis.packages:
                provided.setdefault(_normal(name), step.id)

    _plan_check(result)
    _usage(result)
    result.answer = _answer(result)
    return result


def _node(result: PlanResult, node_id: str, ecosystem: str, name: str) -> DepNode:
    node = result.nodes.get(node_id)
    if node is None:
        node = result.nodes[node_id] = DepNode(id=node_id, ecosystem=ecosystem, name=name)
    return node


def _edge(result: PlanResult, source: str, target: str) -> None:
    if source != target and (source, target) not in result.edges:
        result.edges.append((source, target))


def _need(node: DepNode, step: Step) -> None:
    """Record that this step needs the node: required, or optional behind a flag."""
    if step.optional:
        flag = step.enabled_by or step.id
        if flag not in node.optional_via:
            node.optional_via.append(flag)
    else:
        node.required = True


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
    assert step.manifest is not None
    try:
        roots = load(root / step.manifest, tuple(step.extras))
    except (OSError, ValueError) as exc:
        return StepResult(step, "failed", f"{step.manifest}: {exc}")
    analysis = Analyzer(index, target).run(roots)
    here = f"step:{step.id}"
    for report in analysis.packages.values():
        node = _node(result, f"pypi:{report.name}", "pypi", report.name)
        node.version = node.version or report.version
        _need(node, step)
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
    for package in step.packages:
        node = _node(result, f"debian:{package}", "debian", package)
        _need(node, step)
        node.steps.append(step.id)
        _edge(result, f"step:{step.id}", node.id)
        if not checked:
            _settle(node, UNKNOWN, "archive not checked")
        elif distro is not None and distro.has(package):
            _settle(node, BINARY, f"in {distro.spec.label} for {distro.arch}")
            node.package = package
        else:
            missing += 1
            label = distro.spec.label if distro is not None else "the archive"
            _settle(node, NONE, f"not in {label} for {distro.arch if distro else 'the target'}")
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


def _cmake_configure(
    step: Step, plan: Plan, root: Path, target: Target, distro: Optional[DistroIndex],
    timeout: int, provided: dict[str, str], result: PlanResult,
) -> StepResult:
    source = root / step.source
    outcome = run_pseudobuild(source, timeout=timeout, arch=target.arch, defines=step.defines)
    if outcome is None:
        return StepResult(step, "failed", f"no CMakeLists.txt in {step.source}")
    if outcome.error and not outcome.probes and not outcome.completed:
        return StepResult(step, "failed", outcome.error, pseudobuild=outcome)
    graph = depgraph.build(outcome, f"{plan.repo}/{step.id}", distro)
    here = f"step:{step.id}"
    ids = {key: f"{node.kind}:{key}" for key, node in graph.nodes.items()}
    for key, gnode in graph.nodes.items():
        node = _node(result, ids[key], gnode.kind, gnode.name)
        needed = gnode.status == depgraph.REQUIRED or gnode.asked_required
        if needed:
            _need(node, step)
        elif gnode.status == depgraph.OPTIONAL and not step.optional:
            node.carried_on = True
        if step.id not in node.steps:
            node.steps.append(step.id)
        tier, detail, by = _cmake_tier(gnode, provided)
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
    if not outcome.completed and outcome.error:
        detail += f"; stopped at {outcome.error}"
    return StepResult(step, status, detail, pseudobuild=outcome, graph=graph)


def _cmake_tier(
    node: depgraph.Node, provided: dict[str, str]
) -> tuple[str, str, Optional[str]]:
    for name in [node.key, node.name, *node.aliases]:
        step_id = provided.get(_normal(name))
        if step_id:
            return PROVIDED, f"provided by step {step_id!r}", step_id
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
    if stopped:
        first = stopped[0]
        return PlanAnswer(
            "unknown", f"step {first.step.id!r} {first.status}: {first.detail}", **lists
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
            console.print(f"      {chain}{_usage_note(node)}", style="dim", highlight=False)

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
    if answer.optional or answer.optional_stopped:
        console.print(Text("  optional — only with an extra flag or choice", style="bold blue"))
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
            f"  optional in the configures — absent, and they carried on ({len(carried_on)}): "
            + ", ".join(carried_on),
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
    console.print(
        f"  graph: {len(result.nodes)} nodes, {len(result.edges)} edges — "
        "-f dot | dot -Tsvg > plan.svg",
        style="dim",
    )
    console.print()


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

