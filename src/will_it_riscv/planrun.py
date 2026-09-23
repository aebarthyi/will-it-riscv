"""Run a plan in the pretend environment, and join what its steps need.

Nothing here builds anything, and nothing emulates riscv64. Each step is
answered the cheapest way that is still honest:

  python-install    resolved against the index for the target's wheel tags
  system-packages   looked up in the distro's riscv64 archive
  cmake-configure   configured for real, confined to an empty sysroot that
                    grows a stub for whatever the configure insists on

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

from . import graph as depgraph
from .analyze import Analyzer
from .inputs import load
from .models import Verdict
from .plan import CMAKE_CONFIGURE, PYTHON_INSTALL, SYSTEM_PACKAGES, Plan, Step, check_evidence
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
    steps: list[str] = field(default_factory=list)
    """Every step that asked for it."""
    package: Optional[str] = None
    """The distro package that provides it for the target, when one does."""
    provided_by: Optional[str] = None
    """The id of the step that makes it exist, when the plan does."""


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


@dataclass
class PlanAnswer:
    verdict: str
    """``yes``, ``yes-after-source-builds``, ``probably``, ``no`` or ``unknown``."""
    headline: str
    blockers: list[str] = field(default_factory=list)
    from_source: list[str] = field(default_factory=list)
    install: list[str] = field(default_factory=list)


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
        node.required = True
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
        node.required = True
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
        node.required = node.required or needed
        if step.id not in node.steps:
            node.steps.append(step.id)
        _settle(node, *_cmake_tier(gnode, provided))
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
    required = [n for n in result.nodes.values() if n.required]
    blockers = sorted(n.id for n in required if n.tier == NONE)
    from_source = sorted(n.id for n in required if n.tier == SOURCE)
    unknown = sorted(n.id for n in required if n.tier == UNKNOWN)
    install = sorted({n.package for n in required if n.tier == BINARY and n.package})
    stopped = [s for s in result.steps if s.status in ("stopped", "failed")]
    if blockers:
        names = ", ".join(result.nodes[b].name for b in blockers[:5])
        more = f" and {len(blockers) - 5} more" if len(blockers) > 5 else ""
        return PlanAnswer(
            "no",
            f"{names}{more}: required, and nothing for the target in the index or "
            "archive checked",
            blockers, from_source, install,
        )
    if stopped:
        first = stopped[0]
        return PlanAnswer(
            "unknown",
            f"step {first.step.id!r} {first.status}: {first.detail}",
            blockers, from_source, install,
        )
    if from_source:
        return PlanAnswer(
            "yes-after-source-builds",
            f"every step completes, once {len(from_source)} "
            f"package{'s' if len(from_source) != 1 else ''} are built from source",
            blockers, from_source, install,
        )
    if unknown:
        return PlanAnswer(
            "probably",
            f"every step completes; {len(unknown)} requirement"
            f"{'s' if len(unknown) != 1 else ''} could not be checked",
            blockers, from_source, install,
        )
    return PlanAnswer(
        "yes", "every step completes, and everything it needs exists for the target",
        blockers, from_source, install,
    )


# ------------------------------------------------------------------- output

_TIER_STYLE = {
    BINARY: "green", TOOLCHAIN: "green", PROVIDED: "cyan",
    SOURCE: "yellow", UNKNOWN: "magenta", NONE: "bold red",
}
_VERDICT_STYLE = {
    "yes": "bold green", "yes-after-source-builds": "bold yellow",
    "probably": "bold yellow", "no": "bold red", "unknown": "bold magenta",
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
        style = "green" if mark == "✓" else "red"
        line = Text(f"  {mark} ", style=style)
        line.append(f"{outcome.step.id:<24}", style="bold")
        line.append(f" {outcome.step.kind:<16} ")
        line.append(outcome.detail)
        console.print(line, highlight=False)
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
            console.print(f"      {chain}", style="dim", highlight=False)

    show("nothing public for the target", answer.blockers, "bold red")
    show("to build from source", answer.from_source, "bold yellow")
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
            }
            if answer else None
        ),
        "evidence_problems": result.evidence_problems,
        "steps": [
            {
                "id": s.step.id,
                "kind": s.step.kind,
                "status": s.status,
                "detail": s.detail,
                "hard_requirements": (
                    [s.graph.nodes[k].name for k in s.graph.order] if s.graph else []
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
                "steps": n.steps,
                "package": n.package,
                "provided_by": n.provided_by,
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
    keep = {n.id for n in result.nodes.values() if n.required or not required_only}
    for outcome in result.steps:
        step = outcome.step
        fill = "#e2e2e2" if outcome.status in ("done", "completed") else "#f6c9c9"
        label = f"{step.id}\n{step.kind}"
        lines.append(
            f"  {_quote('step:' + step.id)} [label={_quote(label)}, shape=box, "
            f'style="filled,bold", fillcolor="{fill}"];'
        )
    for node in sorted(result.nodes.values(), key=lambda n: n.id):
        if node.id not in keep:
            continue
        label = f"{node.name}\n{node.ecosystem}" + (f" {node.version}" if node.version else "")
        font = ', fontcolor="white"' if node.tier == NONE else ""
        lines.append(
            f"  {_quote(node.id)} [label={_quote(label)}, fillcolor=\"{_FILL[node.tier]}\""
            f"{font}, tooltip={_quote(f'{node.tier}: {node.detail}')}];"
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

