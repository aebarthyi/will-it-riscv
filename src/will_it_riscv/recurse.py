"""Recurse: fetch what has no riscv64 build, configure it, and build back up.

A plan's answer stops at its frontier: the packages its default build needs
that have only source, or nothing at all, for riscv64. Each of those is a
repository in its own right, and gets the same treatment as the root --
fetched at the version the plan resolved, planned from its own build files,
run in the same pretend environment -- and so on down, until everything
either has a binary for the target or cannot be had at all.

Then the answer is built back up from the leaves:

  buildable   its own configure ran and completed, and everything under it
              is buildable
  probably    its build could only be read -- Meson, Bazel, Cargo are not
              configured yet -- and nothing under it is blocked
  blocked     something it requires has nothing public for the target
  unknown     a configure stopped, a fetch failed, or the budget ran out

and the order to build things in falls out of the walk: dependencies first.
That order is the port plan.

This runs the configures of every package it fetches. They are third-party
build scripts, run on this host -- confined to a scratch sysroot, never
compiled, never emulated, but run. It is opt-in, and bounded in depth,
breadth and time.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from . import planrun
from .autoplan import auto_plan, configured
from .plan import Plan
from .planrun import BINARY, NONE, SOURCE, TOOLCHAIN, DepNode, PlanResult
from .sources import SourceTree, fetch
from .upstream import UPSTREAM
from .upstream import fetch as fetch_upstream

if TYPE_CHECKING:  # pragma: no cover
    from .distro import DistroIndex
    from .index import PackageIndex
    from .target import Target

BUILDABLE = "buildable"
PROBABLY = "probably"
BLOCKED = "blocked"
UNKNOWN = "unknown"
_WORST = {BUILDABLE: 0, PROBABLY: 1, UNKNOWN: 2, BLOCKED: 3}

MAX_DEPTH = 3
MAX_PACKAGES = 40


@dataclass
class TreeNode:
    key: str
    """``pypi:jaxlib@0.11.2``: a package at one version."""
    id: str
    name: str
    version: Optional[str]
    ecosystem: str
    tier: str
    """How the plan that reached it said it could be had: source, or none."""
    detail: str
    depth: int
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    source: Optional[SourceTree] = None
    plan: Optional[Plan] = None
    result: Optional[PlanResult] = None
    status: str = UNKNOWN
    reason: str = ""
    visited: bool = False


@dataclass
class Recursion:
    project: str
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    top: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    """What to build from source, dependencies first."""
    verdict: str = UNKNOWN
    headline: str = ""
    duration: float = 0.0
    fetched: int = 0
    configured: int = 0
    budget_hit: bool = False


def frontier(result: PlanResult) -> list[DepNode]:
    """What the default build requires that has only source, or nothing."""
    return sorted(
        (n for n in result.nodes.values() if n.required and n.tier in (NONE, SOURCE)),
        key=lambda n: n.id,
    )


def recurse(
    root: PlanResult,
    *,
    index: PackageIndex,
    target: Target,
    distro: Optional[DistroIndex],
    cache_root: Path,
    max_depth: int = MAX_DEPTH,
    max_packages: int = MAX_PACKAGES,
    timeout: int = 600,
    pip_cache: Optional[Path] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Recursion:
    started = time.monotonic()
    walk = Recursion(project=root.plan.repo)

    def visit(dep: DepNode, depth: int, path: tuple[str, ...], parent: str) -> str:
        version = dep.version or dep.wants
        key = f"{dep.id}@{version}" if version else dep.id
        node = walk.nodes.get(key)
        if node is not None:
            # Seen before -- or still being worked out further up the path,
            # which makes this a bootstrap cycle, judged by whoever closed it.
            if parent not in node.parents:
                node.parents.append(parent)
            return key
        node = walk.nodes[key] = TreeNode(
            key=key, id=dep.id, name=dep.name, version=version,
            ecosystem=dep.ecosystem, tier=dep.tier, detail=dep.detail, depth=depth,
            parents=[parent],
        )
        _explore(node, dep, depth, path + (key,))
        node.visited = True
        return key

    def _explore(node: TreeNode, dep: DepNode, depth: int, path: tuple[str, ...]) -> None:
        tool = UPSTREAM.get(node.name) if node.ecosystem != "pypi" else None
        if node.ecosystem != "pypi" and (tool is None or node.tier == NONE or not node.version):
            # A distro package with nothing for the target, or with only a
            # source nobody has said how to follow.
            if node.tier == NONE:
                node.status, node.reason = BLOCKED, node.detail or "no package for the target"
            else:
                node.status = UNKNOWN
                node.reason = f"source only: {node.detail} (not recursed into)"
            return
        if depth > max_depth:
            node.reason = f"not visited: deeper than {max_depth}"
            walk.budget_hit = True
            return
        if walk.fetched >= max_packages:
            node.reason = f"not visited: {max_packages} packages already fetched"
            walk.budget_hit = True
            return
        if progress is not None:
            progress(f"fetching {node.name} {node.version or ''}")
        if tool is not None:
            assert node.version is not None
            node.source = fetch_upstream(tool, node.version, cache_root)
        else:
            node.source = fetch(index, node.name, node.version, cache_root)
        if not node.source.ok:
            nothing = node.tier == NONE and "names no repository" in (node.source.error or "")
            node.status = BLOCKED if nothing else UNKNOWN
            node.reason = (
                "nothing public: no wheel for the target, no sdist, and no repository"
                if nothing else f"could not fetch its source: {node.source.error}"
            )
            return
        walk.fetched += 1
        assert node.source.path is not None
        if tool is not None:
            node.plan = tool.planner(node.source.path, node.version or "")
        else:
            node.plan = auto_plan(node.source.path, node.name, target.arch, cache_root)
        if progress is not None:
            progress(f"configuring {node.name} {node.version or ''}")
        node.result = planrun.execute(
            node.plan, node.source.path, index=index, target=target, distro=distro,
            timeout=timeout, pip_cache=pip_cache,
        )
        if configured(node.plan):
            walk.configured += 1
        node.children = [visit(child, depth + 1, path, node.key) for child in frontier(node.result)]
        node.status, node.reason = _judge(node, walk)

    for dep in frontier(root):
        walk.top.append(visit(dep, 1, (), "root"))

    walk.order = _build_order(walk)
    walk.verdict, walk.headline = _verdict(walk)
    walk.duration = time.monotonic() - started
    return walk


def _judge(node: TreeNode, walk: Recursion) -> tuple[str, str]:
    """Its own plan, and then the worst of what is under it."""
    assert node.result is not None and node.plan is not None
    stopped = [
        s for s in node.result.steps
        if not s.step.optional and s.status in ("stopped", "failed")
    ]
    children = [walk.nodes[c] for c in node.children]
    cycle = [c for c in children if not c.visited]
    if cycle:
        return UNKNOWN, (
            f"needs {', '.join(c.name for c in cycle)}, which needs it in turn "
            "(a bootstrap cycle)"
        )
    blocked = [c for c in children if c.status == BLOCKED]
    if blocked:
        return BLOCKED, f"needs {', '.join(c.name for c in blocked)}, which cannot be had"
    if node.plan.blocked:
        return BLOCKED, node.plan.blocked[0]
    if stopped:
        first = stopped[0]
        return UNKNOWN, f"its {first.step.id} step {first.status}: {first.detail}"
    unknown = [c for c in children if c.status == UNKNOWN]
    if unknown:
        source_only = [c for c in unknown if c.reason.startswith("source only")]
        if source_only and len(source_only) == len(unknown):
            return UNKNOWN, (
                f"needs {', '.join(c.name for c in source_only)}, which the target has "
                "only as source"
            )
        return UNKNOWN, f"needs {', '.join(c.name for c in unknown)}, which could not be settled"
    if not configured(node.plan):
        return PROBABLY, _unconfigured(node.plan)
    probably = [c for c in children if c.status == PROBABLY]
    if probably:
        return PROBABLY, (
            "its configure completed; "
            f"{', '.join(c.name for c in probably)} could only be read"
        )
    chain = ""
    for outcome in node.result.steps:
        if outcome.graph is not None and outcome.graph.order:
            chain = " → ".join(outcome.graph.nodes[k].name for k in outcome.graph.order)
    return BUILDABLE, "its configure completed" + (f": {chain}" if chain else "")


def _unconfigured(plan: Plan) -> str:
    """Why a build was only read, in a few words."""
    systems = []
    for note in plan.unsure:
        match = re.match(r"builds with (\w+)", note)
        if match:
            systems.append(match.group(1))
    if systems:
        return f"read, not configured: builds with {' and '.join(systems)}"
    return plan.unsure[0] if plan.unsure else "its build could not be configured"


def _build_order(walk: Recursion) -> list[str]:
    """Dependencies first: what has to be built before what."""
    order: list[str] = []
    seen: set = set()

    def post(key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        for child in walk.nodes[key].children:
            post(child)
        node = walk.nodes[key]
        if node.source is not None and node.source.ok:
            order.append(key)

    for key in walk.top:
        post(key)
    return order


def _verdict(walk: Recursion) -> tuple[str, str]:
    top = [walk.nodes[k] for k in walk.top]
    if not top:
        return "yes", "nothing the default build requires needs building from source"
    worst = max((n.status for n in top), key=lambda s: _WORST[s])
    count = len(walk.order)
    if worst == BLOCKED:
        blocked = [n for n in top if n.status == BLOCKED]
        chains = "; ".join(" → ".join(walk.nodes[k].name for k in chain_to_leaf(walk, n.key))
                           for n in blocked[:3])
        return "no", f"{', '.join(n.name for n in blocked)} cannot be had: {chains}"
    if worst == UNKNOWN:
        unknown = [n for n in top if n.status == UNKNOWN]
        chains = "; ".join(
            " → ".join(walk.nodes[k].name for k in chain_to_leaf(walk, n.key, UNKNOWN))
            + f" ({walk.nodes[chain_to_leaf(walk, n.key, UNKNOWN)[-1]].reason})"
            for n in unknown[:3]
        )
        return "unknown", f"{', '.join(n.name for n in unknown)} could not be settled: {chains}"
    if worst == PROBABLY:
        return "probably", (
            f"after building {count} package{'s' if count != 1 else ''} from source, "
            "some of them only read, not configured"
        )
    return "yes", f"after building {count} package{'s' if count != 1 else ''} from source"


def installed(node: TreeNode) -> list[str]:
    """What has to be installed before it can be built, as its own plan found.

    cantera's configure wants Boost, BLAS and HDF5: none of them is built
    from source for the port, but each comes from the archive first. A tool
    the archive has too old comes from its upstream -- typos wants rustc
    1.95, and rustup has it. Python packages are left out: those are the
    recursion's own children, or wheels.
    """
    if node.result is None:
        return []
    names: list[str] = []
    for dep in node.result.nodes.values():
        if not dep.required or dep.ecosystem == "pypi":
            continue
        if dep.tier == BINARY and dep.package:
            name = dep.package
        elif dep.tier == BINARY:
            upstream = re.search(r";\s*(\S+) ships", dep.detail)
            name = f"{dep.name} ({upstream.group(1) if upstream else 'upstream'})"
        elif dep.tier == TOOLCHAIN:
            name = dep.detail.removeprefix("comes with ")
        else:
            continue
        if name not in names:
            names.append(name)
    return sorted(names)


def chain_to_leaf(walk: Recursion, key: str, status: str = BLOCKED) -> list[str]:
    """Follow the first child with this status down to where it starts."""
    chain = [key]
    seen = {key}
    while True:
        node = walk.nodes[chain[-1]]
        nxt = next(
            (c for c in node.children if walk.nodes[c].status == status and c not in seen), None
        )
        if nxt is None:
            return chain
        chain.append(nxt)
        seen.add(nxt)


# ------------------------------------------------------------------- output

_MARK = {BUILDABLE: "✓", PROBABLY: "~", BLOCKED: "✗", UNKNOWN: "?"}
_STYLE = {BUILDABLE: "green", PROBABLY: "yellow", BLOCKED: "bold red", UNKNOWN: "magenta"}


def render_text(walk: Recursion, console) -> None:
    from rich.text import Text

    console.print(Text("Recursion", style="bold"))
    console.print(
        f"  fetched {walk.fetched} package{'s' if walk.fetched != 1 else ''} and configured "
        f"{walk.configured}, in {walk.duration:.0f}s"
        + ("; the depth or package budget ran out" if walk.budget_hit else ""),
        style="dim",
    )
    console.print(f"  {walk.project}")
    shown: set = set()

    def draw(key: str, prefix: str, last: bool) -> None:
        node = walk.nodes[key]
        branch = "└─ " if last else "├─ "
        line = Text(f"  {prefix}{branch}")
        line.append(f"{_MARK[node.status]} ", style=_STYLE[node.status])
        line.append(f"{node.name} {node.version or ''}".strip(), style="bold")
        if key in shown:
            line.append("  (above)", style="dim")
            console.print(line, highlight=False)
            return
        shown.add(key)
        line.append(f"  {node.status} — {node.reason}")
        if node.source is not None and node.source.ok:
            line.append(f"   [{node.source.kind} {node.source.origin}]", style="dim")
        console.print(line, highlight=False)
        extension = "   " if last else "│  "
        rail = "│  " if node.children else "   "
        for finding in node.plan.read if node.plan is not None else []:
            console.print(f"  {prefix}{extension}{rail}  {finding}", style="dim", highlight=False)
        archive = installed(node)
        if archive:
            console.print(
                f"  {prefix}{extension}{rail}  installed first: " + ", ".join(archive),
                style="dim", highlight=False,
            )
        for index, child in enumerate(node.children):
            draw(child, prefix + extension, index == len(node.children) - 1)

    for index, key in enumerate(walk.top):
        draw(key, "", index == len(walk.top) - 1)

    if walk.order:
        console.print()
        console.print(Text("  build order, dependencies first:", style="bold"))
        names = [f"{walk.nodes[k].name} {walk.nodes[k].version or ''}".strip() for k in walk.order]
        console.print("    " + " → ".join(names), highlight=False)
    console.print()
    style = {"yes": "bold green", "probably": "bold yellow", "no": "bold red"}.get(
        walk.verdict, "bold magenta"
    )
    line = Text("  Will it riscv, all the way down?  ")
    line.append(walk.verdict.upper(), style=style)
    line.append(f" — {walk.headline}")
    console.print(line, highlight=False)
    console.print()


def to_dict(walk: Recursion) -> dict:
    return {
        "project": walk.project,
        "verdict": walk.verdict,
        "headline": walk.headline,
        "fetched": walk.fetched,
        "configured": walk.configured,
        "budget_hit": walk.budget_hit,
        "duration_seconds": round(walk.duration, 1),
        "top": walk.top,
        "build_order": walk.order,
        "nodes": [
            {
                "key": n.key,
                "name": n.name,
                "version": n.version,
                "ecosystem": n.ecosystem,
                "tier": n.tier,
                "status": n.status,
                "reason": n.reason,
                "depth": n.depth,
                "parents": n.parents,
                "children": n.children,
                "installed_first": installed(n),
                "source": (
                    {"kind": n.source.kind, "origin": n.source.origin, "error": n.source.error}
                    if n.source else None
                ),
                "plan": (
                    {"steps": [s.id for s in n.plan.steps], "unsure": n.plan.unsure,
                     "read": n.plan.read, "blocked": n.plan.blocked}
                    if n.plan else None
                ),
                "answer": (
                    {"verdict": n.result.answer.verdict, "headline": n.result.answer.headline}
                    if n.result is not None and n.result.answer is not None else None
                ),
                "steps": [
                    {"id": o.step.id, "kind": o.step.kind, "status": o.status, "detail": o.detail}
                    for o in (n.result.steps if n.result is not None else [])
                ],
            }
            for n in walk.nodes.values()
        ],
    }


_FILL = {BUILDABLE: "#dcefdc", PROBABLY: "#fbefc4", BLOCKED: "#a3122a", UNKNOWN: "#eeeeee"}


def to_dot(walk: Recursion) -> str:
    """The dependency tree, as far down as it went, coloured by status."""
    from .graph import _quote

    lines = [
        f"digraph {_quote(walk.project)} {{",
        f"  graph [rankdir=LR, fontname=Helvetica, labelloc=t, "
        f"label={_quote(f'{walk.project} — all the way down: {walk.verdict}')}];",
        '  node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11];',
        '  edge [color="#9a9a9a", arrowsize=0.6];',
        f'  "root" [label={_quote(walk.project)}, style="filled,bold", fillcolor="#e2e2e2"];',
    ]
    for node in walk.nodes.values():
        label = f"{node.name} {node.version or ''}".strip() + f"\n{node.status}"
        font = ', fontcolor="white"' if node.status == BLOCKED else ""
        lines.append(
            f"  {_quote(node.key)} [label={_quote(label)}, fillcolor=\"{_FILL[node.status]}\""
            f"{font}, tooltip={_quote(node.reason)}];"
        )
    for node in walk.nodes.values():
        for parent in node.parents:
            lines.append(f"  {_quote(parent)} -> {_quote(node.key)};")
    lines.append("}")
    return "\n".join(lines)
