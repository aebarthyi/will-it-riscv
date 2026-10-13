"""The dependency graph, drawn: an interactive page, or a picture.

``will-it-riscv plan --graph deps.html`` writes one self-contained file -- no
Graphviz, and nothing fetched from anywhere when it opens -- with the plan's
steps on the left and what each needs flowing right, coloured as the
report's tiers are. Click a node for why the build needs it (the chain from
a step down to it), what settled it, and the lines the plan cited. Optional
dependencies can be hidden; the page pans, zooms and searches.
``--graph deps.svg`` writes the same picture, still.

The layout is a small layered one, after Sugiyama: ranks by longest path,
long edges through invisible waypoints, barycentre sweeps to cut crossings,
then each node pulled toward its neighbours as far as the nodes above and
below it allow. Plan graphs run from a handful of nodes to a few hundred,
which that handles well.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Optional

from . import planrun

if TYPE_CHECKING:  # pragma: no cover
    from .planrun import PlanResult
    from .recurse import Recursion

#: Tones: what the colours mean, in the report's words.
TONES = {
    "step": "a step of the plan",
    "step-stopped": "a step that stopped",
    "archive": "in the distro's archive",
    "provided": "made by a step of the plan",
    "source": "only source for the target",
    "unknown": "not settled",
    "none": "nothing public for the target",
    "buildable": "builds from source",
    "probably": "probably builds from source",
}
_TIER_TONE = {
    planrun.BINARY: "archive", planrun.TOOLCHAIN: "archive", planrun.PROVIDED: "provided",
    planrun.SOURCE: "source", planrun.UNKNOWN: "unknown", planrun.NONE: "none",
}
_STATUS_TONE = {"buildable": "buildable", "probably": "probably", "blocked": "none",
                "unknown": "unknown"}

TITLE_CHARS = 32
SUBTITLE_CHARS = 40
_TITLE_PX, _SUBTITLE_PX = 7.3, 6.2     # average glyph widths at 12.5 and 10.5 px
_PAD = 14
_RANK_GAP = 76
_NODE_GAP = 14
_WAYPOINT_GAP = 8
_MARGIN = 24


@dataclass
class Node:
    id: str
    title: str
    subtitle: str = ""
    step: bool = False
    tone: str = "unknown"
    optional: bool = False
    info: dict = field(default_factory=dict)
    """What the page shows when the node is clicked."""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


@dataclass
class Edge:
    source: str
    target: str
    kind: str = "needs"
    """``needs``, ``after`` (one step runs after another) or ``provides``."""
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Graph:
    title: str
    verdict: str = ""
    headline: str = ""
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0

    def add(self, node: Node) -> Node:
        return self.nodes.setdefault(node.id, node)

    def link(self, source: str, target: str, kind: str = "needs") -> None:
        if source in self.nodes and target in self.nodes and source != target:
            if not any(e.source == source and e.target == target for e in self.edges):
                self.edges.append(Edge(source, target, kind))


# ------------------------------------------------------------- from a run


def from_plan(result: PlanResult, walk: Optional[Recursion] = None) -> Graph:
    """The plan's steps and what the default build (and its options) needs.

    With a recursion, each package built from source carries what was found
    under it: the packages its own build needs, as far down as it went.
    """
    answer = result.answer
    graph = Graph(
        title=result.plan.repo,
        verdict=(walk.verdict if walk is not None else answer.verdict if answer else "unknown"),
        headline=(walk.headline if walk is not None else answer.headline if answer else ""),
    )
    for outcome in result.steps:
        step = outcome.step
        rows = [["kind", step.kind], ["status", f"{outcome.status}: {outcome.detail}"]]
        if step.optional:
            rows.append(["optional", f"only with {step.enabled_by}"])
        if step.source not in (".", "") and step.kind in ("cmake-configure", "meson-setup"):
            rows.append(["source", step.source])
        if step.defines:
            rows.append(["defines", " ".join(f"-D{k}={v}" for k, v in step.defines.items())])
        if step.script:
            rows.append(["runs", " ".join([step.script, *step.args])])
        if step.manifest:
            rows.append(["installs", step.manifest])
        graph.add(Node(
            id=f"step:{step.id}", title=step.id,
            subtitle=step.kind + (" · optional" if step.optional else ""),
            step=True, optional=step.optional,
            tone="step-stopped" if outcome.status in ("stopped", "failed") else "step",
            info={
                "kind": "Step", "rows": rows,
                "notes": ([step.note] if step.note else []) + outcome.plan_check,
                "evidence": [[str(e), e.quote or ""] for e in step.evidence],
            },
        ))
    keep = [n for n in sorted(result.nodes.values(), key=lambda n: n.id)
            if n.required or n.optional_via or n.carried_on]
    quiet = _quiet(result, keep)
    for dep in keep:
        if dep.id in quiet:
            continue
        rows = [["tier", dep.detail or dep.tier]]
        if dep.required:
            rows.append(["needed", "by the default build"
                         + (f" ({_BASIS[dep.basis]})" if dep.basis in _BASIS else "")])
        elif dep.optional_via:
            rows.append(["needed", "only with " + ", ".join(dep.optional_via)])
        else:
            rows.append(["needed", "no: a configure looked for it, did without it, and "
                                   "completed"])
        if dep.package and dep.package != dep.name:
            rows.append(["package", dep.package])
        if dep.version:
            rows.append(["version", dep.version])
        if dep.wants:
            rows.append(["wanted", dep.wants])
        if dep.usage:
            rows.append(["usage", dep.usage + (f" (by {', '.join(dep.declared_by)})"
                                              if dep.declared_by else "")])
        subtitle = dep.ecosystem + (f" {dep.version}" if dep.version else "")
        graph.add(Node(
            id=dep.id, title=dep.name, subtitle=subtitle, tone=_TIER_TONE[dep.tier],
            optional=not dep.required,
            info={"kind": "Dependency", "rows": rows,
                  "chain": [_chain_entry(result, n) for n in result.why(dep.id)]},
        ))
    _summarise(graph, result, [d for d in keep if d.id in quiet])
    for step in result.plan.steps:
        for before in step.after:
            graph.link(f"step:{before}", f"step:{step.id}", "after")
    for source, target in result.edges:
        graph.link(source, target)
    for dep in keep:
        if dep.provided_by:
            graph.link(f"step:{dep.provided_by}", dep.id, "provides")
    if walk is not None:
        _add_recursion(graph, walk)
    return graph


def _quiet(result: PlanResult, keep: list) -> set:
    """Python packages with a wheel for the target (or pure Python) that lead
    to nothing that is not: MFC's toolchain brings in a hundred of them, and
    one node says it better than a hundred."""
    kept = {d.id: d for d in keep}
    children: dict[str, list[str]] = {}
    for source, target in result.edges:
        if source in kept and target in kept:
            children.setdefault(source, []).append(target)
    answer: dict[str, bool] = {}

    def fine(node_id: str, seen: frozenset = frozenset()) -> bool:
        if node_id in answer:
            return answer[node_id]
        dep = kept[node_id]
        if dep.ecosystem != "pypi" or dep.tier not in (planrun.BINARY, planrun.TOOLCHAIN):
            return False
        ok = all(fine(c, seen | {node_id}) for c in children.get(node_id, []) if c not in seen)
        answer[node_id] = ok
        return ok

    return {node_id for node_id in kept if fine(node_id)}


def _summarise(graph: Graph, result: PlanResult, quiet: list) -> None:
    """One node a step for its quiet Python packages, listing them."""
    by_step: dict[str, list] = {}
    for dep in quiet:
        chain = result.why(dep.id)
        step = chain[0] if chain[0].startswith("step:") else (
            f"step:{dep.steps[0]}" if dep.steps else None)
        if step in graph.nodes:
            by_step.setdefault(step, []).append(dep)
    for step, deps in by_step.items():
        node = graph.add(Node(
            id=f"wheels:{step}", title=f"{len(deps)} Python package{'s' if len(deps) != 1 else ''}",
            subtitle="wheels for the target, or pure Python", tone="archive",
            optional=all(not d.required for d in deps),
            info={"kind": "Python packages", "rows": [[d.name, d.version or ""] for d in
                                                      sorted(deps, key=lambda d: d.name.lower())],
                  "chain": [[step, step.removeprefix("step:")],
                            [f"wheels:{step}", f"{len(deps)} Python packages"]]},
        ))
        graph.link(step, node.id)


_BASIS = {
    planrun.BY_CONFIGURE: "shown by a configure",
    planrun.BY_MANIFEST: "declared by the project",
    planrun.BY_STOPPED: "asked for by a configure that stopped",
    planrun.BY_LIST: "only named by a package list in the plan",
}


def _chain_entry(result: PlanResult, node_id: str) -> list[str]:
    known = result.nodes.get(node_id)
    label = known.name if known is not None else node_id.removeprefix("step:")
    return [node_id, label]


def _add_recursion(graph: Graph, walk: Recursion) -> None:
    """The recursion's packages, under the dependencies they stand for."""
    ids: dict[str, str] = {}
    for key, tree in walk.nodes.items():
        ids[key] = tree.id if tree.id in graph.nodes else f"built:{key}"
    for key, tree in walk.nodes.items():
        rows = [["built from source", f"{tree.status}: {tree.reason}"]]
        if tree.source is not None:
            rows.append(["source", f"{tree.source.kind} {tree.source.origin}".strip()
                         + (f" ({tree.source.error})" if tree.source.error else "")])
        if key in walk.order:
            rows.append(["build order", f"{walk.order.index(key) + 1} of {len(walk.order)}"])
        node = graph.nodes.get(ids[key])
        if node is None:
            node = graph.add(Node(
                id=ids[key], title=tree.name,
                subtitle=f"{tree.ecosystem} {tree.version or ''}".strip() + f" · {tree.status}",
                tone=_STATUS_TONE.get(tree.status, "unknown"),
                info={"kind": "Built from source", "rows": rows},
            ))
        else:
            node.info["rows"] += rows
    for key, tree in walk.nodes.items():
        for parent in tree.parents:
            if parent in ids:
                graph.link(ids[parent], ids[key])
    # A package found only by recursing: its chain is its parent's, and it.
    for key in walk.nodes:
        node = graph.nodes[ids[key]]
        if "chain" in node.info:
            continue
        chain: list[list[str]] = [[node.id, node.title]]
        seen = {key}
        parent = next(iter(walk.nodes[key].parents), None)
        while parent in ids and parent not in seen:
            seen.add(parent)
            above = graph.nodes[ids[parent]]
            if "chain" in above.info:
                chain = above.info["chain"] + chain
                break
            chain.insert(0, [above.id, above.title])
            parent = next(iter(walk.nodes[parent].parents), None)
        node.info["chain"] = chain


# ------------------------------------------------------------------ layout


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _size(node: Node) -> None:
    title = _clip(node.title, TITLE_CHARS)
    subtitle = _clip(node.subtitle, SUBTITLE_CHARS)
    node.w = round(max(64.0, len(title) * _TITLE_PX, len(subtitle) * _SUBTITLE_PX) + 2 * _PAD)
    node.h = 44.0 if subtitle else 30.0


def layout(graph: Graph) -> None:
    """Place every node and route every edge, left to right."""
    order = list(graph.nodes)
    for node in graph.nodes.values():
        _size(node)
    if not order:
        graph.width = graph.height = 2 * _MARGIN
        return
    successors: dict[str, list[str]] = {k: [] for k in order}
    for edge in graph.edges:
        successors[edge.source].append(edge.target)

    # Edges that close a cycle (a bootstrap cycle in a recursion) are laid
    # out reversed, so the rest is a DAG.
    back: set = set()
    state = dict.fromkeys(order, 0)
    for start in order:
        if state[start]:
            continue
        state[start] = 1
        stack = [(start, iter(successors[start]))]
        while stack:
            here, rest = stack[-1]
            nxt = next(rest, None)
            if nxt is None:
                state[here] = 2
                stack.pop()
            elif state[nxt] == 1:
                back.add((here, nxt))
            elif state[nxt] == 0:
                state[nxt] = 1
                stack.append((nxt, iter(successors[nxt])))
    dag: list[tuple[str, str]] = []
    for edge in graph.edges:
        pair = (edge.target, edge.source) if (edge.source, edge.target) in back \
            else (edge.source, edge.target)
        if pair not in dag:
            dag.append(pair)

    # Ranks: the longest path from a node nothing points at.
    preds: dict[str, list[str]] = {k: [] for k in order}
    succs: dict[str, list[str]] = {k: [] for k in order}
    for a, b in dag:
        succs[a].append(b)
        preds[b].append(a)
    rank = dict.fromkeys(order, 0)
    waiting = {k: len(preds[k]) for k in order}
    queue = [k for k in order if not waiting[k]]
    topo: list[str] = []
    while queue:
        here = queue.pop(0)
        topo.append(here)
        for nxt in succs[here]:
            rank[nxt] = max(rank[nxt], rank[here] + 1)
            waiting[nxt] -= 1
            if not waiting[nxt]:
                queue.append(nxt)
    layers: list[list[str]] = [[] for _ in range(max(rank.values()) + 1)]
    for key in topo:
        layers[rank[key]].append(key)

    # Waypoints: an edge across several ranks passes through one in each.
    height = {k: graph.nodes[k].h for k in order}
    chains: dict[tuple[str, str], list[str]] = {}
    up: dict[str, list[str]] = {k: [] for k in order}
    down: dict[str, list[str]] = {k: [] for k in order}
    for a, b in dag:
        chain = [a]
        for r in range(rank[a] + 1, rank[b]):
            point = f"\0{len(height)}"
            height[point] = 2.0
            up[point], down[point] = [], []
            layers[r].append(point)
            rank[point] = r
            chain.append(point)
        chain.append(b)
        chains[(a, b)] = chain
        for x, y in zip(chain, chain[1:]):
            down[x].append(y)
            up[y].append(x)

    # Order within ranks: barycentre sweeps, keeping the fewest crossings.
    def crossings(layered: list[list[str]]) -> int:
        total = 0
        for upper, lower in zip(layered, layered[1:]):
            where = {k: i for i, k in enumerate(lower)}
            pairs = [(i, where[b]) for i, a in enumerate(upper) for b in down[a] if b in where]
            total += sum(1 for n, (a1, b1) in enumerate(pairs) for a2, b2 in pairs[n + 1:]
                         if (a1 - a2) * (b1 - b2) < 0)
        return total

    best, fewest = [list(layer) for layer in layers], crossings(layers)
    for sweep in range(12):
        pos = {k: i for layer in layers for i, k in enumerate(layer)}
        downward = sweep % 2 == 0
        span = range(1, len(layers)) if downward else range(len(layers) - 2, -1, -1)
        for r in span:
            near = up if downward else down
            layers[r].sort(key=lambda k: (sum(pos[n] for n in near[k]) / len(near[k])
                                          if near[k] else pos[k]))
            for i, key in enumerate(layers[r]):
                pos[key] = i
        count = crossings(layers)
        if count < fewest:
            best, fewest = [list(layer) for layer in layers], count
    layers = best

    # x: one column a rank, as wide as its widest node.
    widths = [max((graph.nodes[k].w for k in layer if k in graph.nodes), default=0.0)
              for layer in layers]
    column = [_MARGIN + sum(widths[:r]) + _RANK_GAP * r for r in range(len(layers))]

    # y: packed, then each node drawn toward the median of its neighbours.
    centre: dict[str, float] = {}

    def gap(a: str, b: str) -> float:
        between = _WAYPOINT_GAP if a not in graph.nodes or b not in graph.nodes else _NODE_GAP
        return (height[a] + height[b]) / 2 + between

    def place(layer: list[str], wanted: list[float]) -> None:
        offsets = [0.0]
        for a, b in zip(layer, layer[1:]):
            offsets.append(offsets[-1] + gap(a, b))
        settled = _isotonic([w - o for w, o in zip(wanted, offsets)])
        for key, z, o in zip(layer, settled, offsets):
            centre[key] = z + o

    for layer in layers:
        place(layer, [0.0] * len(layer))
    for sweep in range(8):
        downward = sweep % 2 == 0
        span = range(1, len(layers)) if downward else range(len(layers) - 2, -1, -1)
        for r in span:
            near = up if downward else down
            place(layers[r], [median(centre[n] for n in near[k]) if near[k] else centre[k]
                              for k in layers[r]])
    top = min(centre[k] - height[k] / 2 for k in centre)
    for key in centre:
        centre[key] += _MARGIN - top

    for r, layer in enumerate(layers):
        for key in layer:
            if key in graph.nodes:
                node = graph.nodes[key]
                node.x, node.y = column[r], centre[key] - node.h / 2
    graph.width = column[-1] + widths[-1] + _MARGIN
    graph.height = max(centre[k] + height[k] / 2 for k in centre) + _MARGIN

    for edge in graph.edges:
        a, b = graph.nodes[edge.source], graph.nodes[edge.target]
        forward = (edge.source, edge.target) not in back
        chain = chains[(edge.source, edge.target) if forward else (edge.target, edge.source)]
        middle = [(column[rank[k]] + widths[rank[k]] / 2, centre[k]) for k in chain[1:-1]]
        if forward:
            edge.points = [(a.x + a.w, a.y + a.h / 2), *middle, (b.x, b.y + b.h / 2)]
        else:
            edge.points = [(a.x, a.y + a.h / 2), *reversed(middle), (b.x + b.w, b.y + b.h / 2)]


def _isotonic(values: list[float]) -> list[float]:
    """The non-decreasing sequence nearest these values (pool adjacent violators)."""
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    out: list[float] = []
    for total, count in blocks:
        out += [total / count] * int(count)
    return out


# ------------------------------------------------------------------ drawing


def _path(points: list[tuple[float, float]]) -> str:
    """A smooth curve through the points, leaving and arriving level."""
    (x0, y0), rest = points[0], points[1:]
    d = [f"M{x0:.1f},{y0:.1f}"]
    for x1, y1 in rest:
        bend = (x1 - x0) / 2
        d.append(f"C{x0 + bend:.1f},{y0:.1f} {x1 - bend:.1f},{y1:.1f} {x1:.1f},{y1:.1f}")
        x0, y0 = x1, y1
    return " ".join(d)


def render_svg(graph: Graph, standalone: bool = True) -> str:
    """The laid-out graph as SVG: with its own styles and size when standalone,
    or for the page, which styles it and pans and zooms it."""
    esc = html.escape
    parts = []
    if standalone:
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" class="wir-graph" '
            f'viewBox="0 0 {graph.width:.0f} {graph.height:.0f}" width="{graph.width:.0f}" '
            f'height="{graph.height:.0f}" role="img" '
            f'aria-label="{esc(graph.title)}: dependency graph">'
            f"<style>{_SVG_CSS}</style>"
            f'<rect class="paper" width="100%" height="100%"/>'
        )
    else:
        parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" class="wir-graph" '
                     f'role="img" aria-label="{esc(graph.title)}: dependency graph">')
    parts.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path class="arrowhead" '
        'd="M0,1 L9,5 L0,9 z"/></marker></defs><g id="viewport"><g class="edges">'
    )
    for edge in graph.edges:
        # Hidden with the optional nodes: an edge to or from one.
        hidden = graph.nodes[edge.source].optional or graph.nodes[edge.target].optional
        parts.append(
            f'<path class="edge {edge.kind}{" optional" if hidden else ""}" '
            f'data-from="{esc(edge.source)}" data-to="{esc(edge.target)}" '
            f'd="{_path(edge.points)}" marker-end="url(#arrow)"/>'
        )
    parts.append('</g><g class="nodes">')
    for node in graph.nodes.values():
        classes = ["node", "step" if node.step else "dep", f"tone-{node.tone}"]
        if node.optional:
            classes.append("optional")
        tip = f"{node.title} — {TONES.get(node.tone, node.tone)}"
        title = esc(_clip(node.title, TITLE_CHARS))
        subtitle = esc(_clip(node.subtitle, SUBTITLE_CHARS))
        first = node.h / 2 + (-3 if subtitle else 4.5)
        parts.append(
            f'<g class="{" ".join(classes)}" data-id="{esc(node.id)}" tabindex="0" '
            f'transform="translate({node.x:.1f},{node.y:.1f})"><title>{esc(tip)}</title>'
            f'<rect width="{node.w:.0f}" height="{node.h:.0f}" rx="{4 if node.step else 9}"/>'
            f'<text class="t1" x="{_PAD}" y="{first:.1f}">{title}</text>'
            + (f'<text class="t2" x="{_PAD}" y="{node.h / 2 + 12:.1f}">{subtitle}</text>'
               if subtitle else "")
            + "</g>"
        )
    parts.append("</g></g></svg>")
    return "".join(parts)


def render_html(graph: Graph) -> str:
    """One page, nothing fetched: the graph, a legend, and a panel for the node clicked."""
    present = {n.tone for n in graph.nodes.values()}
    data = {
        "title": graph.title, "verdict": graph.verdict, "headline": graph.headline,
        "width": graph.width, "height": graph.height,
        "tones": TONES,
        "nodes": {
            n.id: {"title": n.title, "subtitle": n.subtitle, "tone": n.tone,
                   "optional": n.optional, "box": [n.x, n.y, n.w, n.h], **n.info}
            for n in graph.nodes.values()
        },
    }
    # Nothing in the data can close the script element it sits in.
    blob = json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e")
    legend = "".join(
        f'<span class="chip tone-{tone}"><i></i>{html.escape(label)}</span>'
        for tone, label in TONES.items() if tone in present
    )
    optional = sum(1 for n in graph.nodes.values() if n.optional)
    page = _PAGE
    for marker, value in (
        ("{{TITLE}}", html.escape(graph.title)),
        ("{{VERDICT}}", html.escape(graph.verdict.upper())),
        ("{{VERDICT_CLASS}}", html.escape(graph.verdict)),
        ("{{HEADLINE}}", html.escape(graph.headline)),
        ("{{LEGEND}}", legend),
        ("{{OPTIONAL}}", str(optional)),
        ("{{CSS}}", _SVG_CSS + _PAGE_CSS),
        ("{{SVG}}", render_svg(graph, standalone=False)),
        ("{{JS}}", _PAGE_JS),
        ("{{DATA}}", blob),
    ):
        page = page.replace(marker, value)
    return page


def write(graph: Graph, path: Path) -> None:
    """``.html``: the page; ``.svg``: the picture."""
    layout(graph)
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        text = render_html(graph)
    elif suffix == ".svg":
        text = render_svg(graph)
    else:
        raise ValueError(f"{path.name}: --graph writes .html (interactive) or .svg")
    path.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ styles

_SVG_CSS = """
.wir-graph,.legend{--ink:#1d1d1f;--muted:#5f6368;--paper:#ffffff;--edge:#a3a3ab;
--step-f:#f4f4f5;--step-s:#52525b;--archive-f:#e3f1e4;--archive-s:#4a8f53;
--provided-f:#ddeef4;--provided-s:#3f8aa3;--source-f:#fbefc8;--source-s:#b8892a;
--unknown-f:#ececee;--unknown-s:#8a8a90;--none-f:#f8d7da;--none-s:#b3263a;
--buildable-f:#e3f1e4;--buildable-s:#2f7a3a;--probably-f:#fbefc8;--probably-s:#a07418;
--focus:#2563eb;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .wir-graph,
:root:not([data-theme=light]) .legend,
.wir-graph:not(.in-page){--ink:#ececf0;--muted:#a0a0a8;--paper:#16161a;--edge:#5d5d66;
--step-f:#26262b;--step-s:#a1a1aa;--archive-f:#1d3322;--archive-s:#6fbf79;
--provided-f:#1b3038;--provided-s:#6ab7d0;--source-f:#3a3016;--source-s:#e0b75a;
--unknown-f:#2a2a2e;--unknown-s:#9a9aa2;--none-f:#47191f;--none-s:#f07a88;
--buildable-f:#1d3322;--buildable-s:#7fd08a;--probably-f:#3a3016;--probably-s:#e8c46a;
--focus:#7aa2ff}}
:root[data-theme=dark] .wir-graph,:root[data-theme=dark] .legend{
--ink:#ececf0;--muted:#a0a0a8;--paper:#16161a;--edge:#5d5d66;
--step-f:#26262b;--step-s:#a1a1aa;--archive-f:#1d3322;--archive-s:#6fbf79;
--provided-f:#1b3038;--provided-s:#6ab7d0;--source-f:#3a3016;--source-s:#e0b75a;
--unknown-f:#2a2a2e;--unknown-s:#9a9aa2;--none-f:#47191f;--none-s:#f07a88;
--buildable-f:#1d3322;--buildable-s:#7fd08a;--probably-f:#3a3016;--probably-s:#e8c46a;
--focus:#7aa2ff}
.wir-graph .paper{fill:var(--paper)}
.wir-graph .edge{fill:none;stroke:var(--edge);stroke-width:1.3}
.wir-graph .edge.after{stroke-dasharray:2 4}
.wir-graph .edge.provides{stroke:var(--provided-s);stroke-dasharray:6 4}
.wir-graph .edge.optional{stroke-dasharray:5 4;opacity:.7}
.wir-graph .arrowhead{fill:var(--edge)}
.wir-graph .node rect{stroke-width:1.4}
.wir-graph .node.step rect{stroke-width:2}
.wir-graph .node.optional rect{stroke-dasharray:5 3}
.wir-graph .node text{fill:var(--ink);font-size:12.5px}
.wir-graph .node .t1{font-weight:600}
.wir-graph .node .t2{fill:var(--muted);font-size:10.5px}
.wir-graph .tone-step rect{fill:var(--step-f);stroke:var(--step-s)}
.wir-graph .tone-step-stopped rect{fill:var(--step-f);stroke:var(--none-s)}
.wir-graph .tone-archive rect{fill:var(--archive-f);stroke:var(--archive-s)}
.wir-graph .tone-provided rect{fill:var(--provided-f);stroke:var(--provided-s)}
.wir-graph .tone-source rect{fill:var(--source-f);stroke:var(--source-s)}
.wir-graph .tone-unknown rect{fill:var(--unknown-f);stroke:var(--unknown-s)}
.wir-graph .tone-none rect{fill:var(--none-f);stroke:var(--none-s)}
.wir-graph .tone-buildable rect{fill:var(--buildable-f);stroke:var(--buildable-s)}
.wir-graph .tone-probably rect{fill:var(--probably-f);stroke:var(--probably-s)}
"""

_PAGE_CSS = """
:root{--bg:#f6f6f7;--panel:#ffffff;--text:#1d1d1f;--sub:#5f6368;--line:#e2e2e6;
--accent:#2563eb;--yes:#2f7a3a;--maybe:#a07418;--no:#b3263a;--unsure:#6b6b75;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f0f12;--panel:#1a1a1f;
--text:#ececf0;--sub:#a0a0a8;--line:#2c2c33;--accent:#7aa2ff;--yes:#7fd08a;--maybe:#e8c46a;
--no:#f07a88;--unsure:#a0a0a8;color-scheme:dark}}
:root[data-theme=dark]{--bg:#0f0f12;--panel:#1a1a1f;--text:#ececf0;--sub:#a0a0a8;--line:#2c2c33;
--accent:#7aa2ff;--yes:#7fd08a;--maybe:#e8c46a;--no:#f07a88;--unsure:#a0a0a8;color-scheme:dark}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",
Roboto,sans-serif;display:grid;grid-template-rows:auto auto 1fr;height:100vh}
header{padding:14px 18px 6px;display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline}
header h1{font-size:17px;margin:0;font-weight:650}
header .tool{color:var(--sub);font-weight:500}
.verdict{font-weight:700;font-size:12px;letter-spacing:.04em;padding:2px 8px;border-radius:999px;
border:1.5px solid currentColor}
.verdict.yes{color:var(--yes)}.verdict.yes-after-source-builds,.verdict.probably{color:var(--maybe)}
.verdict.no{color:var(--no)}.verdict.unknown{color:var(--unsure)}
.headline{color:var(--sub);flex-basis:100%}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:4px 18px 10px;
border-bottom:1px solid var(--line)}
.bar button,.bar input{font:inherit;color:var(--text);background:var(--panel);
border:1px solid var(--line);border-radius:7px;padding:4px 10px}
.bar button{cursor:pointer}.bar button:hover{border-color:var(--accent)}
.bar button[aria-pressed=true]{border-color:var(--accent);color:var(--accent)}
.bar input{min-width:0;width:200px}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;margin-left:auto;color:var(--sub);font-size:12px}
.chip{display:inline-flex;align-items:center;gap:5px}
.chip i{width:18px;height:11px;border-radius:4px;border:2px solid;display:inline-block}
.chip.tone-step i{background:var(--step-f);border-color:var(--step-s)}
.chip.tone-step-stopped i{background:var(--step-f);border-color:var(--none-s)}
.chip.tone-archive i{background:var(--archive-f);border-color:var(--archive-s)}
.chip.tone-provided i{background:var(--provided-f);border-color:var(--provided-s)}
.chip.tone-source i{background:var(--source-f);border-color:var(--source-s)}
.chip.tone-unknown i{background:var(--unknown-f);border-color:var(--unknown-s)}
.chip.tone-none i{background:var(--none-f);border-color:var(--none-s)}
.chip.tone-buildable i{background:var(--buildable-f);border-color:var(--buildable-s)}
.chip.tone-probably i{background:var(--probably-f);border-color:var(--probably-s)}
main{display:grid;grid-template-columns:1fr 340px;min-height:0}
#stage{position:relative;min-height:0;overflow:hidden;cursor:grab;touch-action:none}
#stage.dragging{cursor:grabbing}
#stage svg{width:100%;height:100%;display:block}
aside{border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:16px 18px}
aside h2{font-size:16px;margin:0 0 2px}
aside .kind{color:var(--sub);font-size:12px;margin-bottom:10px}
aside table{border-collapse:collapse;width:100%;font-size:13px}
aside td{padding:4px 0;vertical-align:top;border-top:1px solid var(--line)}
aside td:first-child{color:var(--sub);padding-right:10px;white-space:nowrap}
aside td:last-child{overflow-wrap:anywhere}
aside h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--sub);
margin:16px 0 6px}
aside ol{margin:0;padding-left:20px}
aside ol button{font:inherit;color:var(--accent);background:none;border:0;padding:0;cursor:pointer;
text-align:left}
aside code{display:block;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;
margin:2px 0 8px;
white-space:pre-wrap;overflow-wrap:anywhere}
aside .at{font-size:12px;color:var(--sub)}
aside .empty{color:var(--sub)}
.wir-graph .node{cursor:pointer}
.wir-graph .node:focus{outline:none}
.wir-graph .node:focus-visible rect,.wir-graph .node.selected rect{stroke:var(--focus);
stroke-width:2.6}
.wir-graph .node.match rect{stroke:var(--focus);stroke-width:2.2}
.wir-graph.focus .node:not(.lit){opacity:.28}
.wir-graph.focus .edge:not(.lit){opacity:.15}
.wir-graph .edge.lit{stroke:var(--focus);stroke-width:2}
.wir-graph.hide-optional .optional{display:none}
@media (max-width:760px){main{grid-template-columns:1fr;grid-template-rows:1fr minmax(160px,40vh)}
aside{border-left:0;border-top:1px solid var(--line)}.legend{margin-left:0}.bar input{width:100%}}
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}} · dependency graph</title>
<style>{{CSS}}</style>
</head>
<body>
<header>
<h1><span class="tool">will-it-riscv ·</span> {{TITLE}}</h1>
<span class="verdict {{VERDICT_CLASS}}">{{VERDICT}}</span>
<span class="headline">{{HEADLINE}}</span>
</header>
<div class="bar">
<button type="button" id="fit">Fit</button>
<button type="button" id="zoom-in" aria-label="Zoom in">+</button>
<button type="button" id="zoom-out" aria-label="Zoom out">−</button>
<button type="button" id="optional" aria-pressed="true">Optional ({{OPTIONAL}})</button>
<input id="search" type="search" placeholder="Find a package or step" aria-label="Find">
<div class="legend">{{LEGEND}}</div>
</div>
<main>
<div id="stage">{{SVG}}</div>
<aside id="panel" aria-live="polite"><p class="empty">Click a node to see why the build
needs it, what settled it, and the lines the plan cited. Drag to pan, scroll to zoom.</p></aside>
</main>
<script type="application/json" id="wir-data">{{DATA}}</script>
<script>{{JS}}</script>
</body>
</html>
"""

_PAGE_JS = """
(() => {
  const data = JSON.parse(document.getElementById('wir-data').textContent);
  const stage = document.getElementById('stage');
  const svg = stage.querySelector('svg');
  const view = svg.getElementById('viewport');
  const panel = document.getElementById('panel');
  svg.classList.add('in-page');
  const READABLE = 0.7;
  let s = 1, tx = 0, ty = 0;
  const apply = () => view.setAttribute('transform', `translate(${tx} ${ty}) scale(${s})`);
  const fit = () => {
    const r = stage.getBoundingClientRect();
    s = Math.min(r.width / data.width, r.height / data.height, 1.6) * 0.96;
    tx = (r.width - data.width * s) / 2; ty = (r.height - data.height * s) / 2; apply();
  };
  // A graph too big to read whole opens readable, at the steps: Fit shows it all.
  const start = () => {
    const r = stage.getBoundingClientRect();
    fit();
    if (s >= READABLE) return;
    s = READABLE; tx = 12;
    ty = data.height * s > r.height ? 12 : (r.height - data.height * s) / 2; apply();
  };
  const zoom = (k, px, py) => {
    const n = Math.min(5, Math.max(0.08, s * k));
    tx = px - (px - tx) * n / s; ty = py - (py - ty) * n / s; s = n; apply();
  };
  const centre = () => {
    const r = stage.getBoundingClientRect(); return [r.width / 2, r.height / 2];
  };
  stage.addEventListener('wheel', e => {
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    zoom(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
  }, {passive: false});
  let drag = null;
  stage.addEventListener('pointerdown', e => {
    drag = {x: e.clientX, y: e.clientY, tx, ty, moved: false};
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; stage.classList.add('dragging'); }
    if (drag.moved) { tx = drag.tx + dx; ty = drag.ty + dy; apply(); }
  });
  stage.addEventListener('pointerup', e => {
    const was = drag; drag = null; stage.classList.remove('dragging');
    if (!was || was.moved) return;
    const hit = document.elementFromPoint(e.clientX, e.clientY);
    const node = hit && hit.closest('.node');
    node ? select(node.dataset.id) : clear();
  });
  const nodes = new Map([...svg.querySelectorAll('.node')].map(n => [n.dataset.id, n]));
  const edges = [...svg.querySelectorAll('.edge')];
  const el = (tag, text, cls) => {
    const e = document.createElement(tag);
    if (text != null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  };
  function clear() {
    svg.classList.remove('focus');
    svg.querySelectorAll('.lit,.selected').forEach(n => n.classList.remove('lit', 'selected'));
    panel.replaceChildren(el('p', 'Click a node to see why the build needs it.', 'empty'));
  }
  function select(id) {
    const info = data.nodes[id]; if (!info) return;
    svg.querySelectorAll('.lit,.selected').forEach(n => n.classList.remove('lit', 'selected'));
    svg.classList.add('focus');
    const lit = new Set([id]);
    const chain = (info.chain || []).map(c => c[0]);
    chain.forEach(c => lit.add(c));
    for (const e of edges) {
      const a = e.dataset.from, b = e.dataset.to;
      const onChain = chain.indexOf(a) >= 0 && chain.indexOf(b) === chain.indexOf(a) + 1;
      if (a === id || b === id || onChain) { e.classList.add('lit'); lit.add(a); lit.add(b); }
    }
    lit.forEach(k => nodes.get(k) && nodes.get(k).classList.add('lit'));
    nodes.get(id).classList.add('selected');
    const parts = [el('h2', info.title)];
    parts.push(el('div', `${info.kind || ''} · ${data.tones[info.tone] || info.tone}`
      + (info.optional ? ' · optional' : ''), 'kind'));
    if (info.rows && info.rows.length) {
      const table = el('table');
      for (const [k, v] of info.rows) {
        const tr = el('tr'); tr.append(el('td', k), el('td', v)); table.append(tr);
      }
      parts.push(table);
    }
    if (chain.length > 1) {
      parts.push(el('h3', 'Why the build needs it'));
      const ol = el('ol');
      for (const [cid, label] of info.chain) {
        const li = el('li'); const b = el('button', label);
        b.type = 'button'; b.addEventListener('click', () => select(cid));
        li.append(b); ol.append(li);
      }
      parts.push(ol);
    }
    if (info.notes && info.notes.length) {
      parts.push(el('h3', 'Notes'));
      for (const n of info.notes) parts.push(el('p', n));
    }
    if (info.evidence && info.evidence.length) {
      parts.push(el('h3', 'Cited lines'));
      for (const [at, quote] of info.evidence) {
        parts.push(el('div', at, 'at'));
        if (quote) parts.push(el('code', quote));
      }
    }
    panel.replaceChildren(...parts);
  }
  svg.addEventListener('keydown', e => {
    const node = e.target.closest && e.target.closest('.node');
    if (node && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault(); select(node.dataset.id);
    }
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') clear(); });
  document.getElementById('fit').addEventListener('click', fit);
  document.getElementById('zoom-in').addEventListener('click', () => zoom(1.25, ...centre()));
  document.getElementById('zoom-out').addEventListener('click', () => zoom(0.8, ...centre()));
  const toggle = document.getElementById('optional');
  if (toggle.textContent.includes('(0)')) toggle.hidden = true;
  toggle.addEventListener('click', () => {
    const shown = toggle.getAttribute('aria-pressed') !== 'true';
    toggle.setAttribute('aria-pressed', String(shown));
    svg.classList.toggle('hide-optional', !shown);
  });
  const search = document.getElementById('search');
  const matches = () => {
    const q = search.value.trim().toLowerCase();
    const hits = [];
    nodes.forEach((n, id) => {
      const hit = q && data.nodes[id].title.toLowerCase().includes(q);
      n.classList.toggle('match', !!hit); if (hit) hits.push(id);
    });
    return hits;
  };
  search.addEventListener('input', matches);
  search.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    const hits = matches(); if (!hits.length) return;
    select(hits[0]);
    const [x, y, w, h] = data.nodes[hits[0]].box;
    const [cx, cy] = centre();
    tx = cx - (x + w / 2) * s; ty = cy - (y + h / 2) * s; apply();
  });
  window.addEventListener('resize', start);
  start();
})();
"""
