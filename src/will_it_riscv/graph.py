"""The dependency graph a pseudobuild observed, and the answer it gives.

A pseudobuild's trace records every question the configure asked, who asked
it, and how each one came out. This turns that into a graph -- the project
at the root, an edge from whatever asked to what it asked for -- and checks
the nodes that matter against the target distro's archive, which is what
"will it riscv?" finally comes down to.

Every node carries the reason for its status, because the statuses are not
equally strong. "Stopped the configure" and "shown by experiment" are
demonstrations. "The configure carried on without it" is too, once the
configure was confined to an empty sysroot. "Present on this host" proves
nothing about need, and says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .syslibs import database

if TYPE_CHECKING:  # pragma: no cover
    from .distro import DistroIndex
    from .models import Analysis
    from .pseudobuild import PseudoBuild

REQUIRED = "required"
OPTIONAL = "optional"
PRESENT = "present"
"""A host program the configure found. Whether the build needs it is unknown."""
UNVERIFIED = "unverified"
"""A library a compile-only check claimed. Confined, nothing real was there
to find, so this is the check being fooled, not the library being present."""
UNKNOWN = "unknown"
"""Asked about, but the configure never finished, so its fate is open."""

#: Provided by the target's compiler rather than by a package of its own.
TOOLCHAIN = {
    "openmp": "the compiler (GCC's libgomp)",
    # The target's own Python headers: python3-dev, wherever there is Python.
    "python": "the target's Python (python3-dev)",
    "python3": "the target's Python (python3-dev)",
    "pythonlibs": "the target's Python (python3-dev)",
}

#: FindOpenMP, FindMPI and friends report per language: OpenMP_CXX is OpenMP.
_LANGUAGE_SUFFIX = re.compile(r"_(C|CXX|Fortran|CUDA|HIP)$")

ROOT = ""
"""The project itself, as a parent in :attr:`Graph.edges`."""


@dataclass
class Node:
    key: str
    """Canonical name, shared by every spelling of the same package."""
    name: str
    """The spelling the configure used first, which is what people search for."""
    kind: str = "library"
    status: str = UNKNOWN
    proof: str = ""
    aliases: list[str] = field(default_factory=list)
    via: Optional[str] = None
    site: Optional[str] = None
    round: int = 1
    behind: Optional[str] = None
    """The blocker this was only reachable past."""
    debian: tuple[str, ...] = ()
    guessed: bool = False
    provided_by: Optional[str] = None
    checked: bool = False
    available: Optional[str] = None
    """The distro package that provides it for the target, when checked."""
    asked_required: bool = False
    """Some find_package asked for it with REQUIRED. That is the project's
    claim, not a demonstration: only a blocker has been shown to be needed."""

    @property
    def missing(self) -> bool:
        """Checked against the archive, and nothing there provides it.

        Only for a name the map actually knows. A guessed ``libfoo-dev`` that
        the archive lacks says the guess was wrong, not that the target is.
        """
        return (
            self.checked
            and self.available is None
            and self.provided_by is None
            and bool(self.debian)
            and not self.guessed
        )


@dataclass
class Answer:
    """Will it riscv?"""

    verdict: str
    """``yes``, ``probably``, ``no`` or ``unknown``."""
    headline: str
    install: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    toolchain: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Graph:
    project: str
    platform: str
    completed: bool
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    answer: Optional[Answer] = None
    host_gaps: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    """Keys of the hard requirements, in the order the build demanded them."""

    def children(self, parent: str) -> list[str]:
        return [child for p, child in self.edges if p == parent]

    def with_status(self, status: str) -> list[Node]:
        return sorted(
            (n for n in self.nodes.values() if n.status == status),
            key=lambda n: n.name.lower(),
        )

    def required(self) -> list[Node]:
        ranked = {key: index for index, key in enumerate(self.order)}
        return sorted(
            self.with_status(REQUIRED),
            key=lambda n: (ranked.get(n.key, len(ranked)), n.name.lower()),
        )


def _canonical(raw: str) -> Optional[tuple[str, str, tuple[str, ...], bool]]:
    """(key, kind, debian packages, guessed), or None if it is nobody's package."""
    db = database()
    stem = _LANGUAGE_SUFFIX.sub("", raw)
    candidates = [stem, raw] if stem != raw else [raw]
    for candidate in candidates:
        library = db.lookup(candidate, "library")
        if library is not None and not db.is_guess(library):
            return library.name, "library", library.debian, False
        tool = db.lookup(candidate, "tool")
        if tool is not None and not db.is_guess(tool):
            return tool.name, "tool", tool.debian, False
    library = db.lookup(candidates[0], "library")
    if library is not None:
        return library.name, "library", library.debian, True
    return None


def build(
    result: PseudoBuild, project: str, distro: Optional[DistroIndex] = None
) -> Graph:
    """Turn what the configure did into a graph, and answer the question."""
    graph = Graph(
        project=project,
        platform=result.platform,
        completed=result.completed,
        host_gaps=list(result.host_gaps),
    )
    confirmed = {name for name, ok in result.experiments if ok}

    def node_for(
        raw: str, force: bool = False, spelling_of: Optional[str] = None
    ) -> Optional[Node]:
        known = _canonical(raw)
        if known is None and not force:
            return None
        if spelling_of and (known is None or known[3]):
            # A name the map does not know, looked for inside another
            # package's Find module, is that package's alternative spelling:
            # ssleay32MD inside FindOpenSSL, xerces-c_d inside FindXercesC.
            return graph.nodes.get(_key(spelling_of))
        key, kind, debian, guessed = known or (raw.lower(), "library", (), True)
        node = graph.nodes.get(key)
        if node is None:
            node = graph.nodes[key] = Node(
                key=key, name=raw, kind=kind, debian=debian, guessed=guessed,
                provided_by=TOOLCHAIN.get(key),
            )
        if raw not in node.aliases:
            node.aliases.append(raw)
        return node

    # The hard requirements first, so they keep the spelling that stopped
    # the configure -- "PROJ", not whatever a find_library called it.
    for raw in result.blockers:
        node = node_for(raw, force=True)
        if node is None:
            continue
        node.status = REQUIRED
        node.proof = (
            "shown by experiment: the error moved once it existed"
            if raw in confirmed else "stopped the configure"
        )
        if node.key not in graph.order:
            graph.order.append(node.key)

    # find_package first within a round, so a package keeps its own name
    # rather than that of the first library its module went looking for.
    ordered = sorted(
        result.probes.values(),
        key=lambda p: (p.round, p.command != "find_package", p.name.lower()),
    )
    for probe in ordered:
        nested = probe.parent if probe.command != "find_package" else None
        if probe.command in ("find_library", "check_library_exists") and _is_guess(probe.name):
            # A file name the map does not know -- OpenCV's alapack_r and
            # ptcblas_r -- is a spelling some search tried, not a package.
            # pkg-config module names stay: those are packages' own names.
            continue
        node = node_for(probe.name, spelling_of=nested)
        if node is None:
            continue
        node.asked_required = node.asked_required or probe.required
        if node.site is None or probe.round < node.round:
            node.site, node.via, node.round = probe.site, probe.via, probe.round
            behind = result.reached_behind(probe)
            node.behind = _display(graph, behind) if behind else None
        parent = node_for(probe.parent) if probe.parent else None
        source = parent.key if parent is not None else ROOT
        # FindCURL looking for libcurl is still CURL: no edge to itself.
        if source != node.key and (source, node.key) not in graph.edges:
            graph.edges.append((source, node.key))

    found: dict[str, str] = {}
    for name in result.found:
        found.setdefault(_key(name), result.found_at.get(name, ""))
    soft = {_key(n) for n in result.soft_misses}
    for node in graph.nodes.values():
        if node.status == REQUIRED:
            continue
        if node.key in found:
            where = found[node.key]
            asked = "asked for with REQUIRED; " if node.asked_required else ""
            if _is_program(where) or node.kind == "tool" or not result.confined:
                if _is_program(where) and node.kind != "tool":
                    _as_tool(node)
                node.status = PRESENT
                node.proof = f"{asked}present on this host" + (f" at {where}" if where else "")
            else:
                node.status = UNVERIFIED
                node.proof = (
                    f"{asked}claimed by a compile-only check; "
                    "confined, nothing real was there to find"
                )
        elif node.key in soft and result.completed:
            node.status, node.proof = OPTIONAL, "absent, and the configure carried on"
        elif node.key in soft:
            # It carried on past the miss and then died -- perhaps of the
            # miss. AdaptiveCpp misses LLVM, carries on, and fails wanting
            # clang's headers. Only a configure that finishes proves anything.
            node.status = UNKNOWN
            node.proof = "absent, and the configure carried on, but stopped later"
        elif result.completed and result.confined:
            node.status = OPTIONAL
            node.proof = "never found, and the configure completed anyway"
        else:
            node.status = UNKNOWN
            node.proof = (
                "asked about; the host may have answered"
                if result.completed else "asked about before the configure stopped"
            )

    # Blockers whose only mention was the error itself still hang off the
    # project, so none of them floats free of the graph.
    linked = {child for _, child in graph.edges}
    for key in graph.order:
        if key not in linked:
            graph.edges.append((ROOT, key))

    if distro is not None and distro.available:
        for node in graph.nodes.values():
            if node.debian:
                node.checked = True
                node.available = distro.first_available(node.debian)

    graph.answer = _answer(graph, result, distro)
    return graph


def _as_tool(node: Node) -> None:
    """It turned out to be a program: FindPerl found perl, not libperl."""
    db = database()
    tool = db.lookup(node.name, "tool")
    node.kind = "tool"
    if tool is not None:
        node.debian, node.guessed = tool.debian, db.is_guess(tool)


def _is_program(where: str) -> bool:
    """Whether a "-- Found X: <where>" names a host executable.

    Confined, a library cannot be found for real, so a found path that runs
    is a program on the host -- /opt/homebrew/bin/swig, or TeX's texbin.
    """
    import os

    if not where.startswith("/"):
        return False
    return "/bin/" in where or (os.path.isfile(where) and os.access(where, os.X_OK))


def _is_guess(raw: str) -> bool:
    known = _canonical(raw)
    return known is None or known[3]


def _key(raw: Optional[str]) -> str:
    if not raw:
        return ""
    known = _canonical(raw)
    return known[0] if known else raw.lower()


def _display(graph: Graph, raw: str) -> str:
    node = graph.nodes.get(_key(raw))
    return node.name if node is not None else raw


def _answer(graph: Graph, result: PseudoBuild, distro: Optional[DistroIndex]) -> Answer:
    arch = graph.platform.split("/", 1)[1] if "/" in graph.platform else "the target"
    label = distro.spec.label if distro is not None else "the distro"
    required = graph.required()
    # A tool the configure REQUIRED and this host happened to have: not
    # shown to be needed, but the build expects it, so it goes on the list.
    tools = [
        n for n in graph.with_status(PRESENT) if n.asked_required and n.debian
    ]
    needed = required + tools
    toolchain = [f"{n.name} comes with {n.provided_by}" for n in required if n.provided_by]
    # A guessed name the archive has is at least a real package: libpsl-dev
    # for Libpsl. It goes on the line, and the answer says it was a guess.
    confirmed = [n for n in needed if n.guessed and n.available]
    install = sorted({
        n.available or n.debian[0]
        for n in needed
        if n.debian and not n.provided_by and not n.missing
        and (not n.guessed or n.available)
    })
    missing = [n.name for n in needed if n.missing and not n.guessed]
    unmapped = [
        n.name for n in required
        if n.guessed and not n.provided_by and not n.available
    ]
    notes = [
        f"{n.available} for {n.name} was matched by name, not from the curated map"
        for n in confirmed
    ]

    if not result.completed:
        where = f": {result.error}" if result.error else ""
        return Answer(
            "unknown",
            f"can't tell yet — the configure stopped after {result.rounds} "
            f"round{'s' if result.rounds != 1 else ''}, on something no stub gets past{where}",
            install, missing, unmapped, toolchain, notes,
        )
    hedge = "" if result.confined else " (configured for this host, not for linux)"
    if missing:
        return Answer(
            "no",
            f"not from the archive alone{hedge} — {', '.join(missing)} "
            f"{'has' if len(missing) == 1 else 'have'} no {arch} package in {label}; "
            "build it first",
            install, missing, unmapped, toolchain, notes,
        )
    checked = distro is not None and distro.available
    if not required:
        return Answer(
            "yes" if result.confined else "probably",
            f"a default build demands nothing from the system{hedge}",
            install, missing, unmapped, toolchain, notes,
        )
    if not checked:
        return Answer(
            "probably",
            f"the configure completes once {len(required)} hard "
            f"requirement{'s' if len(required) != 1 else ''} exist{hedge}; "
            "the archive was not checked",
            install, missing, unmapped, toolchain, notes,
        )
    if unmapped or not result.confined:
        why = (
            f"{', '.join(unmapped)} could not be matched to a package"
            if unmapped else "every hard requirement is packaged"
        )
        return Answer(
            "probably", f"{why}{hedge}", install, missing, unmapped, toolchain, notes
        )
    return Answer(
        "yes",
        f"every dependency a default build demands is in {label} for {arch}",
        install, missing, unmapped, toolchain, notes,
    )


def static(analysis: Analysis) -> Graph:
    """The graph static reading alone gives: the project, and what it reads as needing."""
    graph = Graph(project=analysis.root, platform="static", completed=False)
    for requirement in analysis.project_requirements.values():
        if requirement.purpose != "build":
            continue
        graph.nodes[requirement.name] = Node(
            key=requirement.name,
            name=requirement.name,
            kind=requirement.kind,
            status=OPTIONAL if requirement.optional else REQUIRED,
            proof=requirement.gate or ("gated off by default" if requirement.optional
                                       else "read from the build files"),
            debian=requirement.debian,
            site=requirement.found_in[0] if requirement.found_in else None,
        )
        graph.edges.append((ROOT, requirement.name))
    return graph


# ------------------------------------------------------------------- output

_STYLE = {
    REQUIRED: ("#f6c9c9", "#a3122a"),
    OPTIONAL: ("#e6eef8", "#7f9cc0"),
    PRESENT: ("#dcefdc", "#4f8a4f"),
    UNVERIFIED: ("#fbefc4", "#a07d12"),
    UNKNOWN: ("#eeeeee", "#8a8a8a"),
}


def _quote(text: str) -> str:
    """A DOT string. Real newlines become DOT's own line breaks."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


#: A star with more leaves than this is folded into columns, the way
#: Graphviz's own ``unflatten`` does it, rather than drawn as one tall line.
_STAGGER_ABOVE = 12


def to_dot(graph: Graph) -> str:
    """Graphviz source. ``dot -Tsvg`` renders it; hover a node for its proof.

    Nodes are grouped by the round that first reached them, so the drawing
    shows the unblock loop as well as the graph: what the first configure
    asked, and what only came into view once each blocker had been stubbed.
    """
    answer = f": {graph.answer.verdict}" if graph.answer else ""
    title = f"{graph.project} — {graph.platform}{answer}"
    lines = [
        f"digraph {_quote(graph.project)} {{",
        f"  graph [rankdir=LR, fontname=Helvetica, labelloc=t, label={_quote(title)}, "
        "newrank=true];",
        '  node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11];',
        "  edge [color=\"#9a9a9a\", arrowsize=0.6];",
        f'  "__project__" [label={_quote(graph.project)}, style="filled,bold", '
        'fillcolor="#e2e2e2"];',
    ]
    rounds: dict[int, list[Node]] = {}
    for node in sorted(graph.nodes.values(), key=lambda n: n.key):
        rounds.setdefault(node.round, []).append(node)
    clustered = len(rounds) > 1
    for number in sorted(rounds):
        indent = "  "
        if clustered:
            behind = next((n.behind for n in rounds[number] if n.behind), None)
            label = f"round {number}" + (f" · reached once {behind} existed" if behind else "")
            lines.append(f"  subgraph cluster_round_{number} {{")
            lines.append(
                f'    label={_quote(label)}; style="rounded,dashed"; color="#c8c8c8"; '
                "fontsize=10;"
            )
            indent = "    "
        for node in rounds[number]:
            lines.append(f"{indent}{_quote(node.key)} [{_node_attributes(node)}];")
        if clustered:
            lines.append("  }")

    leaves = [c for p, c in graph.edges if p == ROOT and not graph.children(c)]
    stagger = len(leaves) > _STAGGER_ABOVE
    column = 0
    for parent, child in graph.edges:
        source = "__project__" if parent == ROOT else parent
        target = graph.nodes.get(child)
        attributes = []
        if target is not None and target.status == REQUIRED:
            attributes += ["penwidth=1.6", 'color="#a3122a"']
        elif target is not None and target.status == OPTIONAL:
            attributes.append("style=dashed")
        if stagger and parent == ROOT and child in leaves and (
            target is None or target.status != REQUIRED
        ):
            attributes.append(f"minlen={1 + column % 3}")
            column += 1
        suffix = f" [{', '.join(attributes)}]" if attributes else ""
        lines.append(f"  {_quote(source)} -> {_quote(child)}{suffix};")
    lines.append("}")
    return "\n".join(lines)


def _node_attributes(node: Node) -> str:
    fill, border = _STYLE.get(node.status, _STYLE[UNKNOWN])
    label = node.name
    if node.provided_by:
        label += "\ncompiler"
    elif node.available:
        label += f"\n{node.available}"
    elif node.missing:
        label += f"\n{node.debian[0]}: none for the target"
    elif node.debian and not node.guessed:
        label += f"\n{node.debian[0]}"
    attributes = [f"label={_quote(label)}", f'fillcolor="{fill}"', f'color="{border}"']
    needed = node.status == REQUIRED or node.asked_required
    if node.missing and needed:
        attributes += ['fillcolor="#a3122a"', 'fontcolor="white"', "penwidth=2"]
    elif node.missing:
        attributes += ['color="#a3122a"', "penwidth=1.5"]
    elif node.status == REQUIRED:
        attributes.append("penwidth=2")
    tooltip = "\n".join(
        part for part in (
            f"{node.status}: {node.proof}",
            f"asked at {node.site}" if node.site else "",
            f"via {node.via}()" if node.via else "",
            f"reached once {node.behind} existed" if node.behind else "",
            "no package for the target" if node.missing else "",
            "package name guessed" if node.guessed else "",
        ) if part
    )
    attributes.append(f"tooltip={_quote(tooltip)}")
    return ", ".join(attributes)


def to_dict(graph: Graph) -> dict:
    return {
        "platform": graph.platform,
        "completed": graph.completed,
        "answer": (
            {
                "verdict": graph.answer.verdict,
                "headline": graph.answer.headline,
                "install": graph.answer.install,
                "missing": graph.answer.missing,
                "unmapped": graph.answer.unmapped,
                "toolchain": graph.answer.toolchain,
                "notes": graph.answer.notes,
            }
            if graph.answer else None
        ),
        "hard_requirements": [n.name for n in graph.required()],
        "host_gaps": graph.host_gaps,
        "nodes": [
            {
                "id": n.key,
                "name": n.name,
                "aliases": n.aliases,
                "kind": n.kind,
                "status": n.status,
                "proof": n.proof,
                "site": n.site,
                "via": n.via,
                "round": n.round,
                "reached_after": n.behind,
                "debian": list(n.debian),
                "guessed": n.guessed,
                "provided_by": n.provided_by,
                "available": n.available if n.checked else None,
                "missing_for_target": n.missing,
            }
            for n in sorted(graph.nodes.values(), key=lambda n: n.key)
        ],
        "edges": [
            {"from": parent or None, "to": child} for parent, child in graph.edges
        ],
    }
