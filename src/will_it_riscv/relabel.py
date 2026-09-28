"""The executor decides what is required; the teacher only says what the docs say.

A teacher reading a repository copies its install lines -- a CI job's apt
line, a Dockerfile's -- and marks them required. Whether the default build
needs each package is not the teacher's call: configure the plan's default
build, prune what it found, and see. This rewrites one mark only, and only
where a configure settled it:

  required -> optional   the default configure looked for it, did not find
                         it, and completed; or completed with it disabled
  optional -> required   the default configure stopped without it, or failed
                         with it disabled

Packages are never added or removed: the plan still says what the docs
say to install. A package the configure never asked about keeps its mark,
since a header-only library is included without any find_package at all.

The relabelled plan is written beside the teacher's, as
``plan.relabelled.json``, with the proof of every change in ``relabel.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from .plan import CMAKE_CONFIGURE, SYSTEM_PACKAGES, Plan, Step, load_plan, parse_plan, plan_to_dict

OPTIONAL_BECAUSE = (
    "installing it: the default configure uses it when found and completes without it"
)


@dataclass
class Verdicts:
    """What the plan's default configures proved, per package name."""

    required: dict = field(default_factory=dict)
    """Distro package to the dependency and proof that make it required."""
    optional: dict = field(default_factory=dict)
    configures: int = 0
    completed: int = 0


def verdicts(plan: Plan, root: Path, key: str, distro, trace) -> Verdicts:
    """Run the plan's default configures and settle each dependency they asked about.

    ``trace(key, source_dir, defines)`` returns a PseudoBuild (cached by the
    caller); the graph turns it into statuses, the resolver into packages.
    """
    from . import graph as depgraph

    contents = distro.contents() if distro is not None else None
    found = Verdicts()
    for step in plan.steps:
        if step.kind != CMAKE_CONFIGURE or step.optional:
            continue
        source = (root / step.source).resolve()
        if not (source / "CMakeLists.txt").is_file():
            continue
        result = trace(
            key if step.source in (".", "") else f"{key}@{step.source}", source, dict(step.defines)
        )
        if result is None:
            continue
        found.configures += 1
        found.completed += bool(result.completed)
        built = depgraph.build(result, key, distro)
        for node in built.nodes.values():
            if node.status == depgraph.REQUIRED:
                table = found.required
            elif node.status == depgraph.OPTIONAL:
                table = found.optional
            else:
                continue
            for package in _packages(node, result, contents):
                table.setdefault(package, (node.name, node.proof))
    # Required anywhere wins: one default configure needing it is enough.
    for package in found.required:
        found.optional.pop(package, None)
    return found


def _packages(node, result, contents) -> set:
    """Every distro package that would satisfy a node: its own, and the archive's."""
    packages = set(node.debian) | ({node.available} if node.available else set())
    if contents is not None:
        for alias in node.aliases or [node.name]:
            probe = result.probes.get(alias.lower())
            command = probe.command if probe is not None else "find_package"
            packages |= set(
                contents.providers(alias, command, result.lookups.get(alias.lower(), []))[0]
            )
    return packages


def _source_of(package: str, contents) -> str:
    """Debian source package, so libcurl4-gnutls-dev and libcurl4-openssl-dev agree."""
    if contents is not None and package in contents.sources:
        return contents.sources[package]
    return re.sub(r"(-dev|-bin)$", "", package)


def relabel(plan: Plan, found: Verdicts, contents=None) -> tuple[Plan, list[dict]]:
    """The plan with its packages' marks set by what the configures proved."""
    by_source: dict[str, tuple[str, tuple]] = {}
    for verdict, table in (("optional", found.optional), ("required", found.required)):
        for package, why in table.items():
            by_source[_source_of(package, contents)] = (verdict, why)

    def verdict_for(package: str) -> Optional[tuple[str, tuple]]:
        name = re.split(r"[=<>\s]", package.strip(), maxsplit=1)[0]
        return by_source.get(_source_of(name, contents))

    changes: list[dict] = []
    steps: list[Step] = []
    promoted: list[tuple[str, Step]] = []
    removed: dict[str, list[str]] = {}
    for step in plan.steps:
        if step.kind != SYSTEM_PACKAGES:
            steps.append(step)
            continue
        keep, move = [], []
        for package in step.packages:
            verdict = verdict_for(package)
            if verdict and verdict[0] == "optional" and not step.optional:
                move.append(package)
                changes.append(
                    {
                        "package": package,
                        "from": "required",
                        "to": "optional",
                        "dependency": verdict[1][0],
                        "proof": verdict[1][1],
                    }
                )
            elif verdict and verdict[0] == "required" and step.optional:
                promoted.append((package, step))
                changes.append(
                    {
                        "package": package,
                        "from": "optional",
                        "to": "required",
                        "dependency": verdict[1][0],
                        "proof": verdict[1][1],
                    }
                )
            else:
                keep.append(package)
        if step.optional:
            keep = [p for p in keep if p not in {q for q, _ in promoted}]
        if keep:
            steps.append(replace(step, packages=keep))
        if move:
            steps.append(
                replace(
                    step,
                    id=_fresh(f"{step.id}-optional", plan, steps),
                    packages=move,
                    optional=True,
                    enabled_by=OPTIONAL_BECAUSE,
                    note="Found by the default configure if installed; it completes without it.",
                )
            )
        if not keep:
            removed[step.id] = list(step.after)
    # A step whose packages all moved out is gone; whatever ran after it now
    # follows what it followed.
    while removed:
        gone, before = removed.popitem()
        steps = [
            replace(s, after=list(dict.fromkeys([a for a in s.after if a != gone] + before)))
            if gone in s.after
            else s
            for s in steps
        ]
    if promoted:
        target = next((s for s in steps if s.kind == SYSTEM_PACKAGES and not s.optional), None)
        extra = [p for p, _ in promoted]
        if target is not None:
            steps = [
                replace(
                    s,
                    packages=s.packages + extra,
                    evidence=s.evidence + [e for _, o in promoted for e in o.evidence],
                )
                if s is target
                else s
                for s in steps
            ]
        else:
            source = promoted[0][1]
            steps.insert(
                0,
                replace(
                    source,
                    id=_fresh("required-packages", plan, steps),
                    packages=extra,
                    optional=False,
                    enabled_by=None,
                    note="The default configure stops without these.",
                ),
            )
    relabelled = parse_plan(plan_to_dict(replace(plan, steps=steps)))
    return relabelled, changes


def _fresh(wanted: str, plan: Plan, steps: list[Step]) -> str:
    taken = {s.id for s in plan.steps} | {s.id for s in steps}
    name, n = wanted, 2
    while name in taken:
        name, n = f"{wanted}-{n}", n + 1
    return name


def relabel_example(directory: Path, distro, trace) -> Optional[dict]:
    """Relabel one labelled example in place; the report, or None if nothing to run."""
    from . import dataset

    meta = json.loads((directory / "meta.json").read_text())
    root = dataset.tree(meta).root
    if root is None:
        return None
    plan = load_plan(directory / "plan.json")
    found = verdicts(plan, root, directory.name, distro, trace)
    if not found.configures:
        return None
    relabelled, changes = relabel(plan, found, distro.contents() if distro else None)
    (directory / "plan.relabelled.json").write_text(json.dumps(plan_to_dict(relabelled), indent=2))
    report = {"configures": found.configures, "completed": found.completed, "changes": changes}
    (directory / "relabel.json").write_text(json.dumps(report, indent=2))
    return report
