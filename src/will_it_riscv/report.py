"""Rendering an :class:`~will_it_riscv.models.Analysis`."""

from __future__ import annotations

import json
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .distro import DistroIndex
from .models import Analysis, PackageReport, Verdict
from .syslibs import database

VERDICT_STYLE = {
    Verdict.PURE_PYTHON: "green",
    Verdict.WHEEL_AVAILABLE: "cyan",
    Verdict.NEEDS_BUILD: "yellow",
    Verdict.NEEDS_BUILD_BLOCKED: "red",
    Verdict.NO_DISTRIBUTION: "red",
    Verdict.UNRESOLVED: "magenta",
}

VERDICT_LABEL = {
    Verdict.PURE_PYTHON: "pure Python",
    Verdict.WHEEL_AVAILABLE: "wheel available",
    Verdict.NEEDS_BUILD: "must build from source",
    Verdict.NEEDS_BUILD_BLOCKED: "blocked build",
    Verdict.NO_DISTRIBUTION: "nothing installable",
    Verdict.UNRESOLVED: "unresolved",
}


def render_text(
    analysis: Analysis,
    console: Console,
    distro: Optional[DistroIndex] = None,
    verbose: bool = False,
) -> None:
    console.print()
    console.print(Text(f"will-it-riscv  ·  {analysis.root}", style="bold"))
    console.print(f"  target   {analysis.target}")
    if distro is not None:
        state = "ok" if distro.available else f"unavailable ({distro.error})"
        console.print(f"  distro   {distro.spec.label} / {distro.arch}  [{state}]")
    console.print()

    _summary(analysis, console)
    # Before the project section: when a configure stops, what stopped it is
    # the most useful sentence in the report.
    _pseudobuild_section(analysis, console)
    _project_section(analysis, console)

    for verdict in (
        Verdict.UNRESOLVED,
        Verdict.NO_DISTRIBUTION,
        Verdict.NEEDS_BUILD_BLOCKED,
        Verdict.NEEDS_BUILD,
    ):
        packages = analysis.by_verdict(verdict)
        if packages:
            _package_table(verdict, packages, console, distro, verbose)

    if verbose:
        for verdict in (Verdict.WHEEL_AVAILABLE, Verdict.PURE_PYTHON):
            packages = analysis.by_verdict(verdict)
            if packages:
                _package_table(verdict, packages, console, distro, verbose)

    _system_packages(analysis, console, distro)
    _distro_alternatives(analysis, console, distro)

    if analysis.warnings:
        console.print(Text("Warnings", style="bold"))
        for warning in analysis.warnings:
            console.print(f"  • {warning}", style="dim", highlight=False)
        console.print()


def _pseudobuild_section(analysis: Analysis, console: Console) -> None:
    """What running the configure demonstrated, as opposed to what we inferred."""
    result = analysis.pseudobuild
    if result is None:
        return
    console.print(Text("Pseudobuild", style="bold"))
    if result.error and not result.probes:
        console.print(f"  did not run: {result.error}", style="yellow")
        console.print()
        return

    outcome = "configure completed" if result.completed else "configure stopped early"
    rounds = "" if result.rounds <= 1 else f" over {result.rounds} rounds"
    console.print(
        f"  {outcome}{rounds} in {result.duration:.0f}s — "
        f"{len(result.probes)} dependency probes observed"
    )
    if result.blockers:
        console.print(
            Text("  hard requirements, in the order the build demanded them:",
                 style="bold red")
        )
        for index, name in enumerate(result.blockers, 1):
            console.print(f"    {index}. {name}", highlight=False)
        console.print(
            "    each of these stopped a configure; nothing else can be "
            "checked until they exist on the target",
            style="dim",
        )
    if result.soft_misses:
        console.print(
            "  proven optional (absent, and the configure carried on): "
            + ", ".join(sorted(result.soft_misses)),
            highlight=False,
        )
    if result.found:
        console.print(
            "  located on this host: " + ", ".join(sorted(result.found)),
            style="dim",
            highlight=False,
        )
    if not result.completed:
        console.print(
            "    the configure did not finish, so anything after the blocker "
            "was never reached and is missing from this report",
            style="dim yellow",
        )
    console.print()


def _project_section(analysis: Analysis, console: Console) -> None:
    """What building *this* project needs, as distinct from its dependencies."""
    profile = analysis.project_build
    if profile is None:
        return
    console.print(Text("This project", style="bold"))
    builds = ", ".join(sorted(profile.languages | profile.build_systems)) or "nothing"
    console.print(f"  builds   {builds}   ({analysis.files_scanned} files scanned)")
    if analysis.meson_introspect == "ok":
        console.print(
            "  meson    dependency list came from meson introspect", style="dim"
        )
    build_only = [
        r for r in profile.system_requirements
        if r.purpose == "build" and not r.optional
    ]
    libs = [r for r in build_only if r.kind == "library"]
    tools = [r for r in build_only if r.kind == "tool"]
    if libs:
        console.print(f"  links    {', '.join(r.name for r in libs)}", highlight=False)
    if tools:
        console.print(f"  toolchain {', '.join(r.name for r in tools)}", highlight=False)
    if not profile.is_native:
        console.print("  no compiled sources found", style="dim")
    console.print()


def _summary(analysis: Analysis, console: Console) -> None:
    counts = {v: len(analysis.by_verdict(v)) for v in Verdict}
    total = len(analysis.packages)
    if not total:
        if analysis.project_build is not None:
            console.print(
                "No dependency manifest at the repository root — reporting the "
                "project's own build requirements only.",
                style="dim",
            )
            console.print()
        return
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(justify="right")
    table.add_column()
    table.add_row(str(total), "packages reached")
    for verdict in Verdict:
        if counts[verdict]:
            table.add_row(
                Text(str(counts[verdict]), style=VERDICT_STYLE[verdict]),
                VERDICT_LABEL[verdict],
            )
    console.print(table)
    console.print()


def _package_table(
    verdict: Verdict,
    packages: list[PackageReport],
    console: Console,
    distro: Optional[DistroIndex],
    verbose: bool = False,
) -> None:
    title = f"{VERDICT_LABEL[verdict].capitalize()} ({len(packages)})"
    table = Table(
        title=title,
        title_justify="left",
        title_style=f"bold {VERDICT_STYLE[verdict]}",
        expand=True,
        pad_edge=False,
    )
    table.add_column("package", style="bold", ratio=3, no_wrap=True, overflow="ellipsis")
    table.add_column("version", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("builds", ratio=3, overflow="fold")
    table.add_column("needs", ratio=5, overflow="fold")
    if verbose:
        table.add_column("why", style="dim", ratio=6, overflow="fold")

    for report in packages:
        builds = ", ".join(sorted(report.build.languages | report.build.build_systems)) or "—"
        libs = [r.name for r in report.build.system_requirements if r.kind == "library"]
        row = [
            report.name + (" [b]" if report.is_build_dependency else ""),
            report.version or "—",
            builds,
            ", ".join(libs) if libs else "—",
        ]
        if verbose:
            row.append("; ".join(report.reasons))
        table.add_row(*row)
    console.print(table)
    console.print()


def _system_packages(
    analysis: Analysis, console: Console, distro: Optional[DistroIndex]
) -> None:
    requirements = analysis.required_system_requirements()
    if not requirements:
        _optional_section(analysis, console, distro)
        _other_purposes(analysis, console, distro)
        return
    db = database()
    bundled = [r for r in requirements.values() if analysis.is_bundled(r)]
    rest = [r for r in requirements.values() if not analysis.is_bundled(r)]
    known = [r for r in rest if not db.is_guess(r)]
    guessed = [r for r in rest if db.is_guess(r)]

    console.print(Text("System packages needed to build the above", style="bold"))

    checked = distro if distro is not None and distro.available else None
    present: list[str] = []
    missing: list[str] = []
    for req in known:
        for package in req.debian:
            if checked is not None:
                (present if checked.has(package) else missing).append(package)
            else:
                present.append(package)

    if present:
        console.print()
        console.print(f"  sudo apt install {' '.join(sorted(set(present)))}", highlight=False)
        if checked is not None:
            console.print(
                f"    all confirmed present in {checked.spec.label} for {checked.arch}",
                style="dim",
            )
        else:
            console.print("    availability not checked (--no-distro)", style="dim")

    if missing and checked is not None:
        console.print()
        console.print(
            Text(f"  no {checked.arch} package in {checked.spec.label}:", style="bold red")
        )
        for package in sorted(set(missing)):
            console.print(
                f"    • {package}  — needed by {_owners(analysis, package)}",
                highlight=False,
            )
        console.print(
            "    build these from source on the target, or vendor them", style="dim"
        )

    if bundled:
        console.print()
        console.print(
            Text("  optional — the project bundles its own copy:", style="bold cyan")
        )
        for req in sorted(bundled, key=lambda r: r.name):
            packages = " / ".join(req.debian) or "—"
            console.print(f"    • {req.name}  (system package: {packages})",
                          highlight=False)
        console.print(
            "    the build falls back to the bundled source if these are absent",
            style="dim",
        )

    if guessed:
        console.print()
        console.print(
            Text("  referenced by a build, but not in the name map:", style="bold yellow")
        )
        for req in sorted(guessed, key=lambda r: r.name):
            where = ", ".join(sorted(req.found_in)[:3])
            console.print(f"    • {req.name}  — from {where}", highlight=False)
        console.print(
            "    these are raw names scraped from build files; identify the package "
            "yourself before installing",
            style="dim",
        )
    console.print()
    _optional_section(analysis, console, distro)
    _other_purposes(analysis, console, distro)


def _optional_section(
    analysis: Analysis, console: Console, distro: Optional[DistroIndex]
) -> None:
    """Dependencies a default build does not reach.

    A big CMake project spends most of its find_package calls on backends
    nobody enables. Listing them as requirements is how a project that needs
    LLVM and a compiler looks like it needs three vendor GPU stacks.
    """
    optional = analysis.optional_system_requirements()
    if not optional:
        return
    console.print(
        Text(f"Optional — not built unless you ask for it ({len(optional)})",
             style="bold blue")
    )
    for req in sorted(optional.values(), key=lambda r: r.name):
        gate = f"  ← {req.gate}" if req.gate else ""
        packages = " / ".join(p for p in req.debian if p != req.name)
        label = f"{req.name:<24} {packages}" if packages else req.name
        console.print(f"    • {label.rstrip()}{gate}", highlight=False)
    console.print(
        "    left out of the install line above; a default build does not "
        "reach these",
        style="dim",
    )
    console.print()


_PURPOSE_HEADINGS = {
    "test": ("Additionally needed to run the test suite", "cyan"),
    "docs": ("Additionally needed to build the documentation", "cyan"),
}


def _other_purposes(
    analysis: Analysis, console: Console, distro: Optional[DistroIndex]
) -> None:
    """Test and documentation packages, kept out of the build install line."""
    for purpose in ("test", "docs"):
        requirements = analysis.all_system_requirements(purpose)
        if not requirements:
            continue
        heading, style = _PURPOSE_HEADINGS[purpose]
        console.print(Text(f"{heading} ({len(requirements)})", style=f"bold {style}"))
        packages = sorted({p for r in requirements.values() for p in r.debian})
        console.print(f"  sudo apt install {' '.join(packages)}", highlight=False)
        if distro is not None and distro.available:
            missing = [p for p in packages if not distro.has(p)]
            if missing:
                console.print(
                    f"    not in {distro.spec.label} for {distro.arch}: "
                    + ", ".join(missing),
                    style="dim yellow",
                    highlight=False,
                )
        console.print("    not required to compile the project", style="dim")
        console.print()


def _owners(analysis: Analysis, debian_package: str) -> str:
    for req in analysis.all_system_requirements().values():
        if debian_package in req.debian:
            return ", ".join(sorted(req.found_in)[:4])
    return "?"


def _distro_alternatives(
    analysis: Analysis, console: Console, distro: Optional[DistroIndex]
) -> None:
    if distro is None or not distro.available:
        return
    rows = [
        (p.name, p.version or "—", p.distro_packages[distro.spec.id])
        for p in analysis.packages.values()
        if distro.spec.id in p.distro_packages
    ]
    if not rows:
        return
    table = Table(
        title=f"Already packaged by {distro.spec.label} — install instead of building "
        f"({len(rows)})",
        title_justify="left",
        title_style="bold green",
        expand=True,
        pad_edge=False,
    )
    table.add_column("package", style="bold")
    table.add_column("pypi version")
    table.add_column("distro package")
    for row in sorted(rows):
        table.add_row(*row)
    console.print(table)
    console.print(
        "  note: the distro version may differ from what the project pins",
        style="dim",
    )
    console.print()


# ------------------------------------------------------------------ other


def to_dict(analysis: Analysis) -> dict:
    return {
        "root": analysis.root,
        "target": analysis.target,
        "python_version": analysis.python_version,
        "summary": {
            v.value: len(analysis.by_verdict(v)) for v in Verdict if analysis.by_verdict(v)
        },
        "exit_code": analysis.exit_code(),
        "packages": [
            {
                "name": p.name,
                "version": p.version,
                "verdict": p.verdict.value,
                "pure": p.verdict is Verdict.PURE_PYTHON,
                "depth": p.depth,
                "build_dependency": p.is_build_dependency,
                "extras": sorted(p.extras),
                "required_by": p.required_by,
                "matching_wheels": p.matching_wheels,
                "published_platform_tags": p.wheel_platform_tags,
                "has_sdist": p.has_sdist,
                "build": {
                    "backend": p.build.build_backend,
                    "build_requires": p.build.build_requires,
                    "languages": sorted(p.build.languages),
                    "build_systems": sorted(p.build.build_systems),
                    "evidence": sorted(e.value for e in p.build.evidence),
                    "inspected": p.build.inspected,
                    "system_requirements": [
                        {
                            "name": r.name,
                            "kind": r.kind,
                            "debian": list(r.debian),
                            "fedora": list(r.fedora),
                            "found_in": list(r.found_in),
                        }
                        for r in p.build.system_requirements
                    ],
                },
                "distro_packages": p.distro_packages,
                "reasons": p.reasons,
            }
            for p in sorted(analysis.packages.values(), key=lambda r: r.sort_key())
        ],
        "pseudobuild": (
            {
                "completed": analysis.pseudobuild.completed,
                "rounds": analysis.pseudobuild.rounds,
                "blockers": list(analysis.pseudobuild.blockers),
                "stubbed": list(analysis.pseudobuild.unblocked),
                "duration_seconds": round(analysis.pseudobuild.duration, 1),
                "probes": sorted(analysis.pseudobuild.probes),
                "found": sorted(analysis.pseudobuild.found),
                "proven_optional": sorted(analysis.pseudobuild.soft_misses),
                "blocking": analysis.pseudobuild.blocking,
                "error": analysis.pseudobuild.error,
            }
            if analysis.pseudobuild is not None
            else None
        ),
        "project": (
            {
                "files_scanned": analysis.files_scanned,
                "meson_introspect": analysis.meson_introspect,
                "languages": sorted(analysis.project_build.languages),
                "build_systems": sorted(analysis.project_build.build_systems),
                "system_requirements": [
                    _requirement_dict(r)
                    for r in analysis.project_requirements.values()
                ],
            }
            if analysis.project_build is not None
            else None
        ),
        "system_requirements": [
            _requirement_dict(r) for r in analysis.system_requirements.values()
        ],
        "all_system_requirements": [
            _requirement_dict(r) for r in analysis.all_system_requirements().values()
        ],
        "required_system_requirements": [
            _requirement_dict(r) for r in analysis.required_system_requirements().values()
        ],
        "optional_system_requirements": [
            _requirement_dict(r) for r in analysis.optional_system_requirements().values()
        ],
        "system_requirements_by_purpose": {
            purpose: [
                _requirement_dict(r)
                for r in analysis.all_system_requirements(purpose).values()
            ]
            for purpose in ("build", "test", "docs")
        },
        "warnings": analysis.warnings,
    }


def _requirement_dict(r) -> dict:
    return {
        "name": r.name,
        "kind": r.kind,
        "pkgconfig": r.pkgconfig,
        "debian": list(r.debian),
        "fedora": list(r.fedora),
        "declared": r.declared,
        "purpose": r.purpose,
        "optional": r.optional,
        "enabled_by": r.gate,
        "required_by": sorted(r.found_in),
    }


def render_json(analysis: Analysis) -> str:
    return json.dumps(to_dict(analysis), indent=2)


def render_markdown(analysis: Analysis, distro: Optional[DistroIndex] = None) -> str:
    out: list[str] = [f"# will-it-riscv: {analysis.root}", ""]
    out.append(f"- **Target:** {analysis.target}")
    if distro is not None:
        out.append(f"- **Distro:** {distro.spec.label} / {distro.arch}")
    out.append(f"- **Packages reached:** {len(analysis.packages)}")
    out.append("")

    out.append("| verdict | count |")
    out.append("| --- | --- |")
    for verdict in Verdict:
        packages = analysis.by_verdict(verdict)
        if packages:
            out.append(f"| {VERDICT_LABEL[verdict]} | {len(packages)} |")
    out.append("")

    for verdict in (
        Verdict.UNRESOLVED,
        Verdict.NO_DISTRIBUTION,
        Verdict.NEEDS_BUILD_BLOCKED,
        Verdict.NEEDS_BUILD,
    ):
        packages = analysis.by_verdict(verdict)
        if not packages:
            continue
        out.append(f"## {VERDICT_LABEL[verdict].capitalize()} ({len(packages)})")
        out.append("")
        out.append("| package | version | builds | needs | why |")
        out.append("| --- | --- | --- | --- | --- |")
        for report in packages:
            builds = ", ".join(sorted(report.build.languages | report.build.build_systems)) or "—"
            libs = ", ".join(
                r.name for r in report.build.system_requirements if r.kind == "library"
            ) or "—"
            why = report.reasons[0].replace("|", "\\|") if report.reasons else ""
            out.append(
                f"| `{report.name}` | {report.version or '—'} | {builds} | {libs} | {why} |"
            )
        out.append("")

    requirements = analysis.required_system_requirements()
    if requirements:
        db = database()
        known = sorted(
            {p for r in requirements.values() if not db.is_guess(r) for p in r.debian}
        )
        guessed = sorted(r.name for r in requirements.values() if db.is_guess(r))
        out.append("## System packages")
        out.append("")
        if known:
            out.append("```console")
            out.append(f"$ sudo apt install {' '.join(known)}")
            out.append("```")
            out.append("")
        if guessed:
            out.append("Referenced by a build but not in the name map — identify these "
                       "yourself before installing:")
            out.append("")
            out.extend(f"- `{name}`" for name in guessed)
            out.append("")

    optional = analysis.optional_system_requirements()
    if optional:
        out.append("## Optional — not built unless you ask for it")
        out.append("")
        out.append("| dependency | system package | enabled by |")
        out.append("| --- | --- | --- |")
        for req in sorted(optional.values(), key=lambda r: r.name):
            listed = " / ".join(f"`{p}`" for p in req.debian) or "—"
            out.append(f"| `{req.name}` | {listed} | {req.gate or '—'} |")
        out.append("")

    for purpose in ("test", "docs"):
        extra = analysis.all_system_requirements(purpose)
        if not extra:
            continue
        heading, _ = _PURPOSE_HEADINGS[purpose]
        extra_packages = sorted({p for r in extra.values() for p in r.debian})
        out.append(f"## {heading}")
        out.append("")
        out.append("```console")
        out.append(f"$ sudo apt install {' '.join(extra_packages)}")
        out.append("```")
        out.append("")

    if analysis.warnings:
        out.append("## Warnings")
        out.append("")
        out.extend(f"- {w}" for w in analysis.warnings)
        out.append("")
    return "\n".join(out)


def render_list(analysis: Analysis) -> str:
    """Just the non-pure packages, one ``name==version`` per line."""
    lines = []
    for report in sorted(analysis.packages.values(), key=lambda r: r.name):
        if report.verdict is Verdict.PURE_PYTHON:
            continue
        lines.append(f"{report.name}=={report.version}" if report.version else report.name)
    return "\n".join(lines)
