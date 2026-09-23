"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import httpx
from packaging.requirements import InvalidRequirement, Requirement
from rich.console import Console

from . import __version__
from .analyze import Analyzer
from .cache import Cache
from .distro import DEFAULT_DISTRO, KNOWN_DISTROS, DistroIndex, resolve_spec
from .index import IndexError_, PackageIndex
from .inputs import RootRequirements, load
from .models import Analysis, Verdict
from .report import render_dot, render_json, render_list, render_markdown, render_text
from .source import RepositoryInspection, inspect_repository
from .target import Target

USER_AGENT = f"will-it-riscv/{__version__} (+https://github.com/tactcomplabs/will-it-riscv)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="will-it-riscv",
        description=(
            "Find out what a project needs to build on riscv64. Point it at a "
            "repository to scan its sources and CI configuration, or at a "
            "pyproject.toml / requirements.txt to walk its dependency tree. "
            "Reports what has no wheel for the target, what must be compiled, and "
            "which system packages those builds need -- checked against the "
            "target distro's actual archive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes:\n"
            "  0  everything installs from wheels or is pure Python\n"
            "  1  some packages must be built from source\n"
            "  2  something is blocked, unresolvable, or has no distribution\n"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="pyproject.toml, requirements.txt, or a directory containing one "
        "(default: the current directory)",
    )
    parser.add_argument(
        "-p", "--package", action="append", default=[], metavar="SPEC",
        help="analyze this requirement instead of a file; repeatable "
        "(e.g. -p 'numpy>=2' -p pandas)",
    )

    target = parser.add_argument_group("target")
    target.add_argument(
        "-t", "--target", default="riscv64", metavar="ARCH",
        help="target architecture or triple: riscv64 (default), "
        "riscv64-unknown-linux-gnu, riscv64-musl1.2, aarch64, x86_64 ...",
    )
    target.add_argument(
        "--python", metavar="X.Y", default=None,
        help="target Python version (default: whatever this interpreter is)",
    )
    target.add_argument(
        "--free-threaded", action="store_true",
        help="target a free-threaded (no-GIL) interpreter",
    )

    scope = parser.add_argument_group("scope")
    scope.add_argument(
        "-E", "--extra", action="append", default=[], metavar="NAME",
        help="include this optional-dependency group; 'all' for every one",
    )
    scope.add_argument(
        "-G", "--group", action="append", default=[], metavar="NAME",
        help="include this PEP 735 dependency group; 'all' for every one",
    )
    scope.add_argument(
        "--no-build-deps", action="store_true",
        help="skip build-system requirements (they are followed by default, because "
        "an unbuildable build backend is the most common surprise)",
    )
    scope.add_argument(
        "--no-inspect-sdists", action="store_true",
        help="do not download source distributions; classify from wheel tags alone "
        "(much faster, much less accurate)",
    )
    scope.add_argument("--pre", action="store_true", help="consider pre-releases")
    scope.add_argument(
        "--no-scan", action="store_true",
        help="do not scan the source tree; only resolve declared dependencies",
    )
    scope.add_argument(
        "--no-ci-scan", action="store_true",
        help="skip CI configs and Dockerfiles when scanning a source tree",
    )
    scope.add_argument(
        "--pseudobuild", action="store_true",
        help="configure the project for real, as a linux build for the target "
        "architecture confined to an empty scratch sysroot, and watch what it "
        "asks for. Finds what static reading cannot -- a dependency the "
        "configure shrugs off is proven optional, and one that stops it is "
        "proven required -- and answers whether the target's distro has "
        "everything it demands. RUNS THE PROJECT'S BUILD SCRIPTS: only do this "
        "for a repository you trust.",
    )
    scope.add_argument(
        "--plan", metavar="PLAN.json",
        help="run a build plan -- everything the repository's build runs, in "
        "order -- in the pretend environment, and join what every step needs "
        "into one graph. RUNS THE CONFIGURE STEPS IT LISTS. See examples/plans/.",
    )
    scope.add_argument(
        "--pseudobuild-timeout", type=int, default=600, metavar="SECONDS",
        help="time allowed for the whole unblock-and-rerun loop "
        "(default: %(default)s)",
    )
    scope.add_argument(
        "--no-meson-introspect", action="store_true",
        help="do not ask meson about its own dependencies; read the "
        "meson.build files directly instead (meson resolves the project's "
        "languages first, so it runs compiler probes and can fail)",
    )
    scope.add_argument(
        "--max-depth", type=int, default=None, metavar="N",
        help="stop walking below this depth",
    )

    distro = parser.add_argument_group("distro")
    distro.add_argument(
        "--distro", default=DEFAULT_DISTRO, metavar="ID",
        help="distro to check system packages against: "
        + ", ".join(sorted(KNOWN_DISTROS)) + " (default: %(default)s)",
    )
    distro.add_argument(
        "--no-distro", action="store_true",
        help="skip the distro availability check (no large index download)",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-f", "--format", choices=("text", "json", "markdown", "list", "dot"),
        default="text",
        help="'list' prints just the non-pure-Python packages, one per line; "
        "'dot' prints the project's dependency graph for Graphviz "
        "(dot -Tsvg), from the pseudobuild when there is one",
    )
    output.add_argument("-o", "--output", metavar="FILE", help="write the report here")
    output.add_argument(
        "-v", "--verbose", action="store_true",
        help="also table the packages that are fine",
    )
    output.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    output.add_argument(
        "--exit-zero", action="store_true", help="always exit 0, whatever we find"
    )

    net = parser.add_argument_group("network and cache")
    net.add_argument(
        "--index-url", default="https://pypi.org/simple/", metavar="URL",
        help="PEP 691 JSON simple index (default: %(default)s)",
    )
    net.add_argument("--workers", type=int, default=8, metavar="N")
    net.add_argument("--timeout", type=float, default=30.0, metavar="SECONDS")
    net.add_argument("--no-cache", action="store_true")
    net.add_argument(
        "--clear-cache", action="store_true", help="empty the cache and exit"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _python_version(raw: Optional[str]) -> tuple[int, int]:
    if not raw:
        return sys.version_info[0], sys.version_info[1]
    parts = raw.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError as exc:
        raise SystemExit(f"--python: expected X.Y, got {raw!r}") from exc


def _roots(args: argparse.Namespace) -> RootRequirements:
    if args.package:
        roots = RootRequirements(source="command line", project_name="(command line)")
        for spec in args.package:
            try:
                roots.runtime.append(Requirement(spec))
            except InvalidRequirement as exc:
                raise SystemExit(f"-p {spec!r}: {exc}") from exc
        return roots
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"{path}: no such file or directory")
    try:
        return load(path, tuple(args.extra), tuple(args.group))
    except FileNotFoundError:
        # A source tree with no Python manifest -- GROMACS, say. That is a
        # perfectly good thing to analyse; there are just no declared
        # dependencies to resolve.
        return RootRequirements(source=str(path), project_name=path.resolve().name)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: {exc}") from exc


def _adopt_script_installs(
    roots: RootRequirements,
    scan: RepositoryInspection,
    extras: tuple[str, ...] = (),
    groups: tuple[str, ...] = (),
) -> list:
    """Analyse what the project's scripts install, not only what it declares.

    ``./mfc.sh build`` pip-installs toolchain/ before any CMake runs, so that
    manifest is the first thing the build needs, deep in the tree or not.
    Returns the installs adopted; ``roots`` is extended in place.
    """
    adopted = []
    for install in scan.script_installs:
        if install.kind == "requirement":
            try:
                roots.runtime.append(Requirement(install.target))
            except InvalidRequirement:
                continue
            adopted.append(install)
            continue
        path = scan.root / install.target
        if roots.source and Path(roots.source).resolve() == path.resolve():
            continue   # the root manifest, already loaded
        try:
            loaded = load(path, tuple(install.extras) + extras, groups)
        except (OSError, ValueError) as exc:
            roots.warnings.append(f"{install.target}, installed by {install.via}: {exc}")
            continue
        roots.runtime += loaded.runtime
        roots.build += loaded.build
        roots.warnings += loaded.warnings
        adopted.append(install)
    return adopted


def _unanalysed_manifests(scan: RepositoryInspection, adopted: list) -> list[Path]:
    """Manifests below the root that nothing installs, to offer, not adopt.

    A deep manifest is usually for something else -- documentation,
    bindings, a test harness. Unless the project's own scripts install it:
    then it is part of the build, and it was analysed.
    """
    installed = {(scan.root / a.target).resolve() for a in adopted if a.kind == "manifest"}
    return [
        m for m in scan.manifests
        if m.parent != scan.root and m.resolve() not in installed
    ]


def _scan_source_tree(
    args: argparse.Namespace, stderr: Console, arch: str = "riscv64"
) -> Optional[RepositoryInspection]:
    """Scan the repository at args.path, unless told not to."""
    if args.package or args.no_scan:
        return None
    path = Path(args.path)
    if not path.is_dir():
        return None
    scan_ci = not args.no_ci_scan
    use_meson = not args.no_meson_introspect
    if args.quiet:
        return inspect_repository(
            path, scan_ci=scan_ci, use_meson_introspect=use_meson,
            pseudobuild=args.pseudobuild,
            pseudobuild_timeout=args.pseudobuild_timeout,
            pseudobuild_arch=arch,
        )
    label = f"configuring as linux/{arch}" if args.pseudobuild else "scanning"
    with stderr.status(f"{label} {path}…"):
        return inspect_repository(
            path, scan_ci=scan_ci, use_meson_introspect=use_meson,
            pseudobuild=args.pseudobuild,
            pseudobuild_timeout=args.pseudobuild_timeout,
            pseudobuild_arch=arch,
        )


def _annotate_distro(analysis: Analysis, distro: DistroIndex) -> None:
    """Note where the distro already ships a package we would otherwise build."""
    if not distro.available:
        return
    interesting = (
        Verdict.NEEDS_BUILD,
        Verdict.NEEDS_BUILD_BLOCKED,
        Verdict.NO_DISTRIBUTION,
        Verdict.UNRESOLVED,
    )
    for report in analysis.packages.values():
        if report.verdict not in interesting:
            continue
        package = distro.python_package(report.name)
        if package:
            report.distro_packages[distro.spec.id] = package


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stderr = Console(stderr=True, quiet=args.quiet)
    cache = Cache(enabled=not args.no_cache)

    if args.clear_cache:
        removed = cache.clear()
        stderr.print(f"cleared {removed} cached file(s) from {cache.root}")
        return 0

    python_version = _python_version(args.python)
    target = Target.parse(args.target, python_version)
    if args.free_threaded:
        target = Target(
            arch=target.arch,
            libc=target.libc,
            libc_version=target.libc_version,
            python_version=target.python_version,
            implementation=target.implementation,
            free_threaded=True,
        )

    if args.plan:
        return _run_plan(args, target, cache, stderr)

    scan = _scan_source_tree(args, stderr, target.arch)
    roots = _roots(args)
    had_root_manifest = bool(roots)
    adopted = (
        _adopt_script_installs(roots, scan, tuple(args.extra), tuple(args.group))
        if scan is not None else []
    )
    deep_manifests: list[Path] = []
    if scan is not None and not had_root_manifest and scan.manifests:
        deep_manifests = _unanalysed_manifests(scan, adopted)
    if not roots and scan is None:
        stderr.print(f"[yellow]{roots.source}: no dependencies declared[/yellow]")
        return 0

    limits = httpx.Limits(max_connections=args.workers * 2)
    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        limits=limits,
    ) as client:
        index = PackageIndex(client, cache, url=args.index_url)

        distro: Optional[DistroIndex] = None
        if not args.no_distro:
            try:
                spec = resolve_spec(args.distro)
            except KeyError as exc:
                raise SystemExit(str(exc)) from exc
            distro = DistroIndex(client, cache, spec, target.arch)

        analyzer = Analyzer(
            index,
            target,
            include_build_deps=not args.no_build_deps,
            inspect_sdists=not args.no_inspect_sdists,
            allow_prereleases=args.pre,
            max_depth=args.max_depth,
            workers=args.workers,
        )

        try:
            if args.quiet:
                analysis = analyzer.run(roots)
            else:
                with stderr.status("resolving…", spinner="dots") as status:
                    analyzer.progress = lambda event, name: status.update(
                        f"{event} {name}"
                    )
                    analysis = analyzer.run(roots)
        except IndexError_ as exc:
            stderr.print(f"[red]index error:[/red] {exc}")
            return 3
        except KeyboardInterrupt:
            stderr.print("[yellow]interrupted[/yellow]")
            return 130

        if scan is not None:
            analysis.project_build = scan.profile
            analysis.project_requirements = {
                r.name: r for r in scan.profile.system_requirements
            }
            analysis.files_scanned = scan.files_scanned
            analysis.bundled_libraries = scan.bundled
            analysis.meson_introspect = scan.meson_introspect
            analysis.pseudobuild = scan.pseudobuild
            analysis.script_installs = adopted
            for warning in scan.warnings:
                analysis.add_warning(warning)
            analysis.root = scan.name
            for manifest in deep_manifests:
                analysis.add_warning(
                    f"{manifest.relative_to(scan.root)} was not analysed: it is not "
                    "at the repository root, so it probably describes something "
                    "other than this project's own dependencies. Point at it "
                    "directly to analyse it."
                )

        if distro is not None:
            if not args.quiet:
                with stderr.status(f"loading {distro.spec.label} package index…"):
                    distro.names()
            if distro.error:
                analysis.warnings.append(f"distro check skipped: {distro.error}")
            _annotate_distro(analysis, distro)

        _emit(args, analysis, distro)

    return 0 if args.exit_zero else analysis.exit_code()


def _run_plan(
    args: argparse.Namespace, target: Target, cache: Cache, stderr: Console
) -> int:
    """--plan: run each step in the pretend environment, report one graph."""
    from . import planrun
    from .plan import PlanError, load_plan

    root = Path(args.path)
    if not root.is_dir():
        raise SystemExit(f"{root}: --plan needs the repository directory")
    try:
        plan = load_plan(Path(args.plan))
    except PlanError as exc:
        stderr.print(f"[red]{args.plan}: not a plan this can run[/red]")
        for problem in exc.problems:
            stderr.print(f"  • {problem}", highlight=False)
        return 2
    if args.format not in ("text", "json", "dot"):
        raise SystemExit("--plan reports as text, json or dot")

    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        limits=httpx.Limits(max_connections=args.workers * 2),
    ) as client:
        index = PackageIndex(client, cache, url=args.index_url)
        distro: Optional[DistroIndex] = None
        if not args.no_distro:
            try:
                distro = DistroIndex(client, cache, resolve_spec(args.distro), target.arch)
            except KeyError as exc:
                raise SystemExit(str(exc)) from exc
            distro.names()
        try:
            if args.quiet:
                result = planrun.execute(
                    plan, root, index=index, target=target, distro=distro,
                    timeout=args.pseudobuild_timeout,
                )
            else:
                with stderr.status(f"running {plan.repo}'s plan…") as status:
                    result = planrun.execute(
                        plan, root, index=index, target=target, distro=distro,
                        timeout=args.pseudobuild_timeout,
                        progress=lambda what: status.update(f"{plan.repo}: {what}"),
                    )
        except IndexError_ as exc:
            stderr.print(f"[red]index error:[/red] {exc}")
            return 3

    if args.format == "json":
        payload = json.dumps(planrun.to_dict(result), indent=2)
    elif args.format == "dot":
        payload = planrun.to_dot(result)
    else:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                planrun.render_text(result, Console(file=handle, width=120))
        else:
            planrun.render_text(result, Console())
        return 0 if args.exit_zero else planrun.exit_code(result)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0 if args.exit_zero else planrun.exit_code(result)


def _emit(args: argparse.Namespace, analysis: Analysis, distro: Optional[DistroIndex]) -> None:
    if args.format == "text" and not args.output:
        render_text(analysis, Console(), distro, verbose=args.verbose)
        return

    if args.format == "json":
        payload = render_json(analysis, distro)
    elif args.format == "dot":
        payload = render_dot(analysis, distro)
    elif args.format == "markdown":
        payload = render_markdown(analysis, distro)
    elif args.format == "list":
        payload = render_list(analysis)
    else:
        console = Console(file=open(args.output, "w", encoding="utf-8"), width=120)
        render_text(analysis, console, distro, verbose=args.verbose)
        console.file.close()
        return

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
