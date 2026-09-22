"""Command line interface."""

from __future__ import annotations

import argparse
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
from .report import render_json, render_list, render_markdown, render_text
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
        "-f", "--format", choices=("text", "json", "markdown", "list"), default="text",
        help="'list' prints just the non-pure-Python packages, one per line",
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


def _scan_source_tree(
    args: argparse.Namespace, stderr: Console
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
        return inspect_repository(path, scan_ci=scan_ci, use_meson_introspect=use_meson)
    with stderr.status(f"scanning {path}…"):
        return inspect_repository(path, scan_ci=scan_ci, use_meson_introspect=use_meson)


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

    scan = _scan_source_tree(args, stderr)
    roots = _roots(args)
    deep_manifests: list[Path] = []
    if scan is not None and not roots and scan.manifests:
        # Manifests exist, but not at the root. A deep one is usually for
        # something else -- documentation, bindings, a test harness -- so it
        # is offered rather than silently adopted.
        deep_manifests = [m for m in scan.manifests if m.parent != scan.root]
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


def _emit(args: argparse.Namespace, analysis: Analysis, distro: Optional[DistroIndex]) -> None:
    if args.format == "text" and not args.output:
        render_text(analysis, Console(), distro, verbose=args.verbose)
        return

    if args.format == "json":
        payload = render_json(analysis)
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
