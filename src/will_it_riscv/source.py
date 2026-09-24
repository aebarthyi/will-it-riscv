"""Scanning a source tree instead of a source distribution.

The question changes shape when you point this at a repository you are about
to build. The project itself is not in any registry -- you are building it,
that is the point -- so there is no wheel to look for. What you want is:

* what this project needs from the system in order to compile, and
* for whatever it declares as dependencies, the usual per-package question.

The first half is the scrapers in :mod:`will_it_riscv.sdist`, pointed at a
directory instead of a tarball. The second half is the existing analyzer.
"""

from __future__ import annotations

import configparser
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .meson_introspect import MesonDependency, MesonScan, scan_dependencies
from .models import BuildProfile, SystemRequirement
from .pseudobuild import PseudoBuild
from .pseudobuild import run as run_pseudobuild
from .pseudomeson import run as run_pseudomeson
from .scripts import ScriptInstall, find_script_installs
from .sdist import (
    ScanPolicy,
    SdistInspection,
    _apply_implied_tools,
    make_recorder,
    scan_members,
)

Member = tuple[str, Callable[[], bytes]]

#: Directories that hold build output, tooling state or checked-out
#: dependencies rather than this project's source.
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".jj",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".nox", ".venv", "venv", ".env",
    "build", "_build", "builddir", "dist", "target", "out",
    ".eggs", ".cache", ".idea", ".vscode", ".gradle", ".terraform",
    "cmake-build-debug", "cmake-build-release",
})

#: Files worth reading. Everything else is skipped without opening it, which
#: is what keeps a 9,000-file repository fast.
INTERESTING_SUFFIXES = (
    ".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hxx",
    ".pyx", ".pxd", ".pxi", ".rs", ".f", ".f77", ".f90", ".f95", ".f03",
    ".for", ".go", ".zig", ".cmake", ".toml", ".cfg", ".ac", ".am", ".build",
    ".mk",
)
INTERESTING_NAMES = frozenset({
    "cmakelists.txt", "meson.build", "meson_options.txt", "meson.options",
    "configure.ac", "configure.in", "makefile.am", "cargo.toml", "build.rs",
    "sconstruct", "sconscript", "setup.py", "setup.cfg", "pyproject.toml",
    "pkg-info", "makefile", "gnumakefile", "configure",
})

MAX_FILES = 60000
MAX_FILE_BYTES = 512 * 1024


@dataclass
class RepositoryInspection:
    """What a source tree says about how it builds."""

    root: Path
    profile: BuildProfile = field(default_factory=BuildProfile)
    pseudobuild: Optional[PseudoBuild] = None
    """Result of configuring the project for real, when asked for."""
    meson_introspect: Optional[str] = None
    """What Meson's own dependency scan did: None if not attempted, "ok",
    or the reason it could not be used."""
    manifests: list[Path] = field(default_factory=list)
    """Dependency manifests found, for the analyzer to pick up."""
    script_installs: list[ScriptInstall] = field(default_factory=list)
    """What the project's own root scripts pip-install, and how each was
    reached. MFC's toolchain/pyproject.toml is found this way."""
    bundled: set[str] = field(default_factory=set)
    """Libraries the project ships a copy of, so the system package is
    optional. GROMACS bundles sixteen of them."""
    files_scanned: int = 0
    files_seen: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.root.resolve().name


def _is_interesting(name: str) -> bool:
    lower = name.lower()
    return lower in INTERESTING_NAMES or lower.endswith(INTERESTING_SUFFIXES)


def iter_directory(
    root: Path, max_files: int = MAX_FILES
) -> Iterator[tuple[str, Callable[[], bytes]]]:
    """Walk ``root``, yielding ``(relative_posix_path, read)`` pairs.

    Deliberately the same shape :func:`will_it_riscv.sdist._iter_archive`
    yields, so the scan core does not know or care which it is looking at.
    """
    root = Path(root)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d.lower() not in IGNORED_DIRS]
        for filename in sorted(filenames):
            if not _is_interesting(filename):
                continue
            absolute = Path(dirpath) / filename
            try:
                relative = absolute.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - defensive
                continue
            if absolute.is_symlink():
                continue
            count += 1
            if count > max_files:
                return
            yield relative, _reader(absolute)


def _reader(path: Path) -> Callable[[], bytes]:
    def read() -> bytes:
        try:
            with path.open("rb") as fh:
                return fh.read(MAX_FILE_BYTES)
        except OSError:
            return b""

    return read


#: Dependency manifests the analyzer knows how to resolve.
MANIFEST_NAMES = ("pyproject.toml", "requirements.txt", "requirements.in")


def discover_manifests(root: Path, max_depth: int = 3) -> list[Path]:
    """Find Python dependency manifests, nearest the root first."""
    root = Path(root)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        relative = Path(dirpath).relative_to(root)
        depth = 0 if relative == Path(".") else len(relative.parts)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d.lower() not in IGNORED_DIRS and d.lower() not in {"tests", "test"}
        ]
        for name in MANIFEST_NAMES:
            if name in filenames:
                found.append(Path(dirpath) / name)
    return sorted(found, key=lambda p: (len(p.relative_to(root).parts), str(p)))


def find_bundled(root: Path) -> set[str]:
    """Directory names under external/, third_party/ and friends.

    A project that ships a copy of a library can usually build without the
    system one, so these are reported as optional rather than required.
    """
    from .sdist import VENDORED_DIRS, _normalize_dir

    bundled: set[str] = set()
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORED_DIRS]
        if _normalize_dir(Path(dirpath).name) in VENDORED_DIRS:
            bundled.update(d.lower() for d in dirnames)
            dirnames[:] = []
    return bundled


_CMAKE_PROJECT = None


def own_names(root: Path) -> set[str]:
    """Names that refer to this project itself, not to a system library."""
    import re

    names = {Path(root).resolve().name.lower()}
    cmakelists = Path(root) / "CMakeLists.txt"
    if cmakelists.exists():
        try:
            text = cmakelists.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return names
        match = re.search(r"\bproject\s*\(\s*([A-Za-z0-9_+.-]+)", text, re.IGNORECASE)
        if match:
            names.add(match.group(1).lower())
    # Sub-projects in the tree carry the parent's name too (gmxapi, libfoo-py).
    for child in Path(root).iterdir():
        if not child.is_dir() or child.name.lower() in IGNORED_DIRS:
            continue
        if (child / "CMakeLists.txt").exists():
            names.add(child.name.lower())
        for grandchild in child.iterdir():
            if grandchild.is_dir() and (grandchild / "CMakeLists.txt").exists():
                names.add(grandchild.name.lower())
    return names


def check_submodules(root: Path) -> list[str]:
    """Warn about submodules that were never checked out.

    An uninitialised submodule is an empty directory. It scans perfectly
    cleanly and reports nothing, which is the most dangerous way for this
    tool to be wrong -- so it is called out loudly rather than inferred.
    """
    gitmodules = Path(root) / ".gitmodules"
    if not gitmodules.exists():
        return []
    parser = configparser.ConfigParser()
    try:
        parser.read_string(gitmodules.read_text(encoding="utf-8", errors="replace"))
    except (configparser.Error, OSError) as exc:
        return [f".gitmodules could not be parsed ({exc}); submodules were not checked"]

    warnings = []
    for section in parser.sections():
        path = parser[section].get("path")
        if not path:
            continue
        target = Path(root) / path
        if not target.exists() or not any(target.iterdir()):
            warnings.append(
                f"submodule {path!r} is not checked out -- anything it needs is "
                "missing from this report (run: git submodule update --init --recursive)"
            )
    return warnings


def _apply_meson_verdict(scan: MesonScan, found: dict, record) -> None:
    """Let Meson's own answer settle optionality, overriding the scrapers.

    "Required anywhere wins" is the right rule between two inferences. It is
    the wrong rule when one side is the language's own parser and the other
    is a regex over a configure.ac that knows nothing about AC_ARG_WITH. So
    this runs last and overrides, for the names Meson actually reported.

    Meson may mention a dependency more than once with different answers --
    PostgreSQL asks for ICU twice -- and there, required wins.
    """
    from dataclasses import replace

    from .syslibs import database

    db = database()
    verdicts: dict[str, MesonDependency] = {}
    for dependency in scan.dependencies:
        known = db.lookup(dependency.name, "library")
        if known is None:
            continue
        existing = verdicts.get(known.name)
        if existing is None or (existing.optional and not dependency.optional):
            verdicts[known.name] = dependency

    for canonical, dependency in verdicts.items():
        current = found.get(canonical)
        if current is None:
            # The regex reading of dependency() stood aside for Meson, so
            # for a pure-Meson project this is the only thing that knows
            # about it at all.
            record(
                dependency.name,
                "library",
                "meson introspect",
                optional=dependency.optional,
                gate=dependency.gate,
            )
            continue
        found[canonical] = replace(
            current,
            optional=dependency.optional,
            gate=dependency.gate if dependency.optional else None,
            found_in=tuple(dict.fromkeys(current.found_in + ("meson introspect",))),
        )


def _meson_python_config(root: Path) -> tuple[dict, Optional[str], Optional[dict]]:
    """Its meson-python options and Meson, and its build requirements.

    Without a plan nothing has resolved a version, so a build requirement
    the setup needs -- Cython -- is installed for the host at the latest.
    """
    from .autoplan import _build_requires, _meson_python

    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(errors="replace"))
    except (OSError, ValueError):
        return {}, None, None
    options, meson = _meson_python(data)
    requires = _build_requires(root / "pyproject.toml")
    return options, meson, (dict.fromkeys(requires) if requires else None)


def _apply_pseudobuild(
    result: PseudoBuild, found: dict, record, inspection: RepositoryInspection
) -> None:
    """Fold in what running the configure proved.

    Only two of the three outcomes prove anything about necessity:

    * a package the configure could not find and carried on without, all the
      way to the end, is optional, and that is a demonstration rather than an
      inference;
    * a package whose absence stopped the configure is required -- every one
      of them, in every round, including those shown by experiment.

    A package that was *found* proves only that it exists on this machine --
    it says nothing about whether the build would have managed without it --
    so the static verdict is left alone there.

    Confined to an empty sysroot, "could not find" covers far more than the
    misses FPHSA narrates: a quiet pkg-config probe or a bare find_library
    that came back empty was just as absent, and the configure still finished.
    """
    from dataclasses import replace

    from .syslibs import database

    if result.error and not result.probes:
        inspection.profile.notes.append(f"pseudobuild did not run: {result.error}")
        return

    db = database()

    def canonical(name: str) -> Optional[str]:
        known = db.lookup(name, "library")
        return known.name if known is not None else None

    # A miss the configure carried on past is optional only if it then ran
    # to the end. One that stopped later may have stopped *because* of it:
    # AdaptiveCpp misses LLVM, carries on, and dies wanting clang's headers.
    shrugged_off = result.soft_misses if result.completed else set()
    for name in sorted(shrugged_off):
        key = canonical(name)
        current = found.get(key) if key else None
        if current is None:
            continue
        found[key] = replace(
            current,
            optional=True,
            gate=f"configure ran on without it ({name} not found)",
        )

    if result.completed and result.confined:
        present = {canonical(n) for n in result.found} | {
            canonical(n) for n in result.blockers
        }
        for probe in result.probes.values():
            key = canonical(probe.name)
            current = found.get(key) if key else None
            if current is None or key in present or current.optional:
                continue
            found[key] = replace(
                current,
                optional=True,
                gate=f"configure ran on without it ({probe.name} was never there)",
            )

    for blocker in result.blockers or ([result.blocking] if result.blocking else []):
        key = canonical(blocker)
        current = found.get(key) if key else None
        if current is not None:
            found[key] = replace(current, optional=False, gate=None)
        else:
            record(blocker, "library", "pseudobuild: configure stopped here")

    # A probe is a question, not a requirement -- adopting every one of them
    # would inflate the list with things like ssleay32 that the configure
    # merely wondered about. Only REQUIRED says something.
    for probe in result.probes.values():
        if not probe.required:
            continue
        key = canonical(probe.name)
        if key and key not in found:
            record(probe.name, "library", "pseudobuild: required by the configure")

    if result.completed:
        _apply_unreached(result, found)


#: Evidence that the CMake trace is entitled to overrule. A requirement whose
#: only sighting was in a CMake file is one the trace has full view of; one
#: also seen in a Makefile or a CI config is not.
def _only_seen_in_cmake(requirement) -> bool:
    if not requirement.found_in:
        return False
    for origin in requirement.found_in:
        name = origin.rsplit("/", 1)[-1].lower()
        if name != "cmakelists.txt" and not name.endswith(".cmake"):
            return False
    return True


def _apply_unreached(result: PseudoBuild, found: dict) -> None:
    """After a configure that finished, silence means something.

    The configure ran end to end with every pkg-config query denied, so
    anything it never even asked about is not part of a default build. That
    only applies to dependencies the trace could have seen -- ones sighted
    solely in CMake files.
    """
    from dataclasses import replace

    from .syslibs import database

    db = database()
    probed = set()
    for probe in result.probes.values():
        known = db.lookup(probe.name, "library")
        probed.add((known.name if known else probe.name).lower())
    probed |= {n.lower() for n in result.found | result.soft_misses}
    # A blocker was asked for, whether or not the trace saw the question: a
    # header it could not find or an experiment's suspect counts as much.
    for blocker in result.blockers:
        known = db.lookup(blocker, "library")
        probed.add((known.name if known else blocker).lower())

    for name, requirement in list(found.items()):
        if requirement.optional or requirement.kind != "library":
            continue
        if name.lower() in probed or not _only_seen_in_cmake(requirement):
            continue
        found[name] = replace(
            requirement,
            optional=True,
            gate="a completed configure never asked for it",
        )


def _cmake_symbols(members: list) -> object:
    """Learn every CMake option default before judging any condition.

    Options are routinely declared in one file and tested in another, so a
    single pass would call half of them unknown.
    """
    from .cmake_conditions import collect_symbols
    from .sdist import _is_skipped, _is_vendored

    def texts():
        for path, read in members:
            name = path.rsplit("/", 1)[-1].lower()
            if name != "cmakelists.txt" and not name.endswith(".cmake"):
                continue
            if _is_skipped(path) or _is_vendored(path):
                continue
            yield read().decode("utf-8", errors="replace")

    return collect_symbols(texts())


def inspect_repository(
    root: Path,
    policy: Optional[ScanPolicy] = None,
    scan_ci: bool = True,
    use_meson_introspect: bool = True,
    pseudobuild: bool = False,
    pseudobuild_timeout: int = 600,
    pseudobuild_arch: str = "riscv64",
) -> RepositoryInspection:
    """Read a source tree and work out what building it needs from the system."""
    root = Path(root)
    inspection = RepositoryInspection(root=root)
    inspection.profile.inspected = True
    policy = policy or ScanPolicy.for_repository()

    inspection.bundled = find_bundled(root)
    own = own_names(root)
    found: dict[str, SystemRequirement] = {}
    record = make_recorder(found, exclude=own)

    # Ask Meson before scraping, so the scrapers know to stand aside.
    meson_scan = scan_dependencies(root) if use_meson_introspect else None
    if meson_scan is not None:
        if meson_scan.ok:
            inspection.meson_introspect = "ok"
            policy.meson_dependencies_handled = True
        else:
            inspection.meson_introspect = meson_scan.error
            inspection.profile.notes.append(
                f"meson introspect unavailable ({meson_scan.error}); "
                "read the meson.build files directly instead"
            )

    result = SdistInspection(profile=inspection.profile)
    members = list(iter_directory(root))
    inspection.files_scanned = len(members)
    if policy.cmake_symbols is None:
        policy.cmake_symbols = _cmake_symbols(members)
    scan_members(members, result, record, policy)

    if scan_ci:
        from .ci import scan_ci_configuration

        ci = scan_ci_configuration(root, record)
        inspection.warnings.extend(ci.warnings)
        inspection.profile.notes.extend(ci.notes)

    if meson_scan is not None and meson_scan.ok:
        _apply_meson_verdict(meson_scan, found, record)

    if pseudobuild:
        inspection.pseudobuild = run_pseudobuild(
            root, timeout=pseudobuild_timeout, arch=pseudobuild_arch
        )
        if inspection.pseudobuild is None:
            # No CMakeLists.txt at the top: a Meson project is set up instead,
            # with the options and the Meson its [tool.meson-python] names.
            options, meson, requires = _meson_python_config(root)
            inspection.pseudobuild = run_pseudomeson(
                root, timeout=pseudobuild_timeout, arch=pseudobuild_arch,
                options=options, meson=meson, python_dists=requires,
            )
        if inspection.pseudobuild is not None:
            _apply_pseudobuild(inspection.pseudobuild, found, record, inspection)

    _apply_implied_tools(inspection.profile, record)
    inspection.profile.system_requirements = sorted(
        found.values(), key=lambda r: (r.kind, r.name)
    )
    inspection.manifests = discover_manifests(root)
    inspection.script_installs = find_script_installs(root)
    inspection.warnings.extend(check_submodules(root))
    return inspection
