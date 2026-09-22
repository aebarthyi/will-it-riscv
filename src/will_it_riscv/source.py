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

from .models import BuildProfile, SystemRequirement
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
    manifests: list[Path] = field(default_factory=list)
    """Dependency manifests found, for the analyzer to pick up."""
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

    _apply_implied_tools(inspection.profile, record)
    inspection.profile.system_requirements = sorted(
        found.values(), key=lambda r: (r.kind, r.name)
    )
    inspection.manifests = discover_manifests(root)
    inspection.warnings.extend(check_submodules(root))
    return inspection
