"""Write a plan for a fetched dependency, from what its source tree says.

This is where a model will go. Until one does, a fetched package gets the
plan its build files imply, every step citing the line it came from exactly
as a hand-written plan must:

  [build-system].requires   a python-install of what building it needs
  CMakeLists.txt at the top a cmake-configure, run in the pretend environment
  meson.build, Bazel,       a system-packages step for the tool itself -- the
  Cargo, SCons              part that can be checked -- and a note that its
                            configure was not run, because the pretend
                            environment does not configure that system yet

A plan written this way is honest about its reach. A package whose build it
could configure is shown to be buildable or not; one it could only read
stays "probably", and says why.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from packaging.utils import canonicalize_name

from .plan import (
    CMAKE_CONFIGURE,
    PYTHON_INSTALL,
    SYSTEM_PACKAGES,
    Evidence,
    Plan,
    Step,
)

try:  # pragma: no cover - trivial
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

#: Build systems the pretend environment cannot configure yet, the distro
#: packages their tools come in, and the files that give them away.
UNCONFIGURED = {
    "meson": (["meson", "ninja-build"], ["meson.build"]),
    "bazel": (
        ["bazel-bootstrap"], ["MODULE.bazel", "WORKSPACE", "WORKSPACE.bazel", ".bazelversion"]
    ),
    "cargo": (["rustc", "cargo"], ["Cargo.toml"]),
    "scons": (["scons"], ["SConstruct"]),
}

#: Build backends that mean a build system, whatever files are at the top.
BACKENDS = {
    "mesonpy": "meson",
    "maturin": "cargo",
    "setuptools_rust": "cargo",
    "scikit_build_core.build": "cmake",
}


def auto_plan(tree: Path, name: str) -> Plan:
    """The plan a fetched package's own build files imply."""
    tree = Path(tree)
    steps: list[Step] = []
    unsure: list[str] = []
    project_dir = _project_dir(tree, name)
    manifest = project_dir / "pyproject.toml" if project_dir else None
    backend: Optional[str] = None

    if manifest is not None and manifest.is_file():
        text = manifest.read_text(encoding="utf-8", errors="replace")
        relative = manifest.relative_to(tree).as_posix()
        try:
            data = tomllib.loads(text)
        except ValueError:
            data = {}
        build_system = data.get("build-system", {})
        backend = build_system.get("build-backend")
        if build_system.get("requires"):
            steps.append(Step(
                id="build-requires",
                kind=PYTHON_INSTALL,
                manifest=relative,
                section="build-system",
                note="what building it needs, installed the way pip's build isolation would",
                evidence=_cite(text, relative, r"^\s*requires\s*="),
            ))

    systems = _systems(tree, backend)
    after = [s.id for s in steps]
    build_requires = _build_requires(manifest)
    if "cmake" in systems and (tree / "CMakeLists.txt").is_file():
        cmake_text = (tree / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
        defines: dict[str, str] = {}
        if backend == "scikit_build_core.build":
            # What scikit-build-core tells the configure it is running under.
            # Some projects refuse to configure without it.
            defines = {"SKBUILD": "2", "SKBUILD_PROJECT_NAME": str(canonicalize_name(name))}
        steps.append(Step(
            id="configure",
            kind=CMAKE_CONFIGURE,
            after=after,
            defines=defines,
            note="its own CMake build, configured in the pretend environment",
            evidence=_cite(cmake_text, "CMakeLists.txt", r"^\s*project\s*\(", default_line=1),
        ))
    for system in sorted(systems - {"cmake"}):
        packages, markers = UNCONFIGURED.get(system, ([], []))
        unsure.append(
            f"builds with {system}, which the pretend environment does not configure yet: "
            "what that build asks the system for was read, not shown"
        )
        if system == "meson" and build_requires & {"meson-python", "meson"}:
            continue   # meson and ninja come from pip, as build requirements
        packages = _pinned(system, tree, list(packages))
        if not packages:
            continue
        marker = next((m for m in markers if (tree / m).is_file()), None)
        evidence = [Evidence(marker, 1, 1)] if marker else []
        if marker is None and manifest is not None and backend:
            text = manifest.read_text(encoding="utf-8", errors="replace")
            evidence = _cite(text, manifest.relative_to(tree).as_posix(), r"build-backend")
        steps.append(Step(
            id=f"{system}-tools",
            kind=SYSTEM_PACKAGES,
            packages=packages,
            note=f"it builds with {system}; the tool itself is the part that can be checked",
            evidence=evidence,
        ))
    if not systems and _compiles(tree) and (
        (tree / "setup.py").is_file() or (backend or "").startswith("setuptools")
    ):
        unsure.append(
            "builds with setuptools, which compiles its extensions itself: what they "
            "link against was read, not shown"
        )
    if not steps and not unsure:
        unsure.append("nothing in its source tree says how it builds")
    return Plan(repo=str(canonicalize_name(name)), steps=steps, unsure=unsure)


def _build_requires(manifest: Optional[Path]) -> set:
    if manifest is None or not manifest.is_file():
        return set()
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return set()
    names = set()
    for spec in data.get("build-system", {}).get("requires", []):
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", str(spec))
        if match:
            names.add(str(canonicalize_name(match.group(1))))
    return names


def _pinned(system: str, tree: Path, packages: list[str]) -> list[str]:
    """The tool, at the version the build pins, where it pins one.

    jax's .bazelversion says 8.7.0; Debian 13's bazel-bootstrap is 4.2.3. A
    crate's rust-version is the oldest rustc that builds it.
    """
    if system == "bazel":
        pinned = _first_line(tree / ".bazelversion")
        if pinned and re.match(r"^\d+(\.\d+)*$", pinned):
            return [f"bazel-bootstrap>={pinned}"]
    if system == "cargo":
        cargo = tree / "Cargo.toml"
        try:
            data = tomllib.loads(cargo.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            data = {}
        rust = (
            data.get("package", {}).get("rust-version")
            or data.get("workspace", {}).get("package", {}).get("rust-version")
        )
        if isinstance(rust, str) and re.match(r"^\d+(\.\d+)*$", rust):
            return [f"rustc>={rust}", "cargo"]
    return packages


def _first_line(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return None


def _compiles(tree: Path) -> bool:
    """Any C, C++ or Cython source in the top two levels of the tree."""
    suffixes = {".c", ".cc", ".cpp", ".cxx", ".pyx"}
    for depth, pattern in enumerate(("*", "*/*", "*/*/*")):
        if any(p.suffix in suffixes for p in tree.glob(pattern)):
            return True
        if depth >= 2:
            break
    return False


def configured(plan: Plan) -> bool:
    """Whether any step of the plan actually runs the package's build."""
    return any(s.kind == CMAKE_CONFIGURE for s in plan.steps) and not any(
        "does not configure" in u for u in plan.unsure
    )


def _project_dir(tree: Path, name: str) -> Optional[Path]:
    """Where this package's own pyproject.toml is.

    An sdist is one project. A repository can be several -- the jax
    repository builds jaxlib from jaxlib/ -- so prefer the one that names it.
    """
    wanted = canonicalize_name(name)
    for candidate in (tree, tree / str(wanted).replace("-", "_"), tree / str(wanted)):
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            project = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        declared = project.get("project", {}).get("name")
        if declared is None or canonicalize_name(declared) == wanted:
            return candidate
    return None


def _systems(tree: Path, backend: Optional[str]) -> set:
    found: set = set()
    if (tree / "CMakeLists.txt").is_file():
        found.add("cmake")
    for system, (_, markers) in UNCONFIGURED.items():
        if any((tree / m).is_file() for m in markers):
            found.add(system)
    if backend in BACKENDS:
        found.add(BACKENDS[backend])
    if backend == "scikit_build_core.build":
        found.add("cmake")
    return found


def _cite(text: str, path: str, pattern: str, default_line: int = 0) -> list[Evidence]:
    for number, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            return [Evidence(path, number, number)]
    return [Evidence(path, default_line, default_line)] if default_line else []
