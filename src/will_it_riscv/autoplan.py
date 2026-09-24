"""Write a plan for a fetched dependency, from what its source tree says.

This is where a model will go. Until one does, a fetched package gets the
plan its build files imply, every step citing the line it came from exactly
as a hand-written plan must:

  [build-system].requires   a python-install of what building it needs
  CMakeLists.txt at the top a cmake-configure, run in the pretend environment
  meson.build at the top    a meson-setup, likewise, with the options and the
                            Meson that [tool.meson-python] names
  Bazel, Cargo, SCons       a system-packages step for the tool itself -- the
                            part that can be checked -- and a note that its
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
    CONFIGURE_KINDS,
    MESON_SETUP,
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


def auto_plan(
    tree: Path, name: str, arch: str = "riscv64", cache_root: Optional[Path] = None
) -> Plan:
    """The plan a fetched package's own build files imply.

    ``cache_root`` is where a Bazel build's rules_python is read from, to
    see whether its hermetic Python exists for ``arch``.
    """
    tree = Path(tree)
    steps: list[Step] = []
    unsure: list[str] = []
    read: list[str] = []
    blocked: list[str] = []
    project_dir = _project_dir(tree, name)
    manifest = project_dir / "pyproject.toml" if project_dir else None
    backend: Optional[str] = None
    data: dict = {}

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
    # A tree can carry both; the backend says which one the package builds with.
    if backend == "mesonpy":
        systems.discard("cmake")
    elif backend == "scikit_build_core.build":
        systems.discard("meson")
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
    if "meson" in systems and (tree / "meson.build").is_file():
        meson_text = (tree / "meson.build").read_text(encoding="utf-8", errors="replace")
        options, vendored = _meson_python(data)
        steps.append(Step(
            id="setup",
            kind=MESON_SETUP,
            after=after,
            defines=options,
            meson=vendored,
            note="its own Meson build, set up in the pretend environment",
            evidence=_cite(meson_text, "meson.build", r"^\s*project\s*\(", default_line=1),
        ))
    for system in sorted(systems - {"cmake"}):
        if system == "meson" and (tree / "meson.build").is_file():
            if not build_requires & {"meson-python", "meson"}:
                steps.append(_tools_step(system, tree, manifest, backend))
            continue
        unsure.append(
            f"builds with {system}, which the pretend environment does not configure yet: "
            "what that build asks the system for was read, not shown"
        )
        tools = _tools_step(system, tree, manifest, backend)
        if tools.packages:
            steps.append(tools)
        if system == "bazel":
            steps += _bazel_steps(tree, name, arch, cache_root, read, blocked)
    if not systems and _compiles(tree) and (
        (tree / "setup.py").is_file() or (backend or "").startswith("setuptools")
    ):
        unsure.append(
            "builds with setuptools, which compiles its extensions itself: what they "
            "link against was read, not shown"
        )
    if not steps and not unsure:
        unsure.append("nothing in its source tree says how it builds")
    return Plan(
        repo=str(canonicalize_name(name)), steps=steps, unsure=unsure, read=read,
        blocked=blocked,
    )


def _tools_step(
    system: str, tree: Path, manifest: Optional[Path], backend: Optional[str]
) -> Step:
    """The build system's own tool, from the distro, at any version it pins."""
    packages, markers = UNCONFIGURED.get(system, ([], []))
    packages = _pinned(system, tree, list(packages))
    marker = next((m for m in markers if (tree / m).is_file()), None)
    evidence = [Evidence(marker, 1, 1)] if marker else []
    if marker is None and manifest is not None and backend:
        text = manifest.read_text(encoding="utf-8", errors="replace")
        evidence = _cite(text, manifest.relative_to(tree).as_posix(), r"build-backend")
    return Step(
        id=f"{system}-tools",
        kind=SYSTEM_PACKAGES,
        packages=packages,
        note=f"it builds with {system}; the tool itself is the part that can be checked",
        evidence=evidence,
    )


def _bazel_steps(
    tree: Path, name: str, arch: str, cache_root: Optional[Path],
    read: list[str], blocked: list[str],
) -> list[Step]:
    """What reading its MODULE.bazel adds: wheels to build first, a compiler."""
    from . import bazel

    reading = bazel.read(tree, name, arch, cache_root)
    if reading is None:
        return []
    read += reading.read
    blocked += reading.blocked
    steps: list[Step] = []
    if reading.wheels:
        steps.append(Step(
            id="bazel-wheels",
            kind=PYTHON_INSTALL,
            packages=reading.wheels,
            note=f"the wheels its Bazel build takes from dist/, which {arch} has to build first",
            evidence=reading.wheel_evidence,
        ))
    if reading.compiler:
        steps.append(Step(
            id="bazel-cc",
            kind=SYSTEM_PACKAGES,
            packages=[reading.compiler],
            note="the machine's own C++ compiler, in place of a hermetic toolchain",
            evidence=reading.compiler_evidence,
        ))
    return steps


def _meson_python(data: dict) -> tuple[dict, Optional[str]]:
    """The -D options and the Meson a meson-python build is set up with.

    ``[tool.meson-python.args] setup`` is what it adds to ``meson setup``;
    ``[tool.meson-python] meson`` is a Meson the project ships instead of
    the one pip installs -- numpy's vendored fork.
    """
    config = data.get("tool", {}).get("meson-python", {})
    if not isinstance(config, dict):
        return {}, None
    options: dict[str, str] = {}
    setup = config.get("args", {}).get("setup", []) if isinstance(config.get("args"), dict) else []
    for arg in setup if isinstance(setup, list) else []:
        match = re.match(r"^-D([A-Za-z0-9_.:-]+)=(.*)$", str(arg))
        if match:
            options[match.group(1)] = match.group(2)
    meson = config.get("meson")
    return options, meson if isinstance(meson, str) and meson else None


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
    return any(s.kind in CONFIGURE_KINDS for s in plan.steps) and not any(
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
