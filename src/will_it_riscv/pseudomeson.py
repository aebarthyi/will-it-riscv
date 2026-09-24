"""Meson, configured the way the CMake loop configures CMake.

numpy, scipy, matplotlib, pandas: most of the scientific Python stack
builds with Meson, through meson-python. So this runs ``meson setup`` the
way :mod:`pseudobuild` runs a CMake configure -- as a cross build for
linux/riscv64, confined to an empty sysroot, in scratch, never compiled and
never emulated -- and whenever it stops, gives it what it stopped for and
runs it again.

  cross file    host machine linux/riscv64, with this host's compilers.
                Meson runs its checks with them, and needs_exe_wrapper is
                false: what they build runs here, as the CMake loop's
                pass-through emulator lets it. No riscv64 code is run.
  pkg-config    confined to the sysroot's lib/pkgconfig, which is empty
                until a stub .pc is written there
  Python        the build host's interpreter (:mod:`hostpython`), which
                Meson's python module introspects
  build tools   what the build requirements provide -- Cython, pythran,
                meson itself, ninja -- installed for the host from wheels,
                at the versions the plan resolved

What Meson says when it stops, and what it is given:

  Dependency "openblas" not found            openblas.pc and an empty libopenblas.a
  C shared or static library 'm' not found   an empty libm.a
  Program 'pythran' not found                pythran from the plan, else a stub
  Unknown compiler(s): [['cython']]          cython from the plan
  C header 'foo.h' not found                 foo.h, carrying version macros
  Problem encountered: ...                   blame by experiment, as for CMake

A link check still runs against this host's own SDK: on a Mac, Accelerate
answers "blas found: YES". That is reported the way a CMake compile-only
check is: claimed, and unverifiable for the target.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .drive import clone_tree, dist_for_module
from .hostpython import HostPython
from .pseudobuild import (
    FAKE_VERSION_STRING,
    Probe,
    PseudoBuild,
    _fake_header,
    _mentions,
    _signature,
)

TIMEOUT_SECONDS = 600
MAX_SUSPECTS = 3

#: Programs whose distribution is not their name.
PROGRAM_TO_DIST = {
    "cython3": "cython",
    "f2py": "numpy",
    "pybind11-config": "pybind11",
}

#: Languages Meson looks for a compiler for, and the host programs that are one.
_COMPILERS = {
    "c": ("cc", "gcc", "clang"),
    "cpp": ("c++", "g++", "clang++"),
    "fortran": ("gfortran", "flang-new", "flang"),
    "objc": ("cc", "clang"),
    "rust": ("rustc",),
}

_DEPENDENCY = re.compile(
    r"^(?:Run-time |Build-time )?[Dd]ependency (\S+?)"
    r"(?: from subproject (\S+))? found: (YES|NO)\b(.*)$"
)
_LIBRARY = re.compile(r"^Library (\S+) found: (YES|NO)\b")
_PROGRAM = re.compile(r"^Program (\S+) found: (YES|NO)\b(.*)$")
_ERROR = re.compile(r"^(?:(\S+?):(\d+):\d+: )?ERROR: (.*)$")
_SUBPROJECT = re.compile(r"^Executing subproject (\S+)")

_DEP_NOT_FOUND = re.compile(r"""Dependency ["']([^"']+)["'] (?:not found|is required)""")
_DEP_LOOKUP = re.compile(r"Dependency lookup for (\S+) with method")
_DEP_VERSION = re.compile(r"Invalid version of dependency, need '([^']+)'")
_LIB_NOT_FOUND = re.compile(r"library '([^']+)' not found")
_PROG_NOT_FOUND = re.compile(r"Program '([^']+)' not found")
_PROG_VERSION = re.compile(r"Invalid version of program, need '([^']+)'")
_UNKNOWN_COMPILER = re.compile(r"Unknown compiler\(s\): \[\['([^']+)'")
_HEADER_NOT_FOUND = re.compile(r"header '([^']+)' not found")
_MESON_VERSION = re.compile(r"Meson version is (\S+) but project requires (.+?)\.?$")
_NO_MODULE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")

_CALL = re.compile(r"\b(dependency|find_library|find_program)\(\s*'([^']+)'")
_REQUIRED_KWARG = re.compile(r"\brequired\s*:\s*([^,)\s]+)")


def available() -> bool:
    """Whether a meson can be had: this host's, or one installed from a wheel."""
    return shutil.which("meson") is not None or _pip_works()


def _pip_works() -> bool:
    try:
        import pip  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class _Round:
    completed: bool = False
    error: Optional[str] = None
    where: Optional[str] = None
    narration: str = ""
    found: dict = field(default_factory=dict)
    """Name to where it was found, for dependencies, libraries and programs."""
    missing: list = field(default_factory=list)
    """Dependencies and libraries looked for and not found, in order."""
    libraries: set = field(default_factory=set)
    programs: dict = field(default_factory=dict)
    """Programs found, to where: kept apart, since a program and a
    dependency can share a name -- python is both."""
    subprojects: dict = field(default_factory=dict)
    """Dependency to the subproject that provided it."""
    imports: list = field(default_factory=list)
    """Modules a script it ran could not import, from Meson's log."""


def run(
    root: Path,
    timeout: int = TIMEOUT_SECONDS,
    max_rounds: int = 12,
    arch: str = "riscv64",
    options: Optional[dict] = None,
    python_dists: Optional[dict] = None,
    pip_cache: Optional[Path] = None,
    meson: Optional[str] = None,
) -> Optional[PseudoBuild]:
    """Set the project up in scratch, unblocking as it goes.

    ``options`` are the ``-D`` options the build passes and are never
    stubbed over. ``meson`` is a Meson the project ships, relative to the
    root: numpy builds with its own vendored fork. Returns None when there
    is no meson.build at the root.
    """
    root = Path(root)
    if not (root / "meson.build").exists():
        return None

    started = time.monotonic()
    aggregate = PseudoBuild(rounds=0, platform=f"linux/{arch}")
    fixed = {str(k): str(v) for k, v in (options or {}).items()}
    written: set = set()
    trial: Optional[_Trial] = None
    tried: set = set()

    with tempfile.TemporaryDirectory(prefix="will-it-riscv-meson-") as scratch:
        scratch_path = Path(scratch)
        # Meson writes into the source tree -- a wrap it downloads lands in
        # subprojects/ -- so it gets a copy-on-write clone to write on.
        tree = scratch_path / "src"
        clone_tree(root, tree)
        sysroot = scratch_path / "sysroot"
        stubs = scratch_path / "bin"
        for directory in (sysroot / "include", sysroot / "lib" / "pkgconfig", stubs):
            directory.mkdir(parents=True, exist_ok=True)
        interpreter = HostPython.create(scratch_path / "python", python_dists, pip_cache)
        calls = _calls(tree)

        command, note = _meson_command(tree, meson, interpreter)
        if command is None:
            aggregate.error = note
            return aggregate
        if note:
            aggregate.notes.append(note)
        if shutil.which("ninja") is None and not _tool(interpreter, "ninja"):
            aggregate.error = "Meson needs ninja, and neither this host nor a wheel has one"
            return aggregate

        cross = _cross_file(scratch_path, sysroot, arch, interpreter)
        native = _native_file(scratch_path, interpreter)
        env = dict(os.environ)
        env.update({
            "PATH": os.pathsep.join([str(stubs), str(interpreter.scripts), env.get("PATH", "")]),
            "PYTHONPATH": interpreter.path,
            "PKG_CONFIG_LIBDIR": str(sysroot / "lib" / "pkgconfig"),
            "PKG_CONFIG_PATH": str(sysroot / "lib" / "pkgconfig"),
        })

        for attempt in range(1, max_rounds + 1):
            left = timeout - (time.monotonic() - started)
            if left <= 0:
                aggregate.error = f"ran out of time after {attempt - 1} rounds"
                break
            outcome = _setup(
                command, tree, scratch_path / f"build-{attempt}", cross, native,
                fixed, env, max(1, int(left)),
            )
            aggregate.rounds = attempt
            _absorb(aggregate, outcome, calls, attempt)

            if trial is not None:
                if outcome.completed or _signature(outcome.error) != trial.signature:
                    if trial.name not in aggregate.blockers:
                        aggregate.blockers.append(trial.name)
                    aggregate.round_blockers[-1] = trial.name
                    aggregate.experiments.append((trial.name, True))
                    tried = set()
                else:
                    for path in trial.files:
                        Path(path).unlink(missing_ok=True)
                        written.discard(path)
                    aggregate.experiments.append((trial.name, False))
                    outcome = trial.outcome
                trial = None

            if outcome.completed:
                aggregate.error = None
                aggregate.round_blockers.append(None)
                break

            blocker, created, gap = _unblock(outcome, sysroot, stubs, interpreter, written)
            for module in outcome.imports:
                interpreter.missed(module)
            imported = interpreter.provide()
            if gap and gap not in aggregate.host_gaps:
                aggregate.host_gaps.append(gap)
            if blocker and blocker not in aggregate.blockers:
                aggregate.blockers.append(blocker)
            if blocker and _dist_for_program(blocker, interpreter) is not None:
                if blocker not in aggregate.build_tools:
                    aggregate.build_tools.append(blocker)
            label = blocker or gap or (f"import {imported[0]}" if imported else None)
            aggregate.round_blockers.append(label)
            if not created and not imported:
                trial = _next_trial(outcome, tried, aggregate.blockers, sysroot, written)
                if trial is None:
                    break   # nothing left to give it; the setup is stuck here
                tried.add(trial.name)
                created = trial.files
            written.update(created)

        aggregate.python_installed = list(interpreter.installed)
        aggregate.python_stubbed = list(interpreter.stubbed)

    aggregate.blocking = aggregate.blockers[0] if aggregate.blockers else None
    aggregate.soft_misses -= set(aggregate.blockers)
    aggregate.found -= set(aggregate.blockers)
    aggregate.unblocked = sorted(Path(p).name for p in written)
    aggregate.duration = time.monotonic() - started
    return aggregate


# ------------------------------------------------------------------ running


def _meson_command(
    tree: Path, vendored: Optional[str], interpreter: HostPython
) -> tuple[Optional[list[str]], Optional[str]]:
    """The meson to run, and a note saying which it is."""
    if vendored:
        script = tree / vendored
        if script.is_file():
            return [sys.executable, str(script)], f"configured with the Meson it ships ({vendored})"
        return None, f"the plan names a Meson at {vendored}, and there is none"
    version = interpreter.version_of("meson")
    if version and interpreter.install("meson", version):
        return (
            [sys.executable, str(interpreter.scripts / "meson")],
            f"configured with meson {version}, the build requirement's version",
        )
    host = shutil.which("meson")
    if host:
        return [host], None
    if interpreter.install("meson"):
        return [sys.executable, str(interpreter.scripts / "meson")], None
    return None, "meson is not installed, and no wheel of it would install"


def _tool(interpreter: HostPython, dist: str) -> bool:
    return interpreter.install(dist, interpreter.version_of(dist)) is not None


def _cross_file(scratch: Path, sysroot: Path, arch: str, interpreter: HostPython) -> Path:
    def quoted(values) -> str:
        return "[" + ", ".join(repr(str(v)) for v in values) + "]"

    binaries = []
    for language, candidates in _COMPILERS.items():
        found = next((c for c in candidates if shutil.which(c)), None)
        if found:
            binaries.append(f"{language} = {found!r}")
    for tool in ("ar", "strip", "pkg-config"):
        if shutil.which(tool):
            binaries.append(f"{tool} = {tool!r}")
    python = str(interpreter.executable)
    binaries += [f"python = {python!r}", f"python3 = {python!r}"]
    include = ["-isystem", sysroot / "include"]
    link = [f"-L{sysroot / 'lib'}"]
    text = "\n".join([
        "# Written by will-it-riscv: set up as a linux/riscv64 build that can find",
        "# nothing but what this scratch sysroot holds.",
        "[host_machine]",
        "system = 'linux'",
        "kernel = 'linux'",
        f"cpu_family = {arch!r}",
        f"cpu = {arch!r}",
        "endian = 'little'",
        "",
        "[binaries]",
        *binaries,
        "",
        "[properties]",
        "# The compilers are this host's, so what they build runs here.",
        "needs_exe_wrapper = false",
        f"pkg_config_libdir = {quoted([sysroot / 'lib' / 'pkgconfig'])}",
        f"sys_root = {str(sysroot)!r}",
        "",
        "[built-in options]",
        f"c_args = {quoted(include)}",
        f"cpp_args = {quoted(include)}",
        f"c_link_args = {quoted(link)}",
        f"cpp_link_args = {quoted(link)}",
        f"fortran_link_args = {quoted(link)}",
        "",
    ])
    path = scratch / "cross.ini"
    path.write_text(text)
    return path


def _native_file(scratch: Path, interpreter: HostPython) -> Path:
    """What meson-python writes: the interpreter running the build."""
    python = str(interpreter.executable)
    path = scratch / "native.ini"
    path.write_text(f"[binaries]\npython = {python!r}\npython3 = {python!r}\n")
    return path


def _setup(
    command: list[str], tree: Path, build: Path, cross: Path, native: Path,
    options: dict, env: dict, timeout: int,
) -> _Round:
    argv = [
        *command, "setup", str(build), str(tree),
        f"--cross-file={cross}", f"--native-file={native}",
        # What meson-python passes.
        "-Dbuildtype=release", "-Db_ndebug=if-release",
        *(f"-D{key}={value}" for key, value in sorted(options.items())),
    ]
    try:
        process = subprocess.run(
            argv, cwd=tree.parent, env=env, capture_output=True, text=True,
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return _Round(error=f"meson setup did not finish within {timeout}s")
    except OSError as exc:
        return _Round(error=f"could not run meson: {exc}")
    narration = (process.stdout or "") + "\n" + (process.stderr or "")
    outcome = _read(narration, tree)
    outcome.completed = process.returncode == 0
    if not outcome.completed:
        # A run_command's stderr is in the log, not the output.
        try:
            log = (build / "meson-logs" / "meson-log.txt").read_text(errors="replace")
        except OSError:
            log = ""
        outcome.imports = list(dict.fromkeys(_NO_MODULE.findall(log)))
    if not outcome.completed and outcome.error is None:
        tail = [line for line in narration.strip().splitlines() if line.strip()]
        outcome.error = tail[-1] if tail else f"meson setup exited with {process.returncode}"
    return outcome


def _read(narration: str, tree: Path) -> _Round:
    """What one meson setup said it found, missed, and died of."""
    outcome = _Round(narration=narration)
    lines = narration.splitlines()
    for index, line in enumerate(lines):
        line = line.rstrip()
        match = _SUBPROJECT.match(line)
        if match:
            outcome.subprojects.setdefault(match.group(1), f"subprojects/{match.group(1)}")
            continue
        match = _DEPENDENCY.match(line)
        if match:
            name, subproject, verdict, rest = match.groups()
            if subproject:
                outcome.subprojects[name] = subproject
            elif verdict == "YES":
                outcome.found.setdefault(name, "")
            elif name not in outcome.missing:
                outcome.missing.append(name)
            continue
        match = _LIBRARY.match(line)
        if match:
            name, verdict = match.groups()
            outcome.libraries.add(name)
            if verdict == "YES":
                outcome.found.setdefault(name, "")
            elif name not in outcome.missing:
                outcome.missing.append(name)
            continue
        match = _PROGRAM.match(line)
        if match and "/" not in match.group(1) and match.group(2) == "YES":
            paths = re.findall(r"\(([^()]*)\)", match.group(3))
            outcome.programs.setdefault(match.group(1), paths[-1].split()[-1] if paths else "")
            continue
        match = _ERROR.match(line)
        if match and outcome.error is None:
            where, number, message = match.groups()
            detail = [message.strip()]
            for following in lines[index + 1:index + 4]:
                if not following.strip() or following.startswith("A full log"):
                    break
                detail.append(following.strip())
            outcome.error = " ".join(detail)[:320]
            if where:
                outcome.where = f"{_relative(where, tree)}:{number}"
    return outcome


def _relative(path: str, tree: Path) -> str:
    for prefix in (f"{tree}/", f"{tree.name}/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _absorb(aggregate: PseudoBuild, outcome: _Round, calls: dict, attempt: int) -> None:
    for name in [*outcome.found, *outcome.missing]:
        key = name.lower()
        if key in aggregate.probes:
            continue
        command = "find_library" if name in outcome.libraries else "dependency"
        site, required = _site(calls, name)
        aggregate.probes[key] = Probe(
            name=name, command=command, required=required, site=site, round=attempt
        )
    aggregate.found |= set(outcome.found)
    for name, where in outcome.programs.items():
        if "will-it-riscv-meson-" in where or name.lower() in aggregate.probes:
            # The interpreter, or a program this loop installed or stubbed:
            # something it gave the setup, not something the host had.
            continue
        site, required = _site(calls, name)
        aggregate.probes[name.lower()] = Probe(
            name=name, command="find_program", required=required, site=site, round=attempt
        )
        aggregate.found.add(name)
        aggregate.found_at.setdefault(name, where)
    aggregate.soft_misses |= set(outcome.missing)
    built = sorted({Path(sub).name for sub in outcome.subprojects.values()})
    if built:
        # What it builds from its own subprojects -- matplotlib's freetype
        # and qhull -- is part of its build, not something to install first.
        note = "builds from its own subprojects: " + ", ".join(built)
        aggregate.notes = [n for n in aggregate.notes if not n.startswith("builds from its own")]
        aggregate.notes.append(note)
    aggregate.completed = outcome.completed
    aggregate.error = outcome.error
    aggregate.commands_traced += (
        len(outcome.found) + len(outcome.missing) + len(outcome.programs)
    )


# ----------------------------------------------------------------- unblocking


def _unblock(
    outcome: _Round, sysroot: Path, stubs: Path, interpreter: HostPython, written: set
) -> tuple[Optional[str], list[str], Optional[str]]:
    """``(blocker, files written, host gap)`` for what stopped this setup."""
    message = outcome.error or ""
    for pattern in (_DEP_NOT_FOUND, _DEP_LOOKUP, _DEP_VERSION):
        match = pattern.search(message)
        if match:
            name = match.group(1)
            return name, _fresh(_stub_dependency(name, sysroot), written), None
    match = _LIB_NOT_FOUND.search(message)
    if match:
        name = match.group(1)
        return name, _fresh(_stub_library(name, sysroot), written), None
    match = _UNKNOWN_COMPILER.search(message)
    if match:
        name = match.group(1)
        if _dist_for_program(name, interpreter) is None and not _is_build_tool(name):
            # A compiler for a whole language the host lacks: the target's
            # toolchain has it, and nothing here can stand in for one.
            return None, [], f"a {name} compiler"
        return name, _provide_program(name, stubs, interpreter, written), None
    for pattern in (_PROG_NOT_FOUND, _PROG_VERSION):
        match = pattern.search(message)
        if match:
            name = match.group(1)
            return name, _provide_program(name, stubs, interpreter, written), None
    match = _HEADER_NOT_FOUND.search(message)
    if match:
        header = match.group(1)
        from .syslibs import database

        owner = database().header(header)
        created = _fresh(_stub_header(header, sysroot), written)
        if owner is None:
            return None, created, header
        return owner, created, None
    match = _MESON_VERSION.search(message)
    if match:
        return None, [], f"meson {match.group(1)} (the project wants {match.group(2)})"
    return None, [], None


def _fresh(files: list[str], written: set) -> list[str]:
    return [f for f in files if f not in written]


def _dist_for_program(name: str, interpreter: HostPython) -> Optional[str]:
    if interpreter.dists is None:
        return None
    wanted = interpreter.dists
    dist = PROGRAM_TO_DIST.get(name, name)
    return dist_for_module(dist, wanted)


def _is_build_tool(name: str) -> bool:
    return name in ("cython", "cython3")


def _provide_program(name: str, stubs: Path, interpreter: HostPython, written: set) -> list[str]:
    """The program from the plan's build requirements, for the host, else a stub."""
    dist = _dist_for_program(name, interpreter)
    if dist is not None:
        spec = interpreter.install(dist, interpreter.version_of(dist))
        # Installed once already and still not found: stub it instead.
        if spec and spec not in written and (interpreter.scripts / name).exists():
            return [spec]
    return _fresh(_stub_program(name, stubs), written)


def _stub_program(name: str, stubs: Path) -> list[str]:
    if not re.match(r"^[A-Za-z0-9_.+-]+$", name):
        return []
    path = stubs / name
    # Meson reads a version out of what a program says: Cython's own words.
    answer = (
        f"Cython version {FAKE_VERSION_STRING}" if _is_build_tool(name)
        else f"{name} {FAKE_VERSION_STRING}"
    )
    path.write_text(
        "#!/bin/sh\n# stub emitted by will-it-riscv\n"
        f'case "$1" in --version|-V|-v|version) echo "{answer}";; esac\nexit 0\n'
    )
    path.chmod(0o755)
    return [str(path)]


_ARCHIVE = b"!<arch>\n"
"""An archive with nothing in it. Linkers take it; an empty file crashes ld64."""


def _stub_library(name: str, sysroot: Path) -> list[str]:
    if not re.match(r"^[A-Za-z0-9_.+-]+$", name):
        return []
    path = sysroot / "lib" / f"lib{name}.a"
    if path.exists():
        return []
    path.write_bytes(_ARCHIVE)
    return [str(path)]


def _stub_dependency(name: str, sysroot: Path) -> list[str]:
    """A .pc file for a dependency, where pkg-config is confined to look."""
    if not re.match(r"^[A-Za-z0-9_.+-]+$", name):
        return []
    library = name[3:] if name.startswith("lib") and len(name) > 3 else name
    pc = sysroot / "lib" / "pkgconfig" / f"{name}.pc"
    pc.write_text(
        f"prefix={sysroot.as_posix()}\n"
        "includedir=${prefix}/include\n"
        "libdir=${prefix}/lib\n\n"
        f"Name: {name}\n"
        "Description: stub emitted by will-it-riscv\n"
        f"Version: {FAKE_VERSION_STRING}\n"
        "Cflags: -I${includedir}\n"
        f"Libs: -L${{libdir}} -l{library}\n"
    )
    return [str(pc), *_stub_library(library, sysroot)]


def _stub_header(header: str, sysroot: Path) -> list[str]:
    path = sysroot / "include" / header
    if ".." in Path(header).parts or path.exists():
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_fake_header(Path(header).stem))
    return [str(path)]


# ------------------------------------------------------- blame by experiment


@dataclass
class _Trial:
    name: str
    signature: str
    files: list
    outcome: _Round
    """The stuck round, to pick the next suspect from if this one is innocent."""


def _suspects(outcome: _Round) -> list[str]:
    """What the setup missed before it died, likeliest first.

    A miss the error names comes first; after that, the nearer the error,
    the likelier.
    """
    message = (outcome.error or "").lower()
    missed = list(reversed(outcome.missing))
    named = [m for m in missed if _mentions(message, m)]
    return named + [m for m in missed if m not in named]


def _next_trial(
    outcome: _Round, tried: set, blockers: list, sysroot: Path, written: set
) -> Optional[_Trial]:
    if len(tried) >= MAX_SUSPECTS:
        return None
    for name in _suspects(outcome):
        if name in tried or name in blockers:
            continue
        if name in outcome.libraries:
            files = _fresh(_stub_library(name, sysroot), written)
        else:
            files = _fresh(_stub_dependency(name, sysroot), written)
        if not files:
            tried.add(name)
            continue
        return _Trial(name, _signature(outcome.error), files, outcome)
    return None


# ------------------------------------------------------------------ reading


_SUBDIR = re.compile(r"\bsubdir\(\s*'([^']+)'")


def _calls(tree: Path) -> dict[str, list[tuple[str, Optional[bool]]]]:
    """Every ``dependency('x')`` and friends in the project's own meson.build
    files, as ``name -> [(path:line, required)]``.

    Only the files Meson itself would read: the root's, and those its
    ``subdir()`` calls reach. numpy's vendored Meson has hundreds of test
    projects in its tree, and subprojects are their own projects.
    """
    calls: dict[str, list[tuple[str, Optional[bool]]]] = {}
    pending = [tree / "meson.build"]
    seen: set = set()
    while pending:
        path = pending.pop(0)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        relative = path.relative_to(tree)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pending += [path.parent / sub / "meson.build" for sub in _SUBDIR.findall(text)]
        for match in _CALL.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            arguments = _arguments(text, match.end())
            kwarg = _REQUIRED_KWARG.search(arguments)
            if kwarg is None:
                required: Optional[bool] = True
            else:
                value = kwarg.group(1)
                required = True if value == "true" else False if value == "false" else None
            calls.setdefault(match.group(2), []).append((f"{relative.as_posix()}:{line}", required))
    return calls


def _arguments(text: str, start: int) -> str:
    """The rest of a call's argument list, up to its closing parenthesis."""
    depth = 1
    for index in range(start, min(len(text), start + 2000)):
        char = text[index]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return text[start:start + 2000]


def _site(calls: dict, name: str) -> tuple[Optional[str], bool]:
    """Where the project asks for it, and whether any call insists."""
    sites = calls.get(name, [])
    if not sites and name.lower() in ("python", "python3"):
        return None, True   # py.dependency(): an extension module needs it
    if not sites:
        return None, False
    return sites[0][0], any(required is True for _, required in sites)
