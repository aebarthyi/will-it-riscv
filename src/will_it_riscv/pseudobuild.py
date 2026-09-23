"""Pseudobuilds: configure the project, watch what it asks for, throw it away.

Static reading of a build file has a hard ceiling. GDAL wraps every driver in
its own ``gdal_check_package()`` macro, so nothing a regex or an if/else walk
can do will tell you those are optional. But the macro is perfectly visible
while it runs.

So: run the project's *configure* step -- never its build -- in a scratch
directory, with CMake tracing every command it executes and its arguments
already expanded. Deny every ``pkg-config`` query while doing it, because a
configure that still insists on something after being told nothing is
installed is a configure that genuinely needs it.

This executes the project's build scripts. That is a real change of posture
for a tool that otherwise only reads, so it is opt-in, time-bounded, confined
to a temporary directory, and never runs the compiler on the project itself.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TIMEOUT_SECONDS = 600

#: Commands whose execution means the project asked the system for something.
FIND_COMMANDS = {
    "find_package",
    "pkg_check_modules",
    "pkg_search_module",
    "find_library",
    "check_library_exists",
}


@dataclass(frozen=True)
class Probe:
    """One dependency question the configure actually asked."""

    name: str
    command: str
    required: bool = False
    quiet: bool = False


@dataclass
class PseudoBuild:
    """What running the configure told us."""

    probes: dict[str, Probe] = field(default_factory=dict)
    completed: bool = False
    """True when configure ran to the end. Only then does *absence* from the
    trace mean anything -- a configure that stopped early simply never
    reached the rest of the file."""
    error: Optional[str] = None
    duration: float = 0.0
    commands_traced: int = 0

    found: set = field(default_factory=set)
    """Packages the configure located and used. Proven present."""
    soft_misses: set = field(default_factory=set)
    """Packages the configure looked for, did not find, and carried on
    without anyway. That is proof of optionality, not an inference."""
    blocking: Optional[str] = None
    """The package whose absence stopped the configure. For a riscv64 port
    this is the first thing that has to exist."""
    blockers: list = field(default_factory=list)
    """Every package that stopped a round, in the order the build demanded
    them. This is the chain of hard requirements for a port."""
    rounds: int = 1
    """Configure attempts made. More than one means the loop satisfied a
    blocker with a stub and went round again to see what lay behind it."""
    unblocked: list = field(default_factory=list)
    """Cache variables that were faked to get past a blocker."""
    narration: str = ""
    """The configure's own output, kept so the next round can read what it
    asked for. Not part of the report."""

    @property
    def ok(self) -> bool:
        return bool(self.probes) or self.completed

    def reached(self, name: str) -> bool:
        return name.lower() in self.probes


def available() -> bool:
    return shutil.which("cmake") is not None


#: CMake prefixes its STATUS output with "-- ". A miss reported that way is a
#: miss the configure shrugged off; the same words indented inside a CMake
#: Error block are the miss that stopped it.
_STATUS_MISS = re.compile(r"^-- +Could NOT find ([A-Za-z0-9_.+-]+)", re.MULTILINE)
#: FPHSA names the cache variables it wanted: "(missing: PROJ_LIBRARY ...)".
_MISSING_VARS = re.compile(r"\(missing:\s*([^)]*?)\s*\)")
#: A Find module reading a header that is not there names the exact path.
_MISSING_FILE = re.compile(
    r"file failed to open for reading \(No such file or directory\):\s*\n\s*(\S+)"
)
_STATUS_FOUND = re.compile(r"^-- +Found ([A-Za-z0-9_.+-]+)", re.MULTILINE)
_ANY_MISS = re.compile(r"Could NOT find ([A-Za-z0-9_.+-]+)")


def _error_blocks(text: str) -> str:
    """Just the fatal CMake Error stanzas.

    Synthesis must read only these. A "-- Could NOT find MySQL (missing:
    MYSQL_LIBRARY)" status line names variables too, and faking those would
    stub out the very optional dependencies the run exists to identify.
    """
    kept: list[str] = []
    in_error = False
    for line in text.splitlines():
        if line.startswith(("CMake Error", "  CMake Error")):
            in_error = True
        elif in_error and line.startswith("-- "):
            in_error = False
        if in_error:
            kept.append(line)
    return "\n".join(kept)


def _read_outcomes(text: str) -> tuple[set, set, Optional[str]]:
    """Split the configure's own narration into found, shrugged off, and fatal."""
    found = set(_STATUS_FOUND.findall(text))
    soft = set(_STATUS_MISS.findall(text))

    blocking = None
    in_error = False
    for line in text.splitlines():
        if line.startswith("CMake Error"):
            in_error = True
            continue
        if in_error:
            if line.startswith("-- "):
                in_error = False
                continue
            match = _ANY_MISS.search(line)
            if match:
                blocking = match.group(1)
                break
    if blocking:
        # It may have been narrated as a status miss first, then turned fatal.
        soft.discard(blocking)
    return found, soft, blocking


def _parse_trace(path: Path) -> tuple[dict[str, Probe], int]:
    probes: dict[str, Probe] = {}
    traced = 0
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return probes, 0
    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                # The trace opens with a version banner, and a malformed line
                # can still be valid JSON without being an event.
                continue
            command = str(event.get("cmd", "")).lower()
            if not command:
                continue
            traced += 1
            if command not in FIND_COMMANDS:
                continue
            args = [str(a) for a in event.get("args", []) if str(a).strip()]
            if not args:
                continue
            upper = [a.upper() for a in args]
            for name in _subjects(command, args):
                key = name.lower()
                existing = probes.get(key)
                probe = Probe(
                    name=name,
                    command=command,
                    required="REQUIRED" in upper,
                    quiet="QUIET" in upper,
                )
                if existing is None or (probe.required and not existing.required):
                    probes[key] = probe
    return probes, traced


#: Keywords after which a find command stops naming libraries and starts
#: naming places to look. --trace-expand turns those into real paths and
#: even Windows registry keys, so the cut has to be exact.
_STOP_KEYWORDS = {
    "PATHS", "HINTS", "PATH_SUFFIXES", "DOC", "NO_DEFAULT_PATH", "REQUIRED",
    "NO_CMAKE_PATH", "NO_CMAKE_ENVIRONMENT_PATH", "NO_SYSTEM_ENVIRONMENT_PATH",
    "NO_CMAKE_SYSTEM_PATH", "CMAKE_FIND_ROOT_PATH_BOTH", "ONLY_CMAKE_FIND_ROOT_PATH",
    "NO_CMAKE_FIND_ROOT_PATH", "VALIDATOR", "REGISTRY_VIEW", "NO_CACHE",
    "COMPONENTS", "OPTIONAL_COMPONENTS", "CONFIG", "MODULE", "NAMES_PER_DIR",
    "GLOBAL", "BYPASS_PROVIDER", "IMPORTED_TARGET", "NO_PACKAGE_ROOT_PATH",
}
_SKIP_KEYWORDS = {"NAMES", "QUIET", "EXACT", "REQUIRED", "IMPORTED_TARGET",
                  "GLOBAL", "STATIC", "NO_CMAKE_PATH"}


def _subjects(command: str, args: list[str]) -> list[str]:
    """What the command was looking for, skipping variables, flags and paths."""
    from .sdist import _is_plausible_name

    if command in ("find_package", "check_library_exists"):
        candidates = args[:1]
    else:
        # pkg_check_modules(PREFIX [QUIET] mod...) and
        # find_library(VAR [NAMES] name... [PATHS ...]): the first argument is
        # the output variable, and everything from a stop keyword on is noise.
        candidates = []
        for arg in args[1:]:
            upper = arg.upper()
            if upper in _STOP_KEYWORDS:
                break
            if upper in _SKIP_KEYWORDS:
                continue
            candidates.append(arg)

    return [
        c for c in candidates
        if _is_plausible_name(c) and not c.startswith(("(", "[", "-", "$"))
    ]


def run(
    root: Path,
    timeout: int = TIMEOUT_SECONDS,
    deny_pkg_config: bool = True,
    max_rounds: int = 6,
) -> Optional[PseudoBuild]:
    """Configure the project in a scratch directory, unblocking as it goes.

    A configure that stops tells you one thing: what stopped it. Satisfy that
    with a stub and run it again, and it tells you the next thing -- and once
    it finally runs to the end, everything it did *not* ask for is known to be
    optional. GDAL stops at PROJ and yields 7 proven-optional dependencies;
    two rounds later it completes and yields 55.

    Returns None when there is nothing to do (no CMakeLists.txt at the root).
    """
    root = Path(root)
    if not (root / "CMakeLists.txt").exists():
        return None
    if not available():
        return PseudoBuild(error="cmake is not installed")

    started = time.monotonic()
    aggregate = PseudoBuild(rounds=0)
    overrides: dict[str, str] = {}
    written: set = set()

    with tempfile.TemporaryDirectory(prefix="will-it-riscv-") as scratch:
        scratch_path = Path(scratch)
        sysroot = scratch_path / "sysroot"
        env = _environment(scratch_path, deny_pkg_config)

        for attempt in range(1, max_rounds + 1):
            outcome = _configure(
                root, scratch_path, overrides, env, timeout, attempt
            )
            aggregate.rounds = attempt
            aggregate.probes.update(outcome.probes)
            aggregate.found |= outcome.found
            aggregate.soft_misses |= outcome.soft_misses
            aggregate.commands_traced += outcome.commands_traced
            aggregate.completed = outcome.completed
            aggregate.error = outcome.error

            if outcome.completed:
                aggregate.error = None
                break
            if outcome.blocking and outcome.blocking not in aggregate.blockers:
                aggregate.blockers.append(outcome.blocking)

            fresh, created = _synthesize(
                outcome.blocking, outcome.narration, sysroot, written
            )
            fresh = {k: v for k, v in fresh.items() if overrides.get(k) != v}
            if not fresh and not created:
                break   # nothing left to try; the configure is stuck here
            overrides.update(fresh)
            written.update(created)
            aggregate.unblocked = sorted(overrides)

    aggregate.blocking = aggregate.blockers[0] if aggregate.blockers else None
    # A blocker that a later round walked past is still a hard requirement,
    # but it no longer stops anything.
    aggregate.soft_misses -= set(aggregate.blockers)
    aggregate.duration = time.monotonic() - started
    return aggregate


def _configure(
    root: Path,
    scratch: Path,
    overrides: dict,
    env: dict,
    timeout: int,
    attempt: int,
) -> PseudoBuild:
    """One configure attempt, in its own build directory."""
    build = scratch / f"build-{attempt}"
    trace = scratch / f"trace-{attempt}.json"
    build.mkdir(parents=True, exist_ok=True)

    command = [
        "cmake",
        "-S", str(root),
        "-B", str(build),
        # Expanded, so ${_brotli_pc_requires} arrives as the module names it
        # stands for rather than as a variable this tool would reject.
        "--trace-expand",
        "--trace-format=json-v1",
        f"--trace-redirect={trace}",
    ]
    command += [f"-D{name}={value}" for name, value in sorted(overrides.items())]

    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        probes, traced = _parse_trace(trace)
        return PseudoBuild(
            probes=probes,
            error=f"configure did not finish within {timeout}s",
            commands_traced=traced,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PseudoBuild(error=f"could not run cmake: {exc}")

    probes, traced = _parse_trace(trace)
    narration = (process.stdout or "") + "\n" + (process.stderr or "")
    found, soft, blocking = _read_outcomes(narration)
    result = PseudoBuild(
        probes=probes,
        completed=process.returncode == 0,
        commands_traced=traced,
        found=found,
        soft_misses=soft,
        blocking=blocking,
    )
    result.narration = narration
    if not result.completed:
        result.error = _first_error(process.stdout, process.stderr)
    return result


# ----------------------------------------------------------------- unblocking

#: A version no project will consider too old.
FAKE_VERSION = ("99", "9", "9")
FAKE_VERSION_STRING = "99.9.9"

_LIBRARY_VAR = re.compile(r"_(LIBRARY|LIBRARIES|LIB|LIBS)$", re.IGNORECASE)
_INCLUDE_VAR = re.compile(r"_(INCLUDE_DIR|INCLUDE_DIRS|INCLUDEDIR|INCLUDE)$", re.IGNORECASE)
_PROGRAM_VAR = re.compile(r"_(EXECUTABLE|COMMAND|BINARY|PROGRAM|COMPILER)$", re.IGNORECASE)
_VERSION_VAR = re.compile(r"_VERSION", re.IGNORECASE)
#: <Pkg>_DIR is config-mode's hint. Faking it sends CMake looking for a
#: package config file that is not there, which fails worse than not setting it.
_CONFIG_DIR_VAR = re.compile(r"^[A-Za-z0-9_]+_DIR$")

_HEADER_SUFFIXES = (".h", ".hpp", ".hh", ".hxx", ".inc")


def _library_suffix() -> str:
    import sys

    return ".dylib" if sys.platform == "darwin" else ".so"


def _fake_header(name: str) -> str:
    """A header carrying every spelling of a version macro in common use.

    Find modules routinely grep the version out of a header rather than ask
    the library. GDAL's FindPROJ does exactly that, and refuses anything
    below 6.3, so an empty file gets past find_path and then fails.
    """
    stem = re.sub(r"[^A-Za-z0-9]", "_", name).upper().strip("_")
    major, minor, patch = FAKE_VERSION
    lines = [f"/* stub emitted by will-it-riscv for {name} */"]
    for prefix in dict.fromkeys([stem, stem.rstrip("_0123456789")]):
        if not prefix:
            continue
        lines += [
            f"#define {prefix}_VERSION_MAJOR {major}",
            f"#define {prefix}_VERSION_MINOR {minor}",
            f"#define {prefix}_VERSION_PATCH {patch}",
            f"#define {prefix}_VERSION_MICRO {patch}",
            f"#define {prefix}_VERSION_NUM {major}{minor}{patch}",
            f'#define {prefix}_VERSION "{FAKE_VERSION_STRING}"',
            f'#define {prefix}_VERSION_STRING "{FAKE_VERSION_STRING}"',
        ]
    return "\n".join(lines) + "\n"


def _synthesize(
    blocking: Optional[str], narration: str, sysroot: Path, already: set
) -> tuple[dict, list]:
    """Work out what to fake so the next round gets further.

    Everything here is driven by what the configure said it wanted. It names
    the cache variables it could not fill, and when a Find module tries to
    read a header that is not there, it names the exact path.
    """
    # Only what actually stopped the configure. Everything narrated as a
    # status miss is a dependency the build did without, and stubbing it
    # would erase the evidence.
    narration = _error_blocks(narration)

    overrides: dict[str, str] = {}
    include_dir = sysroot / "include"
    lib_dir = sysroot / "lib"
    bin_dir = sysroot / "bin"
    for directory in (include_dir, lib_dir, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. The cache variables FPHSA said were missing.
    for match in _MISSING_VARS.finditer(narration):
        for variable in match.group(1).split():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", variable):
                continue
            if _CONFIG_DIR_VAR.match(variable) and not _INCLUDE_VAR.search(variable):
                continue
            if _LIBRARY_VAR.search(variable):
                stem = _LIBRARY_VAR.sub("", variable).lower() or "stub"
                path = lib_dir / f"lib{stem}{_library_suffix()}"
                path.touch()
                overrides[variable] = str(path)
            elif _INCLUDE_VAR.search(variable):
                overrides[variable] = str(include_dir)
            elif _PROGRAM_VAR.search(variable):
                path = bin_dir / _PROGRAM_VAR.sub("", variable).lower()
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
                overrides[variable] = str(path)
            elif _VERSION_VAR.search(variable):
                overrides[variable] = FAKE_VERSION_STRING
            else:
                # FPHSA only checks the variable is set and not *-NOTFOUND.
                overrides[variable] = "1"

    # 2. Files a Find module tried to read. The error gives the full path.
    created: list = []
    for path_text in _MISSING_FILE.findall(narration):
        path = Path(path_text)
        # Only ever write inside our own scratch directory.
        if not str(path).startswith(str(sysroot)) or str(path) in already:
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() in _HEADER_SUFFIXES:
                path.write_text(_fake_header(blocking or path.stem))
            else:
                path.touch()
        except OSError:
            continue
        created.append(str(path))

    return overrides, created


def _environment(scratch: Path, deny_pkg_config: bool) -> dict:
    import os

    env = dict(os.environ)
    if deny_pkg_config:
        # PKG_CONFIG_LIBDIR replaces the search path outright, so every query
        # misses. Whatever the configure still demands, it truly needs.
        empty = scratch / "no-pkgconfig"
        empty.mkdir(exist_ok=True)
        env["PKG_CONFIG_LIBDIR"] = str(empty)
        env["PKG_CONFIG_PATH"] = str(empty)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"
    return env


def _first_error(stdout: str, stderr: str) -> str:
    for stream in (stderr, stdout):
        for line in (stream or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("CMake Error", "ERROR:")):
                return stripped[:200]
    tail = (stdout or stderr or "").strip().splitlines()
    return tail[-1][:200] if tail else "configure did not complete"
