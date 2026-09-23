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
_STATUS_FOUND = re.compile(r"^-- +Found ([A-Za-z0-9_.+-]+)", re.MULTILINE)
_ANY_MISS = re.compile(r"Could NOT find ([A-Za-z0-9_.+-]+)")


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
) -> Optional[PseudoBuild]:
    """Configure the project in a scratch directory and report what it asked for.

    Returns None when there is nothing to do (no CMakeLists.txt at the root).
    """
    root = Path(root)
    if not (root / "CMakeLists.txt").exists():
        return None
    if not available():
        return PseudoBuild(error="cmake is not installed")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="will-it-riscv-") as scratch:
        scratch_path = Path(scratch)
        build = scratch_path / "build"
        trace = scratch_path / "trace.json"
        build.mkdir()

        env = _environment(scratch_path, deny_pkg_config)
        command = [
            "cmake",
            "-S", str(root),
            "-B", str(build),
            # Expanded, so ${_brotli_pc_requires} arrives as the module names
            # it stands for rather than as a variable this tool would reject.
            "--trace-expand",
            "--trace-format=json-v1",
            f"--trace-redirect={trace}",
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, env=env
            )
        except subprocess.TimeoutExpired:
            probes, traced = _parse_trace(trace)
            return PseudoBuild(
                probes=probes,
                error=f"configure did not finish within {timeout}s",
                duration=time.monotonic() - started,
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
        duration=time.monotonic() - started,
        commands_traced=traced,
        found=found,
        soft_misses=soft,
        blocking=blocking,
    )
    if not result.completed:
        result.error = _first_error(process.stdout, process.stderr)
    return result


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
