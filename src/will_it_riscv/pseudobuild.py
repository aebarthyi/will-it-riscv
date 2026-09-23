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

The configure is run as a cross build for Linux on the target architecture,
with every library, header and package search re-rooted into an empty
scratch sysroot. Otherwise the host answers questions meant for the target:
on a Mac with Homebrew, half of GDAL's dependencies are simply *found*, which
proves nothing about whether the build needed them, and ``if(APPLE)`` sends
the configure down branches a riscv64 build never takes. Confined, the only
things a configure can find are the stubs this module puts there, so every
probe ends in one of two provable states.

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
    round_blockers: list = field(default_factory=list)
    """What stopped each round, by index; None for the round that
    completed. A blocker can stop two rounds running -- GDAL's PROJ does,
    once for its library and again for the header it reads a version from."""
    platform: str = "host"
    """What the configure was run as: ``linux/<arch>`` when confined to the
    scratch sysroot, ``host`` when it would only configure natively."""
    host_gaps: list = field(default_factory=list)
    """Headers the configure demanded that belong to no library -- the ones
    a riscv64 Linux system has and the host's SDK lacks, such as
    ``linux/fs.h``. Stubbed to get past them, and not dependencies."""
    notes: list = field(default_factory=list)
    narration: str = ""
    """The configure's own output, kept so the next round can read what it
    asked for. Not part of the report."""

    @property
    def ok(self) -> bool:
        return bool(self.probes) or self.completed

    @property
    def confined(self) -> bool:
        return self.platform != "host"

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
#: pkg_check_modules(... REQUIRED) lists what it could not find, one per line.
_PKG_REQUIRED = "required packages were not found"
_PKG_MODULE = re.compile(r"^\s+-\s+([A-Za-z0-9_.+-]+)")
#: pkg_search_module(... REQUIRED) names the alternatives it tried.
_PKG_SEARCH = re.compile(r"None of the required '([^']+)' found")
#: The generate step refuses a NOTFOUND variable that a target links. That
#: dependency is not optional -- something is built against it.
_NOTFOUND_HEADER = "are set to NOTFOUND"
_NOTFOUND_VAR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?: \(ADVANCED\))?\s*$")
_VAR_STEM = re.compile(
    r"_(LIBRARY|LIBRARIES|LIB|LIBS|INCLUDE_DIR|INCLUDE_DIRS|INCLUDEDIR|INCLUDE|"
    r"EXECUTABLE|PROGRAM)(_[A-Z]+)?$",
    re.IGNORECASE,
)
#: An error raised from inside a package's own Find module or config file is
#: that package's error, even when its message never names it.
_ERROR_FILE = re.compile(r"^\s*CMake Error at (\S+?):\d+")
_MODULE_FILE = re.compile(r"(?:^|/)(?:Find([A-Za-z0-9_+-]+)|([A-Za-z0-9_+-]+?)-?[Cc]onfig)\.cmake$")
#: FindOpenSSL creates OpenSSL::SSL only if OPENSSL_SSL_LIBRARY exists, and a
#: try_compile that links it then fails naming the target.
_TARGET_NOT_FOUND = re.compile(
    r"links to:\s+([A-Za-z0-9_+.-]+)::([A-Za-z0-9_+.-]+)\s+but the target was not found"
)
#: A header a check could not find, named in the configure's own error.
_HEADER_NAME = re.compile(r"(?<![\w/.])([A-Za-z0-9_+-][A-Za-z0-9_+./-]*\.(?:h|hpp|hh|hxx))\b")
_HEADER_MISSING = re.compile(
    r"not found|cannot find|could not find|missing|no such file|not exist", re.IGNORECASE
)


def _error_stanzas(text: str) -> list[str]:
    """Each fatal CMake Error stanza, separately.

    Synthesis must read only these. A "-- Could NOT find MySQL (missing:
    MYSQL_LIBRARY)" status line names variables too, and faking those would
    stub out the very optional dependencies the run exists to identify.
    """
    stanzas: list[list[str]] = []
    current: Optional[list[str]] = None
    for line in text.splitlines():
        if line.startswith(("CMake Error", "  CMake Error")):
            current = [line]
            stanzas.append(current)
        elif line.startswith(("-- ", "CMake Warning", "CMake Deprecation Warning")):
            current = None
        elif current is not None:
            current.append(line)
    return ["\n".join(stanza) for stanza in stanzas]


def _error_blocks(text: str) -> str:
    return "\n".join(_error_stanzas(text))


def _stanza_blockers(stanza: str) -> list[str]:
    """What a single error stanza says was missing, in the order it says it."""
    names: list[str] = []
    lines = stanza.splitlines()
    if _NOTFOUND_HEADER in stanza:
        for line in lines[1:]:
            match = _NOTFOUND_VAR.match(line)
            if match and not line.startswith("Please"):
                names.append(_VAR_STEM.sub("", match.group(1)) or match.group(1))
        return names
    listing = False
    for line in lines:
        match = _ANY_MISS.search(line)
        if match:
            names.append(match.group(1))
            continue
        search = _PKG_SEARCH.search(line)
        if search:
            names.append(search.group(1).split(";")[0].strip())
            continue
        if _PKG_REQUIRED in line:
            listing = True
            continue
        if listing:
            module = _PKG_MODULE.match(line)
            if module:
                names.append(re.split(r"[<>=]", module.group(1), maxsplit=1)[0])
            elif line.strip():
                listing = False
    for match in _TARGET_NOT_FOUND.finditer(stanza):
        names.append(match.group(1))
    if not names:
        library = _header_blocker(stanza)
        if library:
            names.append(library)
    if not names:
        owner = _raised_by(stanza)
        if owner:
            names.append(owner)
    return names


def _raised_by(stanza: str) -> Optional[str]:
    """The package whose Find module or config file raised this error."""
    match = _ERROR_FILE.match(stanza)
    if not match:
        return None
    module = _MODULE_FILE.search(match.group(1))
    if not module:
        return None
    return module.group(1) or module.group(2)


def _missing_headers(stanza: str) -> list[str]:
    """Relative header paths an error says it could not find."""
    headers: list[str] = []
    for line in stanza.splitlines():
        if not _HEADER_MISSING.search(line):
            continue
        for header in _HEADER_NAME.findall(line):
            if ".." in header.split("/") or header in headers:
                continue
            headers.append(header)
    return headers


def _header_blocker(stanza: str) -> Optional[str]:
    """The library behind a missing header, if it belongs to one.

    ``zlib.h`` is zlib's. ``linux/fs.h`` is nobody's: the target's C library
    and kernel headers ship it, and only the host's SDK lacks it. That is a
    gap in the host, not a dependency, and is kept apart as one.
    """
    from .syslibs import database

    db = database()
    for header in _missing_headers(stanza):
        library = db.header(header)
        if library:
            return library
    return None


def _read_outcomes(text: str) -> tuple[set, set, list]:
    """Split the configure's own narration into found, shrugged off, and fatal."""
    found = set(_STATUS_FOUND.findall(text))
    soft = set(_STATUS_MISS.findall(text))

    blockers: list[str] = []
    for stanza in _error_stanzas(text):
        for name in _stanza_blockers(stanza):
            if name not in blockers:
                blockers.append(name)
    # A blocker may have been narrated as a status miss first, then turned fatal.
    soft -= set(blockers)
    return found, soft, blockers


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
    max_rounds: int = 12,
    arch: str = "riscv64",
    confine: bool = True,
) -> Optional[PseudoBuild]:
    """Configure the project in a scratch directory, unblocking as it goes.

    A configure that stops tells you one thing: what stopped it. Satisfy that
    with a stub and run it again, and it tells you the next thing -- and once
    it finally runs to the end, everything it did *not* ask for is known to be
    optional. GDAL stops at PROJ and yields 7 proven-optional dependencies;
    two rounds later it completes and yields 55.

    ``timeout`` bounds the whole loop, not each round. Returns None when there
    is nothing to do (no CMakeLists.txt at the root).
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
        for directory in ("include", "lib/pkgconfig", "bin"):
            (sysroot / directory).mkdir(parents=True, exist_ok=True)
        env = _environment(scratch_path, deny_pkg_config, sysroot)
        toolchain = _toolchain(scratch_path, sysroot, arch) if confine else None
        aggregate.platform = f"linux/{arch}" if toolchain else "host"

        for attempt in range(1, max_rounds + 1):
            left = timeout - (time.monotonic() - started)
            if left <= 0:
                aggregate.error = f"ran out of time after {attempt - 1} rounds"
                break
            remaining = max(1, int(left))
            outcome = _configure(
                root, scratch_path, overrides, env, remaining, f"build-{attempt}",
                toolchain,
            )
            if attempt == 1 and toolchain and not outcome.completed and not outcome.probes:
                # It would not even start as a cross build. A host answer is
                # worse than a target one, but much better than none.
                host = _configure(
                    root, scratch_path, overrides, env, remaining, "build-host", None
                )
                if host.completed or host.probes:
                    aggregate.notes.append(
                        f"would not configure as a linux/{arch} cross build "
                        f"({outcome.error}), so it was configured for this host "
                        "instead, and the host's own libraries answered for it"
                    )
                    toolchain = None
                    aggregate.platform = "host"
                    outcome = host

            aggregate.rounds = attempt
            aggregate.probes.update(outcome.probes)
            aggregate.found |= outcome.found
            aggregate.soft_misses |= outcome.soft_misses
            aggregate.commands_traced += outcome.commands_traced
            aggregate.completed = outcome.completed
            aggregate.error = outcome.error

            if outcome.completed:
                aggregate.error = None
                aggregate.round_blockers.append(None)
                break
            gaps = _host_gaps(outcome.narration)
            for gap in gaps:
                if gap not in aggregate.host_gaps:
                    aggregate.host_gaps.append(gap)
            stopped_by = outcome.blockers[0] if outcome.blockers else None
            aggregate.round_blockers.append(stopped_by or (gaps[0] if gaps else None))
            for name in outcome.blockers:
                if name not in aggregate.blockers:
                    aggregate.blockers.append(name)

            suffix = ".so" if toolchain else _library_suffix()
            fresh, created = _synthesize(
                outcome.blocking, outcome.narration, sysroot, written,
                shared_suffix=suffix,
            )
            fresh = {k: v for k, v in fresh.items() if overrides.get(k) != v}
            if not fresh and not created:
                break   # nothing left to try; the configure is stuck here
            overrides.update(fresh)
            written.update(created)
            aggregate.unblocked = sorted(overrides)

    aggregate.blocking = aggregate.blockers[0] if aggregate.blockers else None
    # A blocker that a later round walked past is still a hard requirement,
    # but it no longer stops anything -- and it was "found" only because a
    # stub was put there.
    aggregate.soft_misses -= set(aggregate.blockers)
    aggregate.found -= set(aggregate.blockers)
    aggregate.duration = time.monotonic() - started
    return aggregate


#: Written into the scratch directory, never the project.
TOOLCHAIN = """\
# Written by will-it-riscv: configure as a Linux/{arch} build that can find
# nothing but what this scratch sysroot holds.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR {arch})
set(CMAKE_FIND_ROOT_PATH "{sysroot}")
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# Build tools are the host's: a {arch} machine has its own bison and python.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Compile checks only. The Linux platform rules and the host's linker do not mix.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
# Stub headers written to get past a blocker have to be visible to checks.
set(CMAKE_C_FLAGS_INIT "-isystem {sysroot}/include")
set(CMAKE_CXX_FLAGS_INIT "-isystem {sysroot}/include")
# The compiler is the host's, so what it builds runs here. CMake 4.1 and later
# (CMP0190) will not look for an interpreter in a cross build without this.
set(CMAKE_CROSSCOMPILING_EMULATOR "{emulator}")
"""


def _toolchain(scratch: Path, sysroot: Path, arch: str) -> Path:
    emulator = scratch / "run-on-host"
    emulator.write_text('#!/bin/sh\nexec "$@"\n')
    emulator.chmod(0o755)
    path = scratch / "toolchain.cmake"
    path.write_text(
        TOOLCHAIN.format(arch=arch, sysroot=sysroot.as_posix(), emulator=emulator.as_posix())
    )
    return path


def _configure(
    root: Path,
    scratch: Path,
    overrides: dict,
    env: dict,
    timeout: int,
    name: str,
    toolchain: Optional[Path] = None,
) -> PseudoBuild:
    """One configure attempt, in its own build directory."""
    build = scratch / name
    trace = scratch / f"{name}.trace.json"
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
    if toolchain is not None:
        command.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")
    command += [f"-D{key}={value}" for key, value in sorted(overrides.items())]

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
    found, soft, blockers = _read_outcomes(narration)
    result = PseudoBuild(
        probes=probes,
        completed=process.returncode == 0,
        commands_traced=traced,
        found=found,
        soft_misses=soft,
        blocking=blockers[0] if blockers else None,
        blockers=blockers,
    )
    result.narration = narration
    if not result.completed:
        result.error = _first_error(process.stdout, process.stderr)
    return result


# ----------------------------------------------------------------- unblocking

#: A version no project will consider too old.
FAKE_VERSION = ("99", "9", "9")
FAKE_VERSION_STRING = "99.9.9"

_LIBRARY_VAR = re.compile(r"_(LIBRARY|LIBRARIES|LIB|LIBS)(_[A-Z]+)?$", re.IGNORECASE)
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


def _stub_variable(variable: str, sysroot: Path, suffix: str) -> Optional[str]:
    """A value that satisfies a cache variable the configure could not fill."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", variable):
        return None
    if _CONFIG_DIR_VAR.match(variable) and not _INCLUDE_VAR.search(variable):
        return None
    if _LIBRARY_VAR.search(variable):
        stem = _LIBRARY_VAR.sub("", variable).lower() or "stub"
        path = sysroot / "lib" / f"lib{stem}{suffix}"
        path.touch()
        return str(path)
    if _INCLUDE_VAR.search(variable):
        return str(sysroot / "include")
    if _PROGRAM_VAR.search(variable):
        path = sysroot / "bin" / _PROGRAM_VAR.sub("", variable).lower()
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        return str(path)
    if _VERSION_VAR.search(variable):
        return FAKE_VERSION_STRING
    # FPHSA only checks the variable is set and not *-NOTFOUND.
    return "1"


def _stub_pkgconfig(module: str, sysroot: Path, suffix: str) -> list[str]:
    """A .pc file for a module pkg-config was required to find.

    It goes in the directory pkg-config has been confined to, which is empty
    until this writes something there, so every other module still misses.
    """
    name = re.split(r"[<>=\s]", module, maxsplit=1)[0]
    if not re.match(r"^[A-Za-z0-9_.+-]+$", name):
        return []
    library = name[3:] if name.startswith("lib") and len(name) > 3 else name
    lib = sysroot / "lib" / f"lib{library}{suffix}"
    pc = sysroot / "lib" / "pkgconfig" / f"{name}.pc"
    try:
        lib.touch()
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
    except OSError:
        return []
    return [str(pc)]


def _synthesize(
    blocking: Optional[str],
    narration: str,
    sysroot: Path,
    already: set,
    shared_suffix: Optional[str] = None,
) -> tuple[dict, list]:
    """Work out what to fake so the next round gets further.

    Everything here is driven by what the configure said it wanted. It names
    the cache variables it could not fill, the pkg-config modules it was
    required to find, the NOTFOUND variables a target links, and -- when a
    Find module tries to read a header that is not there -- the exact path.
    """
    suffix = shared_suffix or _library_suffix()
    overrides: dict[str, str] = {}
    for directory in ("include", "lib/pkgconfig", "bin"):
        (sysroot / directory).mkdir(parents=True, exist_ok=True)

    created: list = []
    # Only what actually stopped the configure. Everything narrated as a
    # status miss is a dependency the build did without, and stubbing it
    # would erase the evidence.
    for stanza in _error_stanzas(narration):
        variables: list[str] = []
        named = _ANY_MISS.search(stanza)
        package = named.group(1) if named else None
        # 1. The cache variables FPHSA said were missing. A bare word among
        #    them is a component -- "(missing: ... SSL Crypto)" -- and the
        #    variable behind a component is <PKG>_<COMPONENT>_LIBRARY.
        for match in _MISSING_VARS.finditer(stanza):
            for token in match.group(1).split():
                if "_" in token:
                    variables.append(token)
                elif package and token.isidentifier():
                    variables += _component_variables(package, token)
        # 1b. The same, for a component a try_compile tried to link.
        for match in _TARGET_NOT_FOUND.finditer(stanza):
            variables += _component_variables(match.group(1), match.group(2))
        # 2. NOTFOUND variables the generate step refused.
        if _NOTFOUND_HEADER in stanza:
            for line in stanza.splitlines()[1:]:
                found = _NOTFOUND_VAR.match(line)
                if found and not line.startswith("Please"):
                    variables.append(found.group(1))
        for variable in variables:
            value = _stub_variable(variable, sysroot, suffix)
            if value is not None:
                overrides[variable] = value

        # 3. pkg-config modules it was required to find.
        if _PKG_REQUIRED in stanza or _PKG_SEARCH.search(stanza):
            for module in _stanza_blockers(stanza):
                pc = sysroot / "lib" / "pkgconfig" / f"{module}.pc"
                if str(pc) in already:
                    continue
                created += _stub_pkgconfig(module, sysroot, suffix)

        # 4. Files a Find module tried to read. The error gives the full path.
        for path_text in _MISSING_FILE.findall(stanza):
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

        # 5. Headers a check said it could not find. The toolchain file puts
        #    the sysroot's include directory on every compile line, so a stub
        #    there is enough to satisfy check_include_file.
        for header in _missing_headers(stanza):
            path = sysroot / "include" / header
            if str(path) in already or path.exists():
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_fake_header(blocking or path.stem))
            except OSError:
                continue
            created.append(str(path))

    return overrides, created


def _component_variables(package: str, component: str) -> list[str]:
    """The cache variable a Find module fills for one component's library.

    Upper-case is the common spelling (OPENSSL_SSL_LIBRARY); FindBoost keeps
    the package's own case (Boost_FILESYSTEM_LIBRARY). Setting both costs one
    unused-variable warning.
    """
    spellings = [
        f"{package.upper()}_{component.upper()}_LIBRARY",
        f"{package}_{component.upper()}_LIBRARY",
    ]
    return list(dict.fromkeys(spellings))


def _host_gaps(narration: str) -> list[str]:
    """Headers the configure demanded that belong to no library.

    These are what the host's SDK lacks and a riscv64 Linux system has as a
    matter of course -- ``linux/fs.h``, ``sys/epoll.h`` -- so they are
    stubbed, and reported as gaps in the host rather than as dependencies.
    """
    from .syslibs import database

    db = database()
    gaps: list[str] = []
    for stanza in _error_stanzas(narration):
        for header in _missing_headers(stanza):
            if db.header(header) is None and header not in gaps:
                gaps.append(header)
    return gaps


def _environment(scratch: Path, deny_pkg_config: bool, sysroot: Optional[Path] = None) -> dict:
    import os

    env = dict(os.environ)
    if deny_pkg_config:
        # PKG_CONFIG_LIBDIR replaces the search path outright, so every query
        # misses -- until a blocker's stub .pc file is written there.
        # Whatever the configure still demands, it truly needs.
        empty = (sysroot / "lib" / "pkgconfig") if sysroot else scratch / "no-pkgconfig"
        empty.mkdir(parents=True, exist_ok=True)
        env["PKG_CONFIG_LIBDIR"] = str(empty)
        env["PKG_CONFIG_PATH"] = str(empty)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"
    return env


def _first_error(stdout: str, stderr: str) -> str:
    for stream in (stderr, stdout):
        lines = (stream or "").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("CMake Error", "ERROR:")):
                # "CMake Error at x.cmake:12 (message):" says where, not what.
                # The sentence that says what is on the next line.
                detail = next(
                    (n.strip() for n in lines[index + 1:index + 4] if n.strip()), ""
                )
                if stripped.endswith(":") and detail:
                    return f"{stripped} {detail}"[:240]
                return stripped[:240]
    tail = (stdout or stderr or "").strip().splitlines()
    return tail[-1][:200] if tail else "configure did not complete"
