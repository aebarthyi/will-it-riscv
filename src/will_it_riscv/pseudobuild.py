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

import contextvars
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from .hostpython import HostPython

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
    parent: Optional[str] = None
    """The package whose own Find module or config file asked this -- the
    edge from CURL to PkgConfig. None when the project itself asked."""
    via: Optional[str] = None
    """The project's own function or macro that asked, such as GDAL's
    ``gdal_check_package``."""
    site: Optional[str] = None
    """Where in the project the question was asked, as ``path:line``
    relative to the source root."""
    round: int = 1
    """The configure attempt that first reached this probe. Anything past
    round one was only reachable once an earlier blocker had been stubbed."""


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
    found_at: dict = field(default_factory=dict)
    """Where the configure said it found each one. A path under a ``bin``
    directory is a host program; confined, nothing else real can be found."""
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
    lookups: dict = field(default_factory=dict)
    """What each package's Find module looked for in the last round, keyed
    by package: ``(command, variable, names)``. How a suspect gets stubbed
    exactly the way its own module searches for it."""
    searches: list = field(default_factory=list)
    """``(command, variable, names)``: every find_library, find_path,
    find_file and find_program this round, whoever ran it -- the project's
    own ``find_program(EXE_SQLITE3 sqlite3)`` included."""
    exists: list = field(default_factory=list)
    """``(path, module)``: files an ``if(EXISTS)`` asked about, and the
    .cmake file asking -- FindArmadillo checks for its version header
    before it reads one."""
    reads: list = field(default_factory=list)
    """``(file, regex)``: what the configure grepped files for, this round --
    ``file(STRINGS ... REGEX)``, and ``string(REGEX ...)`` on what a stub
    header said. A stub that does not match is given a line that does."""
    experiments: list = field(default_factory=list)
    """Suspects stubbed to find what a configure died of when it did not
    say, as ``(name, confirmed)``. Confirmed means the error moved once the
    suspect existed, which makes it a hard requirement shown by experiment."""
    notes: list = field(default_factory=list)
    python_installed: list = field(default_factory=list)
    """Build requirements put in for the host's interpreter because the
    configure imported them, as ``name==version``."""
    python_stubbed: list = field(default_factory=list)
    """Modules the configure imported that were stubbed instead, with why."""
    build_tools: list = field(default_factory=list)
    """Blockers a Python build requirement answers -- Cython, for a Meson
    build -- and so are the build's own business, not the target's."""
    narration: str = ""
    """The configure's own output, kept so the next round can read what it
    asked for. Not part of the report."""
    pruned: dict = field(default_factory=dict)
    """What taking each found package away did, once the configure had
    completed: name to ``(True, detail)`` when the configure then failed --
    required -- or ``(False, detail)`` when it completed without it."""

    @property
    def ok(self) -> bool:
        return bool(self.probes) or self.completed

    @property
    def confined(self) -> bool:
        return self.platform != "host"

    def reached(self, name: str) -> bool:
        return name.lower() in self.probes

    def reached_behind(self, probe: Probe) -> Optional[str]:
        """The blocker this probe was hidden behind, if it was.

        A probe first reached in round N+1 was only reachable because round
        N's blocker had been stubbed. That is a statement about the order the
        configure runs in, not about what depends on what.
        """
        index = probe.round - 2
        if 0 <= index < len(self.round_blockers):
            return self.round_blockers[index]
        return None


def cmake() -> Optional[str]:
    """The cmake to configure with: $WILL_IT_RISCV_CMAKE, then the one installed
    beside this Python (the pinned wheel), then whatever is on PATH."""
    chosen = os.environ.get("WILL_IT_RISCV_CMAKE")
    if chosen:
        return chosen
    beside = Path(sys.executable).parent / "cmake"
    if beside.is_file() and os.access(beside, os.X_OK):
        return str(beside)
    return shutil.which("cmake")


def available() -> bool:
    return cmake() is not None


#: CMake prefixes its STATUS output with "-- ". A miss reported that way is a
#: miss the configure shrugged off; the same words indented inside a CMake
#: Error block are the miss that stopped it.
_STATUS_MISS = re.compile(r"^-- +Could NOT find ([A-Za-z0-9_.+-]+)", re.MULTILINE)
#: FPHSA names the cache variables it wanted: "(missing: PROJ_LIBRARY ...)".
_MISSING_VARS = re.compile(r"\(missing:\s*([^)]*?)\s*\)")
#: A Find module reading a header that is not there names the exact path.
_MISSING_FILE = re.compile(
    r"file failed to open for reading \(No such file or directory\):\s*\n\s*(\S+)"
    # file(STRINGS) says it differently: FindBoost reading boost/version.hpp.
    r'|file\s+STRINGS\s+file\s+"([^"]+)"\s+cannot\s+be\s+read'
)
_STATUS_FOUND = re.compile(r"^-- +Found ([A-Za-z0-9_.+-]+)", re.MULTILINE)
_STATUS_FOUND_AT = re.compile(r"^-- +Found ([A-Za-z0-9_.+-]+): +(\S+)", re.MULTILINE)
#: FPHSA's two ways of saying so; the second is its HANDLE_COMPONENTS failure.
_ANY_MISS = re.compile(r"(?:Could NOT find|Failed to find all) ([A-Za-z0-9_.+-]+)")
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
#: find_program/find_library/find_path(... REQUIRED), CMake 3.18+: "Could not
#: find FYPP_EXE using the following names: fypp". MFC stops on exactly this.
_REQUIRED_FIND = re.compile(
    r"Could not find ([A-Za-z_][A-Za-z0-9_]*) using the following names:\s*([^\n]+)"
)
#: find_package(Boost 1.70 CONFIG REQUIRED), and nothing to find: cantera's.
_NO_CONFIG = re.compile(
    r'(?:Could not find|to find) a package configuration file provided by\s+"([^"]+)"'
)
#: FPHSA's CONFIG_MODE way of saying the same.
_NEITHER_CONFIG = re.compile(r"found neither ([A-Za-z0-9_+-]+)Config\.cmake nor")
#: The files it searched for, as CMake lists them under "the following names".
_CONFIG_NAMES = re.compile(r"^\s+([A-Za-z0-9_+.-]+?)(?:Config|-config)\.cmake\s*$", re.MULTILINE)
_UNKNOWN_COMMAND = re.compile(r'Unknown CMake command "([A-Za-z_][A-Za-z0-9_]*)"')
_MISSING_MODULE = re.compile(r"include could not find requested file:\s+(\S+)")
#: An error raised from inside a package's own Find module or config file is
#: that package's error, even when its message never names it.
_ERROR_FILE = re.compile(r"^\s*CMake Error at (\S+?):\d+")
_MODULE_FILE = re.compile(r"(?:^|/)(?:Find([A-Za-z0-9_+-]+)|([A-Za-z0-9_+-]+?)-?[Cc]onfig)\.cmake$")
#: FindOpenSSL creates OpenSSL::SSL only if OPENSSL_SSL_LIBRARY exists, and a
#: try_compile that links it then fails naming the target.
_TARGET_NOT_FOUND = re.compile(
    r"links to:\s+([A-Za-z0-9_+.-]+)::([A-Za-z0-9_+.-]+)\s+but the target was not found"
    # deal.II asks for a property of Kokkos::kokkos before anything links it.
    r'|non-existent\s+target\s+"([A-Za-z0-9_+.-]+)::([A-Za-z0-9_+.-]+)"'
)
#: A header a check could not find, named in the configure's own error.
_HEADER_NAME = re.compile(r"(?<![\w/.])([A-Za-z0-9_+-][A-Za-z0-9_+./-]*\.(?:h|hpp|hh|hxx))\b")
_HEADER_MISSING = re.compile(
    r"not found|cannot find|could not find|missing|no such file|not exist|not be found",
    re.IGNORECASE,
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


#: A GPU language the configure enabled, and no compiler for it. Nothing can
#: stub nvcc; that the default build turns CUDA on is the answer itself.
_GPU_LANGUAGE = (
    (re.compile(r"Failed to find nvcc|No CMAKE_CUDA_COMPILER could be found"), "CUDA"),
    (re.compile(r"No CMAKE_HIP_COMPILER could be found"), "HIP"),
)


def _stanza_blockers(stanza: str) -> list[str]:
    """What a single error stanza says was missing, in the order it says it."""
    names: list[str] = []
    lines = stanza.splitlines()
    flat = re.sub(r"\s+", " ", stanza)
    for pattern, language in _GPU_LANGUAGE:
        if pattern.search(flat):
            return [language]
    if _NOTFOUND_HEADER in stanza:
        for line in lines[1:]:
            match = _NOTFOUND_VAR.match(line)
            if match and not line.startswith("Please"):
                names.append(_VAR_STEM.sub("", match.group(1)) or match.group(1))
        return names
    for match in _REQUIRED_FIND.finditer(stanza):
        first = match.group(2).split(",")[0].strip()
        names.append(first or match.group(1))
    names += _NO_CONFIG.findall(re.sub(r"\s+", " ", stanza))
    names += _NEITHER_CONFIG.findall(re.sub(r"\s+", " ", stanza))
    names += _NO_FIND_MODULE.findall(re.sub(r"\s+", " ", stanza))
    # "RAJA_DIR was not provided": RAJA. "set MPI_BASE_DIR": MPI. A _PATH or
    # _HOME is a place, not a package: TRITON_CACHE_PATH.
    names += [_variable_package(v) for v in _named_variables(stanza)
              if re.search(r"_(DIR|ROOT)$", v)]
    for command in _UNKNOWN_COMMAND.findall(stanza):
        # nanobind_add_module is nanobind's: its config defines it.
        names.append(command.split("_", 1)[0] if "_" in command else command)
    if names:
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
        names.append(match.group(1) or match.group(3))
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
    name = module.group(1) or module.group(2)
    # FPHSA raises every Find module's failure from its own file.
    return None if name in ("PackageHandleStandardArgs", "PackageMessage") else name


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


#: Commands a Find module uses to look for the files a package consists of.
LOOKUP_COMMANDS = {
    "find_library", "find_path", "find_file", "pkg_check_modules", "pkg_search_module",
}
_SEARCH_COMMANDS = ("find_library", "find_path", "find_file", "find_program")


def _parse_trace(
    path: Path, root: Optional[Path] = None, lookups: Optional[dict] = None,
    reads: Optional[list] = None, exists: Optional[list] = None,
    searches: Optional[list] = None,
) -> tuple[dict[str, Probe], int]:
    """Every dependency question in the trace, and who asked it.

    The json-v1 trace numbers every command's depth in the whole call stack
    (``global_frame``), so the chain of callers above a find_package can be
    rebuilt exactly: which of the project's macros asked, from which line,
    and whether it was really another package's Find module asking.
    """
    probes: dict[str, Probe] = {}
    chains: dict[str, list[str]] = {}
    traced = 0
    stack: list[dict] = []
    wrappers: set[str] = set()
    prefixes = _prefixes(root)
    last_read: list = []   # the file the last file(READ) read
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
            depth = event.get("global_frame")
            if isinstance(depth, int) and depth >= 1:
                del stack[depth - 1:]
                ancestors = list(stack)
                stack.append(event)
            else:   # CMake < 3.21: no call stack, so no provenance
                ancestors = []
            if command in ("function", "macro"):
                # Only the project's own. CMake's find_dependency is a macro
                # too, and "asked via find_dependency" says nothing.
                defined = event.get("args") or []
                if defined and _inside(str(event.get("file", "")), prefixes):
                    wrappers.add(str(defined[0]).lower())
                continue
            if searches is not None and command in _SEARCH_COMMANDS:
                found_args = [str(a) for a in event.get("args", []) if str(a).strip()]
                if len(found_args) > 1:
                    entry = (command, found_args[0], tuple(_lookup_names(found_args[1:])))
                    if entry[2] and entry not in searches:
                        searches.append(entry)
            if command in ("if", "elseif"):
                args = [str(a) for a in event.get("args", [])]
                for at, word in enumerate(args[:-1]):
                    if word == "EXISTS" and exists is not None:
                        exists.append((args[at + 1], str(event.get("file", ""))))
                    elif word == "MATCHES" and reads is not None:
                        # FindHDF5: if("${HDF5_VERSION_DEFINE}" MATCHES "H5_VERSION ...")
                        _note_read("if", ["MATCHES", args[at + 1], args[at - 1] if at else ""],
                                   reads, last_read)
                continue
            if reads is not None and command in ("file", "string"):
                _note_read(command, [str(a) for a in event.get("args", [])], reads, last_read)
                continue
            if command not in FIND_COMMANDS and command not in LOOKUP_COMMANDS:
                continue
            args = [str(a) for a in event.get("args", []) if str(a).strip()]
            if not args:
                continue
            if lookups is not None and command in LOOKUP_COMMANDS:
                owner = _owner(ancestors)
                if owner:
                    entry = (command, args[0], tuple(_lookup_names(args[1:])))
                    listed = lookups.setdefault(owner, [])
                    if entry not in listed:
                        listed.append(entry)
            if command not in FIND_COMMANDS:
                continue
            upper = [a.upper() for a in args]
            for name in _subjects(command, args):
                key = name.lower()
                existing = probes.get(key)
                required = "REQUIRED" in upper
                if existing is not None:
                    if required and not existing.required:
                        probes[key] = replace(existing, required=True)
                    continue
                parent, via = _asker(key, ancestors, wrappers)
                chains[key] = _project_sites([event, *reversed(ancestors)], prefixes)
                probes[key] = Probe(
                    name=name,
                    command=command,
                    required=required,
                    quiet="QUIET" in upper,
                    parent=parent,
                    via=via,
                )
    sites = _choose_sites(chains)
    for key, site in sites.items():
        probes[key] = replace(probes[key], site=site)
    return probes, traced


#: What every stub header starts with, so a regex run over one is known.
STUB_MARK = "stub emitted by will-it-riscv"


#: How many trace events after reading a file a string(REGEX) is still taken
#: to be parsing what it read.
_READ_REACH = 40


def _note_read(command: str, args: list[str], reads: list, last_read: list) -> None:
    """Record a grep of a file: which file, and for what.

    ``file(STRINGS ... REGEX)`` names both. A ``string(REGEX ...)`` soon
    after a ``file(READ)`` or ``file(STRINGS)`` is taken to be parsing what
    was read when it is over a stub's text, or looks for a version or a
    ``#define`` -- FindArmadillo greps every ARMA_VERSION_ line, then takes
    MAJOR, MINOR and PATCH apart from those lines with REPLACE.
    """
    word = [a.upper() for a in args[:2]]
    if last_read:
        last_read[1] += 1
    if command == "file" and word[:1] in (["READ"], ["STRINGS"]) and len(args) > 1:
        last_read[:] = [args[1], 0]
        if word[0] == "STRINGS" and "REGEX" in args[2:]:
            at = args.index("REGEX", 2)
            if at + 1 < len(args):
                reads.append((args[1], args[at + 1]))
    elif (command == "string" and word[:1] == ["REGEX"] and len(args) > 3
          or command == "if") and last_read:
        # string(REGEX MATCH <rx> <out> <input>...), REPLACE has <replacement> too;
        # if(<input> MATCHES <rx>) arrives as ["MATCHES", <rx>, <input>].
        if command == "if":
            given, regex = args[2:], args[1]
        elif word[1] in ("MATCH", "MATCHALL"):
            given, regex = args[4:], args[2]
        elif word[1] == "REPLACE":
            given, regex = args[5:], args[2]
        else:
            return
        parsing = any(STUB_MARK in a for a in given) or (
            last_read[1] <= _READ_REACH and re.search(r"version|#define", regex, re.IGNORECASE)
        )
        if parsing and (last_read[0], regex) not in reads:
            reads.append((last_read[0], regex))


def _prefixes(root: Optional[Path]) -> tuple[str, ...]:
    if root is None:
        return ()
    spellings = {str(root), str(Path(root).resolve())}
    return tuple(s.rstrip("/") + "/" for s in spellings)


def _inside(file: str, prefixes: tuple[str, ...]) -> bool:
    return bool(prefixes) and file.startswith(prefixes)


def _owner(ancestors: list) -> Optional[str]:
    """The package whose find_package this lookup is running inside."""
    for caller in reversed(ancestors):
        if str(caller.get("cmd", "")).lower() in ("find_package", "find_dependency"):
            asked = (caller.get("args") or [""])[0]
            return str(asked).lower() or None
    return None


def _lookup_names(args: list[str]) -> list[str]:
    """The file or module names a lookup was after, up to its first keyword.

    A list variable arrives as one argument -- FindZLIB's NAMES ${ZLIB_NAMES}
    is "z;zlib;zdll;zlib1;..." -- so arguments are split on semicolons.
    """
    names: list[str] = []
    for arg in [part for given in args for part in given.split(";")]:
        upper = arg.upper()
        if upper in _STOP_KEYWORDS:
            break
        if upper in _SKIP_KEYWORDS or not arg or arg.startswith(("$", "-", "/", "[")):
            continue
        if re.match(r"^[A-Za-z0-9_+.-][A-Za-z0-9_+./<>=-]*$", arg) and ".." not in arg:
            names.append(arg)
    return names


def _asker(key: str, ancestors: list, wrappers: set) -> tuple[Optional[str], Optional[str]]:
    """The package and the project macro that asked, innermost first."""
    via = None
    for caller in reversed(ancestors):
        command = str(caller.get("cmd", "")).lower()
        if command in ("find_package", "find_dependency"):
            asked = (caller.get("args") or [""])[0]
            if str(asked).lower() != key:
                return str(asked), via
            continue   # FindCURL retrying CURL in config mode is still CURL
        if via is None and command in wrappers:
            via = command
    return None, via


def _project_sites(events: list, prefixes: tuple[str, ...]) -> list[str]:
    """``path:line`` of every frame inside the project, innermost first."""
    sites: list[str] = []
    for event in events:
        file = str(event.get("file", ""))
        if not _inside(file, prefixes):
            continue
        relative = file
        for prefix in prefixes:
            if file.startswith(prefix):
                relative = file[len(prefix):]
                break
        site = f"{relative}:{event.get('line', 0)}"
        if not sites or sites[-1] != site:
            sites.append(site)
    return sites


#: A line that asks for this many different things is a wrapper's body, not
#: the place anybody decided to depend on something.
_WRAPPER_FANOUT = 3


def _choose_sites(chains: dict[str, list[str]]) -> dict[str, str]:
    """Pick, for each probe, the line that actually decided to ask.

    The innermost project frame is usually right. When the project routes
    every lookup through one macro, it is the same line for fifty packages
    -- GDAL's ``find_package`` inside ``gdal_check_package`` -- so step out
    to whichever line called the macro, while that narrows things down.
    """
    fanout: dict[str, int] = {}
    for chain in chains.values():
        for site in set(chain):
            fanout[site] = fanout.get(site, 0) + 1
    chosen: dict[str, str] = {}
    for key, chain in chains.items():
        if not chain:
            continue
        index = 0
        while (
            index + 1 < len(chain)
            and fanout[chain[index]] >= _WRAPPER_FANOUT
            and fanout[chain[index + 1]] < fanout[chain[index]]
        ):
            index += 1
        chosen[key] = chain[index]
    return chosen


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
    max_rounds: int = 30,
    arch: str = "riscv64",
    confine: bool = True,
    defines: Optional[dict] = None,
    python_dists: Optional[dict] = None,
    pip_cache: Optional[Path] = None,
    prune: bool = False,
) -> Optional[PseudoBuild]:
    """Configure the project in a scratch directory, unblocking as it goes.

    A configure that stops tells you one thing: what stopped it. Satisfy that
    with a stub and run it again, and it tells you the next thing -- and once
    it finally runs to the end, everything it did *not* ask for is known to be
    optional. GDAL stops at PROJ and yields 7 proven-optional dependencies;
    two rounds later it completes and yields 55.

    ``timeout`` bounds the whole loop, not each round. ``defines`` are -D
    flags the build itself passes -- MFC's -DMFC_MPI=ON -- and are never
    stubbed over. ``python_dists`` is what the plan installs, name to
    version: a configure that imports one of them from the interpreter it is
    given gets it, for the host (see :mod:`hostpython`). Returns None when
    there is nothing to do (no CMakeLists.txt at the root).
    """
    root = Path(root)
    if not (root / "CMakeLists.txt").exists():
        return None
    if not available():
        return PseudoBuild(error="cmake is not installed")

    started = time.monotonic()
    aggregate = PseudoBuild(rounds=0)
    negotiation = _NEGOTIATED.set({})
    overrides: dict[str, str] = {}
    written: set = set()
    trial: Optional[_Trial] = None
    tried: set = set()
    answered: set = set()   # (file, regex) given a matching line already
    fixed = {str(k): str(v) for k, v in (defines or {}).items()}

    with tempfile.TemporaryDirectory(prefix="will-it-riscv-") as scratch:
        scratch_path = Path(scratch)
        sysroot = scratch_path / "sysroot"
        for directory in ("include", "lib/pkgconfig", "bin"):
            (sysroot / directory).mkdir(parents=True, exist_ok=True)
        env = _environment(scratch_path, deny_pkg_config, sysroot)
        toolchain = _toolchain(scratch_path, sysroot, arch, root) if confine else None
        aggregate.platform = f"linux/{arch}" if toolchain else "host"
        # What the build fetches for itself, fetched once for every round:
        # cantera would download eigen, yaml-cpp and SUNDIALS each time round.
        given = {
            "FETCHCONTENT_BASE_DIR": str(scratch_path / "_deps"),
            "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
        }
        # The build host's Python, pinned from the start the way
        # scikit-build-core pins it, so every import a configure makes of it
        # is seen -- and answered.
        interpreter = HostPython.create(scratch_path / "python", python_dists, pip_cache)
        for prefix in ("Python", "Python3", "PYTHON"):
            given[f"{prefix}_EXECUTABLE"] = str(interpreter.executable)

        for attempt in range(1, max_rounds + 1):
            left = timeout - (time.monotonic() - started)
            if left <= 0:
                aggregate.error = f"ran out of time after {attempt - 1} rounds"
                break
            remaining = max(1, int(left))
            outcome = _configure(
                root, scratch_path, {**given, **overrides, **fixed}, env, remaining,
                f"build-{attempt}", toolchain,
            )
            if attempt == 1 and toolchain and not outcome.completed and not outcome.probes:
                # It would not even start as a cross build. A host answer is
                # worse than a target one, but much better than none.
                host = _configure(
                    root, scratch_path, {**given, **overrides, **fixed}, env, remaining,
                    "build-host", None,
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
            for key, probe in outcome.probes.items():
                existing = aggregate.probes.get(key)
                if existing is None:
                    aggregate.probes[key] = replace(probe, round=attempt)
                elif probe.required and not existing.required:
                    aggregate.probes[key] = replace(existing, required=True)
            aggregate.found |= outcome.found
            for name, where in outcome.found_at.items():
                aggregate.found_at.setdefault(name, where)
            aggregate.soft_misses |= outcome.soft_misses
            aggregate.commands_traced += outcome.commands_traced
            aggregate.completed = outcome.completed
            aggregate.error = outcome.error

            if trial is not None:
                # Did stubbing the suspect move the error? Then it was what
                # the previous round died of. If not, it was innocent: take
                # every trace of it back out, or later rounds would find it.
                if outcome.completed or _signature(outcome.error) != trial.signature:
                    for name in (trial.name, *trial.also):
                        if name not in aggregate.blockers:
                            aggregate.blockers.append(name)
                        aggregate.experiments.append((name, True))
                    aggregate.round_blockers[-1] = trial.name
                    tried = set()
                    trial = None
                else:
                    # Not moved yet -- but if its stubs are still being
                    # finished (a version grep to answer, a version header to
                    # write), give it the round that takes before judging.
                    stanzas = _error_stanzas(outcome.narration)
                    pending = _answer_reads(outcome.reads, sysroot, answered,
                                            stanzas[0] if stanzas else "")
                    made = _version_headers(outcome, sysroot, [trial.name, *trial.also])
                    if trial.prefix:
                        for search in outcome.searches:
                            if search[1].startswith(trial.prefix) and search[1] not in overrides \
                                    and search[2]:
                                more, files = _stub_search(
                                    search, sysroot, ".so" if toolchain else _library_suffix())
                                overrides.update(more)
                                trial.overrides.update(more)
                                made += files
                                pending += list(more)
                    if (pending or made) and trial.extended < 3:
                        trial.extended += 1
                        trial.files += [m for m in made if m not in trial.files]
                        written.update(pending + made)
                        continue
                    for key in trial.overrides:
                        overrides.pop(key, None)
                    for path in trial.files:
                        Path(path).unlink(missing_ok=True)
                        written.discard(path)
                    for name in (trial.name, *trial.also):
                        aggregate.experiments.append((name, False))
                    outcome.narration = trial.narration
                    aggregate.unblocked = sorted(overrides)
                    trial = None

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
                shared_suffix=suffix, lookups=outcome.lookups,
                interpreter=str(interpreter.executable),
            )
            stanzas = _error_stanzas(outcome.narration)
            for changed in _answer_reads(outcome.reads, sysroot, answered,
                                         stanzas[0] if stanzas else ""):
                created.append(changed)
            created += _version_headers(outcome, sysroot, aggregate.blockers)
            created += _negotiate(outcome.narration, sysroot, answered, written)
            imported = interpreter.provide()
            if imported and aggregate.round_blockers[-1] is None:
                # Nothing CMake looked for stopped it: its own interpreter
                # did, on an import. That is a build requirement, not a
                # dependency of the target.
                aggregate.round_blockers[-1] = f"import {imported[0]}"
            if _OLD_POLICY.search(outcome.narration) and POLICY_MINIMUM not in fixed:
                # CMake 4 dropped compatibility with projects that ask for
                # less than 3.5; Debian 13's CMake is 3.31 and still has it.
                # This is the host's CMake being newer, not a dependency.
                fresh[POLICY_MINIMUM] = "3.5"
                if aggregate.round_blockers[-1] is None:
                    aggregate.round_blockers[-1] = "CMake < 3.5"
                note = (
                    f"{_policy_culprit(outcome.error)} asks for a CMake older than 3.5, "
                    "which this host's CMake no longer accepts; it was configured "
                    f"with -D{POLICY_MINIMUM}=3.5, as CMake itself suggests"
                )
                if note not in aggregate.notes:
                    aggregate.notes.append(note)
            fresh = {
                k: v for k, v in fresh.items() if overrides.get(k) != v and k not in fixed
            }
            if not fresh and not created and not imported:
                # The configure died without naming anything this can fake.
                # Blame by experiment: stub the likeliest suspect and see.
                trial = _next_trial(
                    outcome, tried, aggregate.blockers, overrides, sysroot, suffix,
                    str(interpreter.executable),
                )
                if trial is None:
                    break   # nothing left to try; the configure is stuck here
                tried.add(trial.name)
                fresh, created = trial.overrides, trial.files
            overrides.update(fresh)
            written.update(created)
            aggregate.unblocked = sorted(overrides)

        if prune and aggregate.completed:
            _prune(
                aggregate, root, scratch_path, {**given, **overrides, **fixed}, env,
                toolchain, lambda: timeout - (time.monotonic() - started),
            )
        aggregate.python_installed = list(interpreter.installed)
        aggregate.python_stubbed = list(interpreter.stubbed)

    _NEGOTIATED.reset(negotiation)
    aggregate.blocking = aggregate.blockers[0] if aggregate.blockers else None
    # A blocker that a later round walked past is still a hard requirement,
    # but it no longer stops anything -- and it was "found" only because a
    # stub was put there.
    aggregate.soft_misses -= set(aggregate.blockers)
    aggregate.found -= set(aggregate.blockers)
    aggregate.duration = time.monotonic() - started
    return aggregate


def _prune(aggregate: PseudoBuild, root: Path, scratch: Path, settings: dict, env: dict,
           toolchain: Optional[Path], left) -> None:
    """Settle every package the completed configure found, by taking it away.

    A found package says nothing about whether the build needs it: nobody saw
    the configure without it. CMAKE_DISABLE_FIND_PACKAGE_<Name> makes one
    find_package act as if nothing were there -- a REQUIRED one then fails --
    so one configure per package answers it: completed means optional,
    failed means required. Stubbed blockers need no pruning; they already
    stopped a round.
    """
    blockers = {b.lower() for b in aggregate.blockers}
    found = {f.lower() for f in aggregate.found}
    names = sorted({
        probe.name for key, probe in aggregate.probes.items()
        if probe.parent is None and probe.command == "find_package"
        and key in found and key not in blockers
    })
    for index, name in enumerate(names):
        remaining = int(left())
        if remaining <= 0:
            aggregate.notes.append(f"ran out of time pruning; {len(names) - index} left unsettled")
            return
        outcome = _configure(
            root, scratch, {**settings, f"CMAKE_DISABLE_FIND_PACKAGE_{name}": "TRUE"}, env,
            remaining, f"prune-{index}", toolchain,
        )
        if outcome.completed:
            aggregate.pruned[name] = (False, "the configure completed with it disabled")
        else:
            aggregate.pruned[name] = (True, "the configure failed with it disabled: "
                                      + (outcome.error or "")[:160])


# --------------------------------------------------------- blame by experiment

#: How many suspects to try at one stuck point before giving up on it.
MAX_SUSPECTS = 3

_STATUS_MISS_LINE = re.compile(r"^-- +Could NOT find ([A-Za-z0-9_.+-]+)(.*)$")


@dataclass
class _Trial:
    """A suspect stubbed to see whether the error moves."""

    name: str
    signature: str
    overrides: dict
    files: list
    narration: str
    """The stuck round's narration, to pick the next suspect from if this
    one turns out to be innocent."""
    extended: int = 0
    """Rounds it was given beyond the first, while its stubs were still being
    finished: FindZLIB greps the zlib.h it was just given for its version."""
    prefix: Optional[str] = None
    """For lookups the error named: the variable prefix whose later lookups
    are its too -- mlpack looks for ARMADILLO_LIBRARY once it has the header."""
    also: list = field(default_factory=list)
    """Stubbed with it and blamed with it: the other dependencies a refused
    package's own module asked for (FindX11's Xft wants Freetype and
    Fontconfig both)."""


def _signature(error: Optional[str]) -> str:
    """An error with the parts that change on every run taken out."""
    text = error or ""
    text = re.sub(r"cmTC_\w+|TryCompile-\w+", "", text)
    text = re.sub(r"\S*will-it-riscv-[^\s/]+\S*", "", text)
    return text.strip()


def _mentions(message: str, name: str) -> bool:
    """Whether an error message names this package, as itself or as a lib --
    or without the major version its name carries: "requires Python >= 3.6"
    names Python3, "Eigen not found" Eigen3."""
    spellings = [name.lower()]
    versioned = re.match(r"^([a-z][a-z0-9_+-]*?[a-z])([0-9]+)$", name.lower())
    if versioned:
        # "Qt 5.15 was not found" names Qt5; "a valid Qt installation", Qt6 and Qt5.
        spellings += [f"{versioned.group(1)} {versioned.group(2)}", versioned.group(1)]
    return any(
        re.search(rf"(?<![a-z0-9])(?:lib)?{re.escape(s)}(?![a-z0-9])", message.lower())
        for s in spellings
    )


def _suspects(narration: str, probes: Optional[dict] = None) -> list[tuple[str, list[str]]]:
    """Packages the configure missed before it died, likeliest first.

    GROMACS narrates "-- Could NOT find OpenMP" and then fails in its own
    words -- "does not support OpenMP parallelism". A miss the error message
    names comes first; after that, the nearer to the error, the likelier.
    """
    misses: list[tuple[str, str]] = []
    for line in narration.splitlines():
        if line.startswith(("CMake Error", "  CMake Error")):
            break
        match = _STATUS_MISS_LINE.match(line)
        if match:
            misses.append((match.group(1), match.group(2)))
    names = {name for name, _ in misses}
    grouped: dict[str, list[str]] = {}
    for name, detail in reversed(misses):
        head = name.split("_", 1)[0]
        root = head if head != name and head in names else name
        grouped.setdefault(root, []).append(detail)
    stanzas = _error_stanzas(narration)
    message = stanzas[0].lower() if stanzas else ""
    named = [root for root in grouped if _mentions(message, root)]
    # A package looked for without FPHSA never says "Could NOT find" at
    # all -- GROMACS's FindFFTW is silent, and only the project's own error
    # names it. The trace still saw the find_package.
    for probe in (probes or {}).values():
        if probe.command != "find_package" or probe.name in grouped:
            continue
        if _mentions(message, probe.name):
            grouped[probe.name] = []
            named.append(probe.name)
    ranked = named + [root for root in grouped if root not in named]
    return [(root, grouped[root]) for root in ranked]


def _named_searches(narration: str, searches: list) -> list[tuple[str, list]]:
    """Lookups the error names in its own words: ``find_program(EXE_SQLITE3
    sqlite3)`` and then "sqlite3 binary not found!", with no package
    involved. Named by the name the error used, which goes first."""
    stanzas = _error_stanzas(narration)
    message = stanzas[0].lower() if stanzas else ""
    named: list[tuple[str, tuple]] = []
    seen: set = set()
    for search in searches:
        command, variable, names = search
        if variable.upper().startswith(("CMAKE_", "_CMAKE")):
            # CMake's own: a stub make program would pass every try_compile,
            # and an error quoting a try_compile's log always names gmake.
            continue
        if re.match(r"^python[0-9.]*$", names[0], re.IGNORECASE):
            continue   # the interpreter is pinned, and its headers are _stub_python's
        for name in names:
            if name.startswith("lib") and command == "find_library" and len(name) > 3:
                name = name[3:]
            stem = re.sub(r"[/.].*$", "", name)
            if stem.lower() in seen or len(stem) < 2:
                continue
            if _mentions(message, stem) or _mentions(message, variable):
                seen.add(stem.lower())
                # Stubbed as the name the error used: pocl's llvm-config, not llvmtce-config.
                first = (command, variable, (name, *[n for n in names if n != name]))
                named.append((stem, [first]))
                break
    # mlpack's own find_armadillo looks for ARMADILLO_INCLUDE_DIR and then
    # ARMADILLO_LIBRARY: the lookups of one prefix are stubbed together.
    for _, group in named:
        prefix = group[0][1].split("_", 1)[0] + "_"
        if len(prefix) > 2:
            group += [s for s in searches if s[1].startswith(prefix) and s[1] != group[0][1]
                      and s[2] and not s[1].upper().startswith(("CMAKE_", "_CMAKE"))]
    return named


def _stub_search(search: tuple, sysroot: Path, suffix: str) -> tuple[dict, list]:
    """Fill a lookup's own variable with a stub of what it looked for."""
    command, variable, names = search
    name = names[0]
    try:
        # Only what this made is the trial's to take back out.
        if command == "find_library":
            base = name[3:] if name.startswith("lib") and len(name) > 3 else name
            path = sysroot / "lib" / f"lib{base}{suffix}"
            made = [] if path.exists() else [str(path)]
            path.touch()
            return {variable: str(path)}, made
        if command == "find_program":
            path = sysroot / "bin" / Path(name).name
            made = [] if path.exists() else [str(path)]
            if made:
                path.write_text(_fake_program(path.name, sysroot.parent))
                path.chmod(0o755)
            return {variable: str(path)}, made
        path = sysroot / "include" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        made = [] if path.exists() else [str(path)]
        if made:
            path.write_text(_fake_header(Path(name).stem))
        value = str(path) if command == "find_file" else str(path.parent)
        if command == "find_path" and "/" in name:
            value = str(sysroot / "include")   # find_path(... openssl/ssl.h) gives the root
        return {variable: value}, made
    except OSError:
        return {}, []


def _next_trial(
    outcome: PseudoBuild,
    tried: set,
    blockers: list,
    overrides: dict,
    sysroot: Path,
    suffix: str,
    interpreter: Optional[str] = None,
) -> Optional[_Trial]:
    if len(tried) >= MAX_SUSPECTS:
        return None
    suspects = _suspects(outcome.narration, outcome.probes)
    stanzas = _error_stanzas(outcome.narration)
    message = stanzas[0].lower() if stanzas else ""
    # Packages the error names, then lookups it names, then the other misses.
    named = [(n, d, None) for n, d in suspects if _mentions(message, n)]
    rest = [(n, d, None) for n, d in suspects if not _mentions(message, n)]
    searched = [(n, [], group) for n, group in _named_searches(outcome.narration,
                                                               outcome.searches)]
    for name, details, search in named + searched:
        if name in tried or name in blockers or name in outcome.found:
            continue
        prefix = None
        if search is not None:
            stubbed, files = {}, []
            for one in search:
                more, made = _stub_search(one, sysroot, suffix)
                stubbed.update(more)
                files += made
            prefix = search[0][1].split("_", 1)[0] + "_"
        else:
            stubbed, files = _stub_package(
                name, details, sysroot, suffix, outcome.lookups.get(name.lower(), []),
                interpreter,
            )
        stubbed = {k: v for k, v in stubbed.items() if overrides.get(k) != v}
        if not stubbed and not files:
            tried.add(name)
            continue
        return _Trial(
            name=name,
            signature=_signature(outcome.error),
            overrides=stubbed,
            files=files,
            narration=outcome.narration,
            prefix=prefix if prefix and len(prefix) > 2 else None,
        )
    group = _nested_trial(outcome, stanzas[0] if stanzas else "", tried, blockers, overrides,
                          sysroot, suffix)
    if group is not None:
        return group
    for name, details, _ in rest:
        if name in tried or name in blockers or name in outcome.found:
            continue
        stubbed, files = _stub_package(
            name, details, sysroot, suffix, outcome.lookups.get(name.lower(), []), interpreter,
        )
        stubbed = {k: v for k, v in stubbed.items() if overrides.get(k) != v}
        if not stubbed and not files:
            tried.add(name)
            continue
        return _Trial(
            name=name,
            signature=_signature(outcome.error),
            overrides=stubbed,
            files=files,
            narration=outcome.narration,
        )
    return None


def _nested_trial(
    outcome: PseudoBuild, stanza: str, tried: set, blockers: list, overrides: dict,
    sysroot: Path, suffix: str,
) -> Optional[_Trial]:
    """Every package a refused package's own module asked for, stubbed together.

    "Could NOT find X11 (missing: Xft)" with Xft's header and library both
    there: FindX11 counts Xft found only if Freetype and Fontconfig are,
    and asks for them QUIET, so nothing names them. One at a time, neither
    moves the error.
    """
    refused = {m.lower() for m in _ANY_MISS.findall(stanza)}
    nested = [
        probe.name for probe in outcome.probes.values()
        if probe.command == "find_package" and probe.parent
        and probe.parent.lower() in refused and probe.name.lower() not in refused
        and probe.name not in tried and probe.name not in blockers
    ]
    nested = list(dict.fromkeys(nested))
    if not nested:
        return None
    stubbed: dict = {}
    files: list = []
    for name in nested:
        more, made = _stub_package(name, [], sysroot, suffix, outcome.lookups.get(name.lower(), []))
        stubbed.update(more)
        files += made
        tried.add(name)
    stubbed = {k: v for k, v in stubbed.items() if overrides.get(k) != v}
    if not stubbed and not files:
        return None
    return _Trial(
        name=nested[0], also=nested[1:], signature=_signature(outcome.error),
        overrides=stubbed, files=files, narration=outcome.narration,
    )


def _stub_package(
    name: str, details: list[str], sysroot: Path, suffix: str, lookups: Optional[list] = None,
    interpreter: Optional[str] = None,
) -> tuple[dict, list]:
    """Make a package the configure shrugged off exist after all."""
    if name.lower() in _PYTHON_PACKAGES:
        return _stub_python("", sysroot, suffix, interpreter)
    if name.lower() == "boost":
        # As when FPHSA names it: FindBoost defers to a BoostConfig.cmake.
        files = _stub_config_package("Boost", sysroot, suffix)
        return {"Boost_DIR": str(sysroot / "lib" / "cmake" / "Boost")}, files
    if name.lower() == "openmp":
        return _stub_openmp(sysroot, suffix)
    if name.lower() == "mpi":
        return _stub_mpi(sysroot, suffix)
    overrides: dict[str, str] = {}
    files: list[str] = _stub_lookups(name, lookups or [], sysroot, suffix)
    wants_config = False
    for detail in details:
        match = _MISSING_VARS.search(detail)
        if not match:
            continue
        for token in match.group(1).split():
            if "_" not in token or token.endswith("_FOUND"):
                continue
            if _CONFIG_DIR_VAR.match(token) and not _INCLUDE_VAR.search(token):
                wants_config = True
                continue
            value = _stub_variable(token, sysroot, suffix)
            if value is None:
                continue
            overrides[token] = value
            if value.startswith(str(sysroot)) and Path(value).is_file():
                files.append(value)
    if wants_config or not (overrides or files):
        # Nothing of its own module's to put in place: a config package then,
        # as paraview's find_package(Qt6 QUIET COMPONENTS Core) looks for.
        files += _stub_config_package(name, sysroot, suffix)
        if files:
            overrides[f"{name}_DIR"] = str(sysroot / "lib" / "cmake" / name)
    return overrides, files


def _stub_lookups(name: str, lookups: list, sysroot: Path, suffix: str) -> list[str]:
    """Put in the sysroot exactly what the package's Find module looked for.

    The find root is the sysroot, so a library at ``<sysroot>/lib`` and a
    header at ``<sysroot>/include`` are found by the module's own search --
    no cache variable needed, and nothing to take back but the files.
    """
    files: list[str] = []

    def write(path: Path, text: Optional[str] = None) -> None:
        if path.exists():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text) if text is not None else path.touch()
        except OSError:
            return
        files.append(str(path))

    for command, _variable, names in lookups:
        if not names:
            continue
        first = names[0]
        if command == "find_library":
            base = first[3:] if first.startswith("lib") and len(first) > 3 else first
            write(sysroot / "lib" / f"lib{base}{suffix}")
        elif command in ("find_path", "find_file"):
            write(sysroot / "include" / first, _fake_header(name))
        else:   # pkg_check_modules / pkg_search_module
            module = re.split(r"[<>=]", first, maxsplit=1)[0]
            if module and not (sysroot / "lib" / "pkgconfig" / f"{module}.pc").exists():
                files += _stub_pkgconfig(module, sysroot, suffix)
    return files


def _stub_config_package(name: str, sysroot: Path, suffix: str) -> list[str]:
    """A <Name>Config.cmake that claims the package, for config-mode lookups.

    It lives under the sysroot the find root is confined to, so a plain
    find_package(Name CONFIG) finds it without any hint.
    """
    if not re.match(r"^[A-Za-z0-9_+-]+$", name):
        return []
    directory = sysroot / "lib" / "cmake" / name
    library = sysroot / "lib" / f"lib{name.lower()}{suffix}"
    major, minor, patch = _fake_version(name)
    dotted = f"{major}.{minor}.{patch}"
    modules = _modules_directory(sysroot).as_posix()
    lines = [
        f"# stub emitted by will-it-riscv for {name}", f"set({name}_FOUND TRUE)",
        # A package that ships CMake modules puts them on the path, or says
        # where they are: ECM_MODULE_PATH, which KDE projects set it from.
        f'list(APPEND CMAKE_MODULE_PATH "{modules}")',
    ]
    use_file = _modules_directory(sysroot) / f"Use{name}.cmake"
    for prefix in dict.fromkeys([name, name.upper()]):
        lines += [
            f'set({prefix}_MODULE_PATH "{modules}")',
            # include(${Geant4_USE_FILE}), as VTK, ITK and ROOT have it too.
            f'set({prefix}_USE_FILE "{use_file.as_posix()}")',
            f'set({prefix}_VERSION "{dotted}")',
            f"set({prefix}_VERSION_MAJOR {major})",
            f"set({prefix}_VERSION_MINOR {minor})",
            f"set({prefix}_VERSION_PATCH {patch})",
            # CGAL_MAJOR_VERSION and CGAL_BUGFIX_VERSION, as CGAL spells them.
            f"set({prefix}_MAJOR_VERSION {major})",
            f"set({prefix}_MINOR_VERSION {minor})",
            f"set({prefix}_PATCH_VERSION {patch})",
            f"set({prefix}_BUGFIX_VERSION {patch})",
            f'set({prefix}_INCLUDE_DIRS "{sysroot.as_posix()}/include")',
            f'set({prefix}_INCLUDE_DIR "{sysroot.as_posix()}/include")',
            f'set({prefix}_LIBRARIES "{library.as_posix()}")',
            f'set({prefix}_LIBRARY_DIRS "{sysroot.as_posix()}/lib")',
        ]
    # Whatever components were asked for are here, each with its target:
    # find_package(Boost COMPONENTS program_options) links Boost::program_options.
    where = f'IMPORTED_LOCATION "{library.as_posix()}" INTERFACE_INCLUDE_DIRECTORIES "{sysroot.as_posix()}/include"'
    targets = [f"{name}::{name}", f"{name}::{name.lower()}"]
    if name == "Boost":
        targets += ["Boost::headers", "Boost::boost", "Boost::disable_autolinking",
                    "Boost::dynamic_linking", "Boost::diagnostic_definitions"]
        lines.append(f'set(Boost_INCLUDE_DIRS "{sysroot.as_posix()}/include")')
    for target in dict.fromkeys(targets):
        if target in ("Boost::headers", "Boost::boost"):
            # Header-only, as the real ones are: spectre adds headers to them.
            lines += [f"if(NOT TARGET {target})", f"  add_library({target} INTERFACE IMPORTED)",
                      f"  set_target_properties({target} PROPERTIES "
                      f'INTERFACE_INCLUDE_DIRECTORIES "{sysroot.as_posix()}/include")', "endif()"]
            continue
        lines += [f"if(NOT TARGET {target})", f"  add_library({target} UNKNOWN IMPORTED)",
                  f"  set_target_properties({target} PROPERTIES {where})", "endif()"]
    lines += [
        f"foreach(_stub_component IN LISTS {name}_FIND_COMPONENTS)",
        f"  set({name}_${{_stub_component}}_FOUND TRUE)",
        # Qt6's components are packages of their own: Qt6Core_VERSION.
        f"  set({name}${{_stub_component}}_FOUND TRUE)",
        f'  set({name}${{_stub_component}}_VERSION "{dotted}")',
        "  string(TOUPPER \"${_stub_component}\" _stub_upper)",
        f'  set({name}_${{_stub_upper}}_FOUND TRUE)',
        f'  set({name}_${{_stub_upper}}_LIBRARY "{library.as_posix()}")',
        f"  list(APPEND {name}_LIBRARIES {name}::${{_stub_component}})",
        f"  if(NOT TARGET {name}::${{_stub_component}})",
        f"    add_library({name}::${{_stub_component}} UNKNOWN IMPORTED)",
        f"    set_target_properties({name}::${{_stub_component}} PROPERTIES {where})",
        "  endif()",
        "endforeach()",
    ]
    qt = re.match(r"^Qt([56])$", name)
    if qt:
        lines += _qt_lines(name, qt.group(1), dotted, sysroot)
    if name.lower() == "llvm":
        lines += _llvm_lines(dotted, sysroot)
    config = directory / f"{name}Config.cmake"
    version = directory / f"{name}ConfigVersion.cmake"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        use_file.parent.mkdir(parents=True, exist_ok=True)
        if not use_file.exists():
            use_file.write_text(f"# stub emitted by will-it-riscv: {name}'s use file\n")
        library.touch()
        config.write_text("\n".join(lines) + "\n")
        # find_package(Hwloc) never looks for HWLOCConfig.cmake, but every
        # spelling looks for hwloc-config.cmake.
        lower = directory / f"{name.lower()}-config.cmake"
        if lower.name != config.name and not lower.exists():
            lower.write_text(f'include("${{CMAKE_CURRENT_LIST_DIR}}/{config.name}")\n')
            (directory / f"{name.lower()}-config-version.cmake").write_text(
                f'include("${{CMAKE_CURRENT_LIST_DIR}}/{version.name}")\n')
        version.write_text(
            f'set(PACKAGE_VERSION "{dotted}")\n'
            "set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
            "set(PACKAGE_VERSION_EXACT FALSE)\n"
        )
    except OSError:
        return []
    return [str(config), str(version), str(library)]


#: The tools a Qt build runs, as targets: AUTORCC wants Qt5::rcc, and KDAB's
#: QtInstallPaths asks Qt::qmake where Qt is.
_QT_TOOLS = ("qmake", "moc", "rcc", "uic", "lrelease", "lupdate", "qmlcachegen")


def _qt_lines(name: str, major: str, dotted: str, sysroot: Path) -> list[str]:
    """What a Qt5Config or Qt6Config says beyond any package's stub."""
    bin_dir = sysroot / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"set(QT_VERSION_MAJOR {major})", f'set(QT_VERSION "{dotted}")',
        f"set(QT_DEFAULT_MAJOR_VERSION {major})",
        f"set(Qt_VERSION_MAJOR {major})",
    ]
    for tool in _QT_TOOLS:
        program = bin_dir / tool
        if not program.exists():
            program.write_text(_fake_program(tool, sysroot.parent, dotted))
            program.chmod(0o755)
        for namespace in dict.fromkeys([name, f"Qt{major}", "Qt"]):
            target = f"{namespace}::{tool}"
            lines += [f"if(NOT TARGET {target})",
                      f"  add_executable({target} IMPORTED)",
                      f'  set_target_properties({target} PROPERTIES IMPORTED_LOCATION '
                      f'"{program.as_posix()}")',
                      "endif()"]
    # Versionless component targets, Qt::Core for Qt6::Core.
    lines += [
        f"foreach(_stub_component IN LISTS {name}_FIND_COMPONENTS)",
        "  if(NOT TARGET Qt::${_stub_component})",
        "    add_library(Qt::${_stub_component} INTERFACE IMPORTED)",
        f"    set_target_properties(Qt::${{_stub_component}} PROPERTIES "
        f"INTERFACE_LINK_LIBRARIES {name}::${{_stub_component}})",
        "  endif()",
        "endforeach()",
    ]
    return lines


def _llvm_lines(dotted: str, sysroot: Path) -> list[str]:
    """What an LLVMConfig says beyond any package's stub: where its tools are."""
    bin_dir = sysroot / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    tool = bin_dir / "llvm-config"
    if not tool.exists():
        tool.write_text(_fake_program("llvm-config", sysroot.parent))
        tool.chmod(0o755)
    root = sysroot.as_posix()
    return [
        f'set(LLVM_PACKAGE_VERSION "{dotted}")',
        f'set(LLVM_TOOLS_BINARY_DIR "{root}/bin")',
        f'set(LLVM_INCLUDE_DIRS "{root}/include")',
        f'set(LLVM_LIBRARY_DIRS "{root}/lib")',
        f'set(LLVM_CMAKE_DIR "{root}/lib/cmake/LLVM")',
        'set(LLVM_DEFINITIONS "")',
        "set(LLVM_TARGETS_TO_BUILD RISCV)",
        "set(LLVM_ENABLE_RTTI ON)",
        "set(LLVM_LINK_LLVM_DYLIB ON)",
    ]


def _add_imported_target(config: Path, target: str, sysroot: Path, suffix: str) -> bool:
    """Give a stub package config the imported target something links to."""
    text = config.read_text()
    if f"add_library({target} " in text:
        return False
    library = sysroot / "lib" / f"lib{target.split('::')[-1].lower()}{suffix}"
    try:
        library.touch()
        config.write_text(
            text
            + f"if(NOT TARGET {target})\n"
            + f"  add_library({target} UNKNOWN IMPORTED)\n"
            + f'  set_target_properties({target} PROPERTIES IMPORTED_LOCATION "{library}"\n'
            + f'    INTERFACE_INCLUDE_DIRECTORIES "{sysroot.as_posix()}/include")\n'
            + "endif()\n"
        )
    except OSError:
        return False
    return True


#: Just enough of omp.h for a check program to compile against.
_OMP_H = """\
/* stub emitted by will-it-riscv */
#ifndef WILL_IT_RISCV_OMP_H
#define WILL_IT_RISCV_OMP_H
#ifdef __cplusplus
extern "C" {
#endif
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_thread_num(void);
int omp_get_num_procs(void);
void omp_set_num_threads(int);
double omp_get_wtime(void);
#ifdef __cplusplus
}
#endif
#endif
"""


def _stub_openmp(sysroot: Path, suffix: str) -> tuple[dict, list]:
    """OpenMP comes with the target's compiler, so stand in for it.

    GCC on riscv64 ships libgomp; AppleClang ships nothing at all, and takes
    the flag only when it is smuggled past the driver with -Xclang.
    """
    import sys

    flags = "-Xclang -fopenmp" if sys.platform == "darwin" else "-fopenmp"
    library = sysroot / "lib" / f"libomp{suffix}"
    header = sysroot / "include" / "omp.h"
    try:
        library.touch()
        header.write_text(_OMP_H)
    except OSError:
        return {}, []
    overrides = {"OpenMP_omp_LIBRARY": str(library)}
    for language in ("C", "CXX"):
        overrides[f"OpenMP_{language}_FLAGS"] = flags
        overrides[f"OpenMP_{language}_LIB_NAMES"] = "omp"
    return overrides, [str(library), str(header)]


#: Just enough of mpi.h and mpif.h for a check to compile against.
_MPI_H = """\
/* stub emitted by will-it-riscv */
#ifndef WILL_IT_RISCV_MPI_H
#define WILL_IT_RISCV_MPI_H
typedef int MPI_Comm;
typedef int MPI_Datatype;
#define MPI_COMM_WORLD 0
#define MPI_VERSION 3
#define MPI_SUBVERSION 1
int MPI_Init(int *, char ***);
int MPI_Finalize(void);
int MPI_Comm_rank(MPI_Comm, int *);
int MPI_Comm_size(MPI_Comm, int *);
#endif
"""
_MPIF_H = """\
! stub emitted by will-it-riscv
      integer MPI_COMM_WORLD, MPI_VERSION, MPI_INTEGER_KIND
      parameter (MPI_COMM_WORLD=0, MPI_VERSION=3, MPI_INTEGER_KIND=4)
"""


_PYTHON_PACKAGES = {"python", "python3", "pythonlibs", "python2"}


def _stub_python(
    narration: str, sysroot: Path, suffix: str, interpreter: Optional[str] = None
) -> tuple[dict, list]:
    """The target's Python headers and library, for one pinned interpreter.

    A scikit-build-core package needs Development.Module: Python.h for the
    target. Confined, there is none -- and FindPython recomputes its include
    directories itself, so the way past is to give it the artifacts it looks
    for. It also insists the interpreter it finds matches their version, so
    the interpreter is pinned -- this one, as scikit-build-core pins the one
    running the build -- and the headers are written at its version. A build
    that wants NumPy's headers (cantera does) gets an include directory for
    them too: numpy is one of its build requirements, which the plan resolves.
    The interpreter is ``interpreter`` when given: this one, holding what the
    configure imports (see :mod:`hostpython`).
    """
    import sys

    major, minor, micro = (str(p) for p in sys.version_info[:3])
    include = sysroot / "include" / f"python{major}.{minor}"
    library = sysroot / "lib" / f"libpython{major}.{minor}{suffix}"
    numpy_include = sysroot / "include" / "numpy-stub"
    patchlevel = (
        "/* stub emitted by will-it-riscv */\n"
        f"#define PY_MAJOR_VERSION {major}\n#define PY_MINOR_VERSION {minor}\n"
        f"#define PY_MICRO_VERSION {micro}\n"
        f'#define PY_VERSION "{major}.{minor}.{micro}"\n'
    )
    stubs = {
        include / "patchlevel.h": patchlevel,
        include / "Python.h": '#include "patchlevel.h"\n',
        # FindPython greps the ABI flags out of it; none set is CPython's default.
        include / "pyconfig.h": "/* stub emitted by will-it-riscv */\n",
    }
    wants_numpy = "NumPy" in narration
    if wants_numpy:
        stubs[numpy_include / "numpy" / "arrayobject.h"] = "/* stub */\n"
        stubs[numpy_include / "numpy" / "numpyconfig.h"] = "#define NPY_API_VERSION 0x00000013\n"
    try:
        for path, text in stubs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        library.touch()
    except OSError:
        return {}, []
    overrides: dict[str, str] = {}
    for prefix in ("Python", "Python3", "PYTHON"):
        overrides[f"{prefix}_EXECUTABLE"] = interpreter or sys.executable
        overrides[f"{prefix}_INCLUDE_DIR"] = str(include)
        overrides[f"{prefix}_LIBRARY"] = str(library)
        if wants_numpy:
            overrides[f"{prefix}_NumPy_INCLUDE_DIR"] = str(numpy_include)
    return overrides, [*(str(p) for p in stubs), str(library)]


_MAKES_TARGET = re.compile(r"(?:^|_)add_(?:\w+_)?(?:module|library|executable|extension)$")

_COMMAND_STUB = """\
function({command} name)
  # stub emitted by will-it-riscv: {command}, which its package's CMake would define.
  # It makes the target it names, from whichever of its arguments are files.
  set(_wir_sources)
  foreach(_wir_arg IN LISTS ARGN)
    if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/${{_wir_arg}}" OR IS_ABSOLUTE "${{_wir_arg}}")
      list(APPEND _wir_sources "${{_wir_arg}}")
    endif()
  endforeach()
  if(NOT _wir_sources)
    file(WRITE "${{CMAKE_CURRENT_BINARY_DIR}}/${{name}}_wir_stub.c" "")
    set(_wir_sources "${{CMAKE_CURRENT_BINARY_DIR}}/${{name}}_wir_stub.c")
  endif()
  add_library(${{name}} {kind} ${{_wir_sources}})
endfunction()
"""


_DEFERRED_TARGETS = """\
# stub emitted by will-it-riscv: targets the build links that nothing it found
# defines. Made once the top directory is done, so a real one always comes first.
cmake_language(DEFER DIRECTORY "${{CMAKE_SOURCE_DIR}}" CALL _wir_deferred_targets)
function(_wir_deferred_targets)
  foreach(_wir_target IN ITEMS {targets})
    if(NOT TARGET ${{_wir_target}})
      add_library(${{_wir_target}} INTERFACE IMPORTED GLOBAL)
    endif()
  endforeach()
endfunction()
"""


_NO_FIND_MODULE = re.compile(r'No "Find([A-Za-z0-9_+-]+)\.cmake" found in CMAKE_MODULE_PATH')
#: An install location an error tells you to set, and the words telling you.
_LOCATION_VARIABLE = re.compile(
    # Not a module's own: FindMPI's _MPI_BASE_DIR is nobody's to set.
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_(?:DIR|ROOT|PATH|HOME))\b"
)
_TELLS_YOU = re.compile(
    r"not provided|not set|not defined|must (?:be )?(?:set|specif|provid)|please (?:set|"
    r"specify|provide|define)|needs? expli?ci?t|specify (?:the )?path|you (?:may )?need to set"
    r"|(?:are|is) provided",
    re.IGNORECASE,
)


def _variable_package(variable: str) -> str:
    """The package an install-location variable is for: MPI_BASE_DIR is MPI's."""
    stem = re.sub(r"_(DIR|ROOT|PATH|HOME)$", "", variable)
    return re.sub(r"_(BASE|INSTALL|PREFIX|TOP|INSTALL_ROOT)$", "", stem, flags=re.IGNORECASE)


def _named_variables(stanza: str) -> list[str]:
    flat = re.sub(r"\s+", " ", stanza)
    if not _TELLS_YOU.search(flat):
        return []
    return [
        v for v in dict.fromkeys(_LOCATION_VARIABLE.findall(flat))
        if not v.upper().startswith(("CMAKE_", "_CMAKE", "ENV_"))
    ]


def _stub_named_variable(
    variable: str, sysroot: Path, suffix: str,
) -> tuple[Optional[str], list]:
    """The sysroot as the install the variable names. A ``<Pkg>_DIR`` is read
    both ways -- the directory of PkgConfig.cmake, or an install prefix -- so
    the sysroot gets a config at its top that leads to the package's stub."""
    if not variable.upper().endswith("_DIR") or _variable_package(variable) != variable[:-4]:
        return str(sysroot), []   # an install prefix: MPI_BASE_DIR, HWLOC_ROOT
    name = variable[:-4]
    files = _stub_config_package(name, sysroot, suffix)
    config = sysroot / "lib" / "cmake" / name / f"{name}Config.cmake"
    top = sysroot / f"{name.lower()}-config.cmake"
    if config.exists() and not top.exists():
        top.write_text(f'include("{config.as_posix()}")\n')
        files.append(str(top))
    return str(sysroot), files


def _modules_directory(sysroot: Path) -> Path:
    """Where modules a stubbed package would ship are written, on CMAKE_MODULE_PATH."""
    return sysroot / "lib" / "cmake" / "will-it-riscv-modules"


def _module_file(module: str, sysroot: Path) -> Optional[Path]:
    if "/" not in module:
        name = module if module.endswith(".cmake") else f"{module}.cmake"
        return _modules_directory(sysroot) / name
    path = Path(module)
    return path if str(path).startswith(str(sysroot) + "/") else None


def _includes_directory(sysroot: Path) -> Path:
    return sysroot / "lib" / "cmake" / "will-it-riscv-commands"


def _top_level_includes(sysroot: Path) -> list[str]:
    """Every stub file that goes in at the first project()."""
    return sorted(str(p) for p in _includes_directory(sysroot).glob("*.cmake"))


def _targets_file(sysroot: Path) -> Path:
    return _includes_directory(sysroot) / "zz-deferred-targets.cmake"


def _defer_target(target: str, sysroot: Path) -> bool:
    """Add a target to those made at the end of the top directory; False if it was."""
    path = _targets_file(sysroot)
    found = re.search(r"IN ITEMS (.*)\)\n", path.read_text()) if path.exists() else None
    targets = found.group(1).split() if found else []
    if target in targets:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFERRED_TARGETS.format(targets=" ".join([*targets, target])))
    return True


def _stub_commands(commands: list[str], sysroot: Path) -> tuple[list[str], list]:
    """Commands the configure called that a stubbed package would have defined.

    nanobind_add_module and pybind11_add_module come from those packages'
    CMake config files -- which, stubbed, define nothing. A command that
    makes a target gets one made from the files it is given; any other is a
    no-op. Each is its own file, included at the first project() before
    anything calls it; a real definition loaded later wins.
    """
    directory = _includes_directory(sysroot)
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    for command in commands:
        path = directory / f"{command}.cmake"
        if path.exists():
            continue
        if _MAKES_TARGET.search(command):
            # A Python extension is a module; anything called a library may be
            # linked into another target, which a module may not (musescore).
            kind = "MODULE" if re.search(r"(module|extension)$", command) else "STATIC"
            path.write_text(_COMMAND_STUB.format(command=command, kind=kind))
        else:
            path.write_text(
                f"function({command})\n  # stub emitted by will-it-riscv\nendfunction()\n"
            )
        created.append(str(path))
    return _top_level_includes(sysroot), created


def _stub_mpi(sysroot: Path, suffix: str) -> tuple[dict, list]:
    """MPI, answered the way FindMPI asks its own questions.

    FindMPI interrogates a compiler wrapper and test-compiles a program per
    language -- Fortran's needs a real mpi module. Instead, give it the
    answers it would have cached: a library, a header or module directory
    per language, and MPI_<LANG>_WORKS, which is what makes it skip the test
    compile. The wrapper search is skipped too, or an mpicc on the host
    would answer for the target.
    """
    library = sysroot / "lib" / f"libmpi{suffix}"
    headers = {sysroot / "include" / "mpi.h": _MPI_H, sysroot / "include" / "mpif.h": _MPIF_H}
    try:
        library.touch()
        for path, text in headers.items():
            path.write_text(text)
    except OSError:
        return {}, []
    include = str(sysroot / "include")
    overrides = {"MPI_SKIP_COMPILER_WRAPPER": "TRUE", "MPI_mpi_LIBRARY": str(library)}
    for language in ("C", "CXX", "Fortran"):
        overrides[f"MPI_{language}_LIB_NAMES"] = "mpi"
        overrides[f"MPI_{language}_WORKS"] = "TRUE"
        # With a version already known, FindMPI does not read it out of the
        # test program it would have built -- which a compile-only toolchain
        # never links, so there is no test_mpi_C.bin to read.
        overrides[f"MPI_{language}_VERSION"] = "3.1"
        overrides[f"MPI_{language}_VERSION_MAJOR"] = "3"
        overrides[f"MPI_{language}_VERSION_MINOR"] = "1"
        overrides[f"MPI_{language}_HEADER_DIR"] = include
    overrides["MPI_Fortran_F77_HEADER_DIR"] = include
    overrides["MPI_Fortran_MODULE_DIR"] = include
    return overrides, [str(library), *(str(p) for p in headers)]


#: Written into the scratch directory, never the project.
TOOLCHAIN = """\
# Written by will-it-riscv: configure as a Linux/{arch} build that can find
# nothing but what this scratch sysroot holds.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR {arch})
# The scratch tree and the source tree are roots too, but only so that a path
# already inside one is searched as written: what the build fetched or made
# for itself, and what the repository ships, is on the target's side. SUNDIALS
# builds a Fortran library in its build tree and then looks for it there.
set(CMAKE_FIND_ROOT_PATH "{sysroot};{scratch};{source}")
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
# FortranCInterface_VERIFY links a program of C and Fortran to prove the two
# compilers agree, and nothing links here; on the target, gcc and gfortran do.
# A plain variable, since detecting the mangling clears the cached answer.
set(FortranCInterface_VERIFIED_C 1)
set(FortranCInterface_VERIFIED_CXX 1)
"""


def _toolchain(scratch: Path, sysroot: Path, arch: str, source: Path) -> Path:
    emulator = scratch / "run-on-host"
    emulator.write_text('#!/bin/sh\nexec "$@"\n')
    emulator.chmod(0o755)
    path = scratch / "toolchain.cmake"
    path.write_text(
        TOOLCHAIN.format(
            arch=arch, sysroot=sysroot.as_posix(), emulator=emulator.as_posix(),
            scratch=scratch.as_posix(), source=Path(source).resolve().as_posix(),
        )
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
        cmake() or "cmake",
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
        # No stdin: nothing a configure runs may sit waiting for input. And
        # run from the scratch directory: gmsh's execute_process writes
        # cmake_options.texi to whatever directory cmake was started in.
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, env=env,
            stdin=subprocess.DEVNULL, cwd=scratch,
        )
    except subprocess.TimeoutExpired:
        probes, traced = _parse_trace(trace, root)
        return PseudoBuild(
            probes=probes,
            error=f"configure did not finish within {timeout}s",
            commands_traced=traced,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PseudoBuild(error=f"could not run cmake: {exc}")

    lookups: dict = {}
    reads: list = []
    exists: list = []
    searches: list = []
    probes, traced = _parse_trace(trace, root, lookups, reads, exists, searches)
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
        lookups=lookups,
        reads=reads,
        exists=exists,
        searches=searches,
    )
    result.narration = narration
    result.found_at = dict(_STATUS_FOUND_AT.findall(narration))
    if not result.completed:
        result.error = _first_error(process.stdout, process.stderr)
    return result


# ----------------------------------------------------------------- unblocking

#: A version no project will consider too old.
FAKE_VERSION = ("99", "9", "9")
FAKE_VERSION_STRING = "99.9.9"

#: Debian 13's LLVM. Projects pin LLVM to ranges -- pocl takes 18.1, 19.1 or
#: 20.1 and nothing else -- so its stub claims a real release.
LLVM_VERSION = ("19", "1", "7")
_LLVM_FAMILY = {"llvm", "clang", "mlir", "lld", "polly", "llvm-config"}

#: Versions a configure turned the stub's down for and said what it wanted
#: instead, this run: package (lower case) to version. Per thread, since the
#: dataset tool configures several projects at once.
_NEGOTIATED: contextvars.ContextVar[dict] = contextvars.ContextVar("negotiated")


def _negotiated(name: str) -> Optional[str]:
    return _NEGOTIATED.get({}).get(name.lower())


def _fake_version(name: str) -> tuple[str, str, str]:
    """The version a stub of ``name`` claims: 99.9.9, except that a name
    ending in its major version -- Qt6, Qt5, Eigen3, SDL2, Gtk3 -- keeps it.
    ``find_package(QT NAMES Qt6 Qt5)`` takes QT_VERSION_MAJOR from the
    version found, and goes on to find_package(Qt${QT_VERSION_MAJOR})."""
    wanted = _negotiated(name)
    if wanted:
        parts = (wanted.split(".") + ["0", "0"])[:3]
        return parts[0], parts[1], parts[2]
    if name.lower() in _LLVM_FAMILY:
        return LLVM_VERSION
    found = re.match(r"^[A-Za-z]+([0-9])$", name)
    return (found.group(1), "99", "9") if found else FAKE_VERSION

_LIBRARY_VAR = re.compile(r"_(LIBRARY|LIBRARIES|LIB|LIBS)(_[A-Z]+)?$", re.IGNORECASE)
_INCLUDE_VAR = re.compile(r"_(INCLUDE_DIR|INCLUDE_DIRS|INCLUDEDIR|INCLUDE)$", re.IGNORECASE)
_PROGRAM_VAR = re.compile(r"_(EXECUTABLE|EXE|COMMAND|BINARY|PROGRAM|COMPILER)$", re.IGNORECASE)
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
    major, minor, patch = _fake_version(name)
    lines = [f"/* stub emitted by will-it-riscv for {name} */"]
    if stem == "BOOST" or name.endswith("boost/version.hpp"):
        # FindBoost reads one packed number: major * 100000 + minor * 100 + patch.
        return "\n".join(lines + [
            f"#define BOOST_VERSION {int(major) * 100000 + int(minor) * 100 + int(patch)}",
            f'#define BOOST_LIB_VERSION "{major}_{minor}"',
        ]) + "\n"
    for prefix in dict.fromkeys([stem, stem.rstrip("_0123456789")]):
        if not prefix:
            continue
        lines += [
            f"#define {prefix}_VERSION_MAJOR {major}",
            f"#define {prefix}_VERSION_MINOR {minor}",
            f"#define {prefix}_VERSION_PATCH {patch}",
            f"#define {prefix}_VERSION_MICRO {patch}",
            f"#define {prefix}_VERSION_NUM {major}{minor}{patch}",
            f'#define {prefix}_VERSION "{major}.{minor}.{patch}"',
            f'#define {prefix}_VERSION_STRING "{major}.{minor}.{patch}"',
        ]
    return "\n".join(lines) + "\n"


#: How a few programs say their version, where a Find module reads it with
#: the words around it. Anything else says the bare number, which is what
#: FindDoxygen insists on (``^[0-9]+\.[0-9]+\.[0-9]+``).
_VERSION_LINES = {
    "git": "git version {}",
    "cython": "Cython version {}",
    "swig": "SWIG Version {}",
    "bison": "bison (GNU Bison) {}",
    "flex": "flex {}",
}

_PROGRAM_STUB = """\
#!/bin/sh
# stub emitted by will-it-riscv for {name}
if [ $# -eq 1 ]; then
  case "$1" in
    --version|-version|-v|-V|version|-dumpversion) echo {version}; exit 0 ;;
  esac
fi
{query}# Asked to write a file -- doxygen -s -g CMakeDoxyfile.tpl, FindDoxygen's
# first act -- it writes an empty one, inside the scratch directory only.
take=
for arg do
  if [ -n "$take" ]; then
    case "$arg" in {scratch}/*) : > "$arg" ;; esac
    take=
  fi
  case "$arg" in -g|-o|--output) take=1 ;; esac
done
exit 0
"""


#: qmake -query, which a Qt project asks where Qt is installed: everything in the sysroot.
_QMAKE_QUERY = """\
if [ "$1" = "-query" ]; then
  if [ -n "$2" ]; then
    case "$2" in QT_VERSION) echo {version} ;; *) echo {sysroot} ;; esac
    exit 0
  fi
  for var in QT_INSTALL_PREFIX QT_INSTALL_ARCHDATA QT_INSTALL_DATA QT_INSTALL_DOCS \\
      QT_INSTALL_HEADERS QT_INSTALL_LIBS QT_INSTALL_LIBEXECS QT_INSTALL_BINS \\
      QT_INSTALL_TESTS QT_INSTALL_PLUGINS QT_INSTALL_QML QT_INSTALL_TRANSLATIONS \\
      QT_INSTALL_CONFIGURATION QT_INSTALL_EXAMPLES QT_HOST_PREFIX QT_HOST_DATA \\
      QT_HOST_BINS QT_HOST_LIBS; do
    echo "$var:"{sysroot}
  done
  echo QT_VERSION:{version}
  exit 0
fi
"""


#: llvm-config, which tvm asks ten questions of: all answered from the sysroot.
_LLVM_CONFIG = """\
sysroot={sysroot}
out=
for arg do
  case "$arg" in
    --version) v={version} ;;
    --prefix) v="$sysroot" ;;
    --bindir) v="$sysroot/bin" ;;
    --libdir) v="$sysroot/lib" ;;
    --includedir) v="$sysroot/include" ;;
    --cmakedir) v="$sysroot/lib/cmake/llvm" ;;
    --libfiles) v="$sysroot/lib/libLLVM.so" ;;
    --libs) v="-lLLVM" ;;
    --libnames) v="libLLVM.so" ;;
    --system-libs) v="" ;;
    --cflags|--cppflags|--cxxflags) v="-I$sysroot/include" ;;
    --ldflags) v="-L$sysroot/lib" ;;
    --targets-built) v="RISCV" ;;
    --host-target) v="riscv64-unknown-linux-gnu" ;;
    --shared-mode) v="shared" ;;
    --build-mode) v="Release" ;;
    --has-rtti) v="YES" ;;
    --assertion-mode) v="OFF" ;;
    --components) v="all" ;;
    --link-shared|--link-static|--ignore-libllvm) continue ;;
    -*) v="" ;;
    *) continue ;;
  esac
  out="${{out:+$out }}$v"
done
echo "$out"
exit 0
"""


def _fake_program(name: str, scratch: Path, version: Optional[str] = None) -> str:
    """A program that runs, says a version, and writes the file it is told to."""
    if re.match(r"^llvm-config(-?[0-9.]+)?$", name):
        sysroot = scratch / "sysroot"
        (sysroot / "lib").mkdir(parents=True, exist_ok=True)
        (sysroot / "lib" / "libLLVM.so").touch()
        return "#!/bin/sh\n# stub emitted by will-it-riscv for llvm-config\n" + _LLVM_CONFIG.format(
            sysroot=shlex.quote(str(sysroot)), version=".".join(LLVM_VERSION))
    version = version or _negotiated(name) or FAKE_VERSION_STRING
    said = _VERSION_LINES.get(name, "{}").format(version)
    query = ""
    if re.match(r"^qmake(6|-qt[56])?$", name):
        query = _QMAKE_QUERY.format(version=shlex.quote(version),
                                    sysroot=shlex.quote(str(scratch / "sysroot")))
    return _PROGRAM_STUB.format(
        name=name, version=shlex.quote(said), scratch=shlex.quote(str(scratch)), query=query,
    )


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
        path.write_text(_fake_program(path.name, sysroot.parent))
        path.chmod(0o755)
        return str(path)
    if _VERSION_VAR.search(variable):
        return _negotiated(_VERSION_VAR.split(variable)[0]) or FAKE_VERSION_STRING
    if variable.upper().endswith("_FLAGS"):
        # Anything put here lands on a compile line, where "1" is a file name.
        return None
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
    fresh_library = not lib.exists()
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
    # The library too, when this made it: an experiment that is taken back
    # must leave nothing a later find_library could turn up.
    return [str(pc), str(lib)] if fresh_library else [str(pc)]


def _synthesize(
    blocking: Optional[str],
    narration: str,
    sysroot: Path,
    already: set,
    shared_suffix: Optional[str] = None,
    lookups: Optional[dict] = None,
    interpreter: Optional[str] = None,
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
        commands = _UNKNOWN_COMMAND.findall(stanza)
        if commands:
            includes, files = _stub_commands(commands, sysroot)
            overrides["CMAKE_PROJECT_TOP_LEVEL_INCLUDES"] = ";".join(includes)
            created += [f for f in files if f not in already]
            continue
        variables: list[str] = []
        named = _ANY_MISS.search(stanza)
        package = named.group(1) if named else None
        if package and package.lower() in _PYTHON_PACKAGES:
            stubbed, files = _stub_python(narration, sysroot, suffix, interpreter)
            overrides.update(stubbed)
            created += [f for f in files if f not in already]
            continue
        if package and package.lower() == "boost":
            # FindBoost second-guesses preset library paths whenever its
            # version or include dir moves, but asks for a BoostConfig.cmake
            # first and defers to it. So Boost is answered as a config package.
            config = sysroot / "lib" / "cmake" / "Boost" / "BoostConfig.cmake"
            if not config.is_file():
                created += _stub_config_package("Boost", sysroot, suffix)
                continue
        if package and package.split("_", 1)[0].lower() == "mpi":
            # FPHSA's "(missing: MPI_Fortran_FOUND Fortran)" names results,
            # not inputs; faking those gets nowhere. Answer FindMPI instead.
            stubbed, files = _stub_mpi(sysroot, suffix)
            overrides.update(stubbed)
            created += [f for f in files if f not in already]
            continue
        made_before, set_before = len(created), set(overrides)
        # 1. The cache variables FPHSA said were missing. A bare word among
        #    them is a component -- "(missing: ... SSL Crypto)" -- and the
        #    variable behind a component is <PKG>_<COMPONENT>_LIBRARY.
        for match in _MISSING_VARS.finditer(stanza):
            for token in match.group(1).split():
                # A cache variable has capitals; FindBoost's components are
                # spelled in lower case, underscores and all: date_time.
                if "_" in token and not token.islower():
                    variables.append(token)
                elif package and token.isidentifier():
                    variables += _component_variables(package, token)
        # 0. What the blocker's own Find module looked for this round, put
        #    where it looked. FindHDF5 recomputes HDF5_INCLUDE_DIRS from its
        #    own find_path(hdf5.h), so faking the variable gets nowhere --
        #    but an hdf5.h in the sysroot is found the way the module finds it.
        for name in _stanza_blockers(stanza):
            looked_for = (lookups or {}).get(name.lower(), [])
            for stub in _stub_lookups(name, looked_for, sysroot, suffix):
                if stub not in already:
                    created.append(stub)
        # 1a. The variable a find_*(... REQUIRED) could not fill.
        for match in _REQUIRED_FIND.finditer(stanza):
            variables.append(match.group(1))
        # 1b. The same, for a component a try_compile tried to link -- or,
        #     for a package that is here as a stub config, the imported
        #     target itself: Boost::headers, which cantera links.
        for match in _TARGET_NOT_FOUND.finditer(stanza):
            package = match.group(1) or match.group(3)
            component = match.group(2) or match.group(4)
            config = sysroot / "lib" / "cmake" / package / f"{package}Config.cmake"
            if config.is_file():
                if _add_imported_target(config, f"{package}::{component}", sysroot, suffix):
                    created.append(f"{config}#{package}::{component}")
            else:
                variables += _component_variables(package, component)
            # A linked target nothing defined -- lcc links BML::bml, which
            # BML's library variables never make, and ginkgo hip::device from a
            # config it never loads -- is made at the end, if still missing.
            if _defer_target(f"{package}::{component}", sysroot):
                created.append(f"{_targets_file(sysroot)}#{package}::{component}")
                overrides["CMAKE_PROJECT_TOP_LEVEL_INCLUDES"] = ";".join(
                    _top_level_includes(sysroot))
        # 1b'. A module a stubbed package would have shipped: FairRoot
        #      include()s FairFormattedOutput after find_package(FairCMakeModules).
        for module in _MISSING_MODULE.findall(stanza):
            path = _module_file(module, sysroot)
            if path is not None and str(path) not in already and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# stub emitted by will-it-riscv: {module}, which a stubbed "
                                "package would ship\n")
                created.append(str(path))
                overrides["CMAKE_MODULE_PATH"] = str(_modules_directory(sysroot))
        # 1d. A variable the project's own error says to set: exacmech's
        #     "RAJA_DIR was not provided", variorum's "needs explict HWLOC_DIR".
        for variable in _named_variables(stanza):
            value, files = _stub_named_variable(variable, sysroot, suffix)
            if value is not None and overrides.get(variable) != value:
                overrides[variable] = value
                created += [f for f in files if f not in already]
        # 1e. A Find module the project insists on, from a package that would
        #     have shipped it: ufs-weather-model's FindESMF.cmake.
        for name in _NO_FIND_MODULE.findall(re.sub(r"\s+", " ", stanza)):
            module = _modules_directory(sysroot) / f"Find{name}.cmake"
            if not module.exists():
                created += _stub_config_package(name, sysroot, suffix)
                module.parent.mkdir(parents=True, exist_ok=True)
                config = sysroot / "lib" / "cmake" / name / f"{name}Config.cmake"
                module.write_text(f"# stub emitted by will-it-riscv: {name}'s Find module\n"
                                  f'include("{config.as_posix()}")\n')
                created.append(str(module))
                overrides["CMAKE_MODULE_PATH"] = str(_modules_directory(sysroot))
        # 1c. A config-mode package required and nowhere to be found -- as
        #     the first file it searched for: find_package(QT NAMES Qt6 Qt5)
        #     looks for Qt6Config.cmake, never QTConfig.cmake.
        flat = re.sub(r"\s+", " ", stanza)
        for asked in _NO_CONFIG.findall(flat) + _NEITHER_CONFIG.findall(flat):
            listed = _CONFIG_NAMES.findall(stanza.split("following names", 1)[-1])
            name = listed[0] if listed and "following names" in stanza else asked
            config = sysroot / "lib" / "cmake" / name / f"{name}Config.cmake"
            if not config.is_file():
                created += _stub_config_package(name, sysroot, suffix)
            # Found wherever it is looked for -- atlas asks for ecbuild before
            # project(), when the toolchain's find roots are not set yet.
            for variable in dict.fromkeys([f"{asked}_DIR", f"{name}_DIR"]):
                if overrides.get(variable) != str(config.parent):
                    overrides[variable] = str(config.parent)
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
        for read, strings in _MISSING_FILE.findall(stanza):
            path = Path(read or strings)
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

        # 4b. Blamed, and nothing of its own to put in place: prusaslicer's
        #     Findcereal says "Cereal library not found" after a quiet
        #     find_package(cereal CONFIG). A config package is what it looked for.
        if len(created) == made_before and set(overrides) == set_before:
            for name in _stanza_blockers(stanza)[:1]:
                config = sysroot / "lib" / "cmake" / name / f"{name}Config.cmake"
                if re.match(r"^[A-Za-z][A-Za-z0-9_+-]*$", name) and not config.is_file():
                    created += _stub_config_package(name, sysroot, suffix)
                    overrides[f"{name}_DIR"] = str(config.parent)
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


#: A grep for a feature switch (H5_HAVE_PARALLEL, ARMA_USE_WRAPPER), as
#: against a version or a setting a configure cannot do without (METIS's
#: IDXTYPEWIDTH): a stub claims the second kind, not the first.
_FEATURE_GREP = re.compile(r"(?:^|[^A-Za-z])(?:HAVE|USE|ENABLE|WITH|HAS)_|_ENABLED\b")
_GREP_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{4,}")
_PLAIN_WORDS = {"define", "include", "undef", "ifdef", "ifndef"}


def _feature_named(regex: str, message: str) -> bool:
    """Whether the error is about what this feature grep looks for: "needs to
    enable parallel support" and ``H5_HAVE_PARALLEL``."""
    words = {w.lower() for w in _GREP_WORD.findall(regex.replace("_", " "))} - _PLAIN_WORDS
    lowered = message.lower()
    # dolfinx: "Found serial HDF5, MPI HDF5 build required."
    words |= {other for w in words for other in _FEATURE_SYNONYMS.get(w, ())}
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", lowered) for w in words)


_FEATURE_SYNONYMS = {"parallel": ("mpi",), "deflate": ("zlib",), "szip": ("szlib",)}


def _answer_reads(reads: list, sysroot: Path, answered: set, message: str = "") -> list[str]:
    """Give each stub file a line for every grep of it that found nothing.

    FindArmadillo greps ``armadillo_bits/arma_version.hpp`` for
    ``#define ARMA_VERSION_MAJOR``, FindPython greps ``patchlevel.h`` for
    ``PY_VERSION``, FindCURL ``curlver.h`` for ``LIBCURL_VERSION``. No guess
    at macro names covers them all, but the trace says exactly what each
    module looked for: write a line it matches. Only files inside the
    sysroot, and each grep once. Returns the files changed.
    """
    changed: list[str] = []
    for file, regex in reads:
        if (file, regex) in answered or not file.startswith(str(sysroot) + "/"):
            continue
        if _FEATURE_GREP.search(regex) and not _feature_named(regex, message):
            # A stub claims its version, not every feature its real one might
            # have been built with: parallel HDF5 would bring MPI along.
            continue
        answered.add((file, regex))
        path = Path(file)
        try:
            text = path.read_text(errors="replace") if path.exists() else ""
        except OSError:
            continue
        if text and STUB_MARK not in text and path.suffix.lower() not in _HEADER_SUFFIXES:
            continue   # a stub config or .pc file: a line of C would not belong
        pattern = _python_regex(regex)
        if pattern is not None and any(pattern.search(line) for line in text.splitlines()):
            continue
        python = "PY_" in regex or path.name == "patchlevel.h"
        owner = re.search(rf"{STUB_MARK} for ([A-Za-z0-9_.+-]+)", text)
        version = _python_version() if python else (
            _fake_version(owner.group(1)) if owner else FAKE_VERSION)
        # One line for each branch it allows: TBB_VERSION_(MAJOR|MINOR) wants both.
        lines = list(dict.fromkeys(
            _as_c(line, regex) for line in (_example(regex, version, pick) for pick in range(4))
            if line is not None and "\n" not in line
        ))
        if not lines:
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not text:
                text = f"/* {STUB_MARK} for {path.stem} */\n"
            path.write_text(
                text + ("" if text.endswith("\n") else "\n") + "\n".join(lines) + "\n"
            )
        except OSError:
            continue
        if file not in changed:
            changed.append(file)
    return changed


#: FPHSA, when the version its module worked out is the wrong one, or none.
_UNSUITABLE = re.compile(
    r"Could NOT find ([A-Za-z0-9_.+-]+): "
    r"(?:Found unsuitable version|\(Required is (?:at least|exact) version)"
)


def _version_headers(
    outcome: PseudoBuild, sysroot: Path, blockers: Optional[list] = None,
) -> list[str]:
    """The headers a package's own module looked for before reading its version.

    FindArmadillo reads ``armadillo_bits/arma_version.hpp`` only ``if(EXISTS)``,
    so a stub include directory gives it no version at all, and FPHSA calls
    that unsuitable. When it does, write the headers that module asked
    after; the next round greps them, and ``_answer_reads`` gives it the
    lines it greps for. Only for the package refused, or one already stubbed
    -- FindHDF5 reads H5pubconf.h if it is there, and with none the version
    is empty and the project's own check refuses it -- inside the sysroot.
    """
    refused = {name.lower() for name in _UNSUITABLE.findall(outcome.narration or "")}
    refused |= {name.lower() for name in blockers or []}
    created: list[str] = []
    for path, module in outcome.exists:
        stem = Path(module).name.lower()
        owner = re.sub(r"^find|-?config\.cmake$|\.cmake$", "", stem)
        if not path.startswith(str(sysroot) + "/"):
            continue
        inside = path[len(str(sysroot)):].lower()
        # Its own Find module asking, or a project macro asking after a
        # header of its: mlpack's find_armadillo and armadillo_bits/arma_version.hpp.
        named = [r for r in refused if len(r) >= 3 and r in inside]
        if owner not in refused and not named:
            continue
        owner = owner if owner in refused else named[0]
        target = Path(path)
        if target.exists() or target.suffix.lower() not in _HEADER_SUFFIXES:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"/* {STUB_MARK} for {owner} */\n")
        except OSError:
            continue
        created.append(path)
    return created


def _as_c(line: str, regex: str) -> str:
    """The line as a #define where the regex allows it, so the header still compiles."""
    if line.startswith("#"):
        return line
    candidate = "#define " + line
    pattern = _python_regex(regex)
    return candidate if pattern is not None and pattern.search(candidate) else line


#: FPHSA saying which version it wanted instead of the one it found.
_WANTED = re.compile(
    r'Could NOT find ([A-Za-z0-9_.+-]+): Found unsuitable version "[^"]*", '
    r'(?:but required is at least|but required is exact version|required range is) '
    r'"([0-9][0-9.]*)'
)


def _negotiate(narration: str, sysroot: Path, answered: set, written: set) -> list[str]:
    """Redo a package's stubs at the version the configure said it wanted.

    openvdb wants TBB "at least 2020.3", and TBB's header said 99.9; espresso
    wants Cython in "3.0.4...<4.0.0". The stub files of that package -- its
    headers, its config, its program -- go, and come back next round at the
    wanted version, the way they came the first time. Once a package.
    """
    negotiated = _NEGOTIATED.get(None)
    if negotiated is None:
        return []
    redone: list[str] = []
    for name, wanted in _WANTED.findall(re.sub(r"\s+", " ", narration)):
        key = name.lower()
        if key in negotiated:
            continue
        negotiated[key] = wanted
        mark = f"{STUB_MARK} for {name}".lower()
        doomed = [sysroot / "bin" / key, sysroot / "lib" / "cmake" / name]
        for path in (sysroot / "include").rglob("*"):
            try:
                if path.is_file() and mark in path.read_text(errors="replace")[:200].lower():
                    doomed.append(path)
            except OSError:
                continue
        for path in doomed:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
            else:
                continue
            gone = str(path)
            written.difference_update({w for w in written if w.startswith(gone)})
            answered.difference_update({a for a in answered if a[0].startswith(gone)})
            # Progress, but not a file made: the stub is to be written again.
            redone.append(f"{gone}#redo-at-{wanted}")
    return redone


def _python_version() -> tuple[str, ...]:
    return tuple(str(p) for p in sys.version_info[:3])


def _python_regex(regex: str) -> Optional[re.Pattern]:
    """CMake's regex in Python's dialect, near enough to ask whether a line matches."""
    try:
        return re.compile(regex.replace("\\t", "\t"))
    except re.error:
        return None


#: Words that say which part of a version a number is.
_PART = (
    (re.compile(r"MAJOR", re.IGNORECASE), 0),
    (re.compile(r"MINOR", re.IGNORECASE), 1),
    (re.compile(r"PATCH|MICRO|SUBMINOR|RELEASE|REVISION", re.IGNORECASE), 2),
)


def _example(
    regex: str, version: tuple[str, ...] = FAKE_VERSION, pick: int = 0,
) -> Optional[str]:
    """One line a CMake regular expression matches, with the version's numbers in it.

    Enough of CMake's syntax for what Find modules grep version headers
    with: literals and escapes, ``.``, bracket classes, groups, ``|`` (the
    ``pick``-th branch, or the last there is), ``* + ?``, and the anchors.
    A run of digits is a part of the version -- the one the words before
    it name (MAJOR, MINOR, PATCH), else the next -- and a run that may hold
    a dot (``[0-9.]+``, ``[^"]+``, a captured ``(.*)``) gets the whole
    dotted version. None when it is beyond this, or when what it made does
    not match after all.
    """
    dotted = ".".join(version)
    parts = {"next": 0}

    def number(before: str) -> str:
        named = [index for word, index in _PART if word.search(before[-24:])]
        if named:
            index = named[0]
        else:
            index = parts["next"]
            parts["next"] += 1
        return version[index] if index < len(version) else "0"

    def atom_at(at: int) -> tuple[str, str, int]:
        char = regex[at]
        if char == "[":
            close = at + 1
            if regex[close:close + 1] == "^":
                close += 1
            if regex[close:close + 1] == "]":
                close += 1   # a literal ] first in the class
            close = regex.index("]", close)
            return regex[at + 1:close], "class", close + 1
        if char == "\\" and at + 1 < len(regex):
            return {"t": "\t", "n": "\n"}.get(regex[at + 1], regex[at + 1]), "literal", at + 2
        if char == ".":
            return ".", "any", at + 1
        return char, "literal", at + 1

    def alternatives(at: int, prefix: str, depth: int) -> tuple[str, int]:
        """The chosen branch of a group (or of the whole); returns (text, index of ')' or end)."""
        texts = []
        while True:
            text, at = branch(at, prefix, depth)
            texts.append(text)
            if at < len(regex) and regex[at] == "|":
                at += 1
                continue
            return texts[min(pick, len(texts) - 1)], at

    def branch(at: int, prefix: str, depth: int) -> tuple[str, int]:
        """Render up to the end of this branch: the index of its '|', ')' or the end."""
        text = ""
        while at < len(regex):
            char = regex[at]
            if char == ")":
                if not depth:
                    raise ValueError("unbalanced")
                return text, at
            if char == "|":
                return text, at
            if char in "^$":
                at += 1
                continue
            if char == "(":
                inner, close = alternatives(at + 1, prefix + text, depth + 1)
                if close >= len(regex):
                    raise ValueError("unbalanced")
                atom, kind, at = inner, "group", close + 1
            else:
                atom, kind, at = atom_at(at)
            quantifier = regex[at] if at < len(regex) and regex[at] in "*+?" else ""
            at += bool(quantifier)
            edge = not (prefix + text) or not regex[at:].strip("$)")
            text += render(atom, kind, quantifier, prefix + text, depth, edge)
        if depth:
            raise ValueError("unbalanced")
        return text, at

    def render(
        atom: str, kind: str, quantifier: str, before: str, depth: int, edge: bool,
    ) -> str:
        if kind in ("literal", "group"):
            return "" if quantifier == "*" else atom
        if kind == "any":
            if quantifier == "*":
                # ".*#define X ([0-9]+).*" pads both ends; "(.*)" and
                # LIBCURL_VERSION "\".*\"" hold the version.
                return "" if edge and not depth else dotted
            return dotted if quantifier == "+" else "9"
        allowed = _allowed(atom)
        many = quantifier in ("*", "+")
        if allowed("9") and allowed("."):
            return dotted if many else "9"
        if allowed("9"):
            return number(before) if many else number(before)[:1]
        if quantifier == "*":
            return ""
        for candidate in " _xA\t":
            if allowed(candidate):
                return candidate
        raise ValueError("no character the class allows")

    try:
        line, _ = alternatives(0, "", 0)
    except (ValueError, IndexError):
        return None
    pattern = _python_regex(regex)
    if pattern is None or not pattern.search(line):
        return None
    return line


def _allowed(spec: str):
    """A bracket class, as a test of one character."""
    negated = spec.startswith("^")
    body = spec[1:] if negated else spec
    body = body.replace("\\t", "\t")
    ranges: list[tuple[str, str]] = []
    at = 0
    while at < len(body):
        if at + 2 < len(body) and body[at + 1] == "-":
            ranges.append((body[at], body[at + 2]))
            at += 3
        else:
            ranges.append((body[at], body[at]))
            at += 1

    def test(char: str) -> bool:
        inside = any(low <= char <= high for low, high in ranges)
        return inside != negated

    return test


def _component_variables(package: str, component: str) -> list[str]:
    """The cache variable a Find module fills for one component's library.

    Upper-case is the common spelling (OPENSSL_SSL_LIBRARY); FindBoost keeps
    the package's own case (Boost_FILESYSTEM_LIBRARY). Setting both costs one
    unused-variable warning.
    """
    spellings = [
        f"{package.upper()}_{component.upper()}_LIBRARY",
        f"{package}_{component.upper()}_LIBRARY",
        # FindBoost rebuilds <Pkg>_<COMP>_LIBRARY from its _RELEASE entry.
        f"{package}_{component.upper()}_LIBRARY_RELEASE",
    ]
    return list(dict.fromkeys(spellings))


#: The escape hatch CMake 4 names when a project asks for a CMake < 3.5.
POLICY_MINIMUM = "CMAKE_POLICY_VERSION_MINIMUM"
_OLD_POLICY = re.compile(r"Compatibility with CMake < 3\.5 has been removed")


def _policy_culprit(error: Optional[str]) -> str:
    """Which project's cmake_minimum_required it was: yaml-cpp, fetched."""
    match = re.search(r"/_deps/([A-Za-z0-9_.+-]+?)-src/", error or "")
    if match:
        return f"{match.group(1)} (fetched by the build)"
    match = re.search(r"CMake Error at (?:\S*/)?([^/\s]+)/CMakeLists\.txt:\d+", error or "")
    if match:
        return f"its {match.group(1)}/ subdirectory"
    return "the project"


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
        if _REQUIRED_FIND.search(stanza):
            continue   # find_path(... REQUIRED) names its own package's header
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
    env["PATH"] = _without_shims(env.get("PATH", ""))
    if sysroot is not None:
        # Last, so a host program comes first: the stubs there are only what a
        # build asked for, and tvm runs its llvm-config by bare name.
        env["PATH"] += os.pathsep + str(sysroot / "bin")
    return env


def _without_shims(path: str) -> str:
    """PATH less pyenv's and asdf's shims: a shim is there for every program any
    installed version has, and fails when the active one has none -- so a
    configure finds a cython that says "pyenv: cython: command not found"
    where it should have found nothing, and been given a stub."""
    shim = re.compile(r"/\.(pyenv|asdf)/shims/?$")
    return os.pathsep.join(d for d in path.split(os.pathsep) if d and not shim.search(d))


def _first_error(stdout: str, stderr: str) -> str:
    for stream in (stderr, stdout):
        lines = (stream or "").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("CMake Error", "ERROR:")):
                # "CMake Error at x.cmake:12 (message):" says where, not what.
                # What follows, wrapped over several indented lines, says what
                # -- and MFC's "Please use NVIDIA or Cray compilers" is on the
                # second of them.
                detail: list[str] = []
                for following in lines[index + 1:index + 6]:
                    if not following.strip():
                        if detail:
                            break
                        continue
                    if not following.startswith(" "):
                        break
                    detail.append(following.strip())
                if stripped.endswith(":") and detail:
                    return f"{stripped} {' '.join(detail)}"[:320]
                return stripped[:320]
    tail = (stdout or stderr or "").strip().splitlines()
    return tail[-1][:200] if tail else "configure did not complete"
