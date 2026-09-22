"""Working out which CMake dependencies a *default* build actually needs.

A large CMake project spends most of its `find_package` calls on things
nobody builds by default. AdaptiveCpp is the clean example: its CUDA, ROCm
and Level Zero backends are each wrapped in `if(WITH_..._BACKEND)`, and those
variables default to whatever autodetection found -- which, on a riscv64
machine, is nothing. Reporting those as dependencies is how a project whose
minimal build needs LLVM and a C++ compiler looks like it needs three vendor
GPU stacks.

So: read the option defaults, then walk the `if`/`else`/`endif` structure and
decide whether each find is reachable when the user passes no `-D` flags at
all. Evaluation is three-valued -- true, false, and unknown -- and unknown is
treated as reachable, because guessing a real dependency away is the error
that matters.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .sdist import strip_cmake_comments


class Tri(Enum):
    """Three-valued logic. UNKNOWN means "we could not tell", not "maybe"."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __and__(self, other: Tri) -> Tri:
        if Tri.FALSE in (self, other):
            return Tri.FALSE
        if Tri.UNKNOWN in (self, other):
            return Tri.UNKNOWN
        return Tri.TRUE

    def __or__(self, other: Tri) -> Tri:
        if Tri.TRUE in (self, other):
            return Tri.TRUE
        if Tri.UNKNOWN in (self, other):
            return Tri.UNKNOWN
        return Tri.FALSE

    def __invert__(self) -> Tri:
        if self is Tri.TRUE:
            return Tri.FALSE
        if self is Tri.FALSE:
            return Tri.TRUE
        return Tri.UNKNOWN


def _cannot_exist(package: str) -> bool:
    """True for things with no build for the target at all -- the vendor GPU
    stacks, which the knowledge base already refuses to map."""
    from .syslibs import database, normalize

    return normalize(package) in database().ignore


TRUE_LITERALS = {"on", "true", "yes", "y", "1"}
FALSE_LITERALS = {"off", "false", "no", "n", "0", "", "ignore", "notfound"}

#: Platform variables CMake sets for us. The target is always Linux here, so
#: a Windows-only or Apple-only branch is dead code for our purposes.
PLATFORM_TRUE = {"unix", "linux", "cmake_host_unix"}
PLATFORM_FALSE = {
    "win32", "msvc", "apple", "mingw", "cygwin", "msys", "wince", "windows_store",
    "android", "ios", "emscripten", "borland", "watcom", "xcode",
    "cmake_host_win32", "cmake_host_apple", "cmake_host_solaris", "msvc_ide",
    "cmake_cross_compiling_emulator",
}

#: Commands whose arguments name something we might have to install.
FIND_COMMANDS = {"find_package", "pkg_check_modules", "pkg_search_module", "find_library"}

_COMMAND = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*\(", re.MULTILINE)
_VARIABLE_REF = re.compile(r"^\$\{([A-Za-z0-9_]+)\}$")
_FOUND_SUFFIX = re.compile(r"^(?P<package>[A-Za-z0-9_+.-]+)_FOUND$", re.IGNORECASE)


@dataclass
class Command:
    name: str
    args: str
    line: int


@dataclass
class Finding:
    """One dependency reference, and whether a default build reaches it."""

    name: str
    command: str
    line: int
    reachable: Tri = Tri.TRUE
    gate: Optional[str] = None
    """The condition that decides it, when it is not simply reachable."""
    required_keyword: bool = False
    quiet: bool = False

    @property
    def optional(self) -> bool:
        """True when a build with no -D flags would not need this."""
        return self.reachable is Tri.FALSE or (self.quiet and not self.required_keyword)


def iter_commands(text: str) -> Iterator[Command]:
    """Yield CMake commands, handling arguments that span lines."""
    text = strip_cmake_comments(text)
    position = 0
    while True:
        match = _COMMAND.search(text, position)
        if match is None:
            return
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth:
            return  # unbalanced; give up on the rest of the file
        yield Command(
            name=match.group("name").lower(),
            args=text[match.end() : index - 1],
            line=text.count("\n", 0, match.start()) + 1,
        )
        position = index


def _split_args(args: str) -> list[str]:
    return [a for a in re.split(r"[\s;]+", args.strip()) if a]


class Symbols:
    """Default values of the switches a user would flip with -D."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def learn(self, command: Command) -> None:
        args = _split_args(command.args)
        if not args:
            return
        if command.name == "option":
            # option(NAME "docstring" [DEFAULT]); CMake defaults to OFF.
            name = args[0]
            default = args[-1] if len(args) >= 3 else "OFF"
            self.values.setdefault(name.upper(), default)
        elif command.name == "set" and "CACHE" in [a.upper() for a in args]:
            cache_at = [a.upper() for a in args].index("CACHE")
            if cache_at >= 1:
                value = " ".join(args[1:cache_at]) if cache_at > 1 else ""
                self.values.setdefault(args[0].upper(), value)

    def evaluate(self, token: str) -> tuple[Tri, Optional[str]]:
        """Resolve one condition token to a truth value, and what decided it."""
        raw = token.strip().strip('"')
        lowered = raw.lower()

        if lowered in TRUE_LITERALS:
            return Tri.TRUE, None
        if lowered in FALSE_LITERALS or lowered.endswith("-notfound"):
            return Tri.FALSE, None
        if lowered in PLATFORM_TRUE:
            return Tri.TRUE, None
        if lowered in PLATFORM_FALSE:
            return Tri.FALSE, f"{raw} (not this platform)"

        reference = _VARIABLE_REF.match(raw)
        name = reference.group(1) if reference else raw
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            return Tri.UNKNOWN, None

        value = self.values.get(name.upper())
        if value is None:
            found = _FOUND_SUFFIX.match(name)
            if found and _cannot_exist(found.group("package")):
                # `if(CUDA_FOUND)` on a target with no CUDA. The variable is
                # set by a find_package we cannot run, but we know the answer.
                return Tri.FALSE, f"{name} (no build for this architecture)"
            # An undefined variable is false in CMake, but we cannot be sure
            # it is undefined -- it may be set somewhere we did not read.
            return Tri.UNKNOWN, name
        return self._value_of(value, name)

    def _value_of(
        self, value: str, name: str, depth: int = 0
    ) -> tuple[Tri, Optional[str]]:
        value = value.strip().strip('"')
        lowered = value.lower()
        if lowered in TRUE_LITERALS:
            return Tri.TRUE, name
        if lowered in FALSE_LITERALS or lowered.endswith("-notfound"):
            return Tri.FALSE, name

        reference = _VARIABLE_REF.match(value)
        if reference and depth < 4:
            inner = reference.group(1)
            found = _FOUND_SUFFIX.match(inner)
            if found:
                # `set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL ...)`: the
                # switch defaults to whatever autodetection turned up. A
                # default build on a clean machine detects nothing, so this
                # is off unless the user asks for it.
                return Tri.FALSE, f"{name} (autodetected from {inner})"
            nested = self.values.get(inner.upper())
            if nested is not None:
                return self._value_of(nested, name, depth + 1)
        return Tri.UNKNOWN, name


#: Condition operators we understand. Anything else makes the whole clause
#: unknown, which keeps its dependencies in the report.
_COMPARISONS = re.compile(
    r"\b(STREQUAL|EQUAL|MATCHES|LESS|GREATER|VERSION_\w+|IN_LIST|PATH_EQUAL)\b"
)


def evaluate_condition(args: str, symbols: Symbols) -> tuple[Tri, Optional[str]]:
    """Evaluate an ``if(...)`` argument list in the default configuration."""
    tokens = _split_args(args)
    if not tokens:
        return Tri.UNKNOWN, None
    upper = [t.upper() for t in tokens]

    # A comparison, a DEFINED test or a file test is beyond us.
    if _COMPARISONS.search(" ".join(upper)) or "DEFINED" in upper:
        return Tri.UNKNOWN, None
    for keyword in ("EXISTS", "COMMAND", "TARGET", "POLICY", "IS_DIRECTORY", "TEST"):
        if keyword in upper:
            return Tri.UNKNOWN, None

    # Split on OR first (lowest precedence), then AND.
    return _evaluate_or(tokens, symbols)


def _evaluate_or(tokens: list[str], symbols: Symbols) -> tuple[Tri, Optional[str]]:
    groups = _split_on(tokens, "OR")
    result: Optional[Tri] = None
    gate: Optional[str] = None
    for group in groups:
        value, group_gate = _evaluate_and(group, symbols)
        result = value if result is None else (result | value)
        gate = gate or group_gate
    return (result or Tri.UNKNOWN), gate


def _evaluate_and(tokens: list[str], symbols: Symbols) -> tuple[Tri, Optional[str]]:
    groups = _split_on(tokens, "AND")
    result: Optional[Tri] = None
    gate: Optional[str] = None
    for group in groups:
        value, group_gate = _evaluate_not(group, symbols)
        result = value if result is None else (result & value)
        if value is Tri.FALSE and group_gate:
            gate = group_gate
        gate = gate or group_gate
    return (result or Tri.UNKNOWN), gate


def _evaluate_not(tokens: list[str], symbols: Symbols) -> tuple[Tri, Optional[str]]:
    negate = False
    while tokens and tokens[0].upper() == "NOT":
        negate = not negate
        tokens = tokens[1:]
    if len(tokens) != 1:
        return Tri.UNKNOWN, None
    value, gate = symbols.evaluate(tokens[0])
    return (~value if negate else value), gate


def _split_on(tokens: list[str], operator: str) -> list[list[str]]:
    groups: list[list[str]] = [[]]
    for token in tokens:
        if token.upper() == operator:
            groups.append([])
        else:
            groups[-1].append(token)
    return groups


def _subject_names(command: str, args: list[str]) -> list[str]:
    """What a find command is actually looking for.

    ``find_package(Foo ...)`` names Foo, but ``find_library(VAR NAMES a b)``
    names a and b -- the first argument is the output variable, and NAMES is
    a keyword, not a library.
    """
    from .sdist import _cmake_tokens

    if command == "find_package":
        return args[:1]
    if command in ("pkg_check_modules", "pkg_search_module"):
        return list(_cmake_tokens(" ".join(args[1:])))
    if command == "find_library":
        return list(_cmake_tokens(" ".join(args[1:])))
    return args[:1]


@dataclass
class _Frame:
    """One if/elseif/else level, and whether we are inside a live branch."""

    reachable: Tri
    gate: Optional[str]
    any_branch_taken: Tri = Tri.FALSE


@dataclass
class FileAnalysis:
    findings: list[Finding] = field(default_factory=list)
    symbols: Symbols = field(default_factory=Symbols)


def analyze(text: str, symbols: Optional[Symbols] = None) -> FileAnalysis:
    """Walk one CMake file, tracking reachability through if/else blocks."""
    symbols = symbols if symbols is not None else Symbols()
    result = FileAnalysis(symbols=symbols)
    stack: list[_Frame] = []

    def current() -> tuple[Tri, Optional[str]]:
        reachable = Tri.TRUE
        gate = None
        for frame in stack:
            reachable = reachable & frame.reachable
            if frame.reachable is not Tri.TRUE and gate is None:
                gate = frame.gate
        return reachable, gate

    for command in iter_commands(text):
        if command.name == "if":
            value, gate = evaluate_condition(command.args, symbols)
            stack.append(_Frame(reachable=value, gate=gate, any_branch_taken=value))
        elif command.name == "elseif" and stack:
            frame = stack[-1]
            value, gate = evaluate_condition(command.args, symbols)
            frame.reachable = value & ~frame.any_branch_taken
            frame.gate = gate
            frame.any_branch_taken = frame.any_branch_taken | value
        elif command.name == "else" and stack:
            frame = stack[-1]
            frame.reachable = ~frame.any_branch_taken
            frame.gate = frame.gate
        elif command.name == "endif" and stack:
            stack.pop()
        elif command.name in ("option", "set"):
            symbols.learn(command)
        elif command.name in FIND_COMMANDS:
            reachable, gate = current()
            args = _split_args(command.args)
            if not args:
                continue
            upper = [a.upper() for a in args]
            for name in _subject_names(command.name, args):
                result.findings.append(
                    Finding(
                        name=name,
                        command=command.name,
                        line=command.line,
                        reachable=reachable,
                        gate=gate,
                        required_keyword="REQUIRED" in upper,
                        quiet="QUIET" in upper,
                    )
                )
    return result


def collect_symbols(texts: Iterator[str]) -> Symbols:
    """Pre-pass: learn every option default before judging any condition."""
    symbols = Symbols()
    for text in texts:
        for command in iter_commands(text):
            if command.name in ("option", "set"):
                symbols.learn(command)
    return symbols
