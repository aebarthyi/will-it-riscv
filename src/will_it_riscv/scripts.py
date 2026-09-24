"""What a project's own install scripts install before they build.

MFC is the case that motivates this. ``./mfc.sh build`` sources
``toolchain/bootstrap/python.sh``, which pip-installs ``toolchain/`` into a
venv before any CMake runs. So ``toolchain/pyproject.toml`` -- which the
manifest finder sets aside, because a manifest below the root usually
describes documentation or bindings -- is in fact the first thing the build
needs. Its jax pulls in jaxlib, which has no riscv64 build at all: the build
stops before it starts.

This reads shell; it never runs it. From each shell script at the root it
follows ``source`` chains and invoked scripts, resolves the handful of ways a
script spells "the repository root", treats a wrapper function that forwards
``"$@"`` to an installer as an installer, and reports every install it finds
with the chain of scripts that led there. Conditions are not evaluated: an
install anywhere in the scripts counts, because the one path a build takes
cannot be told apart statically from the ones it does not.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from packaging.requirements import InvalidRequirement, Requirement

#: Enough to cover real bootstrap chains without wandering a large tree.
MAX_SCRIPTS = 40
MAX_SCRIPT_BYTES = 256 * 1024

#: Root-level scripts that are the build system itself, or its plumbing --
#: never an install recipe. A generated configure is also enormous.
_NOT_ENTRY_POINTS = {
    "configure", "config.status", "config.guess", "config.sub", "libtool",
    "install-sh", "missing", "depcomp", "compile", "ltmain.sh", "mkinstalldirs",
    "py-compile", "test-driver", "ylwrap",
}

_SHEBANG = re.compile(rb"^#!\s*\S*(?:\bsh|\bbash|\bzsh|\bdash|env\s+(?:ba|z|da)?sh)\b")

#: Command words that print or log. "echo pip install x" installs nothing.
_TALKERS = {"echo", "printf", "print", "log", "warn", "error", "ok", "info", "msg", ":"}

#: Words that can sit in front of the command that actually runs.
_PREFIXES = {
    "if", "then", "else", "elif", "do", "while", "until", "!", "{", "}", "time",
    "exec", "command", "builtin", "sudo", "nohup", "env", "eval",
}

_OPERATORS = {";", "&&", "||", "|", "&", "(", ")", ";;", "|&"}
_REDIRECT = re.compile(r"^[0-9]*[<>][<>&|]*$")

#: pip options that take a value, so the value is not mistaken for a target.
_VALUED_OPTIONS = {
    "-c", "--constraint", "-i", "--index-url", "--extra-index-url", "-f",
    "--find-links", "-t", "--target", "--prefix", "--root", "--python", "-p",
    "--upgrade-strategy", "--platform", "--only-binary", "--no-binary",
    "--config-settings", "-C", "--src", "--progress-bar", "--log", "--cache-dir",
    "--trusted-host", "--timeout", "--python-version", "--implementation", "--abi",
    "--report", "--index-strategy", "--keyring-provider", "--resolution",
    "--prerelease", "--exclude-newer", "--override", "--group",
}

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_FUNCTION = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_:.-]*)\s*\(\s*\)\s*\{?")
_EXTRAS = re.compile(r"^(.*?)\[([A-Za-z0-9_,.\s-]+)\]$")


@dataclass(frozen=True)
class ScriptInstall:
    """One thing a project's scripts install."""

    kind: str
    """``manifest`` for a local pyproject.toml or requirements file, or
    ``requirement`` for a package named on the command line."""
    target: str
    """The manifest, relative to the repository root, or the requirement."""
    chain: tuple[str, ...]
    """How it was reached: the entry script, any scripts it sourced or ran,
    and ``path:line`` of the install itself."""
    extras: tuple[str, ...] = ()

    @property
    def via(self) -> str:
        return " → ".join(self.chain)


@dataclass
class _Command:
    file: Path
    line: int
    words: list[str]
    function: Optional[str]
    """The shell function this command sits inside, if any."""
    chain: tuple[str, ...]


@dataclass
class _Reading:
    commands: list[_Command] = field(default_factory=list)
    functions: dict[str, list[_Command]] = field(default_factory=dict)
    seen: set = field(default_factory=set)


def entry_scripts(root: Path) -> list[Path]:
    """Shell scripts at the top of the repository, the ones a person runs."""
    root = Path(root)
    found: list[Path] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        return found
    for path in children:
        if not path.is_file() or path.is_symlink() or path.name in _NOT_ENTRY_POINTS:
            continue
        if path.suffix in (".sh", ".bash"):
            found.append(path)
            continue
        if path.suffix:
            continue
        try:
            with path.open("rb") as handle:
                head = handle.read(128)
        except OSError:
            continue
        if _SHEBANG.match(head):
            found.append(path)
    return found


def find_script_installs(root: Path) -> list[ScriptInstall]:
    """Everything the root scripts install, with how each was reached."""
    root = Path(root).resolve()
    installs: dict[tuple, ScriptInstall] = {}
    for entry in entry_scripts(root):
        reading = _Reading()
        _read(entry, root, entry, {}, reading, ())
        for install in _installs(reading, root):
            key = (install.kind, install.target, install.extras)
            installs.setdefault(key, install)
    return list(installs.values())


def reached_files(root: Path) -> dict[str, tuple[str, ...]]:
    """Every file the root scripts source or run, with how each was reached.

    The scripts themselves, and the Python drivers they hand over to: mfc.sh
    reaches toolchain/bootstrap/python.sh, and then runs toolchain/main.py.
    Keyed by path relative to the root; the entry scripts map to ``()``.
    """
    root = Path(root).resolve()
    found: dict[str, tuple[str, ...]] = {}
    for entry in entry_scripts(root):
        reading = _Reading()
        _read(entry, root, entry, {}, reading, ())
        found.setdefault(_relative(entry.resolve(), root), ())
        for command in reading.commands:
            found.setdefault(_relative(command.file, root), command.chain[:-1])
            for word in command.words:
                if not word.endswith(".py") or "$" in word:
                    continue
                path = Path(word) if os.path.isabs(word) else root / word
                if path.is_file() and _is_inside(path.resolve(), root):
                    found.setdefault(_relative(path.resolve(), root), command.chain)
    return found


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ reading


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Physical lines with backslash continuations joined, numbered by start."""
    lines: list[tuple[int, str]] = []
    pending: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), 1):
        if not pending:
            start = number
        if raw.endswith("\\") and not raw.endswith("\\\\"):
            pending.append(raw[:-1])
            continue
        pending.append(raw)
        lines.append((start, " ".join(pending)))
        pending = []
    if pending:
        lines.append((start, " ".join(pending)))
    return lines


def _substitute(line: str, root: Path, entry: Path, current: Path, variables: dict) -> str:
    """Resolve the spellings of "here" a script uses, and variables it set.

    Entry scripts run from the root -- MFC refuses to run from anywhere
    else -- so the working directory is the root; ``$0`` is the entry
    script; ``BASH_SOURCE`` is whichever file is being read.
    """
    here = str(current.parent)
    replacements = [
        (r"\$\(\s*pwd(?:\s+-[LP])?\s*\)|`pwd`|\$\{PWD\}|\$PWD\b", str(root)),
        (r"\$\(\s*git\s+rev-parse\s+--show-toplevel\s*\)", str(root)),
        (r"\$\{BASH_SOURCE(?:\[0\])?%/\*\}", here),
        (r"\$\{BASH_SOURCE(?:\[0\])?\}|\$BASH_SOURCE\b", str(current)),
        (r"\$\{0\}|\$0\b", str(entry)),
    ]
    for _ in range(4):
        before = line
        for pattern, value in replacements:
            line = re.sub(pattern, _always(value), line)
        line = re.sub(
            r"\$\(\s*dirname\s+(?:--\s+)?\"?([^\"$()`]+?)\"?\s*\)",
            lambda m: os.path.dirname(m.group(1)),
            line,
        )
        line = re.sub(
            r"\$\(\s*cd\s+(?:--\s+)?\"?([^\"$()`]+?)\"?\s*(?:&&|;)\s*pwd(?:\s+-[LP])?\s*\)",
            lambda m: m.group(1),
            line,
        )
        for name, value in variables.items():
            line = re.sub(
                rf"\$\{{{re.escape(name)}\}}|\${re.escape(name)}\b", _always(value), line
            )
        if line == before:
            break
    return line


def _always(value: str) -> Callable[[re.Match], str]:
    """A re.sub replacement that inserts ``value`` literally, backslashes and all."""
    return lambda _match: value


def _words(line: str) -> Optional[list[str]]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError:  # an unbalanced quote: a heredoc, or shell this cannot read
        return None


def _simple_commands(words: list[str]) -> list[list[str]]:
    """Split at operators, and drop redirections: ``2>&1`` is not a package."""
    commands: list[list[str]] = [[]]
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if _REDIRECT.match(word):
            current = commands[-1]
            if current and current[-1].isdigit():
                current.pop()   # the file descriptor: the 2 of 2>&1
            skip = True         # and what it redirects to
            continue
        if word in _OPERATORS:
            commands.append([])
        else:
            commands[-1].append(word)
    return [c for c in commands if c]


def _strip_prefixes(words: list[str]) -> list[str]:
    index = 0
    while index < len(words):
        word = words[index]
        if word in _PREFIXES or _ASSIGNMENT.match(word):
            index += 1
            continue
        if word in ("flock",) and index + 2 < len(words):
            index += 2   # flock LOCKFILE command ...
            continue
        break
    return words[index:]


def _read(
    path: Path, root: Path, entry: Path, variables: dict, reading: _Reading,
    chain: tuple[str, ...],
) -> None:
    """Read one script, following what it sources or runs."""
    resolved = path.resolve()
    if resolved in reading.seen or len(reading.seen) >= MAX_SCRIPTS:
        return
    if not _is_inside(resolved, root):
        return
    reading.seen.add(resolved)
    try:
        text = resolved.read_bytes()[:MAX_SCRIPT_BYTES].decode("utf-8", errors="replace")
    except OSError:
        return

    function: Optional[str] = None
    depth = 0
    for number, raw in _logical_lines(text):
        line = _substitute(raw, root, entry, resolved, variables)
        words = _words(line)
        if not words:
            continue
        started = _FUNCTION.match(line)
        if started and function is None and words[0] not in _PREFIXES:
            function = started.group(1)
            reading.functions.setdefault(function, [])
            depth = 0
        opened = sum(1 for w in words if w == "{")
        closed = sum(1 for w in words if w == "}")
        where = f"{_relative(resolved, root)}:{number}"

        for words_ in _simple_commands(words):
            # Before the empty check: a line that only assigns is how a
            # script says where its root is, ROOT="$(cd "$(dirname "$0")" && pwd)".
            _remember_assignment(words_, variables)
            command = _strip_prefixes(words_)
            if not command:
                continue
            if function is not None and command[0] == function and len(command) == 1:
                continue
            record = _Command(resolved, number, command, function, chain + (where,))
            if function is not None:
                reading.functions[function].append(record)
            reading.commands.append(record)
            _follow(command, root, entry, variables, reading, chain + (where,))

        if function is not None:
            depth += opened - closed
            if depth <= 0 and (opened or closed):
                function = None


def _remember_assignment(words: list[str], variables: dict) -> None:
    """``NAME=/some/path`` makes ``$NAME`` resolvable from here on."""
    for word in words:
        if word in ("export", "local", "readonly", "declare"):
            continue
        match = _ASSIGNMENT.match(word)
        if not match:
            return
        name, value = match.groups()
        if value and "$" not in value and "`" not in value:
            variables[name] = value


def _follow(
    command: list[str], root: Path, entry: Path, variables: dict, reading: _Reading,
    chain: tuple[str, ...],
) -> None:
    """``source x`` and ``. x`` share variables; ``bash x`` and ``./x`` run apart."""
    head = command[0]
    target: Optional[str] = None
    shared = False
    if head in (".", "source") and len(command) > 1:
        target, shared = command[1], True
    elif head in ("bash", "sh", "zsh", "dash") and len(command) > 1:
        target = next((w for w in command[1:] if not w.startswith("-")), None)
    elif head.endswith((".sh", ".bash")) or head.startswith("./"):
        target = head
    if not target or "$" in target:
        return
    path = Path(target) if os.path.isabs(target) else root / target
    if path.is_file():
        _read(path, root, entry, variables if shared else dict(variables), reading, chain)


# ---------------------------------------------------------------- installs


def _install_arguments(words: list[str]) -> Optional[list[str]]:
    """The arguments of a pip install in this command, wherever it starts.

    ``pip install``, ``pip3 install``, ``python3 -m pip install``, ``uv pip
    install`` and ``flock lock uv pip install`` all end in the same two words.
    """
    if not words or os.path.basename(words[0]) in _TALKERS:
        return None
    for index in range(len(words) - 1):
        program = os.path.basename(words[index])
        if re.match(r"^pip(\d+(\.\d+)?)?$", program) and words[index + 1] == "install":
            return words[index + 2:]
    return None


def _forwards(arguments: list[str]) -> bool:
    return any(a in ("$@", "$*", "${@}", "${*}") for a in arguments)


def _installers(reading: _Reading) -> set[str]:
    """Functions that pass their arguments on to an install, however deep.

    MFC's ``uv_install_with_retry`` calls ``uv_install``, which runs
    ``uv pip install "$@"``: both are installers.
    """
    installers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, body in reading.functions.items():
            if name in installers:
                continue
            for command in body:
                arguments = _install_arguments(command.words)
                calls = command.words[0] in installers and _forwards(command.words[1:])
                if (arguments is not None and _forwards(arguments)) or calls:
                    installers.add(name)
                    changed = True
                    break
    return installers


def _installs(reading: _Reading, root: Path) -> list[ScriptInstall]:
    installers = _installers(reading)
    found: list[ScriptInstall] = []
    for command in reading.commands:
        arguments = _install_arguments(command.words)
        if arguments is None and command.words[0] in installers:
            arguments = command.words[1:]
        if arguments is None or _forwards(arguments):
            continue   # a wrapper's own body: its callers say what it installs
        found += _targets(arguments, root, command.chain)
    return found


def _targets(arguments: list[str], root: Path, chain: tuple[str, ...]) -> list[ScriptInstall]:
    """What one install command installs, if it can be read at all."""
    if "--no-deps" in arguments:
        return []   # installs the thing, not what it depends on
    targets: list[ScriptInstall] = []
    index = 0
    while index < len(arguments):
        word = arguments[index]
        index += 1
        value: Optional[str] = None
        requirements_file = False
        if word in ("-r", "--requirement", "-e", "--editable"):
            if index >= len(arguments):
                break
            value, requirements_file = arguments[index], word in ("-r", "--requirement")
            index += 1
        elif word.startswith(("--requirement=", "--editable=")):
            value = word.split("=", 1)[1]
            requirements_file = word.startswith("--requirement=")
        elif word.startswith("-r") and len(word) > 2:
            value, requirements_file = word[2:], True
        elif word in _VALUED_OPTIONS:
            index += 1
            continue
        elif word.startswith("-"):
            continue
        else:
            value = word
        if not value or "$" in value or "`" in value:
            continue
        target = _target(value, requirements_file, root, chain)
        if target is not None:
            targets.append(target)
    return targets


def _target(
    value: str, requirements_file: bool, root: Path, chain: tuple[str, ...]
) -> Optional[ScriptInstall]:
    extras: tuple[str, ...] = ()
    match = _EXTRAS.match(value)
    if match:
        value = match.group(1)
        extras = tuple(e.strip() for e in match.group(2).split(",") if e.strip())
    looks_local = requirements_file or value in (".", "..") or value.startswith(
        ("./", "../", "/")
    ) or "/" in value
    if looks_local:
        path = Path(value) if os.path.isabs(value) else root / value
        try:
            path = path.resolve()
        except OSError:
            return None
        if not _is_inside(path, root):
            return None
        if path.is_dir():
            if not (path / "pyproject.toml").is_file():
                return None
            path = path / "pyproject.toml"
        elif not path.is_file():
            return None
        return ScriptInstall("manifest", _relative(path, root), chain, extras)
    try:
        requirement = Requirement(value if not extras else f"{value}[{','.join(extras)}]")
    except InvalidRequirement:
        return None
    return ScriptInstall("requirement", str(requirement), chain)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
