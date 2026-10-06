"""An agent for the configures the executor's rules cannot get past.

The executor (:mod:`pseudobuild`) unblocks a configure by rule: what the
error names, what the Find module looked for, which suspect moves the
error. Its long tail is one-offs -- ELPA's version file in a format no rule
knows, an export set that wants an imported target the stub lacks, a
version grep with an unusual regex -- and every hand-written rule for them
reached one or two repos. So when the rules run out of suspects, this hands
the stuck point to a model with a workbench:

- read-only: the error, the trace around it, what each Find module searched
  for, the source tree, the sysroot, CMake's own modules;
- write: stub files inside the scratch sysroot, and cache variables that
  are the results of a lookup -- never the plan's -D flags, never a build
  option, never anything of CMake's own, never the source tree;
- run: the configure again, with its stubs, a few times.

What it leaves behind is a trial like any other: the executor's next round
judges it. The error moved -- the stub's package is a hard requirement, shown
by experiment. It did not -- every file is taken back out. An agent that
says the configure needs a flag the plan does not give, or something no stub
can be, leaves a note instead.

The agent is Claude, through ``claude -p``, with the workbench served to it
over MCP (``python -m will_it_riscv.unblock serve SESSION``). It is opt-in:
``WILL_IT_RISCV_UNBLOCK=claude`` (or ``claude:<model>``) turns it on for
every configure the process runs. Each session's prompt, transcript, ledger
and stubs are kept under ``$WILL_IT_RISCV_UNBLOCK_LOG`` (default
``~/.cache/will-it-riscv/unblock``), to read, to mine for rules, or to train
a smaller unblocker on.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import pseudobuild as pb
from .modelplan import claude_environment

#: What a session may end with.
UNBLOCKED = "unblocked"
PLAN_FAULT = "plan_fault"
CANNOT_STUB = "cannot_stub"
GAVE_UP = "gave_up"
OUTCOMES = (UNBLOCKED, PLAN_FAULT, CANNOT_STUB, GAVE_UP)

#: What the blocker was: a package the build needs, or something every
#: riscv64 Linux system has and the sandbox lacked (a header, a CMake quirk).
DEPENDENCY = "dependency"
HOST = "host"

#: Per session.
MAX_CONFIGURES = 8
MAX_FILE_BYTES = 200_000
#: Per configure loop: how many stuck points get a session. Two was too
#: few: in the first benchmark, 19 of the 56 left unknown were still getting
#: further each time when they ran out.
SESSIONS = 6

#: How much of one tool's answer the agent sees.
_ANSWER_CHARS = 12_000


# ------------------------------------------------------------------ the hand-over


@dataclass
class Stuck:
    """Where the executor stopped, and everything needed to carry on from there."""

    root: Path
    scratch: Path
    sysroot: Path
    toolchain: Optional[Path]
    env: dict
    settings: dict
    """Every -D the stuck round was configured with: the executor's stubs,
    what the build fetches for itself, the plan's own flags."""
    fixed: dict
    """The plan's -D flags. Never overridden."""
    outcome: pb.PseudoBuild
    """The stuck round."""
    round: int
    trace: Path
    build: Path
    overrides: dict
    """The cache variables the executor's stubs set so far."""
    blockers: list
    timeout: int
    """Seconds each configure may take."""
    earlier: list = field(default_factory=list)
    """What earlier sessions in this loop did, as records."""
    arch: str = "riscv64"


@dataclass
class Result:
    """What a session came to."""

    outcome: str
    name: Optional[str] = None
    kind: str = DEPENDENCY
    reason: str = ""
    flags: dict = field(default_factory=dict)
    also: list = field(default_factory=list)
    """Other packages stubbed with the blocker and blamed with it."""
    version: Optional[str] = None
    """The version the configure demanded, as the agent read it: ">= 4.10.1"."""
    replayed: Optional[str] = None
    """The log of the session whose stubs this one put back, if it is a replay."""
    overrides: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    """Files it made, to take back out if the executor rejects them."""
    restore: dict = field(default_factory=dict)
    """Files it changed, path to what they said before."""
    log: Optional[str] = None
    cost: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    configures: int = 0
    memo: Optional[tuple] = None
    """Where it is remembered for replay, to forget it if it is rejected."""

    @property
    def proposes(self) -> bool:
        """Whether there is a trial to judge."""
        changed = self.files or self.overrides or self.restore
        return self.outcome == UNBLOCKED and bool(self.name) and bool(changed)

    def record(self) -> dict:
        """What the configure's result keeps of it."""
        return {
            "outcome": self.outcome, "name": self.name, "kind": self.kind, "also": self.also,
            "version": self.version, "replayed": self.replayed,
            "reason": self.reason, "flags": self.flags, "overrides": sorted(self.overrides),
            "files": len(self.files) + len(self.restore), "confirmed": None,
            "configures": self.configures, "cost": self.cost, "log": self.log,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
        }


# ------------------------------------------------------------------ the workbench


@functools.lru_cache(maxsize=1)
def cmake_modules() -> Optional[Path]:
    """CMake's own modules directory: FindHDF5.cmake and the like."""
    binary = pb.cmake()
    if not binary:
        return None
    script = Path(tempfile.gettempdir()) / f"will-it-riscv-cmake-root-{os.getpid()}.cmake"
    try:
        script.write_text('message("${CMAKE_ROOT}")\n')
        done = subprocess.run([binary, "-P", str(script)], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        script.unlink(missing_ok=True)
    found = Path(done.stderr.strip()) / "Modules"
    return found if found.is_dir() else None


class Refused(Exception):
    """A workbench call the rules do not allow; the agent is told why."""


class Workbench:
    """The tools, over one session directory.

    Everything the agent changes is in the ledger, written after every
    change, so the driver can hand it back -- or take it all out again --
    whatever state the agent stopped in.
    """

    def __init__(self, session: Path, finish_tool: str = "finish"):
        self.session = Path(session)
        self.finish_tool = finish_tool   # what ends the work here: the agent's is resume
        state = json.loads((self.session / "state.json").read_text())
        self.state = state
        self.root = Path(state["root"])
        self.scratch = Path(state["scratch"])
        self.sysroot = Path(state["sysroot"]).resolve()
        self.toolchain = Path(state["toolchain"]) if state.get("toolchain") else None
        self.env = {**os.environ, **state["env"]}
        self.settings = state["settings"]
        self.fixed = state["fixed"]
        self.signature = state["signature"]
        self.timeout = state["timeout"]
        self.max_configures = state.get("max_configures", MAX_CONFIGURES)
        self.stuck_trace = Path(state["trace"])
        ledger = self.session / "ledger.json"
        self.ledger = json.loads(ledger.read_text()) if ledger.is_file() else {
            "created": [], "changed": {}, "cache": {}, "configures": [], "finish": None,
        }
        self._vocabulary: Optional[dict] = None

    # -- bookkeeping

    def _save(self) -> None:
        (self.session / "ledger.json").write_text(json.dumps(self.ledger, indent=1))

    @property
    def trace(self) -> Path:
        """The latest configure's trace: the stuck round's, until the agent runs one."""
        runs = self.ledger["configures"]
        return Path(runs[-1]["trace"]) if runs else self.stuck_trace

    def _where(self, file: str) -> str:
        path = Path(file)
        for base, label in ((self.root, ""), (cmake_modules(), "<cmake>/"),
                            (self.sysroot, "<sysroot>/"), (self.scratch, "<scratch>/")):
            if base is not None:
                try:
                    return label + path.resolve().relative_to(base.resolve()).as_posix()
                except (ValueError, OSError):
                    continue
        return file

    def _inside_sysroot(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.sysroot / candidate
        resolved = candidate.resolve()
        if resolved != self.sysroot and self.sysroot not in resolved.parents:
            raise Refused(f"{path} is outside the sysroot ({self.sysroot}); stubs go there only")
        return resolved

    # -- reading

    def show_error(self) -> str:
        state = self.state
        lines = [
            f"project: {self.root}",
            f"stuck at round {state['round']}; this round's build directory: {state['build']}",
            f"trace: {self.stuck_trace}",
            "",
            "the error:",
            _clip("\n\n".join(state["stanzas"]) or state["error"] or "(no CMake Error stanza)",
                  6000),
            "",
            "status lines just before it:",
            "\n".join(state["status"]) or "(none)",
            "",
            "packages the configure asked for by name (find_package / pkg-config), latest "
            "last: " + (", ".join(self.vocabulary()["packages"][-40:]) or "none"),
            "blockers the executor already stubbed, in order: "
            + (", ".join(state["blockers"]) or "none"),
            "cache variables it set: " + (", ".join(sorted(state["overrides"])) or "none"),
            "the plan's -D flags (never changed): "
            + (", ".join(f"{k}={v}" for k, v in sorted(self.fixed.items())) or "none"),
        ]
        if state.get("earlier"):
            lines += ["", "earlier sessions at other stuck points in this configure:"]
            lines += [f"- {e['outcome']} {e.get('name') or ''}: {e.get('reason', '')}"
                      f" (executor confirmed: {e.get('confirmed')})" for e in state["earlier"]]
        return "\n".join(lines)

    def show_trace(self, around: Optional[str] = None, grep: Optional[str] = None,
                   before: int = 40, after: int = 10, limit: int = 80) -> str:
        """The latest configure's trace, as ``file:line cmd(args)``."""
        if not self.trace.is_file():
            return f"no trace at {self.trace}"
        if around:
            match = re.match(r"^(.*?):(\d+)$", around.strip())
            if not match:
                raise Refused('around is "path:line", as a CMake Error names it')
            want_file, want_line = match.group(1), int(match.group(2))
            window: deque = deque(maxlen=max(1, min(before, 200)))
            hit: Optional[list] = None
            tail: list = []
            for event in _events(self.trace):
                text = self._render(event)
                if hit is not None and len(tail) < after:
                    tail.append(text)
                if (int(event.get("line", -1)) == want_line
                        and str(event.get("file", "")).endswith(want_file.lstrip("./"))):
                    hit, tail = list(window), []
                    hit.append(">> " + text)
                window.append(text)
            if hit is None:
                return f"nothing in the trace ran at {around}"
            return _clip("\n".join(hit + tail), _ANSWER_CHARS, keep="tail")
        if grep:
            try:
                pattern = re.compile(grep)
            except re.error as exc:
                raise Refused(f"not a regex: {exc}") from None
            kept: deque = deque(maxlen=max(1, min(limit, 300)))
            count = 0
            for event in _events(self.trace):
                text = self._render(event)
                if pattern.search(text):
                    count += 1
                    kept.append(text)
            shown = f"; the last {len(kept)}:" if count > len(kept) else ":"
            head = f"{count} events match{shown}"
            return _clip(head + "\n" + "\n".join(kept), _ANSWER_CHARS, keep="tail")
        raise Refused("give around (path:line) or grep (a regex)")

    def _render(self, event: dict) -> str:
        args = " ".join(str(a) for a in event.get("args", []))
        if len(args) > 300:
            args = args[:300] + " …"
        where = self._where(str(event.get("file", "")))
        return f"{where}:{event.get('line')} {event.get('cmd')}({args})"

    def show_lookups(self) -> str:
        """What the latest configure looked for, and how."""
        lookups: dict = {}
        reads: list = []
        exists: list = []
        searches: list = []
        probes, _ = pb._parse_trace(self.trace, self.root, lookups, reads, exists, searches)
        lines = ["find_package / pkg-config / find_library questions "
                 "(name, command, required, site):"]
        lines += [f"  {p.name} {p.command}{' REQUIRED' if p.required else ''}"
                  f"{' via ' + p.via if p.via else ''}{' in ' + p.parent if p.parent else ''}"
                  f" at {p.site or '?'}" for p in list(probes.values())[-80:]]
        lines += ["searches (command, result variable, names), last 80:"]
        lines += [f"  {c} {v} {' '.join(n)}" for c, v, n in searches[-80:]]
        lines += ["if(EXISTS ...) asked of (path, asking file), last 40:"]
        lines += [f"  {p}  ({self._where(m)})" for p, m in exists[-40:]]
        lines += ["files grepped and the regex, last 40:"]
        lines += [f"  {self._where(f)}  /{r}/" for f, r in reads[-40:]]
        return _clip("\n".join(lines), _ANSWER_CHARS)

    def show_sysroot(self) -> str:
        made = set(self.ledger["created"]) | set(self.ledger["changed"])
        lines = []
        for path in sorted(self.sysroot.rglob("*")):
            if path.is_file():
                mark = " (yours)" if str(path) in made else ""
                lines.append(f"{path.relative_to(self.sysroot)} {path.stat().st_size}B{mark}")
        return _clip("\n".join(lines) or "(empty)", _ANSWER_CHARS)

    # -- writing

    def write_file(self, path: str, content: str, executable: bool = False) -> str:
        target = self._inside_sysroot(path)
        if len(content.encode()) > MAX_FILE_BYTES:
            raise Refused(f"over {MAX_FILE_BYTES} bytes; a stub needs only what is looked for")
        key = str(target)
        if key not in self.ledger["created"] and key not in self.ledger["changed"]:
            if target.exists():
                # The executor's own stub: kept, to put back if this fails.
                backup = self.session / "backup" / str(len(self.ledger["changed"]))
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, backup)
                self.ledger["changed"][key] = str(backup)
            else:
                self.ledger["created"].append(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        if executable:
            target.chmod(0o755)
        self._save()
        return f"wrote {target.relative_to(self.sysroot)} ({len(content)} chars)"

    def remove_file(self, path: str) -> str:
        target = self._inside_sysroot(path)
        key = str(target)
        if key in self.ledger["created"]:
            target.unlink(missing_ok=True)
            self.ledger["created"].remove(key)
        elif key in self.ledger["changed"]:
            shutil.copyfile(self.ledger["changed"].pop(key), target)
            self._save()
            return f"restored {target.relative_to(self.sysroot)} to what the executor wrote"
        else:
            raise Refused("only files this session wrote can be removed")
        self._save()
        return f"removed {target.relative_to(self.sysroot)}"

    def vocabulary(self) -> dict:
        """What the stuck round's trace says each name is.

        ``options``: the build's own choices -- option(), cmake_dependent_option(),
        set(... CACHE BOOL ...). ``packages``: what find_package and pkg-config
        were asked for. ``results``: the variables a find_* filled.
        ``checks``: the names a check_* command was given, its result among them.
        """
        if self._vocabulary is None:
            words: dict = {"options": set(), "packages": [], "results": set(), "checks": set()}
            for event in _events(self.stuck_trace):
                command = str(event.get("cmd", "")).lower()
                args = [str(a) for a in event.get("args", []) if str(a).strip()]
                if not args:
                    continue
                if command in ("option", "cmake_dependent_option"):
                    words["options"].add(args[0])
                elif command == "set" and "CACHE" in args:
                    at = args.index("CACHE")
                    if at + 1 < len(args) and args[at + 1].upper() == "BOOL":
                        words["options"].add(args[0])
                elif command in pb.FIND_COMMANDS:
                    for name in pb._subjects(command, args):
                        if name not in words["packages"]:
                            words["packages"].append(name)
                elif command in ("find_library", "find_path", "find_file", "find_program"):
                    words["results"].add(args[0])
                elif command.startswith("check_"):
                    words["checks"].update(a for a in args if re.match(r"^[A-Za-z_]\w*$", a))
            self._vocabulary = words
        return self._vocabulary

    def set_cache(self, name: str, value: str) -> str:
        why = self._cache_refusal(name, value)
        if why:
            raise Refused(f"-D{name} is not for a stub to set: {why}")
        self.ledger["cache"][name] = value
        self._save()
        return f"-D{name}={value} from the next configure on"

    def _cache_refusal(self, name: str, value: str) -> Optional[str]:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_.+-]*$", name):
            return "not a variable name"
        if name in self.fixed:
            return "the plan sets it"
        if name.upper().startswith(("CMAKE_", "_CMAKE")):
            return "CMake's own settings are the sandbox's, not a dependency's"
        words = self.vocabulary()
        if name in words["options"] or re.match(r"(?i)^(WITH|ENABLE|USE|BUILD)_", name):
            return ("it is a build option, and the default build is the spec; if the configure "
                    "cannot go on without changing it, finish with plan_fault")
        packages = {p.lower() for p in words["packages"]}
        lookup = (
            name in words["results"] or name in words["checks"] or "HAVE_" in name.upper()
            or pb._VAR_STEM.search(name) is not None or name.upper().endswith("_VERSION")
            or (name.endswith("_DIR") and name[:-4].lower() in packages)
            or self._asked_for(name)
        )
        if not lookup:
            return ("it is neither what a find_* or check_* filled nor a package's _DIR; if "
                    "the configure needs a setting of the project's own, that is plan_fault")
        for token in re.split(r"[;\s]", value):
            if token.startswith("/") and not any(
                Path(token).resolve().is_relative_to(base.resolve())
                for base in (self.sysroot, self.scratch, self.root)
            ):
                return f"{token} is on the host; a found path has to be in the sysroot"
        return None

    def _asked_for(self, name: str) -> bool:
        """Whether the error stopping the configure asks for this install
        location by name: axom's "Sina requires Conduit.  Set CONDUIT_DIR to
        location of a Conduit install." Pointing it at a stub in the sysroot
        is what a person would do with a real install."""
        return (name.upper().endswith(("_DIR", "_ROOT", "_PREFIX", "_PATH"))
                and re.search(rf"\bset\s+{re.escape(name)}\b", self.signature or "", re.I)
                is not None)

    def unset_cache(self, name: str) -> str:
        if self.ledger["cache"].pop(name, None) is None:
            raise Refused(f"{name} is not one this session set")
        self._save()
        return f"-D{name} dropped"

    # -- running

    def configure(self) -> str:
        runs = self.ledger["configures"]
        if len(runs) >= self.max_configures:
            raise Refused(f"all {self.max_configures} configures are spent; "
                          f"{self.finish_tool} now")
        label = f"unblock-{self.state['session']}-{len(runs) + 1}"
        started = time.monotonic()
        outcome = pb._configure(
            self.root, self.scratch, {**self.settings, **self.ledger["cache"]}, self.env,
            self.timeout, label, self.toolchain,
        )
        moved = outcome.completed or pb._signature(outcome.error) != self.signature
        runs.append({
            "trace": str(self.scratch / f"{label}.trace.json"), "build": str(self.scratch / label),
            "completed": outcome.completed, "moved": moved, "error": outcome.error,
            "seconds": round(time.monotonic() - started, 1),
        })
        self._save()
        left = self.max_configures - len(runs)
        if outcome.completed:
            return (f"configure {len(runs)}: COMPLETED. The error moved. "
                    f"Call {self.finish_tool} now "
                    f"({left} configures left).")
        stanzas = pb._error_stanzas(outcome.narration)
        verdict = ("the error MOVED: it is no longer the one you were given. If this is a "
                   f"different problem, call {self.finish_tool} now; the executor takes it "
                   "from here."
                   if moved else "the SAME error as before: not past it yet.")
        return "\n".join([
            f"configure {len(runs)}: stopped; {verdict} ({left} configures left)",
            f"build directory: {self.scratch / label}; trace: {runs[-1]['trace']}",
            "",
            _clip("\n\n".join(stanzas[:2]) or (outcome.error or ""), 5000),
            *([f"(+{len(stanzas) - 2} more error stanzas)"] if len(stanzas) > 2 else []),
            "",
            "status lines just before it:",
            "\n".join(_status_before_error(outcome.narration)) or "(none)",
        ])

    def finish(self, outcome: str, reason: str, blocker: Optional[str] = None,
               kind: str = DEPENDENCY, flags: Optional[dict] = None,
               also: Optional[list] = None, version: Optional[str] = None) -> str:
        if self.ledger["finish"] is not None:
            raise Refused("already finished")
        if outcome not in OUTCOMES:
            raise Refused(f"outcome is one of {', '.join(OUTCOMES)}")
        if kind not in (DEPENDENCY, HOST):
            raise Refused(f"kind is {DEPENDENCY} or {HOST}")
        for name in [blocker, *(also or [])]:
            # The name becomes a node of the dependency graph, and is looked
            # up in the archive: "NetCDF-C (netcdf_meta.h)" finds nothing.
            if name is not None and not _NAME.match(name):
                raise Refused(
                    f"{name!r} is not a package name: one word, as the configure asked for it "
                    "(find_package(HDF5) is HDF5, pkg-config netcdf is netcdf, a header is "
                    "dlfcn.h); put other packages in also and details in reason"
                )
        if outcome == UNBLOCKED:
            if not blocker:
                raise Refused("unblocked needs the blocker's name")
            if not (self.ledger["created"] or self.ledger["changed"] or self.ledger["cache"]):
                raise Refused("unblocked, with nothing written or set?")
        self.ledger["finish"] = {
            "outcome": outcome, "reason": reason, "blocker": blocker, "kind": kind,
            "flags": flags or {}, "also": list(also or []), "version": version,
        }
        self._save()
        return "recorded. Stop here; reply with one line."

    # -- the driver's side

    def result(self) -> Result:
        """The session as a trial for the executor: everything it left, if it
        says it got past; else nothing, and everything taken back out."""
        finish = self.ledger["finish"] or {"outcome": GAVE_UP, "reason": "never called finish"}
        result = Result(
            outcome=finish["outcome"], name=finish.get("blocker"),
            kind=finish.get("kind") or DEPENDENCY, reason=finish.get("reason") or "",
            flags=finish.get("flags") or {}, configures=len(self.ledger["configures"]),
            also=finish.get("also") or [], version=finish.get("version"),
        )
        if result.outcome == UNBLOCKED:
            result.overrides = dict(self.ledger["cache"])
            result.files = list(self.ledger["created"])
            result.restore = {path: Path(backup).read_text(errors="replace")
                              for path, backup in self.ledger["changed"].items()}
        else:
            self.rollback()
        return result

    def rollback(self) -> None:
        for path in self.ledger["created"]:
            Path(path).unlink(missing_ok=True)
        for path, backup in self.ledger["changed"].items():
            shutil.copyfile(backup, path)
        self.ledger["created"], self.ledger["changed"], self.ledger["cache"] = [], {}, {}
        self._save()


#: A package, program or header name, as a configure asks for it.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+/-]*$")


def _events(trace: Path):
    try:
        handle = trace.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("cmd"):
                yield event


def _status_before_error(narration: str, count: int = 20) -> list[str]:
    status: list[str] = []
    for line in narration.splitlines():
        if line.startswith(("CMake Error", "  CMake Error")):
            break
        if line.startswith("-- "):
            status.append(line)
    return status[-count:]


def _clip(text: str, limit: int, keep: str = "head") -> str:
    if len(text) <= limit:
        return text
    if keep == "tail":
        return "[…]\n" + text[-limit:]
    return text[:limit] + "\n[…]"


# ------------------------------------------------------------------ MCP


TOOLS = [
    {
        "name": "show_error",
        "description": "The stuck error in full, the status lines before it, and what the "
                       "executor already did: its blockers, cache variables, the plan's flags.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "show_trace",
        "description": "The latest configure's CMake trace (every command run, arguments "
                       "expanded), one 'file:line cmd(args)' per event. Either around a "
                       "'path:line' -- the last time that line ran, with what ran before it, "
                       "marked '>>' -- or grep: the last events whose line matches a regex.",
        "inputSchema": {"type": "object", "properties": {
            "around": {"type": "string", "description": "path:line, as a CMake Error names it"},
            "grep": {"type": "string", "description": "a Python regex over the rendered line"},
            "before": {"type": "integer", "default": 40},
            "after": {"type": "integer", "default": 10},
            "limit": {"type": "integer", "default": 80},
        }},
    },
    {
        "name": "show_lookups",
        "description": "What the latest configure looked for: each find_package and pkg-config "
                       "question, every find_library/find_path/find_program/find_file with its "
                       "result variable and names, the paths if(EXISTS) asked about, and the "
                       "files it grepped with the regex.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "show_sysroot",
        "description": "Every file in the scratch sysroot, with sizes; yours are marked.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "write_file",
        "description": "Write a stub file inside the sysroot (a path relative to it, or "
                       "absolute inside it): a header, an empty lib<name>.so, a "
                       "<Name>Config.cmake, a .pc file, a program script (executable: true). "
                       "Rewriting one of the executor's stubs is allowed and is undone if the "
                       "session's trial fails.",
        "inputSchema": {"type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "executable": {"type": "boolean", "default": False},
        }},
    },
    {
        "name": "remove_file",
        "description": "Take back a file this session wrote (or restore one it rewrote).",
        "inputSchema": {"type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "set_cache",
        "description": "Pass -D<name>=<value> on later configures: only for a lookup's result "
                       "(FOO_LIBRARY, FOO_INCLUDE_DIR, Foo_DIR, a check's result variable), or an "
                       "install location the error asks for by name (\"Set FOO_DIR to ...\"). "
                       "Refused for the plan's flags, build options, CMAKE_* and host paths.",
        "inputSchema": {"type": "object", "required": ["name", "value"], "properties": {
            "name": {"type": "string"}, "value": {"type": "string"},
        }},
    },
    {
        "name": "unset_cache",
        "description": "Stop passing a -D this session set.",
        "inputSchema": {"type": "object", "required": ["name"],
                        "properties": {"name": {"type": "string"}}},
    },
    {
        "name": "configure",
        "description": "Run the configure again with every stub and -D so far. Says whether the "
                       "error moved. Limited runs.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "finish",
        "description": "End the session. outcome: unblocked (your stubs get past the error; name "
                       "the blocker and its kind), plan_fault (the configure needs a flag or "
                       "setting the plan does not give: put it in flags), cannot_stub (needs "
                       "something no stub can be; name it as blocker if it is a dependency the "
                       "default build demands, such as CUDA), gave_up.",
        "inputSchema": {"type": "object", "required": ["outcome", "reason"], "properties": {
            "outcome": {"type": "string", "enum": list(OUTCOMES)},
            "reason": {"type": "string", "description": "one or two sentences: what stopped "
                       "the configure and what you did"},
            "blocker": {"type": "string", "description": "one word: the package as the "
                        "project's own find_package / pkg-config / find_library names it, or "
                        "the host header or tool"},
            "kind": {"type": "string", "enum": [DEPENDENCY, HOST], "default": DEPENDENCY},
            "also": {"type": "array", "items": {"type": "string"},
                     "description": "other packages this error needed stubbed at once "
                                    "(FindX11's Xft wants Freetype and Fontconfig)"},
            "version": {"type": "string", "description": "the version the configure "
                        "demanded of the blocker, if it checked one: '>= 4.10.1'"},
            "flags": {"type": "object", "description": "for plan_fault: the -D flags it wants",
                      "additionalProperties": {"type": "string"}},
        }},
    },
]


def call(bench: Workbench, name: str, arguments: dict) -> tuple[str, bool]:
    """One tool call: the answer, and whether it was refused."""
    methods = {
        "show_error": bench.show_error, "show_trace": bench.show_trace,
        "show_lookups": bench.show_lookups, "show_sysroot": bench.show_sysroot,
        "write_file": bench.write_file, "remove_file": bench.remove_file,
        "set_cache": bench.set_cache, "unset_cache": bench.unset_cache,
        "configure": bench.configure, "finish": bench.finish,
    }
    method = methods.get(name)
    if method is None:
        return f"no tool {name}", True
    try:
        return method(**(arguments or {})), False
    except Refused as exc:
        return f"refused: {exc}", True
    except TypeError as exc:
        return f"bad arguments: {exc}", True
    except OSError as exc:
        return f"failed: {exc}", True


def serve(session: Path, stdin=None, stdout=None) -> None:
    """The workbench as an MCP server on stdio."""
    bench = Workbench(session)
    serve_tools(TOOLS, lambda name, arguments: call(bench, name, arguments),
                Path(session) / "calls.jsonl", stdin, stdout)


def serve_tools(tools: list, dispatch, log: Path, stdin=None, stdout=None,
                closing=None) -> None:
    """Tools as an MCP server on stdio: newline-delimited JSON-RPC.
    ``dispatch(name, arguments)`` answers ``(text, refused)``; every call is
    logged; ``closing`` runs when the client goes away."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    calls = Path(log).open("a")

    def reply(message_id, result=None, error=None) -> None:
        body = {"jsonrpc": "2.0", "id": message_id}
        body.update({"error": error} if error else {"result": result})
        stdout.write(json.dumps(body) + "\n")
        stdout.flush()

    try:
        for line in stdin:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            method, message_id = message.get("method"), message.get("id")
            if message_id is None:
                continue   # a notification: initialized, cancelled
            if method == "initialize":
                version = (message.get("params") or {}).get("protocolVersion", "2024-11-05")
                reply(message_id, {
                    "protocolVersion": version, "capabilities": {"tools": {}},
                    "serverInfo": {"name": "will-it-riscv", "version": "0.1"},
                })
            elif method == "ping":
                reply(message_id, {})
            elif method == "tools/list":
                reply(message_id, {"tools": tools})
            elif method == "tools/call":
                params = message.get("params") or {}
                started = time.monotonic()
                text, refused = dispatch(params.get("name", ""), params.get("arguments") or {})
                calls.write(json.dumps({
                    "tool": params.get("name"), "arguments": params.get("arguments"),
                    "refused": refused, "seconds": round(time.monotonic() - started, 1),
                    "answer": text[:2000],
                }) + "\n")
                calls.flush()
                reply(message_id, {"content": [{"type": "text", "text": text}], "isError": refused})
            else:
                reply(message_id, error={"code": -32601, "message": f"no method {method}"})
    finally:
        if closing is not None:
            closing()
        calls.close()


# ------------------------------------------------------------------ the agent


SYSTEM = """\
You unblock CMake configures in a pretend {arch} sandbox.

will-it-riscv answers "will this project build on {arch}?" by running the project's \
configure step -- never its build -- as a cross build for Linux/{arch}, confined to an \
empty scratch sysroot. Nothing is installed and nothing is emulated. Whatever the configure \
insists on is given a stub: files that exist only to be found (a header, an empty \
lib<name>.so, a <Name>Config.cmake, a .pc file, a program script). Each package the \
configure would not go on without is a hard requirement of the default build, which is \
the minimal spec.

The executor's rules stub most blockers. This configure stopped somewhere they could not \
get past. Your job: find out why it stopped, and stub what it was looking for so it gets \
past THIS error -- the way the real package, installed on a current Debian or Ubuntu for {arch}, \
would be found.

Rules:
- Write only inside the sysroot (write_file) and set only lookup results (set_cache). Never \
the source tree, never a build option, never the plan's -D flags, never CMAKE_*.
- Stub exactly what this error needs, no more: not the next package you expect it to want. \
Claim no feature nobody asked for (no parallel HDF5, no MPI-enabled anything, no optional \
component) unless this configure demands it.
- Never impersonate a compiler or toolchain: no fake nvcc, hipcc, gfortran or compiler \
identification. A default build that enables CUDA or HIP needs the vendor's compiler, which no \
stub can be: finish with cannot_stub, blocker CUDA (or HIP), kind dependency. Whether {arch} can \
have it is the resolver's call, not yours.
- A version check gets a version that passes: what the target distro ships if you know it and it \
passes, else the oldest the check accepts. Real version formats (ELPA's 2023.05.001, a \
header's #define) as the check parses them. Put what the check demanded in finish's version.
- Test with configure. As soon as the error MOVES -- a different error, or it completes -- \
call finish. Do not chase the next error; the executor takes it from there.
- The plan's -D flags are deliberate: some configures are optional variants (a CUDA build, \
an MPI build), and a flag the plan sets is never a plan fault. If the configure needs a \
setting the plan does not give (a build type it insists on), finish with plan_fault and the \
-D flags in flags. If it needs what no stub can be \
(a compiler for another language, running a program built for the target, files missing \
from the checkout), finish with cannot_stub. Either way, say why in one or two sentences.
- Name the blocker in one word, as the project's own find_package / pkg-config / \
find_library call names it (ELPA, HDF5, Armadillo, wxWidgets, netcdf). Others stubbed for the \
same error go in also. If what stopped it is no dependency but something every {arch} Linux \
system has (dlfcn.h, a CMake quirk of the sandbox), kind is "host".

How the sandbox works: CMAKE_FIND_ROOT_PATH is the sysroot (plus the scratch and source \
trees), libraries/includes/packages are searched there only; programs come from the host \
PATH, then <sysroot>/bin. pkg-config sees only <sysroot>/lib/pkgconfig. Compile checks \
compile (CMAKE_TRY_COMPILE_TARGET_TYPE is STATIC_LIBRARY) with -isystem <sysroot>/include, \
using the host's compiler; nothing links. A failed check's compiler output is in \
<build>/CMakeFiles/CMakeConfigureLog.yaml. The executor's stub headers say "stub emitted \
by will-it-riscv". You can Read, Grep and Glob the source tree, the scratch directory (the \
sysroot, traces, build directories) and CMake's own modules.

Be brief: a few tool calls to understand, one or two configures to test, then finish.
"""


def system_prompt(arch: str = "riscv64") -> str:
    return SYSTEM.replace("{arch}", arch)


def prompt(stuck: Stuck, bench: Workbench, configures: int) -> str:
    return "\n".join([
        f"Source tree: {stuck.root}",
        f"Scratch directory (your working directory): {stuck.scratch}",
        f"Sysroot: {stuck.sysroot}",
        f"CMake's modules: {cmake_modules() or '(unknown)'}",
        f"You have {configures} configures.",
        "",
        bench.show_error(),
        "",
        "Start from the error and the trace around where it was raised.",
    ])


class Replays:
    """What earlier sessions found at a stuck point, kept on disk under the log
    directory's replay/, for any later run to put back without asking again."""

    log_dir: Optional[Path] = None

    def base(self) -> Path:
        return Path(self.log_dir or os.environ.get(
            "WILL_IT_RISCV_UNBLOCK_LOG", Path.home() / ".cache" / "will-it-riscv" / "unblock"))

    def _replay_file(self, key: tuple) -> Path:
        digest = hashlib.sha256("\n".join(key).encode()).hexdigest()[:32]
        return self.base() / "replay" / f"{digest}.json"

    def _recall(self, key: tuple) -> Optional[dict]:
        """A session at this stuck point before: in this process, or in any
        run since -- the reference run, the relabelling, an eval of a
        planner's plan all meet the same stuck points, and get the same stubs."""
        if key in _REPLAYS:
            return _REPLAYS[key]
        try:
            memo = json.loads(self._replay_file(key).read_text())
        except (OSError, ValueError):
            return None
        if [memo.get("root"), memo.get("signature")] != list(key):
            return None
        _REPLAYS[key] = memo
        return memo

    def _store(self, key: tuple, memo: dict) -> None:
        _REPLAYS[key] = memo
        path = self._replay_file(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(f".{os.getpid()}.part")
            partial.write_text(json.dumps({"root": key[0], "signature": key[1], **memo}))
            partial.replace(path)
        except OSError:
            pass

    def _forget(self, key: tuple) -> None:
        _REPLAYS.pop(key, None)
        self._replay_file(key).unlink(missing_ok=True)


def open_session(stuck: Stuck, max_configures: int = MAX_CONFIGURES) -> Path:
    """A session directory at a stuck point: what the workbench serves."""
    number = len(stuck.earlier) + 1
    session = stuck.scratch / "unblock" / f"session-{number}"
    session.mkdir(parents=True, exist_ok=True)
    state = {
        "session": number, "root": str(stuck.root), "scratch": str(stuck.scratch),
        "sysroot": str(stuck.sysroot),
        "toolchain": str(stuck.toolchain) if stuck.toolchain else None,
        # Only what the executor changed: the rest is this process's own
        # environment, and has no business in a log.
        "env": {k: v for k, v in stuck.env.items() if os.environ.get(k) != v},
        "settings": stuck.settings, "fixed": stuck.fixed,
        "overrides": sorted(stuck.overrides),
        "signature": pb._signature(stuck.outcome.error), "error": stuck.outcome.error,
        "stanzas": pb._error_stanzas(stuck.outcome.narration),
        "status": _status_before_error(stuck.outcome.narration),
        "round": stuck.round, "trace": str(stuck.trace), "build": str(stuck.build),
        "blockers": list(stuck.blockers), "timeout": stuck.timeout,
        "max_configures": max_configures, "earlier": stuck.earlier,
    }
    (session / "state.json").write_text(json.dumps(state, indent=1))
    return session


@dataclass
class ClaudeUnblocker(Replays):
    """Sessions with Claude through ``claude -p``, the workbench over MCP."""

    model: str = "sonnet"
    budget_usd: float = 2.0
    """Per session."""
    timeout: int = 1200
    """Seconds per session, configures included."""
    max_configures: int = MAX_CONFIGURES
    sessions: int = SESSIONS
    log_dir: Optional[Path] = None
    command: str = "claude"

    def __call__(self, stuck: Stuck) -> Result:
        key = (str(stuck.root.resolve()), pb._signature(stuck.outcome.error))
        memo = self._recall(key)
        if memo is not None:
            result = _replay(memo, stuck, key)
        else:
            result = self._session(stuck)
            if result.outcome != GAVE_UP:
                self._store(key, _remember(result, stuck))
                result.memo = key
        SPENT.append(result.record())
        return result

    def _session(self, stuck: Stuck) -> Result:
        session = open_session(stuck, self.max_configures)
        bench = Workbench(session)
        text = prompt(stuck, bench, self.max_configures)
        (session / "prompt.txt").write_text(text)
        source = str(Path(__file__).resolve().parents[1])
        mcp = {"mcpServers": {"unblock": {
            "command": sys.executable,
            "args": ["-m", "will_it_riscv.unblock", "serve", str(session)],
            "env": {"PYTHONPATH": os.pathsep.join(
                p for p in (source, os.environ.get("PYTHONPATH", "")) if p)},
        }}}
        (session / "mcp.json").write_text(json.dumps(mcp))
        dirs = [str(stuck.root)] + ([str(cmake_modules())] if cmake_modules() else [])
        command = [
            self.command, "-p", "--model", self.model,
            "--system-prompt", system_prompt(stuck.arch),
            "--tools", "Read,Grep,Glob",
            "--allowedTools", "Read,Grep,Glob,mcp__unblock",
            "--permission-mode", "dontAsk",
            "--strict-mcp-config", "--mcp-config", str(session / "mcp.json"),
            "--add-dir", *dirs,
            "--output-format", "stream-json", "--verbose",
            "--no-session-persistence",
            "--max-budget-usd", f"{self.budget_usd:.2f}",
        ]
        env = {**claude_environment(), "MCP_TOOL_TIMEOUT": str(self.timeout * 1000)}
        transcript = session / "transcript.jsonl"
        started = time.monotonic()
        failure = None
        try:
            with transcript.open("w") as out, (session / "stderr.txt").open("w") as err:
                subprocess.run(command, input=text, stdout=out, stderr=err,
                               text=True, cwd=stuck.scratch, env=env, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            failure = f"the session ran past {self.timeout}s"
        except OSError as exc:
            failure = f"could not run {self.command}: {exc}"
        bench = Workbench(session)   # the server process wrote the ledger
        result = bench.result()
        if failure and result.outcome == GAVE_UP:
            result.reason = failure
        result.cost, result.tokens_in, result.tokens_out = _usage(transcript)
        result.log = self._keep(session, stuck, result, time.monotonic() - started)
        return result

    def _keep(self, session: Path, stuck: Stuck, result: Result, seconds: float) -> Optional[str]:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", stuck.root.name)
        target = self.base() / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{session.name}"
        try:
            target.mkdir(parents=True, exist_ok=True)
            for part in ("prompt.txt", "transcript.jsonl", "ledger.json", "calls.jsonl",
                         "state.json", "stderr.txt"):
                if (session / part).is_file():
                    shutil.copyfile(session / part, target / part)
            for path in result.files + list(result.restore):
                if Path(path).is_file():
                    copy = target / "files" / Path(path).relative_to(stuck.sysroot)
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, copy)
            summary = {**result.record(), "seconds": round(seconds, 1), "root": str(stuck.root),
                       "model": self.model}
            summary.pop("log")
            (target / "result.json").write_text(json.dumps(summary, indent=1))
        except OSError:
            return None
        return str(target)

    def judged(self, result: Result, confirmed: bool) -> None:
        """The executor's verdict on a session's trial, added to its log. A
        rejected one is not replayed."""
        if not confirmed and result.memo:
            self._forget(result.memo)
        if result.log:
            path = Path(result.log) / "result.json"
            try:
                summary = json.loads(path.read_text())
                summary["confirmed"] = confirmed
                path.write_text(json.dumps(summary, indent=1))
            except (OSError, ValueError):
                pass


#: Sessions by project and stuck error, as loaded or made by this process
#: (the same are kept on disk, under the log directory's replay/). A plan
#: configures one project once for each of its steps, and mlpack stops at
#: the same Armadillo check every time: the second time, the first session's
#: stubs are put back without asking again -- and judged again all the same.
_REPLAYS: dict = {}
_SYSROOT, _SCRATCH = "\0sysroot\0", "\0scratch\0"


def _remember(result: Result, stuck: Stuck) -> dict:
    def portable(value: str) -> str:
        # Stubs name their own sysroot -- mpas's pnetcdf-config prints its
        # prefix -- and the next run's is somewhere else.
        return value.replace(str(stuck.sysroot), _SYSROOT).replace(str(stuck.scratch), _SCRATCH)

    files = {}
    for path in [*result.files, *result.restore]:
        try:
            target = Path(path)
            files[target.relative_to(stuck.sysroot).as_posix()] = (
                portable(target.read_text(errors="replace")), os.access(target, os.X_OK))
        except (OSError, ValueError):
            continue

    then = {"outcome": result.outcome, "name": result.name, "kind": result.kind,
            "reason": result.reason, "flags": result.flags, "also": result.also,
            "version": result.version, "log": result.log or result.replayed}
    return {"result": then, "files": files,
            "overrides": {k: portable(v) for k, v in result.overrides.items()}}


def _replay(memo: dict, stuck: Stuck, key: tuple) -> Result:
    then = memo["result"]
    result = Result(
        outcome=then["outcome"], name=then.get("name"), kind=then.get("kind") or DEPENDENCY,
        reason=then.get("reason") or "", flags=dict(then.get("flags") or {}),
        also=list(then.get("also") or []), version=then.get("version"),
        replayed=then.get("log") or "an earlier session", cost=0.0, memo=key,
    )
    if result.outcome != UNBLOCKED:
        return result

    def here(value: str) -> str:
        return value.replace(_SYSROOT, str(stuck.sysroot)).replace(_SCRATCH, str(stuck.scratch))

    for relative, (text, executable) in memo["files"].items():
        target = stuck.sysroot / relative
        if target.exists():
            result.restore[str(target)] = target.read_text(errors="replace")
        else:
            result.files.append(str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(here(text))
        if executable:
            target.chmod(0o755)
    result.overrides = {k: here(v) for k, v in memo["overrides"].items()}
    return result


#: Every session and replay this process has had, as records, in order: what
#: a scoreboard adds up per repository (``reset_spent``, then ``spent``).
SPENT: list = []


def reset_spent() -> None:
    SPENT.clear()


def spent() -> dict:
    """Sessions, replays, tokens and dollars since the last ``reset_spent``."""
    return {
        "sessions": sum(1 for r in SPENT if not r.get("replayed")),
        "replays": sum(1 for r in SPENT if r.get("replayed")),
        "tokens_in": sum(r.get("tokens_in") or 0 for r in SPENT),
        "tokens_out": sum(r.get("tokens_out") or 0 for r in SPENT),
        "cost": round(sum(r.get("cost") or 0 for r in SPENT), 4),
    }


def _usage(transcript: Path) -> tuple[Optional[float], int, int]:
    """Dollars and tokens in and out, from the final message ``claude -p``
    streams. Read from its cache is still read."""
    try:
        for line in reversed(transcript.read_text().splitlines()):
            message = json.loads(line)
            if message.get("type") == "result":
                usage = message.get("usage") or {}
                read = sum(usage.get(k) or 0 for k in (
                    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                return message.get("total_cost_usd"), read, usage.get("output_tokens") or 0
    except (OSError, ValueError):
        pass
    return None, 0, 0


#: An unblocker set for this thread's configures (``using``), ahead of the
#: environment's: the agent pauses its own plan's run at each stuck point.
_CHOSEN: contextvars.ContextVar = contextvars.ContextVar("unblocker", default=None)


@contextlib.contextmanager
def using(unblocker):
    token = _CHOSEN.set(unblocker)
    try:
        yield unblocker
    finally:
        _CHOSEN.reset(token)


def from_environment():
    """What ``using`` set; else ``WILL_IT_RISCV_UNBLOCK=claude[:model]``, or
    ``replay`` (only what earlier sessions found, never a new one); else
    nothing. ``WILL_IT_RISCV_UNBLOCK_BUDGET`` is dollars per session."""
    chosen = _CHOSEN.get()
    if chosen is not None:
        return chosen
    spec = os.environ.get("WILL_IT_RISCV_UNBLOCK", "").strip()
    if not spec:
        return None
    kind, _, model = spec.partition(":")
    if kind == "replay":
        return ReplayOnly()
    if kind != "claude":
        raise ValueError(f"WILL_IT_RISCV_UNBLOCK={spec}: claude[:model] or replay")
    unblocker = ClaudeUnblocker(model=model or "sonnet")
    if os.environ.get("WILL_IT_RISCV_UNBLOCK_BUDGET"):
        unblocker.budget_usd = float(os.environ["WILL_IT_RISCV_UNBLOCK_BUDGET"])
    return unblocker


@dataclass
class ReplayOnly(Replays):
    """Puts back what earlier sessions found and asks nothing new: a run that
    must cost nothing, such as rescoring a plan an agent already ran."""

    log_dir: Optional[Path] = None
    sessions: int = SESSIONS

    def __call__(self, stuck: Stuck) -> Result:
        key = (str(stuck.root.resolve()), pb._signature(stuck.outcome.error))
        memo = self._recall(key)
        result = (_replay(memo, stuck, key) if memo is not None
                  else Result(GAVE_UP, reason="no earlier session at this stuck point"))
        SPENT.append(result.record())
        return result

    def judged(self, result: Result, confirmed: bool) -> None:
        if not confirmed and result.memo:
            self._forget(result.memo)


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "serve":
        serve(Path(argv[1]))
        return 0
    print("usage: python -m will_it_riscv.unblock serve SESSION_DIR", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
