"""An agent that turns a project into the dependency graph of its build, for a target.

The planner (:mod:`modelplan`) writes a plan once, hears what running it
showed was the plan's fault, and revises. It never sees more of the project
than its evidence pack, and never touches the sandbox. This is the same job
as a loop the model drives, with tools:

- read the project: Read, Grep, Glob (Claude Code's own);
- write the plan, and run it (:mod:`planrun`): Python installs resolved
  against the index, system packages looked up in the distro's archive for
  the target, CMake and Meson configured for real without compiling;
- where a configure stops and the executor's rules have nothing left to try,
  the run pauses, and the agent has the unblocker's workbench there
  (:mod:`unblock`). Its stubs go back as a trial the executor judges, as ever;
- fix what the run shows is the plan's fault, run again, finish.

What it leaves is the plan, and its last run's graph. Every fix it made at a
stuck point is kept for replay (:class:`unblock.Replays`), so running the
same plan again -- to score it, or for a report -- asks nothing new.

    python -m will_it_riscv.agent serve SESSION   (the tools, over MCP)
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import evidence as evidence_pack
from . import modelplan, unblock
from . import pseudobuild as pb
from .plan import Plan, PlanError, check_evidence, parse_plan
from .unblock import Refused

#: Runs of the plan per session: each costs the configures, and a plan that
#: has not settled in six is not going to.
MAX_RUNS = 6
#: Pauses per configure loop, as the unblocker's sessions.
PAUSES = 6


class Abort(BaseException):
    """Raised in the run's thread, at a stuck point, when the agent abandons
    the run. Not an Exception: nothing on the way up may catch it and go on."""


_ABORT = object()


class _Pause(unblock.Replays):
    """The run's unblocker: replays what an earlier session found, and hands
    every new stuck point to the agent, waiting until it resumes."""

    sessions = PAUSES

    def __init__(self, session: Session):
        self.session = session
        self.log_dir = None

    def __call__(self, stuck: unblock.Stuck) -> unblock.Result:
        key = (str(stuck.root.resolve()), pb._signature(stuck.outcome.error))
        memo = self._recall(key)
        if memo is not None:
            result = unblock._replay(memo, stuck, key)
            self.session.replayed.append(result.name or result.outcome)
            return result
        self.session.events.put(("stuck", unblock.open_session(stuck), stuck))
        answer = self.session.answers.get()
        if answer is _ABORT:
            raise Abort()
        if answer.outcome != unblock.GAVE_UP:
            self._store(key, unblock._remember(answer, stuck))
            answer.memo = key
        return answer

    def judged(self, result: unblock.Result, confirmed: bool) -> None:
        if not confirmed and result.memo:
            self._forget(result.memo)
        self.session.judgements.append((result.name, confirmed))


class Session:
    """One agent's plan, runs and stuck points, over a session directory.

    ``state.json`` says what to plan for: the project root, the pack, the
    target arch, the distro, the Python. ``agent.json`` keeps the plan and
    what each run showed, written after every change.
    """

    def __init__(self, directory: Path, env=None):
        self.directory = Path(directory)
        self.state = json.loads((self.directory / "state.json").read_text())
        self.root = Path(self.state["root"])
        self._env = env
        pack = self.state.get("pack")
        self.pack = (evidence_pack.EvidencePack.from_dict(json.loads(Path(pack).read_text()))
                     if pack else None)
        self.wire: Optional[dict] = None
        self.plan: Optional[Plan] = None
        self.events: queue.Queue = queue.Queue()
        self.answers: queue.Queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.bench: Optional[unblock.Workbench] = None
        self.replayed: list = []
        self.judgements: list = []
        self.record: dict = {"plans": 0, "runs": [], "finish": None}
        self.pause = _Pause(self)

    @property
    def env(self):
        if self._env is None:
            from . import dataset

            self._env = dataset.environment(
                self.state.get("python", "3.12"), arch=self.state.get("arch", "riscv64"),
                distro=self.state.get("distro", dataset.LABEL_DISTRO),
            )
        return self._env

    def _save(self) -> None:
        (self.directory / "agent.json").write_text(
            json.dumps({"wire": self.wire, **self.record}, indent=1))

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # -- the plan

    def write_plan(self, plan: dict) -> str:
        if self.running:
            raise Refused("a run is paused at a stuck point: resume or abort_run first")
        try:
            parsed = parse_plan(modelplan.to_plan(plan))
        except PlanError as exc:
            raise Refused("not a plan:\n" + "\n".join(f"- {p}" for p in exc.problems)) from None
        notes: list = []
        if self.pack is not None:
            parsed, notes = modelplan.relocate(parsed, self.pack)
        self.plan, self.wire = parsed, plan
        self.record["plans"] += 1
        self._save()
        problems = check_evidence(parsed, self.root)
        lines = [f"plan kept: {len(parsed.steps)} steps."]
        lines += [f"note: {n}" for n in notes]
        if problems:
            lines += ["these citations do not hold; fix them:"] + [f"- {p}" for p in problems]
        else:
            lines.append("run_plan to run it.")
        return "\n".join(lines)

    def show_plan(self) -> str:
        return json.dumps(self.wire, indent=1) if self.wire else "no plan yet: write_plan"

    # -- running

    def run_plan(self) -> str:
        if self.plan is None:
            raise Refused("write_plan first")
        if self.running:
            raise Refused("a run is already paused at a stuck point: resume or abort_run")
        if len(self.record["runs"]) >= self.state.get("max_runs", MAX_RUNS):
            raise Refused("every run is spent: finish")
        self.replayed, self.judgements = [], []
        plan = self.plan

        def work() -> None:
            from . import dataset

            try:
                with unblock.using(self.pause):
                    result = dataset.run_plan(plan, self.root, self.env)
                self.events.put(("done", result))
            except Abort:
                self.events.put(("aborted", None))
            except Exception as exc:  # noqa: BLE001 - the agent hears what broke
                self.events.put(("error", exc))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        return self._next()

    def _next(self) -> str:
        event = self.events.get()
        if event[0] == "stuck":
            _, directory, stuck = event
            self.bench = unblock.Workbench(directory)
            return self._paused(stuck)
        self.bench = None
        if self.thread is not None:
            self.thread.join(timeout=60)
        if event[0] == "done":
            return self._done(event[1])
        if event[0] == "aborted":
            self.record["runs"].append({"aborted": True})
            self._save()
            return "the run was abandoned. write_plan, then run_plan again."
        self.record["runs"].append({"error": str(event[1])[:300]})
        self._save()
        return f"the run failed: {event[1]}"

    def _paused(self, stuck: unblock.Stuck) -> str:
        try:
            where = stuck.root.resolve().relative_to(self.root.resolve()).as_posix() or "."
        except ValueError:
            where = str(stuck.root)
        lines = [
            f"The run paused: the configure of {where!r} stopped at round {stuck.round}, and "
            "the executor's rules have nothing left to try.",
            *self._judged(),
            "",
            self.bench.show_error() if self.bench else "",
            "",
            "Work it out with the workbench, then resume with the outcome. If the plan is at "
            "fault, abort_run and fix it.",
        ]
        return "\n".join(lines)

    def _judged(self) -> list[str]:
        said = [f"your stubs for {name}: "
                + ("kept, the error moved" if ok else "taken out, it did not")
                for name, ok in self.judgements]
        self.judgements = []
        return said

    def _done(self, result) -> str:
        from . import dataset, planrun

        summary = dataset.summary(result)
        faults = dataset.plan_faults(result)
        self.record["runs"].append({
            "verdict": summary["verdict"], "required": len(summary["required"]),
            "faults": len(faults), "replayed": len(self.replayed),
        })
        self._save()
        try:
            (self.directory / "last-run.json").write_text(
                json.dumps(planrun.to_dict(result), indent=1, default=str))
        except (TypeError, ValueError, OSError):
            pass
        lines = [f"verdict: {summary['verdict']}: {summary['headline']}"]
        lines += [f"required ({len(summary['required'])}): "
                  + ", ".join(summary["required"][:80])]
        lines += ["steps:"] + [f"- {s['id']} ({s['kind']}): {s['status']}: {s['detail'][:240]}"
                               for s in summary["steps"]]
        if self.replayed:
            lines.append(f"stuck points answered by earlier fixes: {', '.join(self.replayed)}")
        lines += self._judged()
        if faults:
            lines += ["the plan's faults:"] + [f"- {f}" for f in faults]
            lines.append("Fix them with write_plan and run_plan again, or finish if you cannot.")
        else:
            lines.append("No faults of the plan's. finish, unless you know the plan is wrong.")
        return "\n".join(lines)

    def resume(self, outcome: str, reason: str, blocker: Optional[str] = None,
               kind: str = unblock.DEPENDENCY, also: Optional[list] = None,
               version: Optional[str] = None, flags: Optional[dict] = None) -> str:
        if self.bench is None:
            raise Refused("nothing is paused")
        self.bench.finish(outcome, reason, blocker, kind, flags, also, version)
        result = self.bench.result()
        self.bench = None
        self.answers.put(result)
        return self._next()

    def abort_run(self, reason: str = "") -> str:
        if self.bench is None:
            raise Refused("nothing is paused")
        self.bench.rollback()
        self.bench = None
        self.answers.put(_ABORT)
        return self._next()

    def finish(self, note: str = "") -> str:
        if self.bench is not None:
            self.abort_run()
        if self.plan is None:
            raise Refused("write_plan first")
        self.record["finish"] = {"note": note}
        self._save()
        return "recorded. Stop here."

    def close(self) -> None:
        """The agent went away: let a paused run end."""
        if self.running:
            self.answers.put(_ABORT)
            self.thread.join(timeout=120)


# ------------------------------------------------------------------ MCP

_WORKBENCH = {t["name"]: t for t in unblock.TOOLS if t["name"] != "finish"}
_RESUME = next(t for t in unblock.TOOLS if t["name"] == "finish")

TOOLS = [
    {
        "name": "write_plan",
        "description": "Set the build plan (the format is in your instructions). Citations "
                       "are checked against the files; it says which do not hold.",
        "inputSchema": {"type": "object", "required": ["plan"],
                        "properties": {"plan": modelplan.WIRE_SCHEMA}},
    },
    {
        "name": "show_plan",
        "description": "The plan as it stands.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_plan",
        "description": "Run the plan in the pretend sandbox. Answers with the verdict, what "
                       "the build requires, each step's outcome and the plan's own faults -- "
                       "or, when a configure stops where the executor's rules cannot get past, "
                       "pauses there and says so: use the workbench, then resume.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "resume",
        "description": "At a stuck point: say what you found and let the run go on. The same "
                       "fields as the unblocker's finish: " + _RESUME["description"],
        "inputSchema": _RESUME["inputSchema"],
    },
    {
        "name": "abort_run",
        "description": "At a stuck point: abandon the run, to change the plan.",
        "inputSchema": {"type": "object", "properties": {"reason": {"type": "string"}}},
    },
    *[{**tool, "description": "At a stuck point: " + tool["description"]}
      for tool in _WORKBENCH.values()],
    {
        "name": "finish",
        "description": "End: the plan as it stands is your answer.",
        "inputSchema": {"type": "object", "properties": {"note": {"type": "string"}}},
    },
]


def call(session: Session, name: str, arguments: dict) -> tuple[str, bool]:
    methods = {
        "write_plan": session.write_plan, "show_plan": session.show_plan,
        "run_plan": session.run_plan, "resume": session.resume,
        "abort_run": session.abort_run, "finish": session.finish,
    }
    try:
        if name in methods:
            return methods[name](**(arguments or {})), False
        if name in _WORKBENCH:
            if session.bench is None:
                raise Refused("nothing is paused: the workbench is for stuck points")
            return unblock.call(session.bench, name, arguments)
        return f"no tool {name}", True
    except Refused as exc:
        return f"refused: {exc}", True
    except TypeError as exc:
        return f"bad arguments: {exc}", True
    except OSError as exc:
        return f"failed: {exc}", True


def serve(directory: Path, env=None, stdin=None, stdout=None) -> None:
    session = Session(directory, env)
    unblock.serve_tools(TOOLS, lambda name, arguments: call(session, name, arguments),
                        Path(directory) / "calls.jsonl", stdin, stdout, closing=session.close)


# ------------------------------------------------------------------ the agent

SYSTEM = """\
You work out what a software project needs in order to build on {arch} Linux ({distro}): \
the dependency graph of its default build, and of the optional configurations it offers. You \
do it with tools, and you spend as few tokens as you can.

You are given an evidence pack: the parts of the project that say how it is built, every line \
numbered. Read more of the project with Read, Grep and Glob only where the pack leaves a \
question open, and read narrowly: a line range, a grep, not whole trees.

1. write_plan: the build plan, in the format below.
2. run_plan: the tool runs it in a pretend {arch} sandbox. Python installs are resolved against \
the package index, system packages looked up in the {arch} archive, CMake and Meson configured \
for real, as a cross build into an empty sysroot, without compiling anything. It answers with \
the verdict, what the build requires, and any faults that are the plan's own.
3. When a configure stops where the executor's own rules cannot get past, the run pauses and \
you have a workbench there (show_error, show_trace, show_lookups, show_sysroot, write_file, \
set_cache, configure). Find what the configure looked for, stub it, and resume with the \
outcome. The executor judges your stubs: they stay only if the error moved.
4. If the run shows the plan at fault -- a flag the build insists on, a step that is not there \
-- fix it with write_plan and run_plan again. Otherwise finish.

At a stuck point:
- Write only inside the sysroot, and set only what a lookup or check fills. Never the source \
tree, a build option, the plan's -D flags or CMAKE_*.
- Stub exactly what this error needs. Claim no feature nobody asked for.
- Never impersonate a compiler or toolchain (no fake nvcc, hipcc, gfortran). A default build \
that enables CUDA or HIP is itself the answer: resume with cannot_stub, blocker CUDA (or HIP).
- A version check gets a version that passes: what {distro} ships if you know it and it \
passes, else the oldest the check accepts. Put what it demanded in resume's version.
- As soon as the error moves, resume: do not chase the next error yourself.
- Name the blocker in one word, as the project's find_package or pkg-config call names it; \
others stubbed for the same error go in also; kind "host" for what every Linux system has.
- A setting the plan does not give that the configure insists on is the plan's fault: \
abort_run and add it to the plan.

The plan:
{plan_format}
"""


def system_prompt(arch: str, distro: str) -> str:
    planner = modelplan.SYSTEM_PROMPT
    plan_format = planner[planner.index("Step kinds:"):]
    plan_format = plan_format.replace(
        "Cite only lines the pack shows.", "Cite lines you have read."
    ).replace("the pack does not show", "the project does not show").replace(
        "what the pack leaves you unable to tell", "what you could not tell")
    return SYSTEM.format(arch=arch, distro=distro, plan_format=plan_format)


@dataclass
class Outcome:
    plan: Optional[Plan] = None
    wire: Optional[dict] = None
    finished: bool = False
    runs: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost: Optional[float] = None
    log: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ClaudeAgent:
    """The agent as Claude, through ``claude -p``, with the tools over MCP."""

    model: str = "sonnet"
    budget_usd: float = 4.0
    """Per project."""
    timeout: int = 3600
    max_runs: int = MAX_RUNS
    log_dir: Optional[Path] = None
    command: str = "claude"

    def __call__(self, root: Path, pack: evidence_pack.EvidencePack, name: str,
                 arch: str = "riscv64", distro: str = "debian:trixie", python: str = "3.12",
                 distro_label: Optional[str] = None) -> Outcome:
        root = Path(root)
        base = Path(self.log_dir or unblock.Replays().base()) / "agent"
        directory = base / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pack.json").write_text(json.dumps(pack.to_dict()))
        (directory / "state.json").write_text(json.dumps({
            "root": str(root), "name": name, "arch": arch, "distro": distro, "python": python,
            "pack": str(directory / "pack.json"), "max_runs": self.max_runs,
        }, indent=1))
        source = str(Path(__file__).resolve().parents[1])
        (directory / "mcp.json").write_text(json.dumps({"mcpServers": {"plan": {
            "command": sys.executable,
            "args": ["-m", "will_it_riscv.agent", "serve", str(directory)],
            "env": {"PYTHONPATH": os.pathsep.join(
                p for p in (source, os.environ.get("PYTHONPATH", "")) if p)},
        }}}))
        text = (f"Project: {name}, at {root}\nTarget: {arch}, {distro_label or distro}\n\n"
                + pack.render() + "\n\nWrite the plan, run it, and finish.")
        (directory / "prompt.txt").write_text(text)
        dirs = [str(root)] + ([str(unblock.cmake_modules())] if unblock.cmake_modules() else [])
        command = [
            self.command, "-p", "--model", self.model,
            "--system-prompt", system_prompt(arch, distro_label or distro),
            "--tools", "Read,Grep,Glob",
            "--allowedTools", "Read,Grep,Glob,mcp__plan",
            "--permission-mode", "dontAsk",
            "--strict-mcp-config", "--mcp-config", str(directory / "mcp.json"),
            "--add-dir", *dirs,
            "--output-format", "stream-json", "--verbose",
            "--no-session-persistence",
            "--max-budget-usd", f"{self.budget_usd:.2f}",
        ]
        env = {**modelplan.claude_environment(), "MCP_TOOL_TIMEOUT": str(self.timeout * 1000)}
        outcome = Outcome(log=str(directory))
        transcript = directory / "transcript.jsonl"
        try:
            with transcript.open("w") as out, (directory / "stderr.txt").open("w") as err:
                subprocess.run(command, input=text, stdout=out, stderr=err, text=True,
                               cwd=directory, env=env, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            outcome.error = f"the agent ran past {self.timeout}s"
        except OSError as exc:
            outcome.error = f"could not run {self.command}: {exc}"
        outcome.cost, outcome.tokens_in, outcome.tokens_out = unblock._usage(transcript)
        return _read(directory, root, pack, outcome)


def _read(directory: Path, root: Path, pack, outcome: Outcome) -> Outcome:
    """What the session left: the plan, and whether the agent said it was done."""
    try:
        record = json.loads((directory / "agent.json").read_text())
    except (OSError, ValueError):
        outcome.error = outcome.error or "the agent never wrote a plan"
        return outcome
    outcome.wire, outcome.runs = record.get("wire"), record.get("runs") or []
    outcome.finished = record.get("finish") is not None
    if outcome.wire:
        try:
            plan = parse_plan(modelplan.to_plan(outcome.wire))
            outcome.plan = modelplan.relocate(plan, pack)[0] if pack is not None else plan
        except PlanError as exc:
            outcome.error = f"its plan does not parse: {exc}"
    return outcome


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "serve":
        serve(Path(argv[1]))
        return 0
    print("usage: python -m will_it_riscv.agent serve SESSION_DIR", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

