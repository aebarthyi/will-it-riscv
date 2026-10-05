"""A model writes the plan: from a repository's evidence pack to plan JSON.

The model only plans. It is shown the evidence pack (:mod:`evidence`) and
writes down what the default build runs, citing the lines that say so; the
executor judges. So a wrong plan costs a round, never a wrong verdict: what
it gets wrong -- JSON that does not parse, a step the format refuses, a
citation that does not hold, a line the pack never showed it -- goes back
to it, and it writes the plan again. A citation whose quote the pack shows
on other lines is moved there instead (``relocate``): the quote grounds the
step, and the line number is only where to find it.

The wire format is the plan format made strict enough for schema-constrained
decoding: every object closed, evidence always ``{at, quote}``, a
configure's ``-D`` flags as name/value pairs rather than a free-form map.
The same schema constrains a local model served by vLLM, llama.cpp or
Ollama, the Claude API's structured outputs, and ``claude -p --json-schema``.

Backends:

  openai       any OpenAI-compatible chat endpoint -- vLLM, llama.cpp's
               server, Ollama -- with the schema as ``response_format``.
               What a fine-tuned small model is served by.
  anthropic    the Claude API, through the official SDK (``pip install
               anthropic``), with structured outputs. The teacher.
  claude-cli   ``claude -p``: the same teacher through Claude Code, with its
               tools switched off, for when there is no API key.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from . import evidence as evidence_pack
from .plan import (
    CMAKE_CONFIGURE,
    MESON_SETUP,
    PYTHON_INSTALL,
    PYTHON_RUN,
    SECTIONS,
    STEP_KINDS,
    SYSTEM_PACKAGES,
    Evidence,
    Plan,
    PlanError,
    check_evidence,
    parse_plan,
    plan_to_dict,
)

#: How many times a plan is sent back before the attempt is given up.
ROUNDS = 3
DEFAULT_TEACHER = "claude-opus-5"

_STRINGS = {"type": "array", "items": {"type": "string"}}

WIRE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["repo", "steps", "unsure"],
    "properties": {
        "repo": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind", "evidence"],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"enum": list(STEP_KINDS)},
                    "after": _STRINGS,
                    "optional": {"type": "boolean"},
                    "enabled_by": {"type": "string"},
                    "note": {"type": "string"},
                    "provides": _STRINGS,
                    "manifest": {"type": "string"},
                    "packages": _STRINGS,
                    "section": {"enum": list(SECTIONS)},
                    "extras": _STRINGS,
                    "script": {"type": "string"},
                    "args": _STRINGS,
                    "source": {"type": "string"},
                    "defines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "value"],
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "string"},
                            },
                        },
                    },
                    "meson": {"type": "string"},
                    "evidence": {"$ref": "#/$defs/evidence"},
                },
            },
        },
        "unsure": _STRINGS,
        "entry": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command", "evidence"],
            "properties": {
                "command": {"type": "string"},
                "evidence": {"$ref": "#/$defs/evidence"},
            },
        },
    },
    "$defs": {
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["at", "quote"],
                "properties": {"at": {"type": "string"}, "quote": {"type": "string"}},
            },
        },
    },
}



def in_written_order(schema: Any) -> Any:
    """The schema for a grammar that writes an object's keys in its listed order.

    llama.cpp's grammar writes an object's required properties first and the
    optional ones after, so a required ``evidence`` would come straight after
    ``kind`` -- where no plan the model was trained on puts it, and the model
    closes the step there. Only the required properties that lead the list
    stay required; ``check`` asks for the rest.
    """
    if isinstance(schema, list):
        return [in_written_order(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {key: in_written_order(value) for key, value in schema.items()}
    if "required" in schema and "properties" in schema:
        leading = []
        for name in schema["properties"]:
            if name not in schema["required"]:
                break
            leading.append(name)
        out["required"] = leading
    return out


#: Longest list the grammar lets a small model write, per property: about twice the
#: longest in the labelled plans (packages 66, defines 37, unsure 9, ...). Past it, a
#: model that has started repeating itself is made to close the list.
MAX_ITEMS = {"packages": 100, "defines": 60, "extras": 40, "steps": 30, "evidence": 20,
             "unsure": 12, "args": 20, "after": 20, "provides": 20}
#: Longest string: the labelled plans' longest is an ``unsure`` of 477 characters.
MAX_LENGTH = 600


def bounded(schema: Any, name: str = "") -> Any:
    """The schema with MAX_ITEMS and MAX_LENGTH, for a grammar to enforce.

    A fine-tuned model decoding greedily can repeat one item or phrase until it
    runs out of tokens, and a cut-off answer is not JSON: the whole plan is lost.
    With the limits in the grammar the answer ends shorter but whole.
    """
    if isinstance(schema, list):
        return [bounded(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if "items" in schema:
        out["items"] = bounded(schema["items"])
    if "properties" in schema:
        out["properties"] = {
            key: bounded(value, key) for key, value in schema["properties"].items()
        }
    if "$defs" in schema:
        out["$defs"] = {key: bounded(value, key) for key, value in schema["$defs"].items()}
    if schema.get("type") == "array" and name in MAX_ITEMS:
        out["maxItems"] = MAX_ITEMS[name]
    if schema.get("type") == "string":
        out["maxLength"] = MAX_LENGTH
    return out


SYSTEM_PROMPT = """\
You write build plans for will-it-riscv, a tool that works out what a repository needs in \
order to build on riscv64. You are shown an evidence pack: the parts of one repository that \
say how it is built -- its install docs, the scripts a person runs and what they call, its \
manifests, the top of its build files, its CI -- with every line numbered. You write down, as \
JSON, everything the repository's default build runs, in order.

The tool then runs each step in a pretend environment: it resolves Python installs against \
the package index, looks system packages up in the target distro's archive, and configures \
CMake and Meson builds for real without compiling anything. So you need no knowledge of \
riscv64. You need to read carefully and cite exactly.

Step kinds:
- python-install: Python packages the build installs. "manifest" is the pyproject.toml or \
requirements file installed, relative to the repository root; "section" is "build-system" \
for what building the project needs (its [build-system] requires) rather than its \
dependencies; "extras" are extras installed. Or, when no file lists them, "packages": \
requirement specifiers such as "numpy==2.1.3".
- system-packages: distro packages the build installs or needs, by their Debian names, from \
apt-get lines, Dockerfiles and CI. A version the build must have goes in the name: \
"bazel-bootstrap>=8.7.0".
- cmake-configure: a CMake configure the build runs. "source" is the directory holding its \
CMakeLists.txt, relative to the root ("." for the top). "defines" are the -D flags the \
build passes, as name/value pairs.
- meson-setup: a meson setup the build runs, with "source" and "defines" (its -D options), \
and "meson" when the project ships its own Meson (a meson.py, relative to "source").
- python-run: the project's own build driver, run with Python: "script" relative to the \
root, and "args" as a person would run it.

A package built with a plain "pip install ." is a python-install of its pyproject.toml with \
section "build-system", then the configure its build backend runs: scikit-build-core runs \
CMake, meson-python runs Meson.

The tool runs no make, autotools or Bazel. When the default build is one of those and the \
repository also has a CMakeLists.txt or meson.build at its root, write that configure as a \
default step with its default options: it stands in for the build the tool cannot run, and \
without it package lists are all the tool sees. Say so in its note.

Every step has:
- "id": a short name of letters, digits, dots and dashes, like "configure-simulation".
- "after": the ids of the steps that must run before it.
- "evidence": at least one citation, each {"at": "path:line" or "path:first-last", \
"quote": text copied exactly from those lines}. Cite only lines the pack shows. Keep quotes \
short: the words that say what the step does.
- "note": optional, one sentence.

The default build is the minimal spec. Everything a plain, default build runs is required \
and is an ordinary step. A step that runs only with an extra flag, an option, or a choice \
left to the user -- a GPU backend, a debug build, optional extras -- has "optional": true \
and "enabled_by": the command or flag that turns it on, such as "./mfc.sh build --gpu acc". \
A default step never comes after an optional one. Tests, linting, formatting, docs and \
benchmarks are not the build: leave them out.

"repo" is the repository's short name. "entry" is the command a person runs to build it, \
with its evidence. "unsure" lists, one sentence each, what the pack leaves you unable to \
tell. Say so there rather than guess, and write no step, file, flag or package the pack \
does not show.
"""


# ------------------------------------------------------------------ wire


def to_plan(wire: dict) -> dict:
    """The wire format as plan JSON, for :func:`parse_plan`."""
    data: dict[str, Any] = {"repo": wire.get("repo"), "steps": []}
    if isinstance(wire.get("entry"), dict):
        data["entry"] = wire["entry"]
    if wire.get("unsure"):
        data["unsure"] = wire["unsure"]
    for raw in wire.get("steps") or []:
        if not isinstance(raw, dict):
            data["steps"].append(raw)
            continue
        step = {k: v for k, v in raw.items() if v not in (None, [], "")}
        if isinstance(raw.get("defines"), list):
            step["defines"] = {
                d.get("name"): d.get("value") for d in raw["defines"] if isinstance(d, dict)
            }
            if not step["defines"]:
                del step["defines"]
        if step.get("optional") is False:
            del step["optional"]
        data["steps"].append(step)
    return data


def to_wire(plan: Plan) -> dict:
    """A plan in the wire format: what a model is trained to write."""
    data = plan_to_dict(plan)

    def evidence(items: list) -> list:
        return [
            item if isinstance(item, dict) else {"at": item, "quote": ""} for item in items
        ]

    wire: dict[str, Any] = {"repo": data["repo"], "steps": [], "unsure": data.get("unsure", [])}
    if "entry" in data:
        wire["entry"] = {
            "command": data["entry"]["command"], "evidence": evidence(data["entry"]["evidence"]),
        }
    for raw in data["steps"]:
        step = dict(raw)
        step["evidence"] = evidence(raw["evidence"])
        if isinstance(raw.get("defines"), dict):
            step["defines"] = [{"name": k, "value": v} for k, v in raw["defines"].items()]
        wire["steps"].append(step)
    return wire


# --------------------------------------------------------------- backends


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None


class Backend(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], schema: dict) -> Completion: ...


class OpenAICompatible:
    """A chat endpoint that speaks OpenAI's API: vLLM, llama.cpp, Ollama."""

    def __init__(
        self, url: str, model: str, api_key: Optional[str] = None, max_tokens: int = 8192,
        timeout: float = 600.0, client: Any = None, temperature: float = 0.0,
    ):
        """``temperature`` above 0 samples, for drawing several answers to one pack."""
        import httpx

        self.name = f"openai:{model}"
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)

    def complete(self, system: str, messages: list[dict], schema: dict) -> Completion:
        response = self.client.post(f"{self.url}/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # What vLLM, llama.cpp and Ollama constrain decoding with.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "plan", "schema": bounded(in_written_order(schema))},
            },
        })
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return Completion(
            text=body["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class Shared:
    """Several servers of one model, each answering ``slots`` requests at once.

    Workers forked after this is made share its semaphores: a request waits
    here until some server has a free slot, and goes to that one. So a busy
    card never queues work an idle one could take, and a backend's timeout
    covers answering, not waiting.
    """

    def __init__(self, backends: list[Backend], slots: int = 1):
        import multiprocessing

        context = multiprocessing.get_context("fork")
        self.name = backends[0].name
        self.backends = backends
        self.free = context.BoundedSemaphore(len(backends) * slots)
        self.each = [context.BoundedSemaphore(slots) for _ in backends]

    def complete(self, system: str, messages: list[dict], schema: dict) -> Completion:
        # Whoever holds ``free`` finds a server slot: a slot is only taken
        # while ``free`` is held, and there are as many of one as the other.
        with self.free:
            for backend, gate in zip(self.backends, self.each):
                if gate.acquire(block=False):
                    try:
                        return backend.complete(system, messages, schema)
                    finally:
                        gate.release()
        raise RuntimeError("no server slot was free")  # pragma: no cover - see above


class Anthropic:
    """The Claude API, with the plan's schema as a structured output."""

    def __init__(self, model: str = DEFAULT_TEACHER, effort: str = "high", client: Any = None):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise RuntimeError(
                    "the anthropic backend needs the SDK: pip install 'will-it-riscv[teacher]'"
                ) from exc
            client = anthropic.Anthropic()
        self.name = f"anthropic:{model}"
        self.model = model
        self.effort = effort
        self.client = client

    def complete(self, system: str, messages: list[dict], schema: dict) -> Completion:
        # Long input, long output: streamed, and read once it is whole. The
        # system prompt is the same for every repository, so it is cached.
        with self.client.messages.stream(
            model=self.model,
            max_tokens=32000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.effort,
            },
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} declined to write this plan")
        if message.stop_reason == "max_tokens":
            raise RuntimeError(f"{self.model} ran out of room before the plan was finished")
        text = next((b.text for b in message.content if b.type == "text"), "")
        return Completion(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


class ClaudeCLI:
    """``claude -p``: the teacher through Claude Code, tools off, schema on."""

    def __init__(self, model: str = "opus", budget_usd: float = 2.0, timeout: int = 900):
        self.name = f"claude-cli:{model}"
        self.model = model
        self.budget_usd = budget_usd
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict], schema: dict) -> Completion:
        executable = shutil.which("claude")
        if executable is None:
            raise RuntimeError("the claude-cli backend needs Claude Code's `claude` on PATH")
        command = [
            executable, "-p", "--output-format", "json",
            "--json-schema", json.dumps(schema),
            "--system-prompt", system,
            "--tools", "",
            "--no-session-persistence",
            "--model", self.model,
            "--max-budget-usd", str(self.budget_usd),
        ]
        process = subprocess.run(
            command, input=_as_one_prompt(messages), capture_output=True, text=True,
            timeout=self.timeout, env=claude_environment(),
        )
        try:
            result = json.loads(process.stdout)
        except ValueError as exc:
            detail = (process.stderr or process.stdout or "").strip()[-400:]
            raise RuntimeError(f"claude -p gave no JSON result: {detail}") from exc
        if result.get("is_error"):
            raise RuntimeError(f"claude -p: {result.get('result') or result.get('subtype')}")
        structured = result.get("structured_output")
        text = json.dumps(structured) if structured is not None else result.get("result", "")
        usage = result.get("usage") or {}
        return Completion(
            text=text,
            # What it read from its cache is still what it read.
            input_tokens=usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost_usd=result.get("total_cost_usd"),
        )


#: What ties a ``claude`` process to the Claude Code session that started it.
#: A ``claude -p`` started from inside one is a session of its own.
_PARENT_SESSION = {
    "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_EXECPATH",
}


def claude_environment() -> dict:
    """This process's environment, for a ``claude -p`` of its own."""
    return {k: v for k, v in os.environ.items() if k not in _PARENT_SESSION}


def _as_one_prompt(messages: list[dict]) -> str:
    """A conversation as the one prompt ``claude -p`` takes."""
    if len(messages) == 1:
        return str(messages[0]["content"])
    parts = []
    for message in messages:
        label = "YOUR PLAN" if message["role"] == "assistant" else "USER"
        parts.append(f"--- {label} ---\n{message['content']}")
    return "\n\n".join(parts)


def backend(
    kind: str, model: Optional[str] = None, url: Optional[str] = None, slots: int = 1,
) -> Backend:
    """A backend by name, as the command line names it.

    For ``openai``, ``url`` may name several servers of the same model,
    comma-separated; requests then go to whichever has one of its ``slots``
    free (see Shared).
    """
    if kind == "openai":
        if not url or not model:
            raise ValueError("the openai backend needs --planner-url and --planner-model")
        key = os.environ.get("PLANNER_API_KEY")
        urls = [u.strip() for u in url.split(",") if u.strip()]
        if len(urls) == 1 and slots == 1:
            return OpenAICompatible(urls[0], model, api_key=key)
        return Shared([OpenAICompatible(u, model, api_key=key) for u in urls], slots)
    if kind == "anthropic":
        return Anthropic(model or DEFAULT_TEACHER)
    if kind == "claude-cli":
        return ClaudeCLI(model or "opus")
    raise ValueError(f"no planner backend called {kind!r}")


# ------------------------------------------------------------------ loop


@dataclass
class Attempt:
    text: str
    problems: list[str]
    completion: Optional[Completion] = None
    moved: list[str] = field(default_factory=list)
    """Citations ``relocate`` moved to the lines that say their quote."""


@dataclass
class ModelPlan:
    pack: evidence_pack.EvidencePack
    backend: str
    plan: Optional[Plan] = None
    wire: Optional[dict] = None
    attempts: list[Attempt] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.attempts[-1].problems

    @property
    def tokens(self) -> tuple[int, int]:
        """Input and output tokens, over every attempt."""
        done = [a.completion for a in self.attempts if a.completion]
        return sum(c.input_tokens for c in done), sum(c.output_tokens for c in done)

    @property
    def cost_usd(self) -> Optional[float]:
        costs = [a.completion.cost_usd for a in self.attempts if a.completion]
        return sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None


def check(
    text: str, root: Path, pack: evidence_pack.EvidencePack,
    moved: Optional[list[str]] = None,
) -> tuple[Optional[Plan], Optional[dict], list[str]]:
    """Everything wrong with one answer, as sentences the model can act on.

    The plan comes back with its citations relocated; ``moved`` collects a
    note for each.
    """
    try:
        wire = json.loads(text)
    except ValueError as exc:
        return None, None, [f"the answer is not JSON: {exc}"]
    if not isinstance(wire, dict):
        return None, None, ["the answer must be one JSON object"]
    try:
        plan = parse_plan(to_plan(wire))
    except PlanError as exc:
        return None, wire, list(exc.problems)
    plan, notes = relocate(plan, pack)
    if moved is not None:
        moved += notes
    problems = check_evidence(plan, root)
    cited = [("entry", e) for e in plan.entry_evidence] + [
        (f"step {s.id!r}", e) for s in plan.steps for e in s.evidence
    ]
    for owner, item in cited:
        if not pack.shows(item.path, item.start, item.end):
            problems.append(
                f"{owner}: {item} is not a line the pack shows; cite what you were shown"
            )
    for step in plan.steps:
        if not step.evidence:
            problems.append(f"step {step.id!r} cites nothing")
    return plan, wire, problems


def relocate(plan: Plan, pack: evidence_pack.EvidencePack) -> tuple[Plan, list[str]]:
    """Each citation whose quote is not on its lines, moved to lines the pack shows that say it.

    A model often quotes the right text from the wrong place: gdal's apt
    line, cited at a comment in another part of the workflow; a setup.py
    error message, a line off. Sent back, the same citation tends to come
    back unchanged, and a plan that never holds is never run. The quote is
    what ties a step to the pack, so the citation moves to the nearest
    lines the pack shows that say it -- in the cited file first, then
    anywhere -- and only a quote the pack never shows goes back. Returns the
    plan and a note for each move.
    """
    notes: list[str] = []

    def fixed(owner: str, item: Evidence) -> Evidence:
        if not item.quote or _says(pack, item):
            return item
        found = _find(pack, item)
        if found is None:
            return item
        notes.append(f"{owner}: {item} -> {found}")
        return found

    entry = [fixed("entry", e) for e in plan.entry_evidence]
    steps = [
        replace(s, evidence=[fixed(f"step {s.id!r}", e) for e in s.evidence]) for s in plan.steps
    ]
    if not notes:
        return plan, []
    return replace(plan, entry_evidence=entry, steps=steps), notes


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _says(pack: evidence_pack.EvidencePack, item: Evidence) -> bool:
    excerpt = pack.excerpt(item.path)
    if excerpt is None or not excerpt.shows(item.start, item.end):
        return False
    text = " ".join(t for n, t in excerpt.lines if item.start <= n <= item.end)
    return _squash(item.quote or "") in _squash(text)


def _find(pack: evidence_pack.EvidencePack, item: Evidence) -> Optional[Evidence]:
    """The nearest run of shown lines, up to the cited span or three, that says the quote."""
    want = _squash(item.quote or "")
    if not want:
        return None
    span = max(3, item.end - item.start + 1)
    best: Optional[tuple[tuple, Evidence]] = None
    for order, excerpt in enumerate(pack.excerpts):
        same = excerpt.path == item.path
        shown = excerpt.lines
        for i, (number, _) in enumerate(shown):
            for width in range(1, span + 1):
                run = shown[i:i + width]
                if len(run) < width or run[-1][0] - number != width - 1:
                    break   # past the end, or across lines the pack leaves out
                if want in _squash(" ".join(t for _, t in run)):
                    if width > 1 and want in _squash(" ".join(t for _, t in run[1:])):
                        break   # the lines after this one say it on their own
                    rank = (not same, abs(number - item.start) if same else order, number)
                    if best is None or rank < best[0]:
                        best = (rank, Evidence(excerpt.path, number, run[-1][0], item.quote))
                    break
    return best[1] if best else None


def plan_repository(
    root: Path,
    planner: Backend,
    *,
    name: Optional[str] = None,
    pack: Optional[evidence_pack.EvidencePack] = None,
    rounds: int = ROUNDS,
    verify: Optional[Callable[[Plan], list[str]]] = None,
    history: Optional[list[dict]] = None,
) -> ModelPlan:
    """Ask for a plan, check it, and send it back until it holds or rounds run out.

    ``verify`` is a further check on a plan that holds -- running it, for a
    dataset -- whose problems go back the same way. ``history``: turns after
    the pack that already happened -- an earlier answer and what was wrong
    with it -- so the next answer is a revision.
    """
    root = Path(root)
    pack = pack or evidence_pack.build(root, name)
    outcome = ModelPlan(pack=pack, backend=planner.name)
    messages: list[dict] = [{"role": "user", "content": pack.render()}, *(history or [])]
    for _ in range(max(1, rounds)):
        try:
            completion = planner.complete(SYSTEM_PROMPT, messages, WIRE_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - a backend failing is an outcome
            outcome.error = f"{planner.name}: {exc}"
            break
        moved: list[str] = []
        plan, wire, problems = check(completion.text, root, pack, moved)
        if plan is not None and not problems and verify is not None:
            problems = verify(plan)
        outcome.attempts.append(Attempt(completion.text, problems, completion, moved))
        if plan is not None:
            outcome.plan, outcome.wire = plan, wire
        if not problems:
            break
        messages += [
            {"role": "assistant", "content": completion.text},
            {"role": "user", "content": _feedback(problems)},
        ]
    return outcome


def feedback(problems: list[str]) -> str:
    """What goes back to the planner about an answer, as the next user turn."""
    return _feedback(problems)


def _feedback(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems[:40])
    return (
        f"The plan you wrote has these problems:\n{listed}\n\n"
        "Write the whole plan again, corrected. Everything else about it may stay as it was."
    )


#: The step kinds, for anything that wants to name them without importing plan.
KINDS = (PYTHON_INSTALL, SYSTEM_PACKAGES, CMAKE_CONFIGURE, MESON_SETUP, PYTHON_RUN)
