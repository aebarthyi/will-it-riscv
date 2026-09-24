"""A model writes the plan: from a repository's evidence pack to plan JSON.

The model only plans. It is shown the evidence pack (:mod:`evidence`) and
writes down what the default build runs, citing the lines that say so; the
executor judges. So a wrong plan costs a round, never a wrong verdict: what
it gets wrong -- JSON that does not parse, a step the format refuses, a
citation that does not hold, a line the pack never showed it -- goes back
to it, and it writes the plan again.

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
import shutil
import subprocess
from dataclasses import dataclass, field
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
        "entry": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command", "evidence"],
            "properties": {
                "command": {"type": "string"},
                "evidence": {"$ref": "#/$defs/evidence"},
            },
        },
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
                    "evidence": {"$ref": "#/$defs/evidence"},
                    "after": _STRINGS,
                    "note": {"type": "string"},
                    "optional": {"type": "boolean"},
                    "enabled_by": {"type": "string"},
                    "provides": _STRINGS,
                    "manifest": {"type": "string"},
                    "section": {"enum": list(SECTIONS)},
                    "extras": _STRINGS,
                    "packages": _STRINGS,
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
                    "script": {"type": "string"},
                    "args": _STRINGS,
                },
            },
        },
        "unsure": _STRINGS,
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
        timeout: float = 600.0, client: Any = None,
    ):
        import httpx

        self.name = f"openai:{model}"
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)

    def complete(self, system: str, messages: list[dict], schema: dict) -> Completion:
        response = self.client.post(f"{self.url}/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            # What vLLM, llama.cpp and Ollama constrain decoding with.
            "response_format": {
                "type": "json_schema", "json_schema": {"name": "plan", "schema": schema},
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
            timeout=self.timeout,
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


def _as_one_prompt(messages: list[dict]) -> str:
    """A conversation as the one prompt ``claude -p`` takes."""
    if len(messages) == 1:
        return str(messages[0]["content"])
    parts = []
    for message in messages:
        label = "YOUR PLAN" if message["role"] == "assistant" else "USER"
        parts.append(f"--- {label} ---\n{message['content']}")
    return "\n\n".join(parts)


def backend(kind: str, model: Optional[str] = None, url: Optional[str] = None) -> Backend:
    """A backend by name, as the command line names it."""
    if kind == "openai":
        if not url or not model:
            raise ValueError("the openai backend needs --planner-url and --planner-model")
        return OpenAICompatible(url, model, api_key=os.environ.get("PLANNER_API_KEY"))
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
    def cost_usd(self) -> Optional[float]:
        costs = [a.completion.cost_usd for a in self.attempts if a.completion]
        return sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None


def check(
    text: str, root: Path, pack: evidence_pack.EvidencePack
) -> tuple[Optional[Plan], Optional[dict], list[str]]:
    """Everything wrong with one answer, as sentences the model can act on."""
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


def plan_repository(
    root: Path,
    planner: Backend,
    *,
    name: Optional[str] = None,
    pack: Optional[evidence_pack.EvidencePack] = None,
    rounds: int = ROUNDS,
    verify: Optional[Callable[[Plan], list[str]]] = None,
) -> ModelPlan:
    """Ask for a plan, check it, and send it back until it holds or rounds run out.

    ``verify`` is a further check on a plan that holds -- running it, for a
    dataset -- whose problems go back the same way.
    """
    root = Path(root)
    pack = pack or evidence_pack.build(root, name)
    outcome = ModelPlan(pack=pack, backend=planner.name)
    messages: list[dict] = [{"role": "user", "content": pack.render()}]
    for _ in range(max(1, rounds)):
        try:
            completion = planner.complete(SYSTEM_PROMPT, messages, WIRE_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - a backend failing is an outcome
            outcome.error = f"{planner.name}: {exc}"
            break
        plan, wire, problems = check(completion.text, root, pack)
        if plan is not None and not problems and verify is not None:
            problems = verify(plan)
        outcome.attempts.append(Attempt(completion.text, problems, completion))
        if plan is not None:
            outcome.plan, outcome.wire = plan, wire
        if not problems:
            break
        messages += [
            {"role": "assistant", "content": completion.text},
            {"role": "user", "content": _feedback(problems)},
        ]
    return outcome


def _feedback(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems[:40])
    return (
        f"The plan you wrote has these problems:\n{listed}\n\n"
        "Write the whole plan again, corrected. Everything else about it may stay as it was."
    )


#: The step kinds, for anything that wants to name them without importing plan.
KINDS = (PYTHON_INSTALL, SYSTEM_PACKAGES, CMAKE_CONFIGURE, MESON_SETUP, PYTHON_RUN)
