"""The agent: plan, run, stop at stuck points, fix, finish -- with tools."""

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeIndex
from test_plan import FakeDistro

from will_it_riscv import agent, unblock
from will_it_riscv.agent import ClaudeAgent, Session, call
from will_it_riscv.evidence import EvidencePack
from will_it_riscv.pseudobuild import available

needs_cmake = pytest.mark.skipif(not available(), reason="cmake is not installed")

#: Stops with an error that names nothing: the rules have no suspect.
STUCK = """\
cmake_minimum_required(VERSION 3.16)
project(stuck NONE)
find_file(FROB_DATA frob.dat)
if(NOT FROB_DATA)
  message(FATAL_ERROR "cannot go on without the data")
endif()
"""

WIRE = {
    "repo": "stuck",
    "steps": [{
        "id": "configure", "kind": "cmake-configure", "source": ".", "after": [],
        "evidence": [{"at": "CMakeLists.txt:3", "quote": "find_file(FROB_DATA frob.dat)"}],
    }],
    "unsure": [],
}


@pytest.fixture
def session(tmp_path, monkeypatch, target):
    monkeypatch.setenv("WILL_IT_RISCV_UNBLOCK_LOG", str(tmp_path / "logs"))
    monkeypatch.setattr(unblock, "_REPLAYS", {})
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(STUCK)
    directory = tmp_path / "session"
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps({"root": str(project), "name": "stuck"}))
    env = SimpleNamespace(index=FakeIndex({}), target=target, distro=FakeDistro(set()),
                          timeout=120, pip_cache=tmp_path / "pip")
    made = Session(directory, env)
    yield made
    made.close()


def _stub(session):
    data = str(Path(session.bench.sysroot) / "share" / "frob.dat")
    assert call(session, "write_file", {"path": "share/frob.dat", "content": ""}) == (
        "wrote share/frob.dat (0 chars)", False)
    assert not call(session, "set_cache", {"name": "FROB_DATA", "value": data})[1]


@needs_cmake
def test_a_stuck_run_pauses_for_the_agent_and_goes_on_when_it_resumes(session):
    assert "plan kept: 1 steps" in session.write_plan(WIRE)
    paused = session.run_plan()
    assert "The run paused" in paused and "cannot go on without the data" in paused
    _stub(session)
    assert "COMPLETED" in (moved := call(session, "configure", {})[0]) and "Call resume" in moved
    done = call(session, "resume", {"outcome": "unblocked", "blocker": "frob",
                                    "reason": "find_file wanted frob.dat"})[0]
    assert done.startswith("verdict:") and "configure (cmake-configure): completed" in done
    assert "your stubs for frob: kept" in done
    assert (session.directory / "last-run.json").is_file()
    # The fix is kept: a second run of the same plan is answered without a pause.
    again = session.run_plan()
    assert "answered by earlier fixes: frob" in again
    record = json.loads((session.directory / "agent.json").read_text())
    assert [r["replayed"] for r in record["runs"]] == [0, 1] and record["wire"] == WIRE
    # A run with no fault of the plan's is the end: no finish call, no summary.
    assert "you are done: reply done" in again and record["finish"] == {"note": "", "clean": True}
    session.write_plan({**WIRE, "unsure": ["a note"]})     # unless it writes another plan
    assert json.loads((session.directory / "agent.json").read_text())["finish"] is None


@needs_cmake
def test_a_run_can_be_abandoned_at_a_stuck_point_to_change_the_plan(session):
    session.write_plan(WIRE)
    session.run_plan()
    assert call(session, "write_plan", {"plan": WIRE})[1]   # not while paused
    assert "abandoned" in session.abort_run("the plan needs a flag")
    assert not session.running and session.bench is None
    assert json.loads((session.directory / "agent.json").read_text())["runs"] == [
        {"aborted": True}]


def test_the_workbench_is_only_for_stuck_points(session):
    text, refused = call(session, "show_trace", {"grep": "x"})
    assert refused and "nothing is paused" in text
    assert call(session, "run_plan", {})[1]                  # no plan yet
    text, refused = call(session, "write_plan", {"plan": {"repo": "x", "steps": [{"id": "?"}]}})
    assert refused and "not a plan" in text
    assert "kept" in call(session, "write_plan", {"plan": WIRE})[0]
    assert call(session, "write_plan", {"plan": WIRE})[0].startswith("unchanged: ")
    assert json.loads((session.directory / "agent.json").read_text())["plans"] == 1
    assert "recorded" in call(session, "finish", {"note": "done"})[0]


def test_the_tools_are_served_over_mcp(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"root": str(tmp_path)}))
    lines = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}]
    out = io.StringIO()
    agent.serve(tmp_path, stdin=io.StringIO(json.dumps(lines[0]) + "\n"), stdout=out)
    names = [t["name"] for t in json.loads(out.getvalue())["result"]["tools"]]
    assert names[:5] == ["write_plan", "show_plan", "run_plan", "resume", "abort_run"]
    assert {"show_trace", "write_file", "set_cache", "configure", "finish"} <= set(names)


def test_the_prompt_is_for_the_target_asked_for():
    prompt = agent.system_prompt("aarch64", "Ubuntu 26.04 (resolute)")
    assert "build on aarch64 Linux (Ubuntu 26.04 (resolute))" in prompt
    assert "Cite lines you have read." in prompt and "riscv64" not in prompt.split("The plan:")[0]


#: Plays the agent: starts the MCP server it was configured with, writes a
#: plan and finishes.
AGENT = r'''#!{python}
import json, os, subprocess, sys
args = sys.argv[1:]
assert "--strict-mcp-config" in args and "Read,Grep,Glob,mcp__plan" in args
assert "Target: riscv64" in sys.stdin.read()
server = json.load(open(args[args.index("--mcp-config") + 1]))["mcpServers"]["plan"]
proc = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, text=True, env={{**os.environ, **server["env"]}})
def tool(i, which, **arguments):
    proc.stdin.write(json.dumps({{"jsonrpc": "2.0", "id": i, "method": "tools/call",
                                 "params": {{"name": which, "arguments": arguments}}}}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())["result"]["content"][0]["text"]
assert "kept" in tool(1, "write_plan", plan={wire})
assert "recorded" in tool(2, "finish", note="enough")
proc.stdin.close(); proc.wait()
print(json.dumps({{"type": "result", "total_cost_usd": 0.25,
                   "usage": {{"input_tokens": 100, "cache_read_input_tokens": 900,
                              "output_tokens": 50}}}}))
'''


def test_the_agent_end_to_end_through_claude(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(STUCK)
    fake = tmp_path / "claude"
    fake.write_text(AGENT.format(python=sys.executable, wire=repr(WIRE)))
    fake.chmod(0o755)
    pack = EvidencePack(repo="stuck")
    outcome = ClaudeAgent(command=str(fake), log_dir=tmp_path / "logs")(project, pack, "stuck")
    assert outcome.error is None and outcome.finished
    assert [s.id for s in outcome.plan.steps] == ["configure"]
    assert (outcome.tokens_in, outcome.tokens_out, outcome.cost) == (1000, 50, 0.25)
    assert outcome.tokens_cached == 900
    calls = [json.loads(line)["tool"]
             for line in (Path(outcome.log) / "calls.jsonl").read_text().splitlines()]
    assert calls == ["write_plan", "finish"]


class FakeServer:
    """An OpenAI-compatible endpoint that answers from a script, and keeps what it was sent."""

    def __init__(self, answers):
        self.answers, self.requests = list(answers), []

    def post(self, url, json):
        self.requests.append(copy.deepcopy(json))      # as it went out
        message = self.answers.pop(0)
        if isinstance(message, int):                    # the server could not read the turn
            import httpx

            response = httpx.Response(message, text="does not match the expected format",
                                      request=httpx.Request("POST", url))
            response.raise_for_status()
        body = {"choices": [{"message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
                "timings": {"cache_n": 600, "prompt_ms": 400.0, "predicted_ms": 1600.0}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)


def _calls(*calls):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}
        for i, (name, arguments) in enumerate(calls)]}


@needs_cmake
def test_a_served_model_plans_with_the_same_tools_and_a_clean_run_ends_it(tmp_path, target):
    from will_it_riscv.agent import OpenAIAgent

    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\nproject(ok NONE)\n")
    wire = {"repo": "ok", "steps": [{"id": "configure", "kind": "cmake-configure", "source": ".",
            "after": [], "evidence": [{"at": "CMakeLists.txt:2", "quote": "project(ok NONE)"}]}],
            "unsure": []}
    server = FakeServer([
        500,                                            # asked again
        _calls(("Grep", {"pattern": "project\\(", "glob": "CMakeLists.txt"})),
        # A small model may hand the plan over bare, not under "plan".
        _calls(("write_plan", wire), ("run_plan", {})),
    ])
    env = SimpleNamespace(index=FakeIndex({}), target=target, distro=FakeDistro(set()),
                          timeout=120, pip_cache=tmp_path / "pip")
    outcome = OpenAIAgent("http://x/v1", "m", client=server, log_dir=tmp_path / "logs")(
        project, EvidencePack(repo="ok"), "ok", env=env)
    assert outcome.error is None and outcome.finished      # no turn after the clean run
    assert [s.id for s in outcome.plan.steps] == ["configure"] and len(server.requests) == 3
    names = [t["function"]["name"] for t in server.requests[0]["tools"]]
    plan = next(t["function"]["parameters"] for t in server.requests[0]["tools"]
                if t["function"]["name"] == "write_plan")
    assert "evidence" in plan["$defs"] and "$defs" not in plan["properties"]["plan"]
    assert names[:3] == ["Read", "Grep", "Glob"] and "write_plan" in names
    grep = server.requests[2]["messages"][-1]
    assert grep["role"] == "tool" and grep["content"] == "CMakeLists.txt:2:project(ok NONE)"
    assert (outcome.tokens_in, outcome.tokens_out, outcome.tokens_cached) == (2000, 100, 1200)
    assert outcome.seconds == 4.0


def test_a_served_model_that_repeats_itself_is_stopped(tmp_path):
    from will_it_riscv.agent import OpenAIAgent

    (tmp_path / "state.json").write_text("{}")
    server = FakeServer([_calls(("show_plan", {}))] * 5)
    outcome = OpenAIAgent("http://x/v1", "m", client=server, log_dir=tmp_path / "logs")(
        tmp_path, EvidencePack(repo="x"), "x", env=SimpleNamespace())
    assert outcome.error == "it called show_plan three times over" and len(server.requests) == 3
    assert outcome.plan is None
