"""The unblocker: an agent's workbench at a point the executor's rules cannot get past."""

import io
import json
import sys
from pathlib import Path

import pytest

from will_it_riscv import pseudobuild, unblock
from will_it_riscv.pseudobuild import available, run
from will_it_riscv.unblock import (
    CANNOT_STUB,
    PLAN_FAULT,
    UNBLOCKED,
    ClaudeUnblocker,
    Result,
    Workbench,
    call,
    serve,
)

needs_cmake = pytest.mark.skipif(not available(), reason="cmake is not installed")

#: Stops with an error that names nothing, after a lookup the error does not
#: mention: no rule has a suspect.
STUCK = """\
cmake_minimum_required(VERSION 3.16)
project(stuck NONE)
option(WITH_FROB_GUI "the GUI" OFF)
find_file(FROB_DATA frob.dat)
if(NOT FROB_DATA)
  message(FATAL_ERROR "cannot go on without the data")
endif()
"""


def _session(tmp_path: Path, fixed=None, trace_events=()) -> Workbench:
    scratch = tmp_path / "scratch"
    sysroot = scratch / "sysroot"
    (sysroot / "include").mkdir(parents=True)
    root = tmp_path / "src"
    root.mkdir()
    trace = scratch / "build-1.trace.json"
    trace.write_text("".join(json.dumps(e) + "\n" for e in [{"version": {"major": 1}}, *trace_events]))
    session = scratch / "unblock" / "session-1"
    session.mkdir(parents=True)
    (session / "state.json").write_text(json.dumps({
        "session": 1, "root": str(root), "scratch": str(scratch), "sysroot": str(sysroot),
        "toolchain": None, "env": {}, "settings": dict(fixed or {}), "fixed": dict(fixed or {}),
        "overrides": [], "signature": "boom", "error": "boom", "stanzas": ["CMake Error: boom"],
        "status": [], "round": 1, "trace": str(trace), "build": str(scratch / "build-1"),
        "blockers": [], "timeout": 60, "max_configures": 2, "earlier": [],
    }))
    return Workbench(session)


# -- what the agent may and may not do -----------------------------------------


def test_stubs_go_in_the_sysroot_and_nowhere_else(tmp_path):
    bench = _session(tmp_path)
    assert "wrote include/frob.h" in bench.write_file("include/frob.h", "#define FROB 1\n")
    for outside in ("../escape.h", str(tmp_path / "src" / "CMakeLists.txt"), "/etc/frob.h"):
        text, refused = call(bench, "write_file", {"path": outside, "content": "x"})
        assert refused and "outside the sysroot" in text
    (bench.sysroot / "lib").mkdir()
    (bench.sysroot / "lib" / "out").symlink_to(tmp_path / "src")
    assert call(bench, "write_file", {"path": "lib/out/x.h", "content": "x"})[1]
    assert not (tmp_path / "src" / "x.h").exists()


def test_cache_variables_are_lookup_results_only(tmp_path):
    events = [
        {"cmd": "option", "args": ["FROB_PARALLEL", "MPI", "OFF"], "file": "x", "line": 1},
        {"cmd": "find_package", "args": ["Frob", "REQUIRED"], "file": "x", "line": 2},
        {"cmd": "find_file", "args": ["FROB_DATA", "frob.dat"], "file": "x", "line": 3},
        {"cmd": "check_include_files", "args": ["dlfcn.h", "eckit_dlopen_ok"], "file": "x",
         "line": 4},
    ]
    bench = _session(tmp_path, fixed={"FROB_MPI": "ON"}, trace_events=events)
    lib = bench.sysroot / "lib" / "libfrob.so"
    assert "-DFROB_LIBRARY" in bench.set_cache("FROB_LIBRARY", str(lib))
    for name in ("FROB_DATA", "eckit_dlopen_ok", "Frob_DIR", "FROB_VERSION"):
        assert not call(bench, "set_cache", {"name": name, "value": "1"})[1], name
    for name, value, why in [
        ("FROB_MPI", "OFF", "the plan sets it"),
        ("CMAKE_BUILD_TYPE", "Release", "CMake's own"),
        ("FROB_PARALLEL", "ON", "build option"),
        ("WITH_GUI", "OFF", "build option"),
        ("FROB_INCLUDE_DIR", "/usr/include", "on the host"),
        # hdf5-vol-cache's own install directory: a _DIR, but no package's.
        ("HDF5_VOL_CACHE_INSTALL_BIN_DIR", str(bench.sysroot / "bin"), "plan_fault"),
    ]:
        text, refused = call(bench, "set_cache", {"name": name, "value": value})
        assert refused and why in text, name
    assert set(bench.ledger["cache"]) == {
        "FROB_LIBRARY", "FROB_DATA", "eckit_dlopen_ok", "Frob_DIR", "FROB_VERSION"}


def test_a_blocker_is_named_the_way_the_configure_asked_for_it(tmp_path):
    bench = _session(tmp_path)
    bench.write_file("include/netcdf_meta.h", "#define NC_VERSION_MAJOR 4\n")
    for messy in ("NetCDF-C (netcdf_meta.h)", "ZLIB, Glib", "CUDA (nvcc / CUDAToolkit)"):
        text, refused = call(bench, "finish", {"outcome": UNBLOCKED, "reason": "r",
                                               "blocker": messy})
        assert refused and "not a package name" in text
    assert call(bench, "finish", {"outcome": UNBLOCKED, "reason": "r", "blocker": "netcdf",
                                  "also": ["bad name"]})[1]
    bench.finish(UNBLOCKED, "the meta header", blocker="netcdf", also=["hdf5"],
                 version=">= 4.10.1")
    result = bench.result()
    assert (result.name, result.also, result.version) == ("netcdf", ["hdf5"], ">= 4.10.1")


def test_a_rewritten_executor_stub_is_put_back_when_the_session_is_rolled_back(tmp_path):
    bench = _session(tmp_path)
    theirs = bench.sysroot / "include" / "elpa.h"
    theirs.write_text("/* stub emitted by will-it-riscv */\n")
    bench.write_file("include/elpa.h", "#define ELPA_API_VERSION 20231705\n")
    bench.write_file("include/elpa/elpa_version.h", "x")
    assert call(bench, "remove_file", {"path": "include/other.h"})[1]   # not its own
    result = bench.result()
    assert result.outcome == "gave_up" and result.reason == "never called finish"
    assert theirs.read_text() == "/* stub emitted by will-it-riscv */\n"
    assert not (bench.sysroot / "include" / "elpa" / "elpa_version.h").exists()


def test_an_unblocked_session_hands_back_what_it_made_and_what_it_changed(tmp_path):
    bench = _session(tmp_path)
    theirs = bench.sysroot / "include" / "elpa.h"
    theirs.write_text("old\n")
    bench.write_file("include/elpa.h", "new\n")
    bench.write_file("lib/libelpa.so", "")
    assert call(bench, "finish", {"outcome": UNBLOCKED, "reason": "r"})[1]   # needs a name
    bench.finish(UNBLOCKED, "the version file", blocker="ELPA")
    assert call(bench, "finish", {"outcome": UNBLOCKED, "reason": "again"})[1]
    result = Workbench(bench.session).result()   # as the driver reads it, from the ledger
    assert result.proposes and result.name == "ELPA" and result.kind == "dependency"
    assert result.files == [str(bench.sysroot / "lib" / "libelpa.so")]
    assert result.restore == {str(theirs): "old\n"}
    assert theirs.read_text() == "new\n"


def test_the_trace_is_shown_around_the_line_that_raised_the_error(tmp_path):
    events = [{"cmd": "set", "args": ["A", str(i)], "file": str(tmp_path / "src" / "x.cmake"),
               "line": i} for i in range(1, 30)]
    events.append({"cmd": "message", "args": ["FATAL_ERROR", "boom"],
                   "file": str(tmp_path / "src" / "cmake" / "FindFrob.cmake"), "line": 7})
    bench = _session(tmp_path, trace_events=events)
    shown = bench.show_trace(around="cmake/FindFrob.cmake:7", before=3)
    assert shown.splitlines() == [
        "x.cmake:27 set(A 27)", "x.cmake:28 set(A 28)", "x.cmake:29 set(A 29)",
        ">> cmake/FindFrob.cmake:7 message(FATAL_ERROR boom)",
    ]
    assert "2 events match" in bench.show_trace(grep=r"set\(A 2[89]\)")


def test_configures_are_counted(tmp_path, monkeypatch):
    bench = _session(tmp_path)
    monkeypatch.setattr(pseudobuild, "_configure",
                        lambda *a, **k: pseudobuild.PseudoBuild(error="boom", narration=""))
    assert "SAME error" in bench.configure()
    monkeypatch.setattr(pseudobuild, "_configure",
                        lambda *a, **k: pseudobuild.PseudoBuild(error="other", narration=""))
    assert "MOVED" in bench.configure()
    text, refused = call(bench, "configure", {})
    assert refused and "spent" in text


# -- MCP --------------------------------------------------------------------------


def test_the_workbench_speaks_mcp(tmp_path):
    bench = _session(tmp_path)
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "write_file", "arguments": {"path": "include/a.h", "content": "x"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "set_cache", "arguments": {"name": "CMAKE_X", "value": "1"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
    ]
    out = io.StringIO()
    serve(bench.session, io.StringIO("".join(json.dumps(m) + "\n" for m in lines)), out)
    replies = {r["id"]: r for r in map(json.loads, out.getvalue().splitlines())}
    assert sorted(replies) == [1, 2, 3, 4, 5]
    assert replies[1]["result"]["protocolVersion"] == "2025-06-18"
    assert {t["name"] for t in replies[2]["result"]["tools"]} >= {"configure", "finish"}
    assert replies[3]["result"]["isError"] is False
    assert replies[4]["result"]["isError"] is True
    assert replies[5]["error"]["code"] == -32601
    assert (bench.sysroot / "include" / "a.h").read_text() == "x"


# -- the executor's side ------------------------------------------------------------


class Fake:
    """An unblocker that does what it is told, and remembers what it was shown."""

    sessions = 2

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen = []
        self.verdicts = []

    def __call__(self, stuck):
        self.seen.append(stuck)
        return self.answers.pop(0)(stuck)

    def judged(self, result, confirmed):
        self.verdicts.append((result.name, confirmed))


def _right(stuck):
    data = stuck.sysroot / "share" / "frob.dat"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text("")
    return Result(UNBLOCKED, name="frob", reason="the data file",
                  overrides={"FROB_DATA": str(data)}, files=[str(data)])


def _wrong(stuck):
    decoy = stuck.sysroot / "include" / "decoy.h"
    decoy.write_text("")
    return Result(UNBLOCKED, name="decoy", reason="a guess", files=[str(decoy)])


@needs_cmake
def test_a_stuck_configure_is_handed_over_and_the_executor_judges_the_stubs(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(STUCK)
    fake = Fake(_right)
    result = run(tmp_path, unblocker=fake)
    assert result.completed
    assert result.blockers == ["frob"]
    assert ("frob", True) in result.experiments
    assert result.unblocker[0]["outcome"] == UNBLOCKED and result.unblocker[0]["confirmed"]
    assert fake.verdicts == [("frob", True)]
    stuck = fake.seen[0]
    assert "cannot go on without the data" in "\n".join(pseudobuild._error_stanzas(
        stuck.outcome.narration))
    assert stuck.trace.name == "build-1.trace.json"


@needs_cmake
def test_stubs_that_do_not_move_the_error_are_taken_back_out(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(STUCK)
    fake = Fake(_wrong, _right)
    result = run(tmp_path, unblocker=fake)
    assert result.completed and result.blockers == ["frob"]
    assert ("decoy", False) in result.experiments
    assert fake.verdicts == [("decoy", False), ("frob", True)]
    assert fake.seen[1].earlier[0]["name"] == "decoy"
    assert fake.seen[1].earlier[0]["confirmed"] is False
    assert not (fake.seen[1].sysroot / "include" / "decoy.h").exists()


@needs_cmake
def test_a_plan_fault_is_a_note_not_a_trial(tmp_path):
    """And the end of the line: asking again at the same point gets the same answer."""
    (tmp_path / "CMakeLists.txt").write_text(STUCK)
    fault = Result(PLAN_FAULT, reason="it wants the data path", flags={"FROB_DATA_PATH": "/x"})
    fake = Fake(lambda s: fault)
    result = run(tmp_path, unblocker=fake)
    assert not result.completed and result.blockers == []
    assert [r["outcome"] for r in result.unblocker] == [PLAN_FAULT]
    assert any("plan_fault (-DFROB_DATA_PATH=/x): it wants the data path" in n for n in result.notes)
    assert fake.verdicts == []


@needs_cmake
def test_what_no_stub_can_be_is_still_the_blocker(tmp_path):
    """A default build that turns CUDA on: no stub, and the answer is CUDA."""
    (tmp_path / "CMakeLists.txt").write_text(STUCK)
    cuda = Result(CANNOT_STUB, name="CUDA", reason="enable_language(CUDA) needs nvcc")
    result = run(tmp_path, unblocker=Fake(lambda s: cuda))
    assert not result.completed and result.blockers == ["CUDA"]
    assert result.round_blockers[-1] == "CUDA"


class Remembering(ClaudeUnblocker):
    """The driver with the model taken out: _right, counted, plus a program
    that names the sysroot it was written in."""

    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)
        self.asked = 0

    def _session(self, stuck):
        self.asked += 1
        result = _right(stuck)
        config = stuck.sysroot / "bin" / "frob-config"
        config.write_text(f"#!/bin/sh\necho {stuck.sysroot}/include\n")
        config.chmod(0o755)
        result.files.append(str(config))
        return result


@needs_cmake
def test_the_same_stuck_point_again_is_replayed_not_asked(tmp_path, monkeypatch):
    """A plan configures one project once per step; each run starts afresh."""
    monkeypatch.setattr(unblock, "_REPLAYS", {})
    (tmp_path / "CMakeLists.txt").write_text(STUCK)
    unblocker = Remembering(tmp_path / "logs")
    first = run(tmp_path, unblocker=unblocker)
    second = run(tmp_path, unblocker=unblocker)
    assert first.completed and second.completed and second.blockers == ["frob"]
    assert unblocker.asked == 1
    assert second.unblocker[0]["cost"] == 0.0 and second.unblocker[0]["confirmed"] is True
    # What the scoreboard adds up: one session, then one replay.
    assert [bool(r["replayed"]) for r in unblock.SPENT[-2:]] == [False, True]
    # The replayed program names the second run's sysroot, not the first's.
    replayed = []
    original = unblock._replay

    def spy(memo, stuck, key):
        result = original(memo, stuck, key)
        replayed.append(((stuck.sysroot / "bin" / "frob-config").read_text(), stuck.sysroot))
        return result

    monkeypatch.setattr(unblock, "_replay", spy)
    third = run(tmp_path, unblocker=unblocker)
    assert third.completed and unblocker.asked == 1
    text, sysroot = replayed[0]
    assert text == f"#!/bin/sh\necho {sysroot}/include\n"


@needs_cmake
def test_a_rejected_session_is_not_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(unblock, "_REPLAYS", {})
    (tmp_path / "CMakeLists.txt").write_text(STUCK)

    class Wrong(Remembering):
        def _session(self, stuck):
            self.asked += 1
            return _wrong(stuck)

    unblocker = Wrong(tmp_path / "logs")
    unblocker.sessions = 1
    run(tmp_path, unblocker=unblocker)
    run(tmp_path, unblocker=unblocker)
    assert unblocker.asked == 2
    assert not list((tmp_path / "logs" / "replay").glob("*.json"))


@needs_cmake
def test_what_one_process_found_the_next_one_replays(tmp_path, monkeypatch):
    """The reference run, the relabelling and an eval are separate processes."""
    monkeypatch.setattr(unblock, "_REPLAYS", {})
    (tmp_path / "CMakeLists.txt").write_text(STUCK)
    first = Remembering(tmp_path / "logs")
    run(tmp_path, unblocker=first)
    assert len(list((tmp_path / "logs" / "replay").glob("*.json"))) == 1
    monkeypatch.setattr(unblock, "_REPLAYS", {})   # a new process
    later = Remembering(tmp_path / "logs")
    again = run(tmp_path, unblocker=later)
    assert later.asked == 0 and again.completed and again.blockers == ["frob"]
    monkeypatch.setattr(unblock, "_REPLAYS", {})   # another process, kept elsewhere
    elsewhere = Remembering(tmp_path / "other-logs")
    run(tmp_path, unblocker=elsewhere)
    assert elsewhere.asked == 1


@needs_cmake
def test_time_in_a_session_is_not_the_configures(tmp_path, monkeypatch):
    (tmp_path / "CMakeLists.txt").write_text(STUCK)
    clock = [0.0]
    monkeypatch.setattr(pseudobuild.time, "monotonic", lambda: clock[0])

    def slow(stuck):
        clock[0] += 10_000
        return _right(stuck)

    assert run(tmp_path, unblocker=Fake(slow), timeout=600).completed


# -- end to end, with a stand-in for claude --------------------------------------------

#: Plays the agent: starts the MCP server it was configured with, and does
#: what a model would, through it.
AGENT = r'''#!{python}
import json, re, subprocess, sys
args = sys.argv[1:]
assert args[args.index("--tools") + 1] == "Read,Grep,Glob"
assert args[args.index("--permission-mode") + 1] == "dontAsk"
assert "--strict-mcp-config" in args
prompt = sys.stdin.read()
sysroot = re.search(r"^Sysroot: (.*)$", prompt, re.M).group(1)
server = json.load(open(args[args.index("--mcp-config") + 1]))["mcpServers"]["unblock"]
import os
proc = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, text=True, env={{**os.environ, **server["env"]}})
def rpc(i, method, params=None):
    proc.stdin.write(json.dumps({{"jsonrpc": "2.0", "id": i, "method": method,
                                 "params": params or {{}}}}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())["result"]
def tool(i, which, **arguments):
    return rpc(i, "tools/call", {{"name": which, "arguments": arguments}})["content"][0]["text"]
rpc(1, "initialize", {{"protocolVersion": "2025-06-18"}})
assert "FROB_DATA frob.dat" in tool(2, "show_lookups")
assert "refused" in tool(3, "set_cache", name="WITH_FROB_GUI", value="ON")
tool(4, "write_file", path="share/frob.dat", content="")
tool(5, "set_cache", name="FROB_DATA", value=sysroot + "/share/frob.dat")
assert "COMPLETED" in tool(6, "configure")
tool(7, "finish", outcome="unblocked", blocker="frob", reason="find_file wanted frob.dat")
proc.stdin.close(); proc.wait()
print(json.dumps({{"type": "result", "total_cost_usd": 0.01}}))
'''


@needs_cmake
def test_a_claude_session_end_to_end(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "CMakeLists.txt").write_text(STUCK)
    agent = tmp_path / "claude"
    agent.write_text(AGENT.format(python=sys.executable))
    agent.chmod(0o755)
    logs = tmp_path / "logs"
    unblocker = ClaudeUnblocker(command=str(agent), log_dir=logs)
    result = run(tmp_path / "project", unblocker=unblocker)
    assert result.completed and result.blockers == ["frob"]
    record = result.unblocker[0]
    assert record["confirmed"] is True and record["cost"] == 0.01 and record["configures"] == 1
    kept = Path(record["log"])
    assert kept.parent == logs
    summary = json.loads((kept / "result.json").read_text())
    assert summary["confirmed"] is True and summary["name"] == "frob"
    assert (kept / "files" / "share" / "frob.dat").is_file()
    calls = [json.loads(line)["tool"] for line in (kept / "calls.jsonl").read_text().splitlines()]
    assert calls == ["show_lookups", "set_cache", "write_file", "set_cache", "configure", "finish"]
    # Only what the executor changed of the environment is kept, not the rest of it.
    assert set(json.loads((kept / "state.json").read_text())["env"]) <= {
        "PKG_CONFIG_LIBDIR", "PKG_CONFIG_PATH", "CMAKE_BUILD_PARALLEL_LEVEL", "PATH"}


def test_the_prompt_is_for_the_target_asked_for():
    prompt = unblock.system_prompt("aarch64")
    assert "pretend aarch64 sandbox" in prompt and "Linux/aarch64" in prompt
    assert "riscv64" not in prompt.replace("will-it-riscv", "")


def test_it_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("WILL_IT_RISCV_UNBLOCK", raising=False)
    assert unblock.from_environment() is None
    monkeypatch.setenv("WILL_IT_RISCV_UNBLOCK", "claude:opus")
    monkeypatch.setenv("WILL_IT_RISCV_UNBLOCK_BUDGET", "0.5")
    chosen = unblock.from_environment()
    assert chosen.model == "opus" and chosen.budget_usd == 0.5
