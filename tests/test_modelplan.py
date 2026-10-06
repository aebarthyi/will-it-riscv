"""Evidence packs, the model contract, the dataset and its evaluation."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from will_it_riscv import dataset, evaluate, evidence, modelplan
from will_it_riscv.plan import load_plan, parse_plan, plan_to_dict


def write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


REPO = {
    "README.md": (
        "# demo\n\nA thing.\n\n## Usage\n\nimport demo\n\n"
        "## Building\n\nRun `./build.sh`, which installs the toolchain and configures.\n\n"
        "## License\n\nMIT\n"
    ),
    "build.sh": (
        "#!/bin/bash\n"
        "source scripts/deps.sh\n"
        "python3 tools/driver.py build\n"
    ),
    "lint.sh": "#!/bin/bash\nruff check .\n",
    "scripts/deps.sh": "apt-get install -y libfftw3-dev\npip install -r requirements.txt\n",
    "tools/driver.py": "import subprocess\nsubprocess.run(['cmake', '-S', 'src', '-B', 'b'])\n",
    "requirements.txt": "numpy\n",
    "CMakeLists.txt": "cmake_minimum_required(VERSION 3.18)\nproject(demo C)\nfind_package(FFTW)\n",
    ".github/workflows/ci.yml": "jobs:\n  build:\n    steps:\n      - run: |\n          ./build.sh\n",
    ".github/workflows/docs.yml": "jobs:\n  docs:\n    steps:\n      - run: make docs\n",
}


# -- the evidence pack ------------------------------------------------------


def test_the_pack_shows_what_says_how_it_builds(tmp_path):
    write(tmp_path, REPO)
    pack = evidence.build(tmp_path, "demo")
    paths = [e.path for e in pack.excerpts]
    assert paths[:4] == ["README.md", "build.sh", "scripts/deps.sh", "tools/driver.py"]
    readme = pack.excerpt("README.md")
    shown = [text for _, text in readme.lines]
    assert "## Building" in shown and "## License" not in shown and "## Usage" not in shown
    assert paths.index(".github/workflows/ci.yml") < paths.index(".github/workflows/docs.yml")
    assert paths[-1] == "lint.sh"   # reached, but the last thing a build needs shown
    assert pack.shows("build.sh", 2, 3) and not pack.shows("build.sh", 4, 4)
    text = pack.render()
    assert "    2| source scripts/deps.sh" in text


def test_the_same_tree_gives_the_same_pack(tmp_path):
    write(tmp_path, REPO)
    first, second = evidence.build(tmp_path, "demo"), evidence.build(tmp_path, "demo")
    assert first.digest() == second.digest()
    again = evidence.EvidencePack.from_dict(json.loads(json.dumps(first.to_dict())))
    assert again.render() == first.render()


def test_what_does_not_fit_is_named(tmp_path):
    write(tmp_path, REPO)
    pack = evidence.build(tmp_path, "demo", budget=900)
    assert pack.omitted
    assert "NOT SHOWN, FOR LACK OF ROOM" in pack.render()


# -- the wire format ----------------------------------------------------------


def closed(schema):
    """Every object in a schema closed, as structured outputs require."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False, schema
        for value in schema.values():
            closed(value)
    elif isinstance(schema, list):
        for value in schema:
            closed(value)


def test_the_wire_schema_is_strict_enough_for_constrained_decoding():
    closed(modelplan.WIRE_SCHEMA)
    text = json.dumps(modelplan.WIRE_SCHEMA)
    assert '"type": ["' not in text   # no type arrays


def test_the_schema_lists_keys_in_the_order_the_wire_writes_them():
    # A grammar writes keys in schema order; a model trained on to_wire's order
    # stops early when made to write them in another.
    schema = modelplan.WIRE_SCHEMA
    top = list(schema["properties"])
    step = list(schema["properties"]["steps"]["items"]["properties"])
    for path in sorted(Path("examples/plans").glob("*.json")):
        wire = modelplan.to_wire(load_plan(path))
        assert list(wire) == [k for k in top if k in wire], path
        for raw in wire["steps"]:
            assert list(raw) == [k for k in step if k in raw], (path, raw["id"])


def test_the_grammar_keeps_only_the_leading_keys_required():
    grammar = modelplan.in_written_order(modelplan.WIRE_SCHEMA)
    assert grammar["required"] == ["repo", "steps", "unsure"]
    assert grammar["properties"]["steps"]["items"]["required"] == ["id", "kind"]
    assert grammar["properties"]["entry"]["required"] == ["command", "evidence"]
    assert modelplan.WIRE_SCHEMA["properties"]["steps"]["items"]["required"] == [
        "id", "kind", "evidence"]


def test_the_grammar_bounds_lists_and_strings():
    grammar = modelplan.bounded(modelplan.in_written_order(modelplan.WIRE_SCHEMA))
    assert grammar["properties"]["unsure"]["maxItems"] == 12
    assert grammar["properties"]["unsure"]["items"]["maxLength"] == modelplan.MAX_LENGTH
    step = grammar["properties"]["steps"]["items"]["properties"]
    assert step["packages"]["maxItems"] == 100
    assert step["note"]["maxLength"] == modelplan.MAX_LENGTH
    assert "maxLength" not in step["kind"]
    assert grammar["$defs"]["evidence"]["maxItems"] == 20
    assert grammar["$defs"]["evidence"]["items"]["properties"]["quote"]["maxLength"] == 600
    assert "maxItems" not in modelplan.WIRE_SCHEMA["properties"]["unsure"]   # the teacher's is unchanged


def test_a_plan_goes_to_the_wire_and_back():
    plan = load_plan("examples/plans/mfc.json")
    wire = modelplan.to_wire(plan)
    assert all(isinstance(e, dict) for s in wire["steps"] for e in s["evidence"])
    assert plan_to_dict(parse_plan(modelplan.to_plan(wire))) == plan_to_dict(plan)


# -- the loop -----------------------------------------------------------------


class Scripted:
    """A backend that says what it is told to, and remembers what it was asked."""

    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def complete(self, system, messages, schema):
        self.asked.append(list(messages))
        return modelplan.Completion(json.dumps(self.answers.pop(0)), 10, 5, 0.01)


def wire(at="build.sh:2", quote="source scripts/deps.sh"):
    return {
        "repo": "demo",
        "steps": [
            {"id": "deps", "kind": "system-packages", "packages": ["libfftw3-dev"],
             "evidence": [{"at": "scripts/deps.sh:1", "quote": "libfftw3-dev"}]},
            {"id": "configure", "kind": "cmake-configure", "source": ".", "after": ["deps"],
             "defines": [{"name": "WITH_FFTW", "value": "ON"}],
             "evidence": [{"at": at, "quote": quote}]},
        ],
        "unsure": [],
    }


def test_a_quote_cited_at_the_wrong_line_moves_to_where_the_pack_says_it(tmp_path):
    write(tmp_path, REPO)
    elsewhere = wire(at="README.md:3", quote="source scripts/deps.sh")
    elsewhere["steps"][0]["evidence"] = [{"at": "build.sh:3", "quote": "libfftw3-dev"}]
    planner = Scripted(elsewhere)
    outcome = modelplan.plan_repository(tmp_path, planner, name="demo")
    assert outcome.ok and len(outcome.attempts) == 1
    assert [str(e) for e in outcome.plan.step("configure").evidence] == ["build.sh:2"]
    assert [str(e) for e in outcome.plan.step("deps").evidence] == ["scripts/deps.sh:1"]
    assert outcome.attempts[0].moved == [
        "step 'deps': build.sh:3 -> scripts/deps.sh:1",
        "step 'configure': README.md:3 -> build.sh:2",
    ]


def test_a_quote_moves_to_the_nearest_place_in_the_file_it_cites(tmp_path):
    script = "source scripts/deps.sh\ncmake .\necho\necho\necho\ncmake .\n"
    write(tmp_path, {**REPO, "build.sh": script})
    plan, _, problems = modelplan.check(
        json.dumps(wire(at="build.sh:5", quote="cmake .")), tmp_path, evidence.build(tmp_path))
    assert not problems and str(plan.step("configure").evidence[0]) == "build.sh:6"


def test_a_plan_that_holds_is_taken_first_time(tmp_path):
    write(tmp_path, REPO)
    planner = Scripted(wire())
    outcome = modelplan.plan_repository(tmp_path, planner, name="demo")
    assert outcome.ok and len(outcome.attempts) == 1
    assert outcome.plan.step("configure").defines == {"WITH_FFTW": "ON"}
    assert outcome.cost_usd == pytest.approx(0.01)


def test_what_does_not_hold_goes_back_and_is_fixed(tmp_path):
    write(tmp_path, REPO)
    planner = Scripted(wire(quote="cmake --build"), wire(at="lint.sh:9", quote="ninja install"), wire())
    outcome = modelplan.plan_repository(tmp_path, planner, name="demo")
    assert outcome.ok and len(outcome.attempts) == 3
    assert "does not say 'cmake --build'" in " ".join(outcome.attempts[0].problems)
    assert any("past the end" in p or "not a line the pack shows" in p
               for p in outcome.attempts[1].problems)
    feedback = planner.asked[1][-1]["content"]
    assert feedback.startswith("The plan you wrote has these problems")


def test_a_backend_that_fails_is_an_outcome_not_a_crash(tmp_path):
    write(tmp_path, REPO)

    class Broken:
        name = "broken"

        def complete(self, system, messages, schema):
            raise RuntimeError("no network")

    outcome = modelplan.plan_repository(tmp_path, Broken(), name="demo")
    assert outcome.plan is None and "no network" in outcome.error


# -- the backends ---------------------------------------------------------------


def test_an_openai_compatible_server_is_asked_with_the_schema(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8000/v1/chat/completions",
        json={"choices": [{"message": {"content": '{"repo": "x"}'}}],
              "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    )
    backend = modelplan.OpenAICompatible("http://localhost:8000/v1", "qwen3-8b")
    completion = backend.complete("sys", [{"role": "user", "content": "pack"}], {"type": "object"})
    body = json.loads(httpx_mock.get_request().content)
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert completion.text == '{"repo": "x"}' and completion.input_tokens == 7
    assert body["temperature"] == 0


def test_an_openai_compatible_server_can_be_asked_to_sample(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8000/v1/chat/completions",
        json={"choices": [{"message": {"content": "{}"}}], "usage": {}},
    )
    backend = modelplan.OpenAICompatible("http://localhost:8000/v1", "m", temperature=0.8)
    backend.complete("sys", [], {"type": "object"})
    assert json.loads(httpx_mock.get_request().content)["temperature"] == 0.8


def test_the_claude_api_is_asked_for_a_structured_output():
    seen = {}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text='{"repo": "x"}')],
                usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            )

    class Messages:
        def stream(self, **params):
            seen.update(params)
            return Stream()

    backend = modelplan.Anthropic(client=SimpleNamespace(messages=Messages()))
    completion = backend.complete("sys", [{"role": "user", "content": "pack"}], {"type": "object"})
    assert seen["model"] == modelplan.DEFAULT_TEACHER
    assert seen["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert completion.text == '{"repo": "x"}'


def test_claude_code_is_run_headless_with_its_tools_off(monkeypatch):
    ran = {}

    def run(command, **kwargs):
        ran["command"], ran["input"] = command, kwargs["input"]
        out = {"is_error": False, "structured_output": {"repo": "x"}, "total_cost_usd": 0.02,
               "usage": {"input_tokens": 50, "output_tokens": 9}}
        return subprocess.CompletedProcess(command, 0, json.dumps(out), "")

    monkeypatch.setattr(modelplan.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(modelplan.subprocess, "run", run)
    completion = modelplan.ClaudeCLI().complete("sys", [{"role": "user", "content": "pack"}], {})
    command = ran["command"]
    assert command[command.index("--tools") + 1] == ""
    assert "--json-schema" in command and ran["input"] == "pack"
    assert json.loads(completion.text) == {"repo": "x"} and completion.cost_usd == 0.02


# -- the dataset ---------------------------------------------------------------


def test_a_corpus_says_where_each_repository_comes_from(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"name": "mfc", "git": "https://github.com/MFlowCode/MFC"}\n'
        "# a comment\n"
        '{"name": "numpy", "pypi": "numpy", "version": "2.5.3", "split": "test"}\n'
    )
    mfc, numpy = dataset.load_corpus(corpus)
    assert mfc.key == "mfc" and numpy.key == "numpy-2.5.3" and numpy.held_out == "test"
    assert mfc.held_out == dataset.Entry(name="mfc").held_out   # by name, every time
    corpus.write_text('{"name": "x", "git": "u", "pypi": "x"}\n')
    with pytest.raises(SystemExit):
        dataset.load_corpus(corpus)


def test_a_labelled_repository_becomes_a_training_example(tmp_path):
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    entry = dataset.Entry(name="demo", path=str(repo), split="train")
    meta = dataset.label(entry, repo, Scripted(wire()), out)
    assert meta["tier"] == dataset.VERIFIED and meta["rounds"] == 1
    for name in ("pack.txt", "pack.json", "plan.json", "transcript.json", "meta.json"):
        assert (out / "demo" / name).is_file()
    counts = dataset.export(out, tmp_path / "sft")
    assert counts == {"train": 1}
    example = json.loads((tmp_path / "sft" / "train.jsonl").read_text())
    system, user, assistant = example["messages"]
    assert system["content"] == modelplan.SYSTEM_PROMPT
    assert user["content"].startswith("REPOSITORY demo")
    assert parse_plan(modelplan.to_plan(json.loads(assistant["content"]))).repo == "demo"


def test_every_answer_sent_back_becomes_a_correction_to_the_verified_plan(tmp_path):
    """The teacher's own wrong round, and a model's from an eval's transcripts."""
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    entry = dataset.Entry(name="demo", path=str(repo), split="train")
    wrong = wire(quote="cmake --build")
    dataset.label(entry, repo, Scripted(wrong, wire()), out)
    mined = tmp_path / "mined"
    mined.mkdir()
    (mined / "demo.json").write_text(json.dumps({"attempts": [
        {"text": "{}", "problems": ["the plan names no steps"]},
        {"text": json.dumps(wire()), "problems": []},
    ]}))
    assert dataset.export(out, tmp_path / "plain") == {"train": 1}
    counts = dataset.export(out, tmp_path / "sft", corrections=True, transcripts=(mined,))
    assert counts == {"train": 3}
    rows = [json.loads(line) for line in (tmp_path / "sft" / "train.jsonl").read_text().splitlines()]
    first, teacher, model = rows
    assert [m["role"] for m in teacher["messages"]] == [
        "system", "user", "assistant", "user", "assistant"]
    assert json.loads(teacher["messages"][2]["content"]) == wrong
    assert teacher["messages"][3]["content"].startswith("The plan you wrote has these problems")
    assert teacher["messages"][4] == first["messages"][2]     # the verified plan
    assert teacher["meta"]["of"] == "demo" and teacher["meta"]["key"] == "demo#teacher-0"
    assert model["meta"]["key"] == "demo#model0-0"
    assert "the plan names no steps" in model["messages"][3]["content"]


def test_a_revision_starts_from_the_plan_and_what_running_it_shows(tmp_path, monkeypatch):
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    entry = dataset.Entry(name="demo", path=str(repo), split="train")
    dataset.label(entry, repo, Scripted(wire()), out)
    runs = iter([["Debian has no package named 'fftw'"], []])
    monkeypatch.setattr(dataset, "run_plan", lambda plan, root, env: SimpleNamespace())
    monkeypatch.setattr(dataset, "plan_faults", lambda result: next(runs))
    monkeypatch.setattr(dataset, "summary", lambda result: {"verdict": "yes"})
    teacher = Scripted(wire())
    record = dataset.revise(out / "demo", teacher, env=None)
    assert record["ok"] and record["problems"] == ["Debian has no package named 'fftw'"]
    roles = [m["role"] for m in teacher.asked[0]]
    assert roles == ["user", "assistant", "user"]
    assert "no package named 'fftw'" in teacher.asked[0][2]["content"]
    assert (out / "demo" / "revised-from" / "plan.json").is_file()
    assert json.loads((out / "demo" / "meta.json").read_text())["revised"] is True
    assert json.loads((out / "demo" / "run.json").read_text()) == {"verdict": "yes"}


def test_a_rerun_records_what_todays_executor_says_and_keeps_the_old_run(tmp_path, monkeypatch):
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    dataset.label(dataset.Entry(name="demo", path=str(repo), split="test"), repo,
                  Scripted(wire()), out)
    (out / "demo" / "run.json").write_text(json.dumps({"verdict": "no"}))
    monkeypatch.setattr(dataset, "run_plan", lambda plan, root, env: SimpleNamespace())
    monkeypatch.setattr(dataset, "summary", lambda result: {"verdict": "probably"})
    def env(distro):
        return SimpleNamespace(distro=SimpleNamespace(spec=SimpleNamespace(id=distro)))

    assert dataset.rerun(out / "demo", env("debian:trixie")) == ("no", "probably")
    assert json.loads((out / "demo" / "run.before.json").read_text()) == {"verdict": "no"}
    assert json.loads((out / "demo" / "run.json").read_text()) == {"verdict": "probably"}
    # Another distro's reference is its own file; the label's run stays.
    assert dataset.rerun(out / "demo", env("ubuntu:resolute")) == (None, "probably")
    assert (out / "demo" / "run.ubuntu-resolute.json").is_file()


def test_only_what_the_plan_got_wrong_goes_back_to_the_teacher():
    def outcome(step_id, status, detail, plan_check=None, kind="cmake-configure"):
        return SimpleNamespace(step=SimpleNamespace(id=step_id, kind=kind), status=status,
                               detail=detail, plan_check=plan_check)

    result = SimpleNamespace(evidence_problems=[], steps=[
        outcome("a", "failed", "no CMakeLists.txt in src"),
        outcome("b", "stopped", "3 rounds: MPI; stopped at CMake Error"),   # the build's own
        outcome("c", "failed", "reqs.txt: [Errno 2] No such file or directory",
                kind="python-install"),
        outcome("d", "done", "ran", ["the driver configured src -DX=ON, which no plan step does"]),
        # qemu's meson.build reads what its ./configure writes: the executor's gap.
        outcome("e", "failed", "Failed to load /tmp/x/build-1/config-host.mak: [Errno 2] No "
                "such file or directory", kind="meson-setup"),
    ])
    problems = dataset.plan_faults(result)
    assert len(problems) == 3 and not any("'b'" in p or "'e'" in p for p in problems)


# -- evaluation ------------------------------------------------------------------


def test_steps_match_as_far_as_they_agree():
    plan = parse_plan(modelplan.to_plan(wire()))
    other = parse_plan(modelplan.to_plan({**wire(), "steps": [
        {**wire()["steps"][0], "packages": ["libfftw3-dev", "cmake"]},
        {**wire()["steps"][1], "defines": []},
    ]}))
    assert evaluate.similarity(plan.steps[0], other.steps[0]) == pytest.approx(0.5)
    assert evaluate.similarity(plan.steps[1], other.steps[1]) == pytest.approx(0.5)
    assert evaluate.similarity(plan.steps[0], plan.steps[1]) == 0


def test_a_plan_scores_perfectly_against_itself_and_order_counts(tmp_path):
    repo = write(tmp_path, REPO)
    pack = evidence.build(repo, "demo")
    plan = parse_plan(modelplan.to_plan(wire()))
    same = evaluate.score(plan, plan, repo, pack)
    assert (same.precision, same.recall, same.order, same.optional, same.cited) == (1, 1, 1, 1, 1)
    backwards = parse_plan(modelplan.to_plan({**wire(), "steps": [
        {**wire()["steps"][1], "after": []}, {**wire()["steps"][0], "after": ["configure"]},
    ]}))
    assert evaluate.score(backwards, plan, repo, pack).order == 0
    assert not evaluate.score(None, plan, repo, pack).parsed


def test_the_harness_scores_baselines_on_held_out_repositories(tmp_path):
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    entry = dataset.Entry(name="demo", path=str(repo), split="test")
    dataset.label(entry, repo, Scripted(wire(quote="nope"), wire()), out)
    scores = evaluate.evaluate(out, {"autoplan": evaluate.autoplan_planner})
    rows = {(row["planner"], row["kind"]): row for row in evaluate.table(scores)}
    # One local, non-git repository: a row over all, and one for its kind.
    assert set(rows) == {(p, k) for p in ("autoplan", "teacher-first") for k in ("all", "pypi")}
    teacher = rows["teacher-first", "all"]
    assert teacher["recall"] == 1.0 and teacher["cited"] == 0.5


def test_the_harness_scores_only_the_keys_it_is_given(tmp_path):
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    entry = dataset.Entry(name="demo", path=str(repo), split="test")
    dataset.label(entry, repo, Scripted(wire()), out)
    planners = {"autoplan": evaluate.autoplan_planner}
    assert evaluate.evaluate(out, planners, keys=set()) == []
    (key,) = [d.name for d in out.iterdir()]
    assert {s.key for s in evaluate.evaluate(out, planners, keys={key})} == {key}


def test_the_harness_scores_the_same_in_parallel(tmp_path):
    out = tmp_path / "out"
    for name in ("one", "two", "three"):
        repo = write(tmp_path / name, REPO)
        dataset.label(dataset.Entry(name=name, path=str(repo), split="test"), repo,
                      Scripted(wire(quote="nope"), wire()), out)
    planners = {"autoplan": evaluate.autoplan_planner}
    keys = sorted(d.name for d in out.iterdir())[::-1]
    seen = []

    def plain(scores):
        return [{**asdict(s), "seconds": None} for s in scores]

    alone = evaluate.evaluate(out, planners)
    together = evaluate.evaluate(out, planners, keys=keys, jobs=2, progress=seen.append)
    assert plain(together) == plain(alone) and len(alone) == 6
    assert all(s.seconds is not None for s in together)
    assert sum("done" in line for line in seen) == 3


def test_shared_servers_send_a_request_to_whichever_is_free():
    used = []

    class Server:
        def __init__(self, name):
            self.name = name

        def complete(self, system, messages, schema):
            used.append(self.name)
            if len(used) == 1:
                # While this one is busy, the next request goes to the other.
                shared.complete(system, messages, schema)
            return modelplan.Completion(text=self.name)

    shared = modelplan.Shared([Server("a"), Server("b")])
    assert shared.complete("sys", [], {}).text == "a"
    assert used == ["a", "b"]
    assert shared.complete("sys", [], {}).text == "a"


def test_several_urls_make_a_shared_backend():
    one = modelplan.backend("openai", "m", "http://a/v1")
    two = modelplan.backend("openai", "m", "http://a/v1, http://b/v1", slots=2)
    assert isinstance(one, modelplan.OpenAICompatible)
    assert isinstance(two, modelplan.Shared) and two.name == "openai:m"
    assert [b.url for b in two.backends] == ["http://a/v1", "http://b/v1"]


def test_the_harness_keeps_what_the_model_wrote(tmp_path):
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    entry = dataset.Entry(name="demo", path=str(repo), split="test")
    dataset.label(entry, repo, Scripted(wire()), out)
    kept = tmp_path / "transcripts"
    planner = evaluate.model_planner(Scripted(wire(quote="nope"), wire()), 3, kept)
    scores = evaluate.evaluate(out, {"model": planner})
    assert [s.parsed for s in scores if s.planner == "model"] == [True]
    (saved,) = kept.iterdir()
    attempts = json.loads(saved.read_text())["attempts"]
    assert len(attempts) == 2 and attempts[0]["problems"] and not attempts[1]["problems"]


def test_the_harness_counts_what_every_example_cost(tmp_path, monkeypatch):
    """Two attempts at 10 in, 5 out and a cent each; and an unblocker that
    spent on its own during the run, counted with the example it ran for."""
    from will_it_riscv import unblock

    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    dataset.label(dataset.Entry(name="demo", path=str(repo), split="test"), repo,
                  Scripted(wire()), out)

    def planner(root, pack, meta):
        unblock.SPENT.append({"cost": 0.5, "tokens_in": 1000, "tokens_out": 200})
        unblock.SPENT.append({"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "replayed": "x"})
        return evaluate.model_planner(Scripted(wire(quote="nope"), wire()), 3)(root, pack, meta)

    unblock.SPENT.append({"cost": 9.0})   # an earlier example's, not this one's
    (score,) = evaluate.evaluate(out, {"model": planner}, baselines=False)
    assert (score.planner_tokens_in, score.planner_tokens_out) == (20, 10)
    assert score.planner_usd == pytest.approx(0.02)
    assert (score.unblock_sessions, score.unblock_replays) == (1, 1)
    assert score.tokens == 20 + 10 + 1000 + 200 and score.usd == pytest.approx(0.52)
    (row, _) = evaluate.table([score])
    assert row["ktokens"] == 1.23 and row["usd"] == 0.52 and row["sessions"] == 1.0


def test_the_command_line_runs_the_plan_a_model_wrote(tmp_path, monkeypatch):
    from will_it_riscv import cli
    from will_it_riscv.pseudobuild import available

    if not available():
        pytest.skip("cmake is not installed")
    repo = write(tmp_path / "repo", REPO)
    monkeypatch.setattr(modelplan, "backend", lambda *a: Scripted(wire(quote="nope"), wire()))
    saved, report = tmp_path / "plan.json", tmp_path / "report.json"
    cli.main([str(repo), "--plan-from-model", "--save-plan", str(saved), "--no-distro",
              "-f", "json", "-o", str(report), "--exit-zero"])
    assert load_plan(saved).step("configure").defines == {"WITH_FFTW": "ON"}
    steps = {s["id"]: s for s in json.loads(report.read_text())["steps"]}
    assert steps["configure"]["status"] == "completed"


def test_the_command_line_sends_back_what_running_the_plan_showed(tmp_path, monkeypatch):
    """As the eval does: run each plan that holds, return its faults, and
    report the run the last plan was taken on rather than running it again."""
    from will_it_riscv import cli, planrun
    from will_it_riscv.pseudobuild import available

    if not available():
        pytest.skip("cmake is not installed")
    repo = write(tmp_path / "repo", REPO)
    elsewhere = wire()
    elsewhere["steps"][1]["source"] = "src"   # there is no CMakeLists.txt there
    planner = Scripted(elsewhere, wire())
    monkeypatch.setattr(modelplan, "backend", lambda *a: planner)
    runs = []
    execute = planrun.execute
    monkeypatch.setattr(planrun, "execute", lambda plan, *a, **k: runs.append(plan) or execute(plan, *a, **k))
    report = tmp_path / "report.json"
    cli.main([str(repo), "--plan-from-model", "--no-distro", "-f", "json", "-o", str(report),
              "--exit-zero"])
    assert "no CMakeLists.txt in src" in planner.asked[1][-1]["content"]
    assert [p.step("configure").source for p in runs] == ["src", "."]
    steps = {s["id"]: s for s in json.loads(report.read_text())["steps"]}
    assert steps["configure"]["status"] == "completed"


def test_a_gguf_planner_is_served_for_the_run(tmp_path, monkeypatch):
    import argparse
    import contextlib

    from rich.console import Console

    from will_it_riscv import cli, serve

    served = []

    @contextlib.contextmanager
    def fake(model):
        served.append(("up", model))
        yield "http://127.0.0.1:9/v1"
        served.append(("down", model))

    monkeypatch.setattr(serve, "llama_server", fake)
    args = argparse.Namespace(planner=None, planner_model="/m/planner-318.gguf", planner_url=None)
    with contextlib.ExitStack() as stack:
        planner = cli._planner(args, Console(quiet=True), stack)
        assert isinstance(planner, modelplan.OpenAICompatible)
        assert (planner.url, planner.model) == ("http://127.0.0.1:9/v1", "planner-318")
        assert served == [("up", Path("/m/planner-318.gguf"))]
    assert served[-1] == ("down", Path("/m/planner-318.gguf"))
    with pytest.raises(SystemExit, match="needs --planner-model"):
        cli._planner(argparse.Namespace(planner="llama-server", planner_model=None,
                                        planner_url=None), Console(quiet=True),
                     contextlib.ExitStack())


FAKE_LLAMA_SERVER = '''
import http.server, sys
args = sys.argv[1:]
if "--fail" in args:
    print("error: failed to load model"); sys.exit(1)
port = int(args[args.index("--port") + 1])
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers()
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
'''


def test_llama_server_is_started_waited_for_and_stopped(tmp_path, monkeypatch):
    import sys

    import httpx

    from will_it_riscv import serve

    script = tmp_path / "llama-server"
    script.write_text(f"#!{sys.executable}\n{FAKE_LLAMA_SERVER}")
    script.chmod(0o755)
    model = tmp_path / "planner.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setenv("LLAMA_SERVER", str(script))
    monkeypatch.setenv("LLAMA_SERVER_ARGS", "-ngl 0")
    command = serve.command(model, 1234)
    assert command[0] == str(script) and command[-2:] == ["-ngl", "0"]
    assert command[command.index("-a") + 1] == "planner"
    with serve.llama_server(model) as url:
        assert httpx.get(url.removesuffix("/v1") + "/health").status_code == 200
    with pytest.raises(httpx.HTTPError):
        httpx.get(url.removesuffix("/v1") + "/health", timeout=2)
    monkeypatch.setenv("LLAMA_SERVER_ARGS", "--fail")
    with pytest.raises(RuntimeError, match="failed to load model"):
        with serve.llama_server(model):
            pass
    with pytest.raises(RuntimeError, match="no such model file"):
        with serve.llama_server(tmp_path / "missing.gguf"):
            pass


def test_the_pack_can_be_shown_without_a_model(tmp_path, capsys):
    from will_it_riscv import cli

    repo = write(tmp_path / "repo", REPO)
    assert cli.main([str(repo), "--show-pack"]) == 0
    assert capsys.readouterr().out.startswith("REPOSITORY repo")


def test_a_run_stops_when_the_teacher_stops_answering_and_picks_up_again(tmp_path):
    """A usage limit mid-run: nothing it failed on is recorded as done."""

    class Limited:
        name = "limited"

        def complete(self, system, messages, schema):
            raise RuntimeError("usage limit reached")

    entries = []
    for number in range(4):
        repo = write(tmp_path / f"repo{number}", REPO)
        entries.append(dataset.Entry(name=f"demo{number}", path=str(repo)))
    out = tmp_path / "out"
    said = []
    first = dataset.label_corpus(entries, out, Limited(), execute=False, progress=said.append)
    assert [m["tier"] for m in first] == [dataset.TEACHER_ERROR] * 2   # then it stopped
    assert any(line.startswith("stopping:") for line in said)
    assert not (out / "demo2").exists()

    class Answers:
        name = "answers"

        def complete(self, system, messages, schema):
            return modelplan.Completion(json.dumps(wire()), 1, 1, 0.01)

    again = dataset.label_corpus(entries, out, Answers(), execute=False, progress=said.append)
    assert [m["tier"] for m in again] == [dataset.VERIFIED] * 4


def test_the_harness_rescores_what_a_model_wrote_without_asking_it_again(tmp_path):
    repo = write(tmp_path / "repo", REPO)
    out = tmp_path / "out"
    dataset.label(dataset.Entry(name="demo", path=str(repo), split="test"), repo,
                  Scripted(wire()), out)
    kept = tmp_path / "transcripts"
    evaluate.evaluate(out, {"model": evaluate.model_planner(Scripted(wire()), 1, kept)},
                      baselines=False)
    (saved,) = kept.iterdir()
    # The plan kept is the last answer that parsed, as plan_repository keeps it.
    record = json.loads(saved.read_text())
    record["attempts"].append({"text": "not a plan"})
    saved.write_text(json.dumps(record))
    (score,) = evaluate.evaluate(out, {"saved": evaluate.saved_planner([kept])},
                                 baselines=False)
    assert score.parsed and score.precision == 1.0 and score.tokens == 0
    # A later directory's answer for the key wins: here, none that parses.
    later = tmp_path / "later"
    later.mkdir()
    (later / saved.name).write_text(json.dumps({"attempts": []}))
    (score,) = evaluate.evaluate(out, {"saved": evaluate.saved_planner([kept, later])},
                                 baselines=False)
    assert not score.parsed


def test_required_is_scored_on_what_the_runs_prove():
    """An apt line the reference copied from CI is no longer the candidate's
    loss; a reference that proves nothing scores nobody."""
    reference = {"required": ["debian:ccache", "debian:libboost-dev", "library:boost"],
                 "proven": ["library:boost"]}
    ran = {"required": ["debian:cmake", "library:boost"], "proven": ["library:boost"]}
    assert evaluate.required_overlap(ran, reference) == (1.0, 0.25)
    assert evaluate.required_overlap({"required": [], "proven": []}, reference)[0] == 0.0
    assert evaluate.required_overlap(ran, {"required": ["debian:gcc"], "proven": []})[0] is None
    assert evaluate.required_overlap(ran, {"required": ["debian:gcc"]})[0] is None
    # amqp-cpp's configure ran to the end and needed nothing: that is the answer.
    nothing = {"required": ["debian:gcc"], "proven": []}
    assert evaluate.required_overlap(ran, nothing, shown=True)[0] == 0.0
    assert evaluate.required_overlap({"required": [], "proven": []}, nothing, shown=True)[0] == 1.0


def test_a_reference_shows_nothing_needed_only_when_its_configures_completed():
    from will_it_riscv.plan import parse_plan

    reference = parse_plan({"repo": "r", "steps": [
        {"id": "apt", "kind": "system-packages", "packages": ["gcc"], "evidence": []},
        {"id": "cfg", "kind": "cmake-configure", "evidence": []},
        {"id": "gpu", "kind": "cmake-configure", "optional": True, "enabled_by": "-DGPU=ON",
         "evidence": []}]})

    def run(cfg, gpu="stopped"):
        return {"steps": [{"id": "apt", "status": "done"}, {"id": "cfg", "status": cfg},
                          {"id": "gpu", "status": gpu}]}

    assert evaluate.configured(reference, run("completed"))
    assert not evaluate.configured(reference, run("stopped", "completed"))
    apt_only = parse_plan({"repo": "r", "steps": [
        {"id": "apt", "kind": "system-packages", "packages": ["gcc"], "evidence": []}]})
    assert not evaluate.configured(apt_only, run("completed"))
