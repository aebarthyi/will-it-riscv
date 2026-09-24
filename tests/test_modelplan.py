"""Evidence packs, the model contract, the dataset and its evaluation."""

import json
import subprocess
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


def test_a_plan_that_holds_is_taken_first_time(tmp_path):
    write(tmp_path, REPO)
    planner = Scripted(wire())
    outcome = modelplan.plan_repository(tmp_path, planner, name="demo")
    assert outcome.ok and len(outcome.attempts) == 1
    assert outcome.plan.step("configure").defines == {"WITH_FFTW": "ON"}
    assert outcome.cost_usd == pytest.approx(0.01)


def test_what_does_not_hold_goes_back_and_is_fixed(tmp_path):
    write(tmp_path, REPO)
    planner = Scripted(wire(quote="cmake --build"), wire(at="lint.sh:9"), wire())
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


def test_only_what_the_plan_got_wrong_goes_back_to_the_teacher():
    def outcome(step_id, status, detail, plan_check=None):
        return SimpleNamespace(step=SimpleNamespace(id=step_id), status=status, detail=detail,
                               plan_check=plan_check)

    result = SimpleNamespace(evidence_problems=[], steps=[
        outcome("a", "failed", "no CMakeLists.txt in src"),
        outcome("b", "stopped", "3 rounds: MPI; stopped at CMake Error"),   # the build's own
        outcome("c", "failed", "reqs.txt: [Errno 2] No such file or directory"),
        outcome("d", "done", "ran", ["the driver configured src -DX=ON, which no plan step does"]),
    ])
    problems = dataset.plan_faults(result)
    assert len(problems) == 3 and not any("'b'" in p for p in problems)


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
    rows = {row["planner"]: row for row in evaluate.table(scores)}
    assert set(rows) == {"autoplan", "teacher-first"}
    assert rows["teacher-first"]["recall"] == 1.0 and rows["teacher-first"]["cited"] == 0.5


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
