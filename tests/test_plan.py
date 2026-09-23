"""Build plans: reading them, checking their evidence, and running them."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeIndex, metadata

from will_it_riscv import cli, planrun
from will_it_riscv.plan import (
    _PLAN_FIELDS,
    _STEP_FIELDS,
    PLAN_SCHEMA,
    STEP_KINDS,
    PlanError,
    check_evidence,
    load_plan,
    parse_plan,
)
from will_it_riscv.pseudobuild import available

needs_cmake = pytest.mark.skipif(not available(), reason="cmake is not installed")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "plans" / "mfc.json"


class FakeDistro:
    def __init__(self, names):
        self.names_ = set(names)
        self.spec = SimpleNamespace(label="Debian 13 (trixie)", id="debian:trixie")
        self.arch = "riscv64"
        self.available = True

    def first_available(self, candidates):
        return next((c for c in candidates if c in self.names_), None)

    def has(self, package):
        return package in self.names_

    def python_package(self, name):
        return None


def plan(*steps, **extra):
    return {"version": 1, "repo": "demo", "steps": list(steps), **extra}


# -- reading ----------------------------------------------------------------


def test_a_plan_runs_its_steps_in_dependency_order():
    parsed = parse_plan(plan(
        {"id": "b", "kind": "cmake-configure", "after": ["a"], "evidence": []},
        {"id": "a", "kind": "python-install", "manifest": "pyproject.toml", "evidence": []},
    ))
    assert [s.id for s in parsed.ordered()] == ["a", "b"]


def test_every_problem_is_reported_not_just_the_first():
    with pytest.raises(PlanError) as caught:
        parse_plan(plan(
            {"id": "x", "kind": "make-coffee", "evidence": []},
            {"id": "y", "kind": "python-install", "evidence": ["README.md:3"]},
            {"id": "y", "kind": "system-packages", "packages": ["gcc"], "evidence": ["nope"]},
            {"id": "z", "kind": "cmake-configure", "after": ["ghost"], "colour": "red",
             "evidence": []},
            flavour="vanilla",
        ))
    text = " | ".join(caught.value.problems)
    assert "unknown field 'flavour'" in text
    assert "'kind' must be one of" in text
    assert "names its 'manifest'" in text
    assert "evidence 'nope'" in text
    assert "used twice" in text
    assert "unknown step 'ghost'" in text
    assert "unknown field 'colour'" in text


def test_a_cycle_between_steps_is_refused():
    with pytest.raises(PlanError, match="cycle"):
        parse_plan(plan(
            {"id": "a", "kind": "cmake-configure", "after": ["b"], "evidence": []},
            {"id": "b", "kind": "cmake-configure", "after": ["a"], "evidence": []},
        ))


def test_booleans_become_cmake_switches():
    parsed = parse_plan(plan({
        "id": "cfg", "kind": "cmake-configure", "evidence": [],
        "defines": {"MFC_MPI": True, "MFC_GCov": False, "JOBS": 8},
    }))
    assert parsed.steps[0].defines == {"MFC_MPI": "ON", "MFC_GCov": "OFF", "JOBS": "8"}


def test_the_schema_and_the_parser_agree():
    """The schema constrains whatever writes a plan; the parser checks it."""
    assert set(PLAN_SCHEMA["properties"]) == _PLAN_FIELDS
    step = PLAN_SCHEMA["properties"]["steps"]["items"]
    assert set(step["properties"]) == _STEP_FIELDS
    assert step["properties"]["kind"]["enum"] == list(STEP_KINDS)


def test_the_mfc_example_is_a_valid_plan():
    parsed = load_plan(EXAMPLE)
    assert parsed.repo == "MFC"
    assert parsed.step("post_process").after[-1] == "dep-lapack"


# -- evidence ---------------------------------------------------------------


def test_evidence_that_holds_is_silent(tmp_path):
    (tmp_path / "build.sh").write_text("#!/bin/sh\npip install ./tools\ncmake -S . -B b\n")
    parsed = parse_plan(plan({
        "id": "tools", "kind": "python-install", "manifest": "tools/pyproject.toml",
        "evidence": [{"at": "build.sh:2", "quote": "pip   install ./tools"}, "build.sh:2-3"],
    }))
    assert check_evidence(parsed, tmp_path) == []


def test_evidence_that_does_not_hold_says_why(tmp_path):
    (tmp_path / "build.sh").write_text("#!/bin/sh\npip install ./tools\n")
    parsed = parse_plan(plan({
        "id": "tools", "kind": "python-install", "manifest": "tools/pyproject.toml",
        "evidence": [
            {"at": "build.sh:2", "quote": "pip install ./other"},
            "build.sh:9",
            "missing.sh:1",
            "../outside.sh:1",
        ],
    }))
    problems = check_evidence(parsed, tmp_path)
    assert any("does not say 'pip install ./other'" in p for p in problems)
    assert any("past the end" in p for p in problems)
    assert any("missing.sh does not exist" in p for p in problems)
    assert any("outside the repository" in p for p in problems)


# -- running ----------------------------------------------------------------


def index_for(**projects):
    """A fake index: name -> (filenames, requires)."""
    files, meta = {}, {}
    for name, (filenames, requires) in projects.items():
        files[name] = filenames
        for filename in filenames:
            if filename.endswith(".whl"):
                meta[filename] = metadata(*requires, name=name, version="1.0")
    return FakeIndex(files, meta)


def run_plan(tmp_path, target, raw, index=None, distro=None):
    return planrun.execute(
        parse_plan(raw), tmp_path, index=index or FakeIndex({}), target=target,
        distro=distro, timeout=120,
    )


def test_system_packages_are_checked_against_the_archive(tmp_path, target):
    result = run_plan(tmp_path, target, plan({
        "id": "apt", "kind": "system-packages", "packages": ["gcc", "libnope-dev"],
        "evidence": [],
    }), distro=FakeDistro({"gcc"}))
    assert result.nodes["debian:gcc"].tier == planrun.BINARY
    assert result.nodes["debian:libnope-dev"].tier == planrun.NONE
    assert result.answer.verdict == "no"
    assert result.why("debian:libnope-dev") == ["step:apt", "debian:libnope-dev"]


def test_a_python_package_with_nothing_for_the_target_blocks_the_plan(tmp_path, target):
    """MFC's shape: pure-Python jax pulls in jaxlib, which has no riscv64 wheel."""
    (tmp_path / "toolchain").mkdir()
    (tmp_path / "toolchain" / "pyproject.toml").write_text(
        '[project]\nname = "t"\nversion = "1"\ndependencies = ["jax"]\n'
    )
    index = index_for(
        jax=(["jax-1.0-py3-none-any.whl"], ["jaxlib"]),
        jaxlib=(["jaxlib-1.0-cp312-cp312-manylinux_2_27_x86_64.whl"], []),
    )
    result = run_plan(tmp_path, target, plan({
        "id": "toolchain", "kind": "python-install", "manifest": "toolchain/pyproject.toml",
        "evidence": [],
    }), index=index)
    assert result.answer.verdict == "no"
    assert result.answer.blockers == ["pypi:jaxlib"]
    assert result.why("pypi:jaxlib") == ["step:toolchain", "pypi:jax", "pypi:jaxlib"]


FIND_THING = (
    "include(FindPackageHandleStandardArgs)\n"
    "find_library(THING_LIBRARY NAMES thing)\n"
    "find_package_handle_standard_args(Thing REQUIRED_VARS THING_LIBRARY)\n"
)


@needs_cmake
def test_what_an_earlier_step_provides_is_not_looked_for_in_the_archive(tmp_path, target):
    """MFC's CMake wants fypp, which its Python toolchain installs, and FFTW,
    which its own dependency target builds from source."""
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindThing.cmake").write_text(FIND_THING)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_program(FYPP_EXE NAMES fypp REQUIRED)\n"
        "find_package(Thing REQUIRED)\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "t"\nversion = "1"\ndependencies = ["fypp"]\n'
    )
    raw = plan(
        {"id": "py", "kind": "python-install", "manifest": "pyproject.toml", "evidence": []},
        {"id": "deps", "kind": "system-packages", "packages": ["cmake"], "provides": ["thing"],
         "evidence": []},
        {"id": "cfg", "kind": "cmake-configure", "after": ["py", "deps"], "evidence": [],
         "defines": {"DEMO_OPTION": True}},
    )
    index = index_for(fypp=(["fypp-1.0-py3-none-any.whl"], []))
    result = run_plan(tmp_path, target, raw, index=index, distro=FakeDistro({"cmake"}))
    cfg = next(s for s in result.steps if s.step.id == "cfg")
    assert cfg.status == "completed", cfg.detail
    assert [cfg.graph.nodes[k].name for k in cfg.graph.order] == ["fypp", "Thing"]
    thing = next(n for n in result.nodes.values() if n.name == "Thing")
    fypp = next(n for n in result.nodes.values() if n.name == "fypp" and n.ecosystem != "pypi")
    assert (thing.tier, thing.provided_by) == (planrun.PROVIDED, "deps")
    assert (fypp.tier, fypp.provided_by) == (planrun.PROVIDED, "py")
    assert result.answer.verdict == "yes"
    assert '"step:deps" -> ' in planrun.to_dot(result)


def test_what_the_plan_builds_itself_is_not_installed(tmp_path, target):
    """SILO is in Debian, but MFC builds its own: it stays off the apt line."""
    result = planrun.PlanResult(plan=parse_plan(plan(
        {"id": "a", "kind": "system-packages", "packages": ["x"], "evidence": []}
    )), root=tmp_path)
    node = planrun._node(result, "library:silo", "library", "SILO")
    node.required = True
    planrun._settle(node, planrun.BINARY, "libsilo-dev ✓")
    node.package = "libsilo-dev"
    planrun._settle(node, planrun.PROVIDED, "provided by step 'dep-silo'", "dep-silo")
    assert (node.tier, node.package) == (planrun.PROVIDED, None)
    assert planrun._answer(result).install == []


def test_a_plan_result_serialises(tmp_path, target):
    result = run_plan(tmp_path, target, plan({
        "id": "apt", "kind": "system-packages", "packages": ["gcc"], "evidence": [],
    }), distro=FakeDistro({"gcc"}))
    data = planrun.to_dict(result)
    assert data["answer"]["verdict"] == "yes"
    assert data["answer"]["install"] == ["gcc"]
    assert {"from": "step:apt", "to": "debian:gcc"} in data["edges"]
    json.dumps(data)


def test_a_plan_that_cannot_be_run_is_reported_and_refused(tmp_path, capsys):
    bad = tmp_path / "plan.json"
    bad.write_text(json.dumps({"repo": "x", "steps": [{"id": "a", "kind": "nope"}]}))
    assert cli.main([str(tmp_path), "--plan", str(bad), "--no-distro"]) == 2
    assert "'kind' must be one of" in capsys.readouterr().err
