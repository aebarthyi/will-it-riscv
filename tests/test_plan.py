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
        self.versions_ = dict(names) if isinstance(names, dict) else {}
        self.spec = SimpleNamespace(label="Debian 13 (trixie)", id="debian:trixie")
        self.arch = "riscv64"
        self.available = True

    def first_available(self, candidates):
        return next((c for c in candidates if c in self.names_), None)

    def has(self, package):
        return package in self.names_

    def version(self, package):
        return self.versions_.get(package)

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


def test_a_tool_the_archive_has_too_old_is_source_only_when_upstream_ships_none(
    tmp_path, target
):
    """jax pins Bazel 8.7.0; Debian 13's bazel-bootstrap is 4.2.3, and Bazel
    publishes no riscv64 binaries -- it has to be bootstrapped from source."""
    result = run_plan(tmp_path, target, plan({
        "id": "tools", "kind": "system-packages", "packages": ["bazel-bootstrap>=8.7.0"],
        "evidence": [],
    }), distro=FakeDistro({"bazel-bootstrap": "4.2.3+ds-11"}))
    node = result.nodes["debian:bazel-bootstrap"]
    assert node.tier == planrun.SOURCE
    assert "has 4.2.3+ds-11; the build wants >= 8.7.0" in node.detail
    assert "bootstrapped from its source" in node.detail
    assert result.answer.verdict == "yes-after-source-builds"


def test_a_tool_the_archive_has_too_old_is_fine_when_upstream_ships_it(tmp_path, target):
    """Debian 13's rustc is 1.85; orjson wants 1.95; rustup ships riscv64 toolchains."""
    result = run_plan(tmp_path, target, plan({
        "id": "tools", "kind": "system-packages", "packages": ["rustc>=1.95", "cargo"],
        "evidence": [],
    }), distro=FakeDistro({"rustc": "1.85.1+dfsg1-1", "cargo": "1.85.1+dfsg1-1"}))
    rustc = result.nodes["debian:rustc"]
    assert rustc.tier == planrun.BINARY
    assert "rustup ships riscv64gc-unknown-linux-gnu" in rustc.detail
    assert result.answer.install == ["cargo"]   # rustup is not apt


def test_a_package_the_archive_has_too_old_and_nobody_else_ships_is_missing(tmp_path, target):
    result = run_plan(tmp_path, target, plan({
        "id": "tools", "kind": "system-packages", "packages": ["frobnicate>=9"],
        "evidence": [],
    }), distro=FakeDistro({"frobnicate": "2.0-1"}))
    assert result.nodes["debian:frobnicate"].tier == planrun.NONE
    assert result.answer.verdict == "no"


def test_a_package_at_a_new_enough_version_is_fine(tmp_path, target):
    result = run_plan(tmp_path, target, plan({
        "id": "tools", "kind": "system-packages", "packages": ["rustc>=1.80"],
        "evidence": [],
    }), distro=FakeDistro({"rustc": "1.85.0+dfsg1-1"}))
    assert result.nodes["debian:rustc"].tier == planrun.BINARY


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


# -- the default build is the minimal spec -----------------------------------


def test_an_optional_step_says_what_turns_it_on():
    with pytest.raises(PlanError, match="says what turns it on"):
        parse_plan(plan({"id": "gpu", "kind": "cmake-configure", "optional": True,
                         "evidence": []}))
    with pytest.raises(PlanError, match="only means something on an optional step"):
        parse_plan(plan({"id": "cfg", "kind": "cmake-configure", "enabled_by": "--gpu",
                         "evidence": []}))


def test_the_default_build_cannot_wait_on_an_optional_step():
    with pytest.raises(PlanError, match="comes after optional step 'gpu'"):
        parse_plan(plan(
            {"id": "gpu", "kind": "cmake-configure", "optional": True, "enabled_by": "--gpu",
             "evidence": []},
            {"id": "cfg", "kind": "cmake-configure", "after": ["gpu"], "evidence": []},
        ))


def test_what_only_an_optional_step_needs_is_optional(tmp_path, target):
    """The GPU toolchain is not in the archive; the default build does not need it."""
    result = run_plan(tmp_path, target, plan(
        {"id": "apt", "kind": "system-packages", "packages": ["gcc"], "evidence": []},
        {"id": "gpu", "kind": "system-packages", "packages": ["nvhpc", "gcc"],
         "optional": True, "enabled_by": "./build.sh --gpu", "evidence": []},
    ), distro=FakeDistro({"gcc"}))
    assert result.nodes["debian:gcc"].required
    nvhpc = result.nodes["debian:nvhpc"]
    assert not nvhpc.required and nvhpc.optional_via == ["./build.sh --gpu"]
    assert result.answer.verdict == "yes"
    assert result.answer.optional == {"./build.sh --gpu": ["debian:nvhpc"]}


@needs_cmake
def test_an_optional_step_that_stops_does_not_decide_the_answer(tmp_path, target):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(demo NONE)\n"
        'if(DEMO_GPU)\n  message(FATAL_ERROR "GPU builds need a vendor compiler")\nendif()\n'
    )
    result = run_plan(tmp_path, target, plan(
        {"id": "cpu", "kind": "cmake-configure", "evidence": []},
        {"id": "gpu", "kind": "cmake-configure", "defines": {"DEMO_GPU": True},
         "optional": True, "enabled_by": "--gpu", "evidence": []},
    ))
    assert result.answer.verdict == "yes"
    assert result.answer.optional_stopped == ["gpu"]


def test_a_python_install_can_name_its_packages_instead_of_a_file():
    """jaxlib's Bazel build takes numpy, scipy and ml_dtypes wheels from dist/."""
    plan = parse_plan({"repo": "r", "steps": [{
        "id": "wheels", "kind": "python-install", "packages": ["numpy==2.1.3"], "evidence": [],
    }]})
    assert plan.steps[0].packages == ["numpy==2.1.3"] and plan.steps[0].manifest is None
    with pytest.raises(PlanError, match="'manifest' or 'packages'"):
        parse_plan({"repo": "r", "steps": [
            {"id": "w", "kind": "python-install", "evidence": []},
        ]})


def test_a_meson_setup_names_the_meson_it_ships():
    plan = parse_plan({"repo": "r", "steps": [{
        "id": "setup", "kind": "meson-setup", "meson": "vendored-meson/meson/meson.py",
        "defines": {"blas": "openblas"}, "evidence": [],
    }]})
    assert plan.steps[0].meson == "vendored-meson/meson/meson.py"
    assert plan.steps[0].defines == {"blas": "openblas"}


def test_cmake_names_are_as_loose_as_cmake_is():
    """LAMMPS documents -D PKG_ML-PACE=on; pyproject cmake.args write -DX:BOOL=ON."""
    plan = parse_plan({"repo": "r", "steps": [{
        "id": "c", "kind": "cmake-configure", "evidence": [],
        "defines": {"PKG_ML-PACE": "on", "SUNDIALS_ENABLE_PYTHON:BOOL": "ON"},
    }]})
    assert plan.steps[0].defines == {"PKG_ML-PACE": "on", "SUNDIALS_ENABLE_PYTHON": "ON"}
    with pytest.raises(PlanError, match="not a CMake variable name"):
        parse_plan({"repo": "r", "steps": [{
            "id": "c", "kind": "cmake-configure", "evidence": [], "defines": {"-bad": "1"},
        }]})
