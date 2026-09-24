"""Meson setups: cross for linux/riscv64, confined, and unblocked as they go."""

import shutil

import pytest

from will_it_riscv import graph as depgraph
from will_it_riscv.pseudomeson import _calls, _read, run

needs_meson = pytest.mark.skipif(
    shutil.which("meson") is None or shutil.which("ninja") is None,
    reason="meson and ninja are not installed",
)
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")


def project(tmp_path, body, languages="'c'"):
    (tmp_path / "meson.build").write_text(f"project('demo', {languages})\n{body}")
    return tmp_path


# -- reading what Meson said -------------------------------------------------


def test_what_meson_found_and_missed_is_read(tmp_path):
    outcome = _read(
        "Run-time dependency python found: YES 3.14\n"
        "Run-time dependency openblas found: NO  (tried pkg-config and cmake)\n"
        "Library m found: YES\n"
        "Program cython found: YES (/usr/bin/cython)\n"
        "Dependency c-siphash from subproject subprojects/c-siphash found: YES 1\n"
        "src/meson.build:12:4: ERROR: Dependency \"openblas\" not found, tried pkg-config\n",
        tmp_path / "src",
    )
    assert set(outcome.found) == {"python", "m"}
    assert outcome.programs == {"cython": "/usr/bin/cython"}
    assert outcome.missing == ["openblas"]
    assert outcome.libraries == {"m"}
    assert outcome.subprojects == {"c-siphash": "subprojects/c-siphash"}
    assert _read("Executing subproject qhull_r method meson\n", tmp_path).subprojects == {
        "qhull_r": "subprojects/qhull_r"
    }
    assert outcome.where == "meson.build:12"
    assert outcome.error.startswith('Dependency "openblas" not found')


def test_call_sites_are_read_from_the_files_meson_reads(tmp_path):
    """subdir() is followed; a vendored Meson's test projects are not."""
    project(tmp_path, "dependency('a')\nsubdir('lib')\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "meson.build").write_text(
        "dependency('b',\n  required: false)\ndependency('c', required: get_option('c'))\n"
    )
    (tmp_path / "vendored" / "test cases").mkdir(parents=True)
    (tmp_path / "vendored" / "test cases" / "meson.build").write_text("dependency('d')\n")
    calls = _calls(tmp_path)
    assert calls["a"] == [("meson.build:2", True)]
    assert calls["b"] == [("lib/meson.build:1", False)]
    assert calls["c"] == [("lib/meson.build:3", None)]
    assert "d" not in calls


# -- against a real meson ----------------------------------------------------


@needs_meson
def test_it_is_set_up_as_linux_on_the_target(tmp_path):
    project(tmp_path, (
        "if host_machine.system() != 'linux' or host_machine.cpu_family() != 'riscv64'\n"
        "  error('set up for the host')\nendif\n"
    ), languages="[]")
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.platform == "linux/riscv64"


@needs_meson
def test_a_required_dependency_is_stubbed_and_an_optional_one_proven_optional(tmp_path):
    project(tmp_path, (
        "dependency('wirfoo', version: '>=2.1')\n"
        "dependency('wirbar', required: false)\n"
    ), languages="[]")
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["wirfoo"]
    assert result.probes["wirfoo"].site == "meson.build:2"
    assert result.soft_misses == {"wirbar"}
    graph = depgraph.build(result, "demo")
    assert graph.nodes["wirbar"].status == depgraph.OPTIONAL


@needs_meson
@needs_cc
def test_a_library_is_stubbed_with_an_archive_the_host_linker_takes(tmp_path):
    project(tmp_path, (
        "cc = meson.get_compiler('c')\n"
        "lib = cc.find_library('wirlib')\n"
        "if not cc.links('int main(void) { return 0; }', dependencies: lib)\n"
        "  error('the stub does not link')\nendif\n"
    ))
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["wirlib"]


@needs_meson
def test_a_program_gets_a_stub_that_answers_its_version(tmp_path):
    project(tmp_path, "find_program('wirprog', version: '>=2')\n", languages="[]")
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["wirprog"]


@needs_meson
def test_what_an_error_does_not_name_is_found_by_experiment(tmp_path):
    project(tmp_path, (
        "dep = dependency('wirz', required: false)\n"
        "dependency('wirother', required: false)\n"
        "if not dep.found()\n  error('this build cannot go on')\nendif\n"
    ), languages="[]")
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["wirz"]
    assert ("wirother", False) in result.experiments   # nearest first, and innocent
    assert ("wirz", True) in result.experiments
    assert result.soft_misses == {"wirother"}


@needs_meson
def test_the_extension_modules_python_is_the_build_hosts(tmp_path):
    """numpy asks Meson's python module for the interpreter and its headers."""
    project(tmp_path, (
        "py = import('python').find_installation(pure: false)\n"
        "py.dependency()\n"
        "r = run_command(py, '-c', 'import wirtestmod', check: true)\n"
    ))
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert "python" in result.found
    assert result.probes["python"].required
    assert result.python_stubbed == ["wirtestmod (no plan says which version)"]


@needs_meson
def test_nothing_it_did_not_ask_for_is_written_to_the_source_tree(tmp_path):
    project(tmp_path, "dependency('wirfoo')\n", languages="[]")
    before = sorted(p.name for p in tmp_path.iterdir())
    run(tmp_path, timeout=120)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@needs_meson
def test_an_import_a_script_meson_runs_could_not_make_is_given(tmp_path):
    """pandas' version comes from generate_version.py, which imports versioneer."""
    (tmp_path / "gen.py").write_text("#!/usr/bin/env python3\nimport wirvers\nprint('1.0')\n")
    (tmp_path / "meson.build").write_text(
        "project('demo', [], version: run_command(['gen.py'], check: true).stdout().strip())\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.round_blockers[0] == "import wirvers"
    assert result.python_stubbed == ["wirvers (no plan says which version)"]


@needs_meson
def test_a_program_a_build_requirement_provides_is_installed_and_said_so(tmp_path, monkeypatch):
    """Cython, for numpy: installed for the host, and not something the target lacks."""
    from will_it_riscv import hostpython

    def install(spec, site, cache, wheels_only=False):
        script = site / "bin" / "wirtool"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/sh\necho 'wirtool 3.0'\n")
        script.chmod(0o755)
        return True

    monkeypatch.setattr(hostpython, "_pip_install", install)
    project(tmp_path, "find_program('wirtool', version: '>=2')\n", languages="[]")
    result = run(tmp_path, timeout=120, python_dists={"wirtool": "3.0"})
    assert result.completed, result.error
    assert result.python_installed == ["wirtool==3.0"]
    assert result.build_tools == ["wirtool"]
    graph = depgraph.build(result, "demo")
    assert graph.nodes["wirtool"].provided_by == depgraph.BUILD_REQUIREMENTS
