"""Pseudobuilds: configure the project, watch what it asks for, throw it away."""

import json
import subprocess

import pytest

from will_it_riscv import pseudobuild
from will_it_riscv.pseudobuild import (
    _parse_trace,
    _read_outcomes,
    _subjects,
    available,
    run,
)

needs_cmake = pytest.mark.skipif(not available(), reason="cmake is not installed")


# -- reading the configure's narration --------------------------------------


def test_a_status_miss_is_proof_of_optionality():
    """CMake prefixes STATUS with "-- ". The configure shrugged and moved on."""
    found, soft, blocking = _read_outcomes(
        "-- Found ZLIB: /usr/lib/libz.so\n"
        "-- Could NOT find MySQL (missing: MYSQL_LIBRARY)\n"
        "-- Configuring done\n"
    )
    assert found == {"ZLIB"}
    assert soft == {"MySQL"}
    assert blocking is None


def test_a_miss_inside_an_error_block_is_the_blocker():
    found, soft, blocking = _read_outcomes(
        "-- Found ZLIB: /usr/lib/libz.so\n"
        "CMake Error at FindPackageHandleStandardArgs.cmake:290 (message):\n"
        "  Could NOT find PROJ (missing: PROJ_LIBRARY)\n"
    )
    assert blocking == "PROJ"
    assert found == {"ZLIB"}


def test_the_blocker_wins_over_its_own_status_line():
    """GDAL narrates PROJ as a status miss first, then fatals on it."""
    _, soft, blocking = _read_outcomes(
        "-- Could NOT find PROJ (missing: PROJ_DIR)\n"
        "CMake Error at CMakeLists.txt:287 (message):\n"
        "  Could NOT find PROJ\n"
    )
    assert blocking == "PROJ"
    assert "PROJ" not in soft


def test_status_output_after_an_error_block_ends_it():
    _, _, blocking = _read_outcomes(
        "CMake Error at x.cmake:1 (message):\n"
        "  some unrelated complaint\n"
        "-- Could NOT find Later (missing: X)\n"
    )
    assert blocking is None


# -- what a find command was actually looking for ---------------------------


def test_find_package_names_the_package():
    assert _subjects("find_package", ["ZLIB", "REQUIRED"]) == ["ZLIB"]


def test_find_library_skips_the_output_variable_and_stops_at_paths():
    """--trace-expand turns PATHS into real directories and registry keys."""
    args = [
        "ZLIB_LIBRARY", "NAMES", "z", "zlib",
        "PATHS", "/usr/lib", "/usr/local/lib",
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\Zlib;InstallPath]",
    ]
    assert _subjects("find_library", args) == ["z", "zlib"]


def test_pkg_check_modules_skips_the_prefix_and_keywords():
    assert _subjects("pkg_check_modules", ["PC_CURL", "QUIET", "libcurl"]) == ["libcurl"]


def test_paths_and_flags_are_never_libraries():
    assert _subjects("find_library", ["V", "NAMES", "/usr/lib/libz.so"]) == []
    assert _subjects("find_library", ["V", "NAMES", "-lz"]) == []


# -- the trace --------------------------------------------------------------


def test_trace_parsing(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"cmd": "set", "args": ["X", "1"]},
                {"cmd": "find_package", "args": ["ZLIB", "REQUIRED"]},
                {"cmd": "find_package", "args": ["Boost", "QUIET"]},
                {"cmd": "pkg_check_modules", "args": ["PC_XML", "QUIET", "libxml-2.0"]},
                {"cmd": "if", "args": ["TRUE"]},
                "not json at all",
            ]
        )
    )
    probes, traced = _parse_trace(trace)
    assert set(probes) == {"zlib", "boost", "libxml-2.0"}
    assert probes["zlib"].required
    assert probes["boost"].quiet and not probes["boost"].required
    assert traced == 5


def test_missing_trace_file_is_not_an_error(tmp_path):
    probes, traced = _parse_trace(tmp_path / "nope.json")
    assert probes == {} and traced == 0


# -- the subprocess boundary ------------------------------------------------


def test_no_cmakelists_means_nothing_to_do(tmp_path):
    assert run(tmp_path) is None


def test_missing_cmake_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    monkeypatch.setattr(pseudobuild.shutil, "which", lambda _: None)
    assert "not installed" in run(tmp_path).error


def test_timeout_keeps_whatever_the_trace_already_held(tmp_path, monkeypatch):
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    monkeypatch.setattr(pseudobuild.shutil, "which", lambda _: "/usr/bin/cmake")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="cmake", timeout=1)

    monkeypatch.setattr(pseudobuild.subprocess, "run", timeout)
    result = run(tmp_path, timeout=1)
    assert not result.completed and "within 1s" in result.error


def test_pkg_config_is_denied_so_optionality_is_demonstrated(tmp_path):
    """A configure told nothing is installed only insists on what it needs."""
    env = pseudobuild._environment(tmp_path, deny_pkg_config=True)
    assert env["PKG_CONFIG_LIBDIR"].startswith(str(tmp_path))
    assert env["PKG_CONFIG_PATH"].startswith(str(tmp_path))


# -- against a real cmake ---------------------------------------------------

FIND_MODULE = (
    "include(FindPackageHandleStandardArgs)\n"
    "find_package_handle_standard_args(Thing REQUIRED_VARS THING_LIBRARY)\n"
)


@needs_cmake
def test_real_configure_that_completes_without_an_optional_package(tmp_path):
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindThing.cmake").write_text(FIND_MODULE)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_package(Thing)\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert "Thing" in result.soft_misses      # absent, and it carried on
    assert result.blocking is None


@needs_cmake
def test_real_configure_that_stops_is_unblocked_and_rerun(tmp_path):
    """The blocker is recorded, stubbed, and the configure run again."""
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindThing.cmake").write_text(FIND_MODULE)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_package(Thing REQUIRED)\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.blockers == ["Thing"]        # it was a hard requirement
    assert result.blocking == "Thing"
    assert result.rounds == 2                  # stubbed, then it got through
    assert result.completed
    assert "THING_LIBRARY" in result.unblocked


@needs_cmake
def test_a_blocker_is_never_reported_as_optional(tmp_path):
    """Round one narrates the blocker as a status miss before fatalling."""
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindThing.cmake").write_text(FIND_MODULE)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_package(Thing REQUIRED)\n"
    )
    result = run(tmp_path, timeout=120)
    assert "Thing" not in result.soft_misses


@needs_cmake
def test_the_loop_stops_when_it_cannot_help(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        'message(FATAL_ERROR "no amount of stubbing fixes this")\n'
    )
    result = run(tmp_path, timeout=120, max_rounds=4)
    assert not result.completed
    assert result.rounds == 1      # nothing to synthesise, so no second try


@needs_cmake
def test_the_source_tree_is_left_alone(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(demo NONE)\n"
    )
    before = sorted(p.name for p in tmp_path.iterdir())
    run(tmp_path, timeout=120)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# -- synthesis --------------------------------------------------------------


def synth(narration, tmp_path, blocking=None, already=None):
    from will_it_riscv.pseudobuild import _synthesize

    return _synthesize(blocking, narration, tmp_path, already or set())


ERROR = "CMake Error at FindPackageHandleStandardArgs.cmake:290 (message):\n  "


def test_missing_library_and_include_variables_are_stubbed(tmp_path):
    overrides, _ = synth(
        ERROR + "Could NOT find PROJ (missing: PROJ_LIBRARY PROJ_INCLUDE_DIR)\n",
        tmp_path,
    )
    assert set(overrides) == {"PROJ_LIBRARY", "PROJ_INCLUDE_DIR"}
    from pathlib import Path

    assert Path(overrides["PROJ_LIBRARY"]).exists()
    assert Path(overrides["PROJ_INCLUDE_DIR"]).is_dir()


def test_status_misses_are_never_stubbed(tmp_path):
    """Stubbing an optional dependency would erase the evidence it is one."""
    narration = (
        "-- Could NOT find MySQL (missing: MYSQL_LIBRARY MYSQL_INCLUDE_DIR)\n"
        "-- Could NOT find ODBC (missing: ODBC_INCLUDE_DIR)\n"
        + ERROR + "Could NOT find PROJ (missing: PROJ_LIBRARY)\n"
    )
    overrides, _ = synth(narration, tmp_path)
    assert set(overrides) == {"PROJ_LIBRARY"}


def test_a_version_variable_gets_a_generous_version(tmp_path):
    overrides, _ = synth(
        ERROR + "Could NOT find X (missing: X_VERSION)\n", tmp_path
    )
    assert overrides["X_VERSION"] == "99.9.9"


def test_an_executable_variable_gets_a_runnable_stub(tmp_path):
    from pathlib import Path

    overrides, _ = synth(
        ERROR + "Could NOT find X (missing: X_EXECUTABLE)\n", tmp_path
    )
    path = Path(overrides["X_EXECUTABLE"])
    assert path.exists() and path.stat().st_mode & 0o111


def test_a_config_mode_dir_hint_is_left_alone(tmp_path):
    """Faking <Pkg>_DIR sends CMake after a config file that is not there."""
    overrides, _ = synth(ERROR + "Could NOT find PROJ (missing: PROJ_DIR)\n", tmp_path)
    assert "PROJ_DIR" not in overrides


def test_an_unrecognised_variable_is_merely_made_truthy(tmp_path):
    overrides, _ = synth(
        ERROR + "Could NOT find X (missing: CRYPTOPP_TEST_KNOWNBUG)\n", tmp_path
    )
    assert overrides["CRYPTOPP_TEST_KNOWNBUG"] == "1"


def test_a_header_the_module_wanted_to_read_is_written_with_versions(tmp_path):
    """GDAL's FindPROJ greps the version out of proj.h and rejects old ones."""
    target = tmp_path / "include" / "proj.h"
    narration = (
        "CMake Error at FindPROJ.cmake:48 (file):\n"
        "  file failed to open for reading (No such file or directory):\n"
        f"    {target}\n"
    )
    overrides, created = synth(narration, tmp_path, blocking="PROJ")
    assert created == [str(target)]
    body = target.read_text()
    assert "#define PROJ_VERSION_MAJOR 99" in body
    assert '#define PROJ_VERSION "99.9.9"' in body


def test_writing_a_file_counts_as_progress_even_without_an_override(tmp_path):
    target = tmp_path / "include" / "thing.h"
    narration = (
        "CMake Error at F.cmake:1 (file):\n"
        "  file failed to open for reading (No such file or directory):\n"
        f"    {target}\n"
    )
    overrides, created = synth(narration, tmp_path)
    assert overrides == {} and created


def test_nothing_is_written_outside_the_scratch_directory(tmp_path):
    narration = (
        "CMake Error at F.cmake:1 (file):\n"
        "  file failed to open for reading (No such file or directory):\n"
        "    /etc/definitely-not-ours.h\n"
    )
    _, created = synth(narration, tmp_path)
    assert created == []


def test_a_file_already_written_is_not_rewritten(tmp_path):
    target = tmp_path / "include" / "x.h"
    narration = (
        "CMake Error at F.cmake:1 (file):\n"
        "  file failed to open for reading (No such file or directory):\n"
        f"    {target}\n"
    )
    _, created = synth(narration, tmp_path, already={str(target)})
    assert created == []
