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
def test_real_configure_that_stops_identifies_the_blocker(tmp_path):
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindThing.cmake").write_text(FIND_MODULE)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_package(Thing REQUIRED)\n"
    )
    result = run(tmp_path, timeout=120)
    assert not result.completed
    assert result.blocking == "Thing"


@needs_cmake
def test_the_source_tree_is_left_alone(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(demo NONE)\n"
    )
    before = sorted(p.name for p in tmp_path.iterdir())
    run(tmp_path, timeout=120)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
