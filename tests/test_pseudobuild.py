"""Pseudobuilds: configure the project, watch what it asks for, throw it away."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

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
    found, soft, blockers = _read_outcomes(
        "-- Found ZLIB: /usr/lib/libz.so\n"
        "-- Could NOT find MySQL (missing: MYSQL_LIBRARY)\n"
        "-- Configuring done\n"
    )
    assert found == {"ZLIB"}
    assert soft == {"MySQL"}
    assert blockers == []


def test_a_miss_inside_an_error_block_is_the_blocker():
    found, soft, blockers = _read_outcomes(
        "-- Found ZLIB: /usr/lib/libz.so\n"
        "CMake Error at FindPackageHandleStandardArgs.cmake:290 (message):\n"
        "  Could NOT find PROJ (missing: PROJ_LIBRARY)\n"
    )
    assert blockers == ["PROJ"]
    assert found == {"ZLIB"}


def test_the_blocker_wins_over_its_own_status_line():
    """GDAL narrates PROJ as a status miss first, then fatals on it."""
    _, soft, blockers = _read_outcomes(
        "-- Could NOT find PROJ (missing: PROJ_DIR)\n"
        "CMake Error at CMakeLists.txt:287 (message):\n"
        "  Could NOT find PROJ\n"
    )
    assert blockers == ["PROJ"]
    assert "PROJ" not in soft


def test_status_output_after_an_error_block_ends_it():
    _, _, blockers = _read_outcomes(
        "CMake Error at x.cmake:1 (message):\n"
        "  some unrelated complaint\n"
        "-- Could NOT find Later (missing: X)\n"
    )
    assert blockers == []


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
    monkeypatch.setattr(pseudobuild, "cmake", lambda: None)
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


@needs_cmake
def test_a_configure_writing_to_its_working_directory_writes_to_scratch(tmp_path, monkeypatch):
    """gmsh's execute_process(... OUTPUT_FILE cmake_options.texi) names no directory."""
    source, elsewhere = tmp_path / "src", tmp_path / "cwd"
    source.mkdir()
    elsewhere.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(demo NONE)\n"
        "execute_process(COMMAND echo hi OUTPUT_FILE leaked.txt)\n"
    )
    monkeypatch.chdir(elsewhere)
    run(source, timeout=120)
    assert list(elsewhere.iterdir()) == []


@needs_cmake
def test_pruning_settles_what_was_found_by_taking_it_away(tmp_path):
    """Found says nothing; configuring without it says required or optional."""
    for name in ("Foo", "Bar"):
        (tmp_path / "cmake" / f"Find{name}.cmake").parent.mkdir(exist_ok=True)
        (tmp_path / "cmake" / f"Find{name}.cmake").write_text(
            f'set({name}_FOUND TRUE)\nmessage(STATUS "Found {name}: yes")\n'
        )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(demo NONE)\n"
        'list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")\n'
        "find_package(Foo)\nfind_package(Bar)\n"
        'if(NOT Bar_FOUND)\n  message(FATAL_ERROR "Bar is needed")\nendif()\n'
    )
    result = run(tmp_path, timeout=120, prune=True)
    assert result.completed
    assert result.pruned["Foo"][0] is False
    assert result.pruned["Bar"][0] is True and "Bar is needed" in result.pruned["Bar"][1]


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


def test_a_stub_program_says_a_version_and_writes_what_it_is_told_to(tmp_path):
    sysroot = tmp_path / "sysroot"
    (sysroot / "bin").mkdir(parents=True)
    overrides, _ = synth(ERROR + "Could NOT find Doxygen (missing: DOXYGEN_EXECUTABLE)\n", sysroot)
    doxygen = overrides["DOXYGEN_EXECUTABLE"]
    said = subprocess.run([doxygen, "--version"], capture_output=True, text=True)
    assert said.stdout.strip() == "99.9.9"
    template = tmp_path / "build" / "CMakeDoxyfile.tpl"
    template.parent.mkdir()
    subprocess.run([doxygen, "-s", "-g", str(template)], check=True)
    assert template.exists()
    outside = tmp_path.parent / "not-scratch.tpl"
    subprocess.run([doxygen, "-g", str(outside)], check=True)
    assert not outside.exists()
    git = pseudobuild._stub_variable("GIT_EXECUTABLE", sysroot, ".so")
    assert subprocess.run([git, "--version"], capture_output=True, text=True).stdout.strip() \
        == "git version 99.9.9"


@needs_cmake
def test_doxygen_is_stubbed_far_enough_for_find_doxygen(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "find_package(Doxygen 1.8 REQUIRED)\n"
        "doxygen_add_docs(docs ${CMAKE_SOURCE_DIR})\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["Doxygen"]


@pytest.mark.parametrize("regex, line", [
    ('^#define[ \t]+PY_VERSION[ \t]+"[^"]+"', '#define PY_VERSION "99.9.9"'),
    ("#define ARMA_VERSION_MINOR[ \t]+([0-9]+)", "#define ARMA_VERSION_MINOR 9"),
    ('#define[ \t]+ARES_VERSION_STR[ \t]+"([0-9.]+)"', '#define ARES_VERSION_STR "99.9.9"'),
    ('PY_VERSION[ \t]+"(.*)"', 'PY_VERSION "99.9.9"'),
    ("^Version: ([0-9]+\\.[0-9]+)$", "Version: 99.9"),
    ("[", None),
])
def test_a_line_is_made_that_a_cmake_regex_matches(regex, line):
    assert pseudobuild._example(regex) == line


def test_each_branch_of_an_alternation_gets_its_own_line():
    regex = "TBB_VERSION_(MAJOR|MINOR)[ \t]+[0-9]+"
    assert [pseudobuild._example(regex, pick=k) for k in (0, 1)] == [
        "TBB_VERSION_MAJOR 99", "TBB_VERSION_MINOR 9"]


VERSION_GREP = """\
find_path(ARMA_INCLUDE_DIR NAMES armadillo)
if(ARMA_INCLUDE_DIR)
  file(STRINGS "${ARMA_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _major
       REGEX "#define ARMA_VERSION_MAJOR[ \\t]+[0-9]+")
  string(REGEX REPLACE ".*#define ARMA_VERSION_MAJOR[ \\t]+([0-9]+).*" "\\\\1" _major "${_major}")
  file(READ "${ARMA_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _text)
  string(REGEX REPLACE ".*#define ARMA_VERSION_MINOR[ \\t]+([0-9]+).*" "\\\\1" _minor "${_text}")
  set(ARMADILLO_VERSION_STRING "${_major}.${_minor}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Armadillo
  REQUIRED_VARS ARMA_INCLUDE_DIR VERSION_VAR ARMADILLO_VERSION_STRING)
"""


@needs_cmake
def test_a_stub_header_gets_the_version_lines_its_module_greps_for(tmp_path):
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindArmadillo.cmake").write_text(VERSION_GREP)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_package(Armadillo 4.0 REQUIRED)\n"
        'if(NOT ARMADILLO_VERSION_STRING VERSION_EQUAL "99.9")\n'
        '  message(FATAL_ERROR "version ${ARMADILLO_VERSION_STRING}")\n'
        "endif()\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["Armadillo"]


@needs_cmake
def test_a_version_header_checked_for_before_it_is_read_is_written(tmp_path):
    """CMake's own FindArmadillo reads arma_version.hpp only if(EXISTS)."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo CXX)\n"
        "find_package(Armadillo 4.0 REQUIRED)\n"
    )
    result = run(tmp_path, timeout=180)
    assert result.completed, result.error
    assert "Armadillo" in result.blockers


@needs_cmake
def test_a_lookup_the_error_names_in_its_own_words_is_tried(tmp_path):
    """No package and no "Could NOT find": the project's own message names it."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo C)\n"
        "find_program(EXE_FROBNICATE frobnicate-tool)\n"
        "if(NOT EXE_FROBNICATE)\n"
        '  message(FATAL_ERROR "frobnicate-tool binary not found!")\n'
        "endif()\n"
        "find_library(MGIS_DL_LIBRARY dl)\n"
        "if(NOT MGIS_DL_LIBRARY)\n"
        '  message(FATAL_ERROR "the libdl library not found. Set `MGIS_DL_LIBRARY` manually")\n'
        "endif()\n"
    )
    result = run(tmp_path, timeout=180)
    assert result.completed, result.error
    assert result.blockers == ["frobnicate-tool", "dl"]
    assert ("frobnicate-tool", True) in result.experiments


@needs_cmake
def test_a_linked_target_nothing_defines_is_made_at_the_end(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\n"
        "project(demo C)\n"
        "file(WRITE ${CMAKE_BINARY_DIR}/main.c \"int main(void){return 0;}\")\n"
        "add_executable(main ${CMAKE_BINARY_DIR}/main.c)\n"
        "target_link_libraries(main PRIVATE Widget::widget)\n"
    )
    result = run(tmp_path, timeout=180)
    assert result.completed, result.error


@needs_cmake
def test_qt_is_found_by_the_name_it_was_searched_for_at_its_own_major(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)\n"
        "find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Widgets)\n"
        'if(Qt6Core_VERSION VERSION_LESS "6.2.4")\n'
        '  message(FATAL_ERROR "Qt version 6.2.4 or higher is required")\n'
        "endif()\n"
    )
    result = run(tmp_path, timeout=180)
    assert result.completed, result.error
    assert result.blockers[0] in ("QT", "Qt6")


def test_cmakes_own_lookups_and_the_interpreter_are_never_suspects():
    narration = (
        "CMake Error at CMakeLists.txt:9 (message):\n"
        "  failed: Run Build Command(s): /usr/bin/gmake -f Makefile; python3 not found;"
        " frob not found\n"
    )
    searches = [
        ("find_program", "CMAKE_MAKE_PROGRAM", ("gmake", "make")),
        ("find_program", "PYTHON_EXECUTABLE", ("python3",)),
        ("find_program", "FROB_EXE", ("frob",)),
    ]
    assert [n for n, _ in pseudobuild._named_searches(narration, searches)] == ["frob"]


CMAKES_OWN_MODULES = {
    # FindCURL greps curlver.h for LIBCURL_VERSION "\".*\"".
    "curl-version": ("C", "find_package(CURL 7.32 REQUIRED)\n", ["CURL"]),
    # FairRoot: the modules a package of CMake modules would have put on the path.
    "shipped-modules": ("NONE", "find_package(FairCMakeModules 0.2 REQUIRED)\n"
                        "include(FairFormattedOutput)\n"
                        "find_package(ECM REQUIRED NO_MODULE)\n"
                        "set(CMAKE_MODULE_PATH ${ECM_MODULE_PATH})\n"
                        "include(KDEInstallDirs)\n", ["FairCMakeModules", "ECM"]),
    # MuseScore asks Qt::qmake where Qt is installed; colmap's AUTORCC wants Qt5::rcc.
    "qt6-qmake": ("CXX", "find_package(Qt6 REQUIRED COMPONENTS Core)\n"
                  "if(NOT TARGET Qt::qmake)\n"
                  '  message(FATAL_ERROR "No supported Qt version found.")\n'
                  "endif()\n"
                  "get_target_property(QMAKE Qt${QT_VERSION_MAJOR}::qmake LOCATION)\n"
                  "execute_process(COMMAND ${QMAKE} -query OUTPUT_VARIABLE ALL RESULT_VARIABLE rc)\n"
                  'string(REPLACE "\\n" ";" VARS ${ALL})\n'
                  "if(NOT rc EQUAL 0 OR NOT ALL MATCHES QT_INSTALL_LIBS)\n"
                  '  message(FATAL_ERROR "qmake -query: ${ALL}")\n'
                  "endif()\n", ["Qt6"]),
    "qt5-autorcc": ("CXX", "find_package(Qt5 REQUIRED COMPONENTS Core Widgets)\n"
                    "set(CMAKE_AUTOMOC ON)\nset(CMAKE_AUTORCC ON)\n"
                    'file(WRITE ${CMAKE_BINARY_DIR}/r.qrc "<RCC><qresource/></RCC>")\n'
                    'file(WRITE ${CMAKE_BINARY_DIR}/m.cpp "int main(){return 0;}")\n'
                    "add_executable(m ${CMAKE_BINARY_DIR}/m.cpp ${CMAKE_BINARY_DIR}/r.qrc)\n"
                    "target_link_libraries(m Qt5::Widgets Qt::Core)\n", ["Qt5"]),
    # open3d: a Python3 it can do without, until its own error says "Python".
    "python-named-bare": ("C", "find_package(Python3 3.6 COMPONENTS Interpreter Development)\n"
                          "if(NOT Python3_FOUND)\n"
                          '  message(FATAL_ERROR "BUILD_PYTHON_MODULE=ON requires Python >= 3.6.")\n'
                          "endif()\n", ["Python3"]),
    # atlas asks for ecbuild before project(); paraview's error only says "Qt".
    "config-before-project": ("NONE", "", ["ecbuild"]),
    "qt-auto": ("CXX", "find_package(Qt6 QUIET COMPONENTS Core)\n"
                "if(NOT Qt6_FOUND)\n  find_package(Qt5 QUIET COMPONENTS Core)\n"
                "  if(NOT Qt5_FOUND)\n"
                '    message(FATAL_ERROR "Could not find a valid Qt installation.")\n'
                "  endif()\nendif()\n", ["Qt6"]),
    # vgm includes Geant4's use file; openscad compares CGAL_MAJOR_VERSION.
    "use-file-and-spellings": ("NONE", "find_package(Geant4 REQUIRED)\ninclude(${Geant4_USE_FILE})\n"
                               "find_package(CGAL REQUIRED COMPONENTS Core)\n"
                               "if (${CGAL_MAJOR_VERSION} LESS 5)\n"
                               '  message(FATAL_ERROR "CGAL too old")\nendif()\n',
                               ["Geant4", "CGAL"]),
    # tramonto: CMake's other wording of a config package it could not find.
    "by-not-providing": ("NONE", "find_package(Trilinos REQUIRED)\n", ["Trilinos"]),
    # exacmech: the error names the variable to set.
    "named-variable": ("NONE", "if(NOT RAJA_DIR)\n"
                       '  message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")\n'
                       "endif()\nfind_package(RAJA REQUIRED PATHS ${RAJA_DIR})\n", ["RAJA"]),
    # ufs-weather-model: a Find module only a package would have shipped.
    "find-module": ("NONE", "find_package(ESMF MODULE REQUIRED)\n", ["ESMF"]),
    # pocl takes LLVM 18.1, 19.1 or 20.1 only, else llvm-config; tvm asks llvm-config.
    "llvm-range": ("NONE", "find_package(LLVM 20.1.0...<20.2 CONFIG)\n"
                   "if(NOT LLVM_FOUND)\n  find_package(LLVM 19.1.0...<19.2 CONFIG)\nendif()\n"
                   "if(NOT LLVM_FOUND)\n"
                   "  find_program(LLVM_CONFIG_BIN NAMES llvmtce-config llvm-config)\nendif()\n"
                   "if(NOT LLVM_CONFIG_BIN AND NOT LLVM_PACKAGE_VERSION)\n"
                   '  message(FATAL_ERROR "Could not find either llvm-config or LLVMConfig.cmake !")\n'
                   "endif()\n", ["LLVM"]),
    "llvm-config": ("NONE", "find_package(LLVM REQUIRED CONFIG)\n"
                    "set(LLVM_CONFIG ${LLVM_TOOLS_BINARY_DIR}/llvm-config)\n"
                    "foreach(q --libfiles --system-libs --cxxflags --version --prefix --libdir "
                    "--cmakedir --targets-built)\n"
                    "  execute_process(COMMAND ${LLVM_CONFIG} ${q} RESULT_VARIABLE rc OUTPUT_VARIABLE o)\n"
                    "  if(rc)\n"
                    '    message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} ${q}")\n'
                    "  endif()\nendforeach()\n", ["LLVM"]),
    # rstudio: a QUIET Boost with components, and its own error after.
    "boost-quiet": ("CXX", "find_package(Boost 1.91.0 QUIET COMPONENTS filesystem system)\n"
                    "if(NOT Boost_FOUND)\n"
                    '  message(FATAL_ERROR "Could not find Boost 1.91.0 or newer")\n'
                    "endif()\n", ["Boost"]),
    # gdl: FindZLIB passes its library names as one list argument.
    "zlib-quiet": ("C", "find_package(ZLIB QUIET)\n"
                   "if(NOT ZLIB_FOUND)\n"
                   '  message(FATAL_ERROR "ZLib library is required but was not found.")\n'
                   "endif()\n", ["ZLIB"]),
    # musescore links what its framework's add-library command made.
    "linked-command-target": ("CXX", "find_package(Muse REQUIRED)\n"
                              'file(WRITE ${CMAKE_BINARY_DIR}/a.cpp "int a(){return 0;}")\n'
                              "muse_add_library(core ${CMAKE_BINARY_DIR}/a.cpp)\n"
                              "add_library(app STATIC ${CMAKE_BINARY_DIR}/a.cpp)\n"
                              "target_link_libraries(app core)\n", ["Muse", "muse"]),
    # tvm runs llvm-config by bare name.
    "stub-on-path": ("NONE", "find_package(LLVM REQUIRED CONFIG)\n"
                     "execute_process(COMMAND llvm-config --version RESULT_VARIABLE rc "
                     "OUTPUT_VARIABLE v)\n"
                     "if(rc)\n"
                     '  message(FATAL_ERROR "Fatal error executing: llvm-config --version")\n'
                     "endif()\n", ["LLVM"]),
    # FPHSA's HANDLE_COMPONENTS: "Failed to find all ICU components (missing: i18n)".
    "icu-components": ("CXX", "find_package(ICU REQUIRED COMPONENTS uc i18n)\n", ["ICU"]),
    # Xft counts only with Freetype and Fontconfig, which FindX11 asks for QUIET.
    "x11-xft": ("C", "find_package(X11 REQUIRED COMPONENTS Xft)\n",
                ["X11", "Freetype", "Fontconfig"]),
    # FindHDF5 reads H5pubconf.h only if it is there, and its version by if(MATCHES).
    "hdf5-version": ("C", "find_package(HDF5 REQUIRED COMPONENTS C)\n"
                     "if(HDF5_VERSION VERSION_LESS 1.14)\n"
                     '  message(FATAL_ERROR "required HDF5 version is at least 1.14")\n'
                     "endif()\n"
                     "if(HDF5_IS_PARALLEL)\n"
                     '  message(FATAL_ERROR "a stub claimed a feature nobody asked for")\n'
                     "endif()\n", ["HDF5"]),
    "hdf5-parallel": ("C", "find_package(HDF5 REQUIRED COMPONENTS C)\n"
                      "if(NOT HDF5_IS_PARALLEL)\n"
                      '  message(FATAL_ERROR "HDF5 library needs to enable parallel support.")\n'
                      "endif()\n", ["HDF5"]),
}


@needs_cmake
@pytest.mark.parametrize("case", sorted(CMAKES_OWN_MODULES))
def test_cmakes_own_find_modules_are_satisfied(case, tmp_path):
    language, body, blockers = CMAKES_OWN_MODULES[case]
    before = "find_package(ecbuild 3.4 REQUIRED)\n" if case == "config-before-project" else ""
    (tmp_path / "CMakeLists.txt").write_text(
        f"cmake_minimum_required(VERSION 3.16)\n{before}project(demo {language})\n{body}"
    )
    result = run(tmp_path, timeout=240)
    assert result.completed, result.error
    assert result.blockers == blockers


FIND_TBB = """\
find_path(TBB_INCLUDE_DIR tbb/version.h)
find_library(TBB_LIBRARY tbb)
if(TBB_INCLUDE_DIR)
  file(STRINGS "${TBB_INCLUDE_DIR}/tbb/version.h" _v REGEX "#define TBB_VERSION_(MAJOR|MINOR) ")
  string(REGEX REPLACE ".*#define TBB_VERSION_MAJOR ([0-9]+).*" "\\\\1" _major "${_v}")
  string(REGEX REPLACE ".*#define TBB_VERSION_MINOR ([0-9]+).*" "\\\\1" _minor "${_v}")
  set(TBB_VERSION "${_major}.${_minor}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB REQUIRED_VARS TBB_INCLUDE_DIR TBB_LIBRARY VERSION_VAR TBB_VERSION)
"""

FIND_CYTHON = """\
find_program(CYTHON_EXECUTABLE NAMES cython)
if(CYTHON_EXECUTABLE)
  execute_process(COMMAND ${CYTHON_EXECUTABLE} --version OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
  string(REGEX REPLACE ".*Cython version ([0-9.]+).*" "\\\\1" CYTHON_VERSION "${_out}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cython REQUIRED_VARS CYTHON_EXECUTABLE VERSION_VAR CYTHON_VERSION
                                  HANDLE_VERSION_RANGE)
"""


@needs_cmake
@pytest.mark.parametrize("module, wants", [
    (FIND_TBB, "find_package(TBB 2020.3 REQUIRED)"),          # openvdb: years, not 99
    (FIND_CYTHON, "find_package(Cython 3.0.4...<4.0.0 REQUIRED)"),   # espresso: a range
])
def test_a_stub_is_redone_at_the_version_the_configure_asked_for(module, wants, tmp_path):
    (tmp_path / "cmake").mkdir()
    name = "TBB" if module is FIND_TBB else "Cython"
    (tmp_path / "cmake" / f"Find{name}.cmake").write_text(module)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\nproject(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n" + wants + "\n"
    )
    result = run(tmp_path, timeout=240)
    assert result.completed, result.error
    assert name in result.blockers


def test_a_setting_is_answered_and_a_feature_switch_only_when_asked_for(tmp_path):
    sysroot = tmp_path / "sysroot"
    header = sysroot / "include" / "metis.h"
    header.parent.mkdir(parents=True)
    header.write_text("/* stub emitted by will-it-riscv for METIS */\n")
    width = "^#define[\t ]+IDXTYPEWIDTH[\t ]+([0-9\\.]+[0-9\\.]+)$"
    parallel = "HAVE_PARALLEL 1"
    answered: set = set()
    pseudobuild._answer_reads([(str(header), width), (str(header), parallel)], sysroot, answered)
    assert "IDXTYPEWIDTH" in header.read_text() and "HAVE_PARALLEL" not in header.read_text()
    pseudobuild._answer_reads([(str(header), parallel)], sysroot, answered,
                              "HDF5 library needs to enable parallel support.")
    assert "HAVE_PARALLEL 1" in header.read_text()


def test_only_stub_files_in_the_sysroot_are_answered(tmp_path):
    sysroot = tmp_path / "sysroot"
    (sysroot / "include").mkdir(parents=True)
    ours = sysroot / "include" / "foo.h"
    ours.write_text("/* stub emitted by will-it-riscv for foo */\n")
    config = sysroot / "lib" / "cmake" / "Foo" / "FooConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("set(Foo_FOUND 1)\n")
    outside = tmp_path / "real.h"
    outside.write_text("/* stub emitted by will-it-riscv */\n")
    reads = [(str(p), "#define FOO_VERSION_MAJOR[ \t]+[0-9]+") for p in (ours, config, outside)]
    answered: set = set()
    assert pseudobuild._answer_reads(reads, sysroot, answered) == [str(ours)]
    assert "#define FOO_VERSION_MAJOR 99" in ours.read_text()
    assert pseudobuild._answer_reads(reads, sysroot, answered) == []   # each grep once


@needs_cmake
@pytest.mark.skipif(not shutil.which("gfortran"), reason="gfortran is not installed")
def test_fortran_and_c_are_taken_to_agree_without_linking(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo C CXX Fortran)\n"
        "include(FortranCInterface)\n"
        "FortranCInterface_VERIFY()\n"
        "FortranCInterface_VERIFY(CXX)\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error


def test_shims_that_only_fail_are_not_on_the_configures_path():
    path = "/home/u/.pyenv/shims:/usr/bin:/home/u/.asdf/shims/:/opt/bin"
    assert pseudobuild._without_shims(path) == "/usr/bin:/opt/bin"


def test_a_config_mode_dir_hint_leads_to_a_config_that_is_there(tmp_path):
    """Faking <Pkg>_DIR alone sends CMake after a config file that is not
    there; it only ever points at the config stub written for it."""
    overrides, _ = synth(ERROR + "Could NOT find PROJ (missing: PROJ_DIR)\n", tmp_path)
    if "PROJ_DIR" in overrides:
        assert (Path(overrides["PROJ_DIR"]) / "PROJConfig.cmake").is_file()


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


def test_a_file_strings_read_is_stubbed_and_boost_gets_its_packed_version(tmp_path):
    target = tmp_path / "include" / "boost" / "version.hpp"
    narration = (
        "CMake Error at /usr/share/cmake-3.28/Modules/FindBoost.cmake:1842 (file):\n"
        # CMake wraps it: the path gets a line of its own.
        f'  file STRINGS file\n  "{target}" cannot be\n  read.\n'
    )
    _, created = synth(narration, tmp_path, blocking="Boost")
    assert created == [str(target)]
    assert "#define BOOST_VERSION 9900909" in target.read_text()


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


# -- more shapes of "what stopped it" ---------------------------------------

from will_it_riscv.pseudobuild import _host_gaps, _stanza_blockers  # noqa: E402


def test_a_required_pkg_config_module_is_the_blocker():
    stanza = (
        "CMake Error at /usr/share/cmake/Modules/FindPkgConfig.cmake:1093 (message):\n"
        "  The following required packages were not found:\n"
        "\n"
        "   - libpsl\n"
        "   - libfoo>=1.2\n"
        "\n"
        "Call Stack (most recent call first):\n"
    )
    assert _stanza_blockers(stanza) == ["libpsl", "libfoo"]


def test_pkg_search_module_names_its_first_alternative():
    stanza = "CMake Error at FindPkgConfig.cmake:9 (message):\n  None of the required 'a;b' found\n"
    assert _stanza_blockers(stanza) == ["a"]


def test_a_notfound_variable_a_target_links_is_a_blocker():
    stanza = (
        "CMake Error: The following variables are used in this project, but they are set to NOTFOUND.\n"
        "Please set them or make sure they are set and tested correctly in the CMake files:\n"
        "FOO_LIBRARY (ADVANCED)\n"
        '    linked by target "bar" in directory /src\n'
    )
    assert _stanza_blockers(stanza) == ["FOO"]


def test_a_missing_imported_target_names_its_package():
    stanza = (
        "CMake Error at /tmp/x/TryCompile-ab12/CMakeLists.txt:32 (target_link_libraries):\n"
        '  Target "cmTC_fb258" links to:\n\n    OpenSSL::SSL\n\n'
        "  but the target was not found.\n"
    )
    assert _stanza_blockers(stanza) == ["OpenSSL"]


def test_an_error_inside_a_find_module_is_that_packages():
    stanza = "CMake Error at cmake/modules/FindPROJ.cmake:48 (file):\n  something odd\n"
    assert _stanza_blockers(stanza) == ["PROJ"]


def test_a_missing_library_header_is_its_librarys_blocker():
    stanza = "CMake Error at CMakeLists.txt:9 (message):\n  zlib.h not found, install zlib\n"
    assert _stanza_blockers(stanza) == ["zlib"]


def test_a_missing_system_header_is_a_host_gap_not_a_dependency():
    """GDAL demands linux/fs.h, which the Mac SDK lacks and riscv64 has."""
    narration = (
        "CMake Error at port/CMakeLists.txt:156 (message):\n"
        "  linux/fs.h header not found.  Impact will be lack of sparse file detection.\n"
    )
    assert _stanza_blockers(narration) == []
    assert _host_gaps(narration) == ["linux/fs.h"]


def test_a_missing_header_is_stubbed_inside_the_sysroot(tmp_path):
    overrides, created = synth(
        "CMake Error at port/CMakeLists.txt:156 (message):\n  linux/fs.h header not found.\n",
        tmp_path,
    )
    assert created == [str(tmp_path / "include" / "linux" / "fs.h")]


def test_a_component_named_by_fphsa_gets_its_library_variable(tmp_path):
    overrides, _ = synth(
        ERROR + "Could NOT find OpenSSL (missing: OPENSSL_CRYPTO_LIBRARY SSL Crypto)\n",
        tmp_path,
    )
    assert "OPENSSL_SSL_LIBRARY" in overrides
    assert "SSL" not in overrides and "Crypto" not in overrides


def test_a_default_build_that_enables_cuda_is_blocked_by_cuda():
    from will_it_riscv.pseudobuild import _stanza_blockers

    stanza = (
        "CMake Error at /usr/share/cmake-3.31/Modules/Internal/CMakeCUDAFindToolkit.cmake:104"
        " (message):\n  Failed to find nvcc.\n\n  Compiler requires the CUDA toolkit.\n"
    )
    assert _stanza_blockers(stanza) == ["CUDA"]


def test_lower_case_components_are_components_though_they_have_underscores(tmp_path):
    overrides, _ = synth(ERROR + "Could NOT find Qux (missing: date_time QUX_INCLUDE_DIR)\n", tmp_path)
    assert "Qux_DATE_TIME_LIBRARY_RELEASE" in overrides and "QUX_INCLUDE_DIR" in overrides
    assert "date_time" not in overrides


def test_boost_is_answered_as_a_config_package_with_its_components(tmp_path):
    """FindBoost second-guesses preset paths, but defers to a BoostConfig.cmake."""
    _, created = synth(
        ERROR + "Could NOT find Boost (missing: date_time program_options) "
        '(found suitable version "99.9.9", minimum required is "1.66")\n',
        tmp_path,
    )
    config = tmp_path / "lib" / "cmake" / "Boost" / "BoostConfig.cmake"
    assert str(config) in created
    text = config.read_text()
    assert "foreach(_stub_component IN LISTS Boost_FIND_COMPONENTS)" in text
    assert "add_library(Boost::headers INTERFACE IMPORTED)" in text


def test_a_property_asked_of_a_missing_stub_target_adds_the_target(tmp_path):
    from will_it_riscv.pseudobuild import _stub_config_package

    _stub_config_package("Kokkos", tmp_path, ".so")
    _, created = synth(
        "CMake Error at cmake/macros/macro_copy_target_properties.cmake:61 (get_target_property):\n"
        '  get_target_property() called with non-existent target "Kokkos::cuda".\n',
        tmp_path,
    )
    assert any(c.endswith("#Kokkos::cuda") for c in created)


def test_a_missing_imported_target_gets_its_component_library(tmp_path):
    overrides, _ = synth(
        "CMake Error at /tmp/TryCompile-x/CMakeLists.txt:32 (target_link_libraries):\n"
        '  Target "cmTC_1" links to:\n\n    OpenSSL::SSL\n\n  but the target was not found.\n',
        tmp_path,
    )
    assert "OPENSSL_SSL_LIBRARY" in overrides


def test_a_required_pkg_config_module_gets_a_stub_pc_file(tmp_path):
    _, created = synth(
        "CMake Error at FindPkgConfig.cmake:1 (message):\n"
        "  The following required packages were not found:\n\n   - libpsl>=0.16\n\n",
        tmp_path,
    )
    pc = tmp_path / "lib" / "pkgconfig" / "libpsl.pc"
    assert str(pc) in created
    assert "Version: 99.9.9" in pc.read_text()


# -- confinement, against a real cmake --------------------------------------


@needs_cmake
def test_the_configure_is_run_as_linux_on_the_target(tmp_path):
    """Not the host: APPLE is false and the processor is the target's."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "if(APPLE OR WIN32)\n  message(FATAL_ERROR \"configured for the host\")\nendif()\n"
        'if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")\n'
        '  message(FATAL_ERROR "wrong processor ${CMAKE_SYSTEM_PROCESSOR}")\nendif()\n'
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.platform == "linux/riscv64"


@needs_cmake
def test_nothing_on_the_host_can_be_found(tmp_path):
    """Confined, a host library cannot answer for the target."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "find_library(Z_LIB NAMES z c System)\n"
        "if(Z_LIB)\n  message(FATAL_ERROR \"found ${Z_LIB} on the host\")\nendif()\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error


@needs_cmake
@pytest.mark.skipif(not pseudobuild.shutil.which("pkg-config"), reason="no pkg-config")
def test_a_required_pkg_config_module_is_unblocked_with_a_pc_file(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "find_package(PkgConfig REQUIRED)\n"
        "pkg_check_modules(FOO REQUIRED libwirtest>=1.0)\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["libwirtest"]


@needs_cmake
def test_a_missing_system_header_is_stubbed_and_reported_as_a_host_gap(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo C)\n"
        "include(CheckIncludeFile)\n"
        'check_include_file("linux/wirtest.h" HAVE_IT)\n'
        'if(NOT HAVE_IT)\n  message(FATAL_ERROR "linux/wirtest.h header not found")\nendif()\n'
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.host_gaps == ["linux/wirtest.h"]
    assert result.blockers == []


# -- blame by experiment ----------------------------------------------------

from will_it_riscv.pseudobuild import (  # noqa: E402
    _signature,
    _stub_lookups,
    _stub_variable,
    _suspects,
)


def test_a_flags_variable_is_never_filled_with_a_placeholder(tmp_path):
    """Whatever goes in *_FLAGS lands on a compile line, where "1" is a file."""
    assert _stub_variable("OpenMP_CXX_FLAGS", tmp_path, ".so") is None


def test_a_miss_the_error_names_is_the_first_suspect():
    narration = (
        "-- Could NOT find MPI (missing: MPI_CXX_LIBRARIES)\n"
        "-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)\n"
        "-- Could NOT find OpenMP (missing: OpenMP_CXX_FOUND)\n"
        "-- Could NOT find Python3 (missing: Python3_EXECUTABLE)\n"
        "CMake Error at cmake/gmxManageOpenMP.cmake:49 (message):\n"
        "  The compiler you are using does not support OpenMP parallelism\n"
    )
    names = [name for name, _ in _suspects(narration)]
    # Named first; then nearest to the error. OpenMP_CXX folds into OpenMP.
    assert names == ["OpenMP", "Python3", "MPI"]


def test_a_silent_find_package_the_error_names_is_a_suspect():
    """GROMACS's FindFFTW never says "Could NOT find": only the error names it."""
    from will_it_riscv.pseudobuild import Probe

    narration = (
        "CMake Error at cmake/gmxManageFFTLibraries.cmake:70 (message):\n"
        "  Cannot find FFTW 3 (with correct precision - libfftw3f)\n"
    )
    probes = {"fftw": Probe(name="FFTW", command="find_package")}
    assert [name for name, _ in _suspects(narration, probes)] == ["FFTW"]


def test_a_suspect_is_stubbed_the_way_its_module_looked_for_it(tmp_path):
    lookups = [
        ("pkg_check_modules", "PC_FFTWF", ("fftw3f",)),
        ("find_path", "FFTWF_INCLUDE_DIR", ("fftw3.h",)),
        ("find_library", "FFTWF_LIBRARY", ("fftw3f",)),
    ]
    for directory in ("include", "lib/pkgconfig"):
        (tmp_path / directory).mkdir(parents=True)
    files = _stub_lookups("FFTW", lookups, tmp_path, ".so")
    assert (tmp_path / "lib" / "libfftw3f.so").exists()
    assert (tmp_path / "include" / "fftw3.h").exists()
    assert (tmp_path / "lib" / "pkgconfig" / "fftw3f.pc").exists()
    # Every file is listed, so an innocent suspect can be taken back out.
    assert len(files) == 3


def test_an_error_signature_ignores_what_changes_every_run():
    first = "CMake Error at /var/x/will-it-riscv-abc123/b/TryCompile-q1 cmTC_1: boom"
    second = "CMake Error at /var/x/will-it-riscv-zzz999/b/TryCompile-r2 cmTC_2: boom"
    assert _signature(first) == _signature(second)


def test_lookups_are_recorded_against_the_package_running_them(tmp_path):
    events = [
        {"cmd": "find_package", "args": ["FFTW"], "file": "/p/CMakeLists.txt", "line": 1, "global_frame": 1},
        {"cmd": "find_path", "args": ["FFTWF_INCLUDE_DIR", "fftw3.h", "HINTS", "/x"], "file": "/p/FindFFTW.cmake", "line": 2, "global_frame": 2},
    ]
    trace = tmp_path / "trace.json"
    trace.write_text("\n".join(json.dumps(e) for e in events))
    lookups: dict = {}
    _parse_trace(trace, lookups=lookups)
    assert lookups == {"fftw": [("find_path", "FFTWF_INCLUDE_DIR", ("fftw3.h",))]}


@needs_cmake
def test_a_miss_the_configure_later_dies_of_is_found_by_experiment(tmp_path):
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindThing.cmake").write_text(FIND_MODULE)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_package(Thing)\n"
        "if(NOT Thing_FOUND)\n  message(FATAL_ERROR \"cannot go on without Thing\")\nendif()\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["Thing"]
    assert result.experiments == [("Thing", True)]


@needs_cmake
def test_an_innocent_suspect_is_taken_back_out(tmp_path):
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "FindThing.cmake").write_text(FIND_MODULE)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)\n"
        "find_package(Thing)\n"
        'message(FATAL_ERROR "something else entirely")\n'
    )
    result = run(tmp_path, timeout=120)
    assert not result.completed
    assert result.experiments == [("Thing", False)]
    assert result.blockers == []
    assert "THING_LIBRARY" not in result.unblocked


# -- provenance: who asked --------------------------------------------------


def test_the_trace_says_who_asked_and_from_where(tmp_path):
    """global_frame rebuilds the call stack: macro, parent package, call site."""
    root = tmp_path / "src"
    root.mkdir()
    events = [
        {"cmd": "macro", "args": ["proj_check"], "file": f"{root}/helpers.cmake", "line": 1, "global_frame": 1},
    ]
    # proj_check(X) called from three lines; its body's find_package is one line.
    for index, name in enumerate(["CURL", "EXPAT", "GEOS"], start=10):
        events += [
            {"cmd": "proj_check", "args": [name], "file": f"{root}/CMakeLists.txt", "line": index, "global_frame": 1},
            {"cmd": "find_package", "args": [name], "file": f"{root}/helpers.cmake", "line": 3, "global_frame": 2},
        ]
    # FindCURL (outside the project) asks for PkgConfig: an edge CURL -> PkgConfig.
    events[3:3] = [
        {"cmd": "find_package", "args": ["PkgConfig"], "file": "/usr/share/cmake/FindCURL.cmake", "line": 5, "global_frame": 3},
    ]
    trace = tmp_path / "trace.json"
    trace.write_text("\n".join(json.dumps(e) for e in events))
    probes, _ = _parse_trace(trace, root)
    assert probes["curl"].via == "proj_check"
    # The macro body's line asks for everything; the call site says who decided.
    assert probes["curl"].site == "CMakeLists.txt:10"
    assert probes["geos"].site == "CMakeLists.txt:12"
    assert probes["pkgconfig"].parent == "CURL"


# -- what a plan's configure steps needed -----------------------------------


REQUIRED_FIND = (
    "CMake Error at cmake/Fypp.cmake:6 (find_program):\n"
    "  Could not find FYPP_EXE using the following names: fypp\n"
)


def test_a_required_find_program_names_the_program():
    """MFC's first stop: find_program(FYPP_EXE fypp REQUIRED)."""
    assert _stanza_blockers(REQUIRED_FIND) == ["fypp"]


def test_a_required_find_program_gets_a_runnable_stub(tmp_path):
    overrides, _ = synth(REQUIRED_FIND, tmp_path)
    stub = Path(overrides["FYPP_EXE"])
    assert stub.name == "fypp" and stub.stat().st_mode & 0o111


def test_a_header_a_required_find_path_names_is_not_a_host_gap():
    narration = (
        "CMake Error at FindFoo.cmake:3 (find_path):\n"
        "  Could not find FOO_INCLUDE_DIR using the following names: foo.h\n"
    )
    assert _host_gaps(narration) == []


def test_mpi_is_answered_the_way_findmpi_asks(tmp_path):
    """FPHSA names MPI's results, not its inputs; faking those gets nowhere."""
    overrides, created = synth(
        "CMake Error at FindPackageHandleStandardArgs.cmake:290 (message):\n"
        "  Could NOT find MPI (missing: MPI_Fortran_FOUND Fortran)\n",
        tmp_path,
    )
    assert overrides["MPI_Fortran_WORKS"] == "TRUE"
    assert overrides["MPI_C_VERSION"] == "3.1"  # or FindMPI reads its test binary
    assert overrides["MPI_SKIP_COMPILER_WRAPPER"] == "TRUE"
    assert "MPI_Fortran_FOUND" not in overrides
    assert {Path(p).name for p in created} >= {"mpi.h", "mpif.h"}


def test_a_blocker_gets_what_its_own_module_looked_for(tmp_path):
    """FindHDF5 recomputes HDF5_INCLUDE_DIRS from its own find_path(hdf5.h)."""
    for directory in ("include", "lib/pkgconfig"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    from will_it_riscv.pseudobuild import _synthesize

    _, created = _synthesize(
        None,
        "CMake Error at FindPackageHandleStandardArgs.cmake:290 (message):\n"
        "  Could NOT find HDF5 (missing: HDF5_INCLUDE_DIRS)\n",
        tmp_path,
        set(),
        shared_suffix=".so",
        lookups={"hdf5": [
            ("find_path", "HDF5_C_INCLUDE_DIR", ("hdf5.h",)),
            ("find_library", "HDF5_C_LIBRARY_hdf5", ("hdf5",)),
        ]},
    )
    assert str(tmp_path / "include" / "hdf5.h") in created
    assert str(tmp_path / "lib" / "libhdf5.so") in created


@needs_cmake
def test_a_plans_defines_are_passed_and_never_stubbed_over(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(demo NONE)\n"
        'if(NOT DEMO_MPI STREQUAL "ON")\n  message(FATAL_ERROR "DEMO_MPI is ${DEMO_MPI}")\nendif()\n'
        "find_program(FYPP_EXE NAMES fypp REQUIRED)\n"
    )
    result = run(tmp_path, timeout=120, defines={"DEMO_MPI": "ON"})
    assert result.completed, result.error
    assert result.blockers == ["fypp"]
    assert "DEMO_MPI" not in result.unblocked


def test_an_error_summary_keeps_all_of_a_wrapped_message():
    """MFC's reason is on the message's second line: say all of it."""
    from will_it_riscv.pseudobuild import _first_error

    out = (
        "CMake Error at CMakeLists.txt:92 (message):\n"
        "  ERROR: MFC with GPU processing is not currently compatible with GNU\n"
        "  compilers.  Please use NVIDIA or Cray compilers.\n"
        "\n"
        "-- Configuring incomplete, errors occurred!\n"
    )
    assert _first_error(out, "").endswith("Please use NVIDIA or Cray compilers.")


def test_python_is_answered_at_the_interpreters_version(tmp_path):
    """FindPython holds the headers' version against the interpreter it found."""
    overrides, created = synth(
        "CMake Error at /usr/share/cmake/Modules/FindPackageHandleStandardArgs.cmake:290 "
        "(message):\n"
        "  Could NOT find Python (missing: Python_INCLUDE_DIRS Development.Module)\n"
        '  (found version "3.14.3")\n',
        tmp_path,
    )
    # Pinned to the interpreter running the build, not the version the error names.
    major, minor, micro = sys.version_info[:3]
    include = Path(overrides["Python_INCLUDE_DIR"])
    assert include.name == f"python{major}.{minor}"
    assert f'#define PY_VERSION "{major}.{minor}.{micro}"' in (
        include / "patchlevel.h").read_text()
    assert overrides["Python3_LIBRARY"].endswith(
        (f"libpython{major}.{minor}.so", f"libpython{major}.{minor}.dylib"))
    assert "Python_INCLUDE_DIRS" not in overrides   # FindPython's result, not its input


# -- what a fetched package's configure needs --------------------------------


@needs_cmake
def test_a_config_mode_package_is_stubbed_with_the_targets_it_is_linked_by(tmp_path):
    """cantera's Boost: config mode, REQUIRED, then linked as Boost::headers."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        "find_package(WirBoost CONFIG REQUIRED)\n"
        "add_library(demo INTERFACE)\n"
        "target_link_libraries(demo INTERFACE WirBoost::headers)\n"
        "add_custom_target(uses ALL)\n"
        "add_dependencies(uses demo)\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["WirBoost"]


def _cmake_major():
    out = subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout
    return int(out.split()[2].split(".")[0])


@needs_cmake
def test_an_old_cmake_minimum_is_the_hosts_cmake_not_a_dependency(tmp_path):
    """CMake 4 refuses cmake_minimum_required(VERSION 2.8); Debian 13's 3.31 does not."""
    if _cmake_major() < 4:
        pytest.skip("this CMake still accepts old minimums")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(demo NONE)\nadd_subdirectory(old)\n"
    )
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 2.8)\nproject(old NONE)\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == []
    assert "CMAKE_POLICY_VERSION_MINIMUM" in result.unblocked
    assert any("old/ subdirectory" in note for note in result.notes)
    assert result.round_blockers == ["CMake < 3.5", None]


@needs_cmake
def test_what_the_build_made_for_itself_can_be_found(tmp_path):
    """SUNDIALS builds a library in its build tree and then looks for it there."""
    (tmp_path / "vendored").mkdir()
    (tmp_path / "vendored" / "vendored.h").write_text("")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(demo NONE)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/made/libflib.a" "")\n'
        'find_library(FLIB flib "${CMAKE_BINARY_DIR}/made" NO_DEFAULT_PATH)\n'
        'find_path(VENDORED vendored.h PATHS "${CMAKE_SOURCE_DIR}/vendored" NO_DEFAULT_PATH)\n'
        'if(NOT FLIB OR NOT VENDORED)\n  message(FATAL_ERROR "${FLIB} ${VENDORED}")\nendif()\n'
        "find_library(Z_LIB NAMES z c System)\n"
        'if(Z_LIB)\n  message(FATAL_ERROR "found ${Z_LIB} on the host")\nendif()\n'
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == []


_IMPORTS = (
    "cmake_minimum_required(VERSION 3.18)\n"
    "project(demo NONE)\n"
    "find_package(Python REQUIRED COMPONENTS Interpreter)\n"
    'execute_process(COMMAND "${Python_EXECUTABLE}" -c\n'
    '  "import wirtestmod; print(wirtestmod.get_include())"\n'
    "  OUTPUT_VARIABLE OUT COMMAND_ERROR_IS_FATAL ANY)\n"
)


@needs_cmake
def test_what_a_configure_imports_with_no_plan_is_stubbed(tmp_path):
    """ml-dtypes asks its interpreter for numpy's include directory."""
    (tmp_path / "CMakeLists.txt").write_text(_IMPORTS)
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.python_stubbed == ["wirtestmod (no plan says which version)"]
    assert result.round_blockers[0] == "import wirtestmod"


@needs_cmake
def test_what_a_configure_imports_is_installed_at_the_plans_version(tmp_path, monkeypatch):
    from will_it_riscv import hostpython

    def install(spec, site, cache, wheels_only=False):
        (site / "wirtestmod.py").write_text("def get_include():\n    return '/inc'\n")
        return True

    monkeypatch.setattr(hostpython, "_pip_install", install)
    (tmp_path / "CMakeLists.txt").write_text(
        _IMPORTS + 'if(NOT OUT MATCHES "/inc")\n  message(FATAL_ERROR "got ${OUT}")\nendif()\n'
    )
    result = run(tmp_path, timeout=120, python_dists={"wirtestmod": "1.0"})
    assert result.completed, result.error
    assert result.python_installed == ["wirtestmod==1.0"]
    assert result.python_stubbed == []


@needs_cmake
def test_a_command_a_stubbed_package_would_define_is_stubbed_too(tmp_path):
    """sundials4py calls nanobind_add_module, which nanobind's own config defines."""
    (tmp_path / "ext.cpp").write_text("")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(demo CXX)\n"
        "find_package(wirnano CONFIG REQUIRED)\n"
        "wirnano_add_module(demo_ext NB_STATIC ext.cpp)\n"
        "target_compile_definitions(demo_ext PRIVATE X=1)\n"
        "wirnano_add_stub(demo_ext_stub MODULE demo_ext)\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed, result.error
    assert result.blockers == ["wirnano"]


@needs_cmake
def test_a_linker_flag_check_is_answered_as_the_link_would(tmp_path):
    """Kokkos checks -DKOKKOS_DEPENDENCE with check_linker_flag. A compile check
    of a static library hands it to ar, which refuses it; the link would not."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.22)\n"
        "project(t CXX)\n"
        "find_package(ZLIB QUIET)\n"   # asked something: no fallback to a host configure
        "include(CheckLinkerFlag)\n"
        'check_linker_flag(CXX "-DKOKKOS_DEPENDENCE" OK)\n'
        "if(NOT OK)\n"
        '  message(FATAL_ERROR "The linker for CXX can not consume flags")\n'
        "endif()\n"
    )
    result = run(tmp_path, timeout=120)
    assert result.completed and result.platform == "linux/riscv64"
