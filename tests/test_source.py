from pathlib import Path

from will_it_riscv.source import (
    check_submodules,
    discover_manifests,
    find_bundled,
    inspect_repository,
    iter_directory,
    own_names,
)


def build_repo(root: Path, files: dict) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def libs(inspection):
    return sorted(r.name for r in inspection.profile.system_requirements
                  if r.kind == "library")


def tools(inspection):
    return sorted(r.name for r in inspection.profile.system_requirements
                  if r.kind == "tool")


def test_iter_directory_yields_relative_posix_paths(tmp_path):
    build_repo(tmp_path, {"src/a.c": "", "CMakeLists.txt": ""})
    assert sorted(p for p, _ in iter_directory(tmp_path)) == ["CMakeLists.txt", "src/a.c"]


def test_iter_directory_skips_build_output_and_vcs(tmp_path):
    build_repo(tmp_path, {
        "src/a.c": "",
        "build/generated.c": "",
        ".git/config.c": "",
        "node_modules/pkg/index.c": "",
        ".venv/lib/x.c": "",
    })
    assert [p for p, _ in iter_directory(tmp_path)] == ["src/a.c"]


def test_iter_directory_ignores_uninteresting_files(tmp_path):
    build_repo(tmp_path, {"README.md": "", "logo.png": "", "main.c": ""})
    assert [p for p, _ in iter_directory(tmp_path)] == ["main.c"]


def test_scans_build_files_at_any_depth(tmp_path):
    """The sdist depth limit would miss this; a repo is not flat."""
    build_repo(tmp_path, {
        "CMakeLists.txt": "project(demo)\n",
        "src/gromacs/fileio/CMakeLists.txt": "find_package(HDF5)\n",
        "cmake/deep/a/b/FindThing.cmake": "pkg_check_modules(X libxml-2.0)\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "hdf5" in libs(inspection)
    assert "libxml2" in libs(inspection)


def test_vendored_directories_are_not_this_projects_build(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": "project(demo)\nfind_package(ZLIB)\n",
        "src/external/someone_else/CMakeLists.txt": "find_package(OpenSSL)\n",
        "third_party/other/meson.build": "dependency('libcurl')\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert libs(inspection) == ["zlib"]


def test_find_bundled_lists_vendored_projects(tmp_path):
    build_repo(tmp_path, {
        "src/external/tinyxml2/x.cpp": "",
        "src/external/lmfit/y.c": "",
        "src/gromacs/real.cpp": "",
    })
    assert find_bundled(tmp_path) == {"tinyxml2", "lmfit"}


def test_own_names_covers_the_project_and_its_subprojects(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": "project(gromacs)\n",
        "python_packaging/gmxapi/CMakeLists.txt": "",
    })
    names = own_names(tmp_path)
    assert "gromacs" in names
    assert "gmxapi" in names


def test_self_references_are_not_reported_as_system_libraries(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": "project(gromacs)\nfind_package(ZLIB)\n",
        "python_packaging/gmxapi/CMakeLists.txt": "find_package(gromacs)\nfind_package(gmxapi)\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert libs(inspection) == ["zlib"]


def test_cmake_comments_do_not_become_library_names(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": (
            "find_library(ITT\n"
            "    NAMES libittnotify.a # We need the static library\n"
            "    HINTS ENV VTUNE_DIR)\n"
            "find_package(ZLIB)\n"
        ),
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert libs(inspection) == ["zlib"]


def test_library_filenames_are_reduced_to_names(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": "find_library(Z NAMES libz.so.1)\n",
    })
    assert libs(inspect_repository(tmp_path, scan_ci=False)) == ["zlib"]


def test_toolchain_is_implied_from_languages(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": "project(demo)\n",
        "src/a.cpp": "",
        "src/b.f90": "",
    })
    found = tools(inspect_repository(tmp_path, scan_ci=False))
    assert {"cmake", "g++", "gfortran", "ninja"} <= set(found)


def test_ci_declarations_merge_with_scraped_findings(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": "find_package(OpenSSL)\n",
        ".github/workflows/ci.yml": "run: apt-get install -y libssl-dev libhwloc-dev\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=True)
    # libssl-dev resolves backwards onto openssl rather than duplicating it.
    assert libs(inspection) == ["hwloc", "openssl"]


def test_ci_scan_can_be_disabled(tmp_path):
    build_repo(tmp_path, {
        ".github/workflows/ci.yml": "run: apt-get install -y libhwloc-dev\n",
        "CMakeLists.txt": "project(x)\n",
    })
    assert libs(inspect_repository(tmp_path, scan_ci=False)) == []


def test_discover_manifests_orders_shallowest_first(tmp_path):
    build_repo(tmp_path, {
        "docs/requirements.txt": "sphinx\n",
        "pyproject.toml": "[project]\nname='x'\n",
    })
    found = discover_manifests(tmp_path)
    assert [p.name for p in found] == ["pyproject.toml", "requirements.txt"]


def test_uninitialised_submodule_is_a_loud_warning(tmp_path):
    (tmp_path / ".gitmodules").write_text(
        '[submodule "vendor/dep"]\n\tpath = vendor/dep\n\turl = https://example.invalid\n'
    )
    (tmp_path / "vendor" / "dep").mkdir(parents=True)
    warnings = check_submodules(tmp_path)
    assert len(warnings) == 1
    assert "not checked out" in warnings[0]


def test_populated_submodule_is_silent(tmp_path):
    (tmp_path / ".gitmodules").write_text(
        '[submodule "vendor/dep"]\n\tpath = vendor/dep\n\turl = https://example.invalid\n'
    )
    (tmp_path / "vendor" / "dep").mkdir(parents=True)
    (tmp_path / "vendor" / "dep" / "CMakeLists.txt").write_text("")
    assert check_submodules(tmp_path) == []


def test_no_gitmodules_means_nothing_to_warn_about(tmp_path):
    assert check_submodules(tmp_path) == []


def test_pure_python_repo_reports_no_native_build(tmp_path):
    build_repo(tmp_path, {
        "pyproject.toml": '[project]\nname="x"\ndependencies=["click"]\n',
        "x/__init__.py": "",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert not inspection.profile.is_native
    assert inspection.manifests == [tmp_path / "pyproject.toml"]


# -- build systems beyond CMake and meson -----------------------------------


def test_makefile_linker_variables(tmp_path):
    build_repo(tmp_path, {
        "Makefile": (
            "FINAL_LIBS=-lm -lssl -lz\n"
            "LDFLAGS+=-lhwloc\n"
            "CFLAGS=-Wall\n"
        ),
        "main.c": "",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "make" in inspection.profile.build_systems
    assert libs(inspection) == ["hwloc", "openssl", "zlib"]


def test_makefile_comments_are_not_dependencies(tmp_path):
    """Redis: '# Detect libzstd via pkg-config, fall back to -lzstd'."""
    build_repo(tmp_path, {
        "Makefile": (
            "# Detect libzstd via pkg-config, fall back to -lzstd\n"
            "# if available or fall back to something else\n"
            "FINAL_LIBS=-lzstd\n"
        ),
    })
    assert libs(inspect_repository(tmp_path, scan_ci=False)) == ["zstd"]


def test_makefile_ignores_l_flags_outside_linker_variables(tmp_path):
    build_repo(tmp_path, {
        "Makefile": "help:\n\t@echo 'pass -lfoo to link'\nLIBS=-lz\n",
    })
    assert libs(inspect_repository(tmp_path, scan_ci=False)) == ["zlib"]


def test_handwritten_configure_script(tmp_path):
    """FFmpeg's DSL: require_pkg_config <feature> <module>, and -l flags."""
    build_repo(tmp_path, {
        "configure": (
            "#!/bin/sh\n"
            "require_pkg_config libx264 x264 x264.h x264_encoder_open\n"
            "check_pkg_config zlib zlib\n"
            "require gmp gmp.h mpz_export -lgmp\n"
        ),
        "main.c": "",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "configure" in inspection.profile.build_systems
    found = libs(inspection)
    assert "zlib" in found and "gmp" in found
    # The header and the symbol are arguments, not libraries.
    assert not any(name.endswith(".h") for name in found)
    assert "x264_encoder_open" not in found


def test_autoconf_generated_configure_is_skipped(tmp_path):
    """A generated configure repeats what configure.ac already said."""
    build_repo(tmp_path, {
        "configure": "#! /bin/sh\n# Generated by GNU Autoconf 2.71.\n-lnonsense\n",
        "configure.ac": "AC_CHECK_LIB([z], [deflate])\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "configure" not in inspection.profile.build_systems
    assert libs(inspection) == ["zlib"]


def test_scons_is_flagged_as_unscraped(tmp_path):
    build_repo(tmp_path, {"SConstruct": "env = Environment()\n"})
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "scons" in inspection.profile.build_systems
    assert any("not scraped" in n for n in inspection.profile.notes)


# -- CMake reading, hardened by real repositories ---------------------------


def test_cmake_bracket_comments_are_stripped(tmp_path):
    """Every Find module CMake ships opens with pages of .rst prose."""
    build_repo(tmp_path, {
        "cmake/FindIconv.cmake": (
            "#[=======================================================[.rst:\n"
            "FindIconv\n"
            "---------\n"
            "These functions might be provided in the regular C library or\n"
            "externally in the form of an additional library.\n"
            "]=======================================================]\n"
            "find_package(ZLIB)\n"
        ),
    })
    assert libs(inspect_repository(tmp_path, scan_ci=False)) == ["zlib"]


def test_cmake_variable_names_are_not_libraries(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": (
            "find_library(BLAS_LIB NAMES openblas HINTS ${OpenBLAS_HOME} "
            "PATHS ${VMD_PATHS} PATH_SUFFIXES lib64 lib)\n"
        ),
    })
    assert libs(inspect_repository(tmp_path, scan_ci=False)) == ["openblas"]


def test_soname_digits_are_only_stripped_when_safe(tmp_path):
    """libpng16 is libpng, but dc1394 is not "dc" and x264 is not "x"."""
    build_repo(tmp_path, {
        "CMakeLists.txt": (
            'find_library(A NAMES "libpng16")\n'
            'find_library(B NAMES "dc1394")\n'
            'find_library(C NAMES "x264")\n'
        ),
    })
    found = libs(inspect_repository(tmp_path, scan_ci=False))
    assert "libpng" in found
    assert "dc1394" in found and "dc" not in found
    assert "x264" in found and "x" not in found


def test_link_variant_suffixes_collapse(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": 'find_library(A NAMES zstd_static zstd)\n',
    })
    assert libs(inspect_repository(tmp_path, scan_ci=False)) == ["zstd"]


def test_vendored_ci_is_not_this_projects_ci(tmp_path):
    """Redis vendors hiredis; hiredis's workflow builds hiredis."""
    build_repo(tmp_path, {
        "Makefile": "LIBS=-lssl\n",
        ".github/workflows/ci.yml": "run: apt-get install -y libhwloc-dev\n",
        "deps/hiredis/.github/workflows/build.yml":
            "run: apt-get install -y libevent-dev\n",
    })
    found = libs(inspect_repository(tmp_path, scan_ci=True))
    assert "hwloc" in found and "openssl" in found
    assert "libevent" not in found


def test_test_packages_are_kept_out_of_the_build_set(tmp_path):
    from will_it_riscv.models import Analysis

    build_repo(tmp_path, {
        "CMakeLists.txt": "find_package(ZLIB)\n",
        ".github/workflows/ci.yml":
            "run: apt-get install -y libssl-dev valgrind tclx doxygen\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=True)
    analysis = Analysis(target="t", root="r", python_version="3.12")
    analysis.project_requirements = {
        r.name: r for r in inspection.profile.system_requirements
    }
    assert {"zlib", "openssl"} <= set(analysis.all_system_requirements("build"))
    assert set(analysis.all_system_requirements("test")) == {"valgrind", "tclx"}
    assert set(analysis.all_system_requirements("docs")) == {"doxygen"}
    assert "valgrind" not in analysis.all_system_requirements("build")


# -- optional dependencies --------------------------------------------------


def optional_names(inspection):
    return sorted(r.name for r in inspection.profile.system_requirements if r.optional)


def required_names(inspection):
    return sorted(
        r.name for r in inspection.profile.system_requirements
        if not r.optional and r.kind == "library"
    )


def test_cmake_gated_backends_are_optional(tmp_path):
    """AdaptiveCpp's shape: the backend switch defaults to autodetection."""
    build_repo(tmp_path, {
        "CMakeLists.txt": (
            "project(demo)\n"
            "find_package(CUDA QUIET)\n"
            'set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "CUDA support")\n'
            'option(WITH_VULKAN "Vulkan support" OFF)\n'
            "find_package(ZLIB REQUIRED)\n"
            "if(WITH_CUDA_BACKEND)\n  find_package(HDF5 REQUIRED)\nendif()\n"
            "if(WITH_VULKAN)\n  find_package(OpenSSL REQUIRED)\nendif()\n"
        ),
        "main.cpp": "",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "zlib" in required_names(inspection)
    assert "hdf5" not in required_names(inspection)
    assert "openssl" not in required_names(inspection)
    assert {"hdf5", "openssl"} <= set(optional_names(inspection))


def test_options_gate_across_files(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": 'option(WITH_EXTRA "" OFF)\nfind_package(ZLIB REQUIRED)\n',
        "src/CMakeLists.txt": "if(WITH_EXTRA)\n  find_package(HDF5)\nendif()\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "zlib" in required_names(inspection)
    assert "hdf5" in optional_names(inspection)


def test_unknown_gate_keeps_a_dependency_required(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": "if(SOMETHING_UNKNOWABLE)\n  find_package(ZLIB)\nendif()\n",
    })
    assert "zlib" in required_names(inspect_repository(tmp_path, scan_ci=False))


def test_meson_required_false_is_optional(tmp_path):
    build_repo(tmp_path, {
        "meson.build": (
            "project('demo', 'c')\n"
            "zlib = dependency('zlib')\n"
            "curl = dependency('libcurl', required: false)\n"
        ),
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "zlib" in required_names(inspection)
    assert "libcurl" in optional_names(inspection)


def test_configure_enable_gates_are_optional(tmp_path):
    """FFmpeg: `enabled libx264 && require_pkg_config libx264 x264 ...`."""
    build_repo(tmp_path, {
        "configure": (
            "#!/bin/sh\n"
            "require_pkg_config zlib zlib\n"
            "enabled libx264 && require_pkg_config libx264 x264\n"
        ),
        "main.c": "",
    })
    inspection = inspect_repository(tmp_path, scan_ci=False)
    assert "zlib" in required_names(inspection)
    x264 = next(r for r in inspection.profile.system_requirements if r.name == "x264")
    assert x264.optional and x264.gate == "--enable-libx264"


def test_accelerator_packages_from_ci_are_optional(tmp_path):
    """A CI job installing ROCm builds a backend nobody enables by default."""
    build_repo(tmp_path, {
        "CMakeLists.txt": "find_package(ZLIB REQUIRED)\n",
        ".github/workflows/ci.yml":
            "run: apt-get install -y libssl-dev rocm-dev nvidia-cuda-toolkit\n",
    })
    inspection = inspect_repository(tmp_path, scan_ci=True)
    assert {"zlib", "openssl"} <= set(required_names(inspection))
    assert {"rocm-dev", "nvidia-cuda-toolkit"} <= set(optional_names(inspection))


def test_required_anywhere_beats_optional_elsewhere(tmp_path):
    build_repo(tmp_path, {
        "CMakeLists.txt": (
            'option(EXTRA "" OFF)\n'
            "if(EXTRA)\n  find_package(ZLIB)\nendif()\n"
            "find_package(ZLIB REQUIRED)\n"
        ),
    })
    assert "zlib" in required_names(inspect_repository(tmp_path, scan_ci=False))
