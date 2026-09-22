from conftest import make_sdist

from will_it_riscv.sdist import inspect_sdist


def names(profile, kind="library"):
    return sorted(r.name for r in profile.system_requirements if r.kind == kind)


def test_pure_python_sdist_is_not_native():
    blob = make_sdist("purelib", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["hatchling"]\n'
                          'build-backend="hatchling.build"\n',
        "purelib/__init__.py": "",
        "PKG-INFO": "Metadata-Version: 2.1\nName: purelib\nVersion: 1.0\n"
                    "Requires-Dist: click\n\n",
    })
    result = inspect_sdist(blob, "purelib-1.0.tar.gz")
    assert result.profile.inspected
    assert not result.profile.is_native
    assert result.profile.build_backend == "hatchling.build"
    assert result.requires_dist == ["click"]
    assert result.metadata_source == "PKG-INFO"


def test_c_extension_with_libraries_kwarg():
    blob = make_sdist("speedy", "2.0", {
        "setup.py": 'from setuptools import setup, Extension\n'
                    'setup(ext_modules=[Extension("s", ["s.c"], libraries=["ssl", "z"])])\n',
        "s.c": "#include <stdio.h>\n",
    })
    profile = inspect_sdist(blob, "speedy-2.0.tar.gz").profile
    assert profile.is_native
    assert "c" in profile.languages
    assert names(profile) == ["openssl", "zlib"]
    assert "gcc" in names(profile, "tool")
    assert "python3-dev" in names(profile, "tool")


def test_includes_reveal_libraries_a_bespoke_setup_py_hides():
    """The Pillow case: the library list is assembled at runtime, never literal."""
    blob = make_sdist("imglib", "1.0", {
        "setup.py": "from setuptools import setup\nsetup()\n",
        "src/_img.c": '#include "Python.h"\n#include <jpeglib.h>\n'
                      '#include <freetype/ftglyph.h>\n#include <webp/encode.h>\n',
    })
    profile = inspect_sdist(blob, "imglib-1.0.tar.gz").profile
    assert names(profile) == ["freetype", "libjpeg", "libwebp"]


def test_vendored_build_system_is_not_this_packages_build_system():
    """numpy vendors Google Highway's CMakeLists; numpy builds with meson."""
    blob = make_sdist("np", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["meson-python"]\n'
                          'build-backend="mesonpy"\n',
        "meson.build": "project('np', 'c')\ndependency('openblas')\n",
        "np/_core/src/highway/CMakeLists.txt": "find_package(OpenSSL)\nfind_package(Boost)\n",
        "np/_core/src/f.c": "#include <stdio.h>\n",
    })
    profile = inspect_sdist(blob, "np-1.0.tar.gz").profile
    assert profile.build_systems == {"meson"}
    assert names(profile) == ["openblas"]


def test_meson_test_corpus_is_skipped():
    """A vendored meson ships thousands of deliberately bogus dependencies."""
    blob = make_sdist("np", "1.0", {
        "meson.build": "project('np', 'c')\ndependency('zlib')\n",
        "vendored-meson/meson/test cases/common/1 trivial/meson.build":
            "dependency('definitely-doesnt-exist')\n",
        "vendored-meson/meson/test cases/frameworks/meson.build":
            "dependency('totally_made_up_dep')\n",
    })
    profile = inspect_sdist(blob, "np-1.0.tar.gz").profile
    assert names(profile) == ["zlib"]


def test_windows_only_build_files_are_skipped():
    """Pillow's winbuild/fribidi.cmake does not mean Pillow needs CMake on Linux."""
    blob = make_sdist("pil", "1.0", {
        "setup.py": "from setuptools import setup\nsetup()\n",
        "src/_i.c": "#include <zlib.h>\n",
        "winbuild/fribidi.cmake": "find_package(OpenSSL)\n",
    })
    profile = inspect_sdist(blob, "pil-1.0.tar.gz").profile
    assert "cmake" not in profile.build_systems
    assert names(profile) == ["zlib"]


def test_rust_crate_below_the_root_is_found():
    """cryptography keeps its crate at src/rust/Cargo.toml."""
    blob = make_sdist("crypt", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["setuptools","setuptools-rust"]\n',
        "src/rust/Cargo.toml": '[package]\nname="c"\n\n[dependencies]\n'
                               'openssl-sys = "0.9"\n',
    })
    profile = inspect_sdist(blob, "crypt-1.0.tar.gz").profile
    assert "rust" in profile.languages
    assert "cargo" in profile.build_systems
    assert "openssl" in names(profile)
    assert "cargo" in names(profile, "tool")


def test_vendored_cargo_dir_is_skipped():
    blob = make_sdist("crate", "1.0", {
        "Cargo.toml": '[package]\nname="c"\n\n[dependencies]\nlibz-sys = "1"\n',
        "vendor/other/Cargo.toml": '[package]\nname="o"\n\n[dependencies]\n'
                                   'openssl-sys = "0.9"\n',
    })
    profile = inspect_sdist(blob, "crate-1.0.tar.gz").profile
    assert names(profile) == ["zlib"]


def test_cmake_and_meson_scrapers():
    blob = make_sdist("cm", "1.0", {
        "pyproject.toml": '[build-system]\nbuild-backend="scikit_build_core.build"\n'
                          'requires=["scikit-build-core"]\n',
        "CMakeLists.txt": (
            "find_package(ZLIB REQUIRED)\n"
            "pkg_check_modules(LXML REQUIRED libxml-2.0)\n"
            "find_library(M_LIB NAMES png PATHS /usr/lib)\n"
            "find_package(Python3 COMPONENTS Development.Module)\n"
            "find_package(${SOME_VAR})\n"
        ),
        "src/m.cpp": "int main(){}\n",
    })
    profile = inspect_sdist(blob, "cm-1.0.tar.gz").profile
    assert profile.build_systems == {"cmake"}
    assert names(profile) == ["libpng", "libxml2", "zlib"]
    assert "cmake" in names(profile, "tool") and "ninja" in names(profile, "tool")


def test_autoconf_scraper():
    blob = make_sdist("ac", "1.0", {
        "configure.ac": "AC_CHECK_LIB([pq], [PQconnectdb])\n"
                        "PKG_CHECK_MODULES([X], [libcurl >= 7.0])\n",
        "s.c": "int main(){}\n",
    })
    profile = inspect_sdist(blob, "ac-1.0.tar.gz").profile
    assert profile.build_systems == {"autotools"}
    assert names(profile) == ["libcurl", "libpq"]


def test_pep725_external_table_is_authoritative():
    blob = make_sdist("ext", "1.0", {
        "pyproject.toml": (
            '[build-system]\nrequires=["meson-python"]\nbuild-backend="mesonpy"\n'
            "[external]\n"
            'build-requires = ["pkg:generic/cmake"]\n'
            'host-requires = ["pkg:generic/openssl", "pkg:generic/zlib"]\n'
        ),
    })
    profile = inspect_sdist(blob, "ext-1.0.tar.gz").profile
    assert names(profile) == ["openssl", "zlib"]
    assert "cmake" in names(profile, "tool")


def test_build_requires_imply_toolchain():
    blob = make_sdist("cy", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["setuptools","Cython>=3","cffi"]\n',
    })
    profile = inspect_sdist(blob, "cy-1.0.tar.gz").profile
    assert {"cython", "c"} <= profile.languages
    assert "libffi" in names(profile)


def test_tests_directory_does_not_make_a_package_native():
    blob = make_sdist("pure", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["flit_core"]\n'
                          'build-backend="flit_core.buildapi"\n',
        "pure/__init__.py": "",
        "tests/fixtures/sample.c": "#include <openssl/ssl.h>\n",
        "docs/example.c": "int main(){}\n",
    })
    profile = inspect_sdist(blob, "pure-1.0.tar.gz").profile
    assert not profile.is_native
    assert profile.system_requirements == []


def test_project_table_is_the_metadata_fallback():
    blob = make_sdist("meta", "1.0", {
        "pyproject.toml": (
            '[project]\nname="meta"\nversion="1.0"\nrequires-python=">=3.9"\n'
            'dependencies = ["requests>=2", "click"]\n'
            "[project.optional-dependencies]\n"
            'fast = ["orjson"]\n'
        ),
    })
    result = inspect_sdist(blob, "meta-1.0.tar.gz")
    assert result.metadata_source == "pyproject.toml"
    assert "requests>=2" in result.requires_dist
    assert result.requires_python == ">=3.9"
    assert "fast" in result.provides_extra
    assert any("extra ==" in d and "orjson" in d for d in result.requires_dist)


def test_corrupt_archive_is_reported_not_raised():
    result = inspect_sdist(b"this is not a tarball", "bad-1.0.tar.gz")
    assert not result.profile.inspected
    assert result.profile.notes
