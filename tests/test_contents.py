"""Which package ships what a build looks for, from the archive's file lists."""

from will_it_riscv.contents import ContentsIndex

CONTENTS = [
    "usr/lib/riscv64-linux-gnu/cmake/TBB/TBBConfig.cmake    libdevel/libtbb-dev\n",
    "usr/lib/riscv64-linux-gnu/pkgconfig/gmodule-2.0.pc    libdevel/libgio-2.0-dev\n",
    "usr/include/zlib.h    libdevel/zlib1g-dev\n",
    "usr/lib/riscv64-linux-gnu/libz.so    libdevel/zlib1g-dev\n",
    "usr/lib/riscv64-linux-gnu/libz.so.1    libs/zlib1g\n",
    "usr/bin/swig    interpreters/swig\n",
    "usr/share/doc/tbb/README    doc/tbb-doc\n",
]
PACKAGES = (
    "Package: libgio-2.0-dev\nSource: glib2.0\nDepends: libc6\n\n"
    "Package: libglib2.0-dev\nSource: glib2.0 (2.86-1)\nDepends: libgio-2.0-dev (= 2.86-1)\n\n"
    "Package: libunrelated-dev\nSource: other\nDepends: libgio-2.0-dev\n"
)


def index():
    built = ContentsIndex()
    built.read_contents(CONTENTS)
    built.read_packages(PACKAGES)
    return built


def test_lookups_resolve_the_way_cmake_searches():
    contents = index()
    assert contents.providers("TBB") == (("libtbb-dev",), "ships TBBConfig.cmake")
    headers = [("find_path", "ZLIB_INCLUDE_DIR", ("zlib.h",))]
    assert contents.providers("ZLIB", "find_package", headers)[0] == ("zlib1g-dev",)
    libraries = [("find_library", "ZLIB_LIBRARY", ("z", "zlib"))]
    assert contents.providers("ZLIB", "find_package", libraries)[0] == ("zlib1g-dev",)
    assert contents.providers("swig", "find_program")[0] == ("swig",)
    assert contents.providers("Nope") == ((), "")


def test_the_dev_package_people_know_is_accepted_too():
    packages, how = index().providers("gmodule-2.0", "pkg_check_modules")
    assert packages == ("libgio-2.0-dev", "libglib2.0-dev") and how == "ships gmodule-2.0.pc"


def test_the_index_survives_the_cache():
    again = ContentsIndex.from_json(index().to_json())
    assert again.providers("TBB")[0] == ("libtbb-dev",)
    assert again.providers("gmodule-2.0", "pkg_check_modules")[0][1] == "libglib2.0-dev"


def test_the_package_the_others_pull_in_comes_first():
    contents = ContentsIndex()
    contents.read_contents(
        [
            "usr/lib/riscv64-linux-gnu/cmake/Qt5/Qt5Config.cmake    libdevel/libqt5opengl5-dev,libdevel/qtbase5-dev\n",
            "usr/bin/doxygen    devel/doxygen\n",
        ]
    )
    contents.read_packages(
        "Package: libqt5opengl5-dev\nDepends: qtbase5-dev (= 5.15)\n\nPackage: qtbase5-dev\nDepends: libc6\n"
    )
    assert contents.providers("Qt5")[0] == ("qtbase5-dev", "libqt5opengl5-dev")
    assert contents.providers("Doxygen") == (("doxygen",), "ships doxygen")
