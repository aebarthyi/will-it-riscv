import pytest

from will_it_riscv.syslibs import database, normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ssl", "openssl"),
        ("OpenSSL", "openssl"),
        ("OpenSSL::SSL", "openssl"),
        ("libcrypto", "openssl"),
        ("z", "zlib"),
        ("libpng16", "libpng"),
        ("png >= 1.6", "libpng"),
        ("-ljpeg", "libjpeg"),
        ("PkgConfig::LIBXML2", "libxml2"),
        ("avcodec", "ffmpeg"),
        ("scipy-openblas", "openblas"),
        ("qhull_r", "qhull"),
    ],
)
def test_curated_lookup(raw, expected):
    req = database().lookup(raw)
    assert req is not None and req.name == expected
    assert not database().is_guess(req)


def test_namespaced_target_prefers_the_namespace():
    """Boost::python is boost, not the ignored 'python'."""
    req = database().lookup("Boost::python")
    assert req is not None and req.name == "boost"


@pytest.mark.parametrize(
    "raw",
    ["Python3", "Threads", "pybind11", "PkgConfig", "atomic", "gcc_s", "ws2_32",
     "Accelerate", "nvcc", "java", "GTest"],
)
def test_ignored(raw):
    assert database().lookup(raw) is None


def test_unknown_is_guessed_and_flagged():
    req = database().lookup("frobnicator")
    assert req is not None
    assert req.debian == ("libfrobnicator-dev",)
    assert database().is_guess(req)


def test_tools_guess_without_the_lib_prefix():
    req = database().lookup("someunknowntool", kind="tool")
    assert req.debian == ("someunknowntool",)


@pytest.mark.parametrize(
    "include, expected",
    [
        ("jpeglib.h", "libjpeg"),
        ("zlib.h", "zlib"),
        ("openssl/ssl.h", "openssl"),
        ("openssl/evp.h", "openssl"),
        ("ft2build.h", "freetype"),
        ("freetype/ftglyph.h", "freetype"),
        ("webp/encode.h", "libwebp"),
        ("libpq-fe.h", "libpq"),
        ("./zlib.h", "zlib"),
        ("unicode/uchar.h", "icu"),
    ],
)
def test_header_mapping(include, expected):
    assert database().header(include) == expected


@pytest.mark.parametrize("include", ["stdio.h", "Python.h", "string.h", "numpy/arrayobject.h"])
def test_unknown_headers_are_not_mapped(include):
    assert database().header(include) is None


def test_normalize_strips_decoration():
    assert normalize("  'OpenBLAS >= 0.3.20' ") == "openblas"
