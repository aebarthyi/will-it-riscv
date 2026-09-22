"""Static inspection of a source distribution.

We never run a build. Instead we read the archive and answer two questions:

1. Does this package compile anything? (extensions, Rust, Fortran, CMake...)
2. If so, what does it need from the system that pip cannot provide?

Everything here is inference, and the caller is expected to say so in its
output. It is still far better than the status quo, which is running
``pip install`` on the target and reading the tail of a traceback.
"""

from __future__ import annotations

import io
import posixpath
import re
import tarfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from email.parser import BytesParser
from typing import Callable, Optional

from .models import BuildProfile, Evidence, SystemRequirement
from .syslibs import database

try:  # pragma: no cover - trivial
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]

#: Source extensions that imply a compiler runs.
#: CUDA (.cu) and Objective-C++ (.mm) are deliberately absent: both are
#: conditionally compiled, and neither is ever built for a Linux riscv64
#: target, so detecting them only produces toolchain noise.
LANGUAGE_EXTENSIONS = {
    ".c": "c",
    ".h": "c",
    ".cc": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".c++": "c++",
    ".hpp": "c++",
    ".hxx": "c++",
    ".pyx": "cython",
    ".pxd": "cython",
    ".pxi": "cython",
    ".rs": "rust",
    ".f": "fortran",
    ".f77": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".for": "fortran",
    ".go": "go",
    ".zig": "zig",
}

#: Directories whose contents do not tell us how the package is built.
#: Compared after stripping spaces, hyphens and underscores, so "test cases",
#: "test-cases" and "testcases" all match -- numpy vendors meson, and meson's
#: test corpus is thousands of meson.build files full of deliberately bogus
#: dependencies.
SKIP_DIRS = {
    "test", "tests", "testing", "testcases", "testsuite", "manualtests",
    "unittests", "regressiontests",
    "doc", "docs", "documentation",
    "example", "examples", "demo", "demos", "sample", "samples",
    "benchmark", "benchmarks", "bench",
    "github", "gitlab", "nodemodules", "pycache", "fixtures",
    # Vendored copies of build tools, and meson wrap fallbacks: a subproject
    # exists precisely so the build works *without* the system package.
    "vendoredmeson", "vendoredninja", "subprojects", "mesonwrap",
    # Build support for platforms that are not the target. Pillow's
    # winbuild/fribidi.cmake is not evidence that Pillow needs CMake on Linux.
    "winbuild", "msvc", "vsprojects", "visualstudio", "xcode",
    "android", "ios", "emscripten", "wasm", "pyodide",
}

#: Build backends whose presence settles the question on its own.
BACKEND_SIGNALS = {
    "maturin": ({"rust"}, {"cargo"}),
    "setuptools_rust": ({"rust"}, {"cargo"}),
    "scikit_build_core": (set(), {"cmake"}),
    "scikit_build": (set(), {"cmake"}),
    "mesonpy": (set(), {"meson"}),
    "meson_python": (set(), {"meson"}),
    "py_build_cmake": (set(), {"cmake"}),
    "cmeel": (set(), {"cmake"}),
}

#: Backends that build pure-Python wheels and nothing else.
PURE_BACKENDS = {
    "flit_core", "flit", "hatchling", "poetry", "poetry_core", "pdm",
    "pdm_backend", "pdm_pep517", "whey", "uv_build", "enscons",
}

#: PyPI build requirements that imply a toolchain or language.
BUILD_REQUIRE_SIGNALS = {
    "cython": ({"cython", "c"}, set(), ()),
    "cython3": ({"cython", "c"}, set(), ()),
    "cffi": ({"c"}, set(), ("libffi",)),
    "pybind11": ({"c++"}, set(), ()),
    "nanobind": ({"c++"}, set(), ()),
    "numpy": ({"c"}, set(), ()),
    "oldest-supported-numpy": ({"c"}, set(), ()),
    "setuptools-rust": ({"rust"}, {"cargo"}, ()),
    "maturin": ({"rust"}, {"cargo"}, ()),
    "cmake": (set(), {"cmake"}, ()),
    "ninja": (set(), {"ninja"}, ()),
    "meson": (set(), {"meson"}, ()),
    "meson-python": (set(), {"meson"}, ()),
    "scikit-build": (set(), {"cmake"}, ()),
    "scikit-build-core": (set(), {"cmake"}, ()),
    "swig": (set(), set(), ("swig",)),
    "pkgconfig": (set(), set(), ("pkg-config",)),
}

#: Tools each build system needs present on the machine doing the build.
BUILD_SYSTEM_TOOLS = {
    "cmake": ("cmake", "ninja"),
    "meson": ("meson", "ninja"),
    "autotools": ("autoconf", "make"),
    "cargo": ("cargo",),
    "make": ("make",),
    "scons": ("scons",),
}

#: Extra tools implied by a language.
LANGUAGE_TOOLS = {
    "c": ("gcc",),
    "c++": ("g++",),
    "cython": ("gcc",),
    "fortran": ("gfortran",),
    "rust": ("cargo",),
    "swig": ("swig",),
}

MAX_TEXT_BYTES = 256 * 1024
MAX_MEMBERS = 30000

#: Includes live at the top of a file, and a big sdist has thousands of them.
MAX_HEADER_FILES = 400
MAX_HEADER_BYTES = 8 * 1024

#: How deep a build-system file can sit and still describe *this* package's
#: build. numpy vendors Google Highway, which ships its own CMakeLists.txt --
#: but numpy builds it with meson, so that CMake file describes nobody's build.
#: The PEP 517 backend drives the build system at the root; anything deeper
#: belongs to vendored code, whose real needs the #include scan already finds.
#:
#: This is a proxy for "not vendored" that works because sdists are flat. A
#: source repository is not flat -- GROMACS spreads 245 CMake files over six
#: levels -- so repo scanning names the vendored directories outright instead
#: (see :data:`VENDORED_DIRS`) and lifts the depth limit.
MAX_BUILD_CONFIG_DEPTH = 1

#: Directories holding somebody else's source, bundled into this tree. A build
#: file here configures the bundled copy, and a project that bundles a library
#: usually does so precisely so it does *not* need the system one.
VENDORED_DIRS = frozenset({
    "external", "externals", "thirdparty", "3rdparty", "extern", "deps",
    "vendor", "vendored", "contrib", "bundled", "submodules", "subprojects",
    "importedcode",
})

#: Cargo is the exception: a Cargo.toml inside a Python sdist is the project's
#: own crate, conventionally at src/rust/, not someone else's build system.
MAX_CARGO_DEPTH = 3

#: Extensions worth reading ``#include`` lines out of.
_INCLUDE_SCAN_EXTENSIONS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hxx", ".pyx"}

#: A real library or tool name. Rejects ``${VAR}``, paths, and CMake noise.
_PLAUSIBLE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_+.:-]{0,39}$")


#: libfoo.so.1 / libfoo.a / foo.dylib -- a filename, not a bare library name.
_LIBRARY_FILENAME = re.compile(
    r"^(?:lib)?(?P<stem>[A-Za-z][A-Za-z0-9_+.-]*?)"
    r"(?:\.so(?:\.\d+)*|\.a|\.dylib|\.dll|\.lib)$"
)


def strip_library_filename(raw: str) -> str:
    """``libittnotify.a`` -> ``ittnotify``; anything else is returned as-is."""
    match = _LIBRARY_FILENAME.match(raw.strip().strip("\"'"))
    return match.group("stem") if match else raw


def _is_plausible_name(raw: str) -> bool:
    name = strip_library_filename(raw)
    if not _PLAUSIBLE_NAME.match(name):
        return False
    return not any(c in name for c in "$/\\{}")


_CMAKE_KEYWORDS = {
    "names", "paths", "hints", "required", "quiet", "no_default_path", "optional",
    "path_suffixes", "doc", "components", "config", "module", "global", "static",
    "no_module", "exact", "name", "no_cmake_path", "imported_target",
    "no_cmake_system_path", "env", "registry_view", "validator", "namespaces",
    "no_package_root_path", "no_cmake_environment_path", "cmake_find_root_path_both",
    "only_cmake_find_root_path", "no_cmake_find_root_path", "no_system_environment_path",
    "no_cmake_builds_path", "no_cmake_install_prefix", "no_cmake_package_registry",
    # values of PATH_SUFFIXES, which name directories rather than libraries
    "lib", "lib64", "lib32", "bin", "include", "share", "usr", "local",
}


@dataclass
class SdistInspection:
    """What we learned from one sdist."""

    profile: BuildProfile = field(default_factory=BuildProfile)
    requires_dist: list[str] = field(default_factory=list)
    provides_extra: list[str] = field(default_factory=list)
    requires_python: Optional[str] = None
    metadata_source: str = "none"


# --------------------------------------------------------------- archives


def _iter_archive(blob: bytes, filename: str) -> Iterator[tuple[str, Callable[[], bytes]]]:
    """Yield ``(path_without_top_dir, read)`` for every regular file."""
    lower = filename.lower()
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(blob))
        names = zf.namelist()[:MAX_MEMBERS]
        prefix = _common_prefix(names)
        for name in names:
            if name.endswith("/"):
                continue
            yield _strip(name, prefix), _zip_reader(zf, name)
        return

    # "r:*" lets tarfile sniff gzip/bzip2/xz rather than trusting the suffix.
    tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
    count = 0
    names = []
    members = []
    for member in tf:
        if not member.isfile():
            continue
        members.append(member)
        names.append(member.name)
        count += 1
        if count >= MAX_MEMBERS:
            break
    prefix = _common_prefix(names)
    for member in members:
        yield _strip(member.name, prefix), _tar_reader(tf, member)


def _zip_reader(zf: zipfile.ZipFile, name: str) -> Callable[[], bytes]:
    def read() -> bytes:
        with zf.open(name) as fh:
            return fh.read(MAX_TEXT_BYTES)

    return read


def _tar_reader(tf: tarfile.TarFile, member: tarfile.TarInfo) -> Callable[[], bytes]:
    def read() -> bytes:
        fh = tf.extractfile(member)
        return fh.read(MAX_TEXT_BYTES) if fh else b""

    return read


def _common_prefix(names: list[str]) -> str:
    """sdists are required to have a single top-level directory; find it."""
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    return next(iter(tops)) if len(tops) == 1 else ""


def _strip(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix + "/"):
        return name[len(prefix) + 1 :]
    return name


def _normalize_dir(part: str) -> str:
    return re.sub(r"[\s_.-]+", "", part.lower())


def _is_skipped(path: str, skip_dirs: frozenset = frozenset()) -> bool:
    against = skip_dirs or SKIP_DIRS
    return any(_normalize_dir(part) in against for part in path.split("/")[:-1])


def _is_vendored(path: str) -> bool:
    """True if this path sits under a directory holding bundled third-party code."""
    return any(_normalize_dir(part) in VENDORED_DIRS for part in path.split("/")[:-1])


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


# -------------------------------------------------------------- inspection


@dataclass
class ScanPolicy:
    """How to interpret a tree of files.

    An sdist and a source repository need different readings of the same
    scrapers. An sdist is a flat, curated subset, so depth stands in for
    "is this ours". A repo is neither, so it names vendored directories
    outright and scans to any depth.
    """

    build_config_depth: Optional[int] = MAX_BUILD_CONFIG_DEPTH
    """Maximum depth for a build-system file, or None for no limit."""
    honour_vendored_dirs: bool = False
    """Skip build files under external/, third_party/, vendor/ and friends."""
    read_python_metadata: bool = True
    max_header_files: int = MAX_HEADER_FILES
    root_only_setup_py: bool = True

    @classmethod
    def for_sdist(cls) -> ScanPolicy:
        return cls()

    @classmethod
    def for_repository(cls) -> ScanPolicy:
        return cls(
            build_config_depth=None,
            honour_vendored_dirs=True,
            root_only_setup_py=False,
            max_header_files=MAX_HEADER_FILES * 4,
        )


class Recorder:
    """Collects system requirements, merging duplicates as they arrive.

    Two ways in. :meth:`__call__` takes a raw name scraped out of a build
    file and resolves it through the curated map -- that is inference.
    :meth:`declare` takes a requirement somebody wrote down verbatim, which
    is evidence, and is never reported as a guess.
    """

    def __init__(self, found: dict, exclude: Optional[set] = None):
        self.found = found
        self.exclude = {e.lower() for e in (exclude or set())}
        self._db = database()

    def __call__(self, raw_name: str, kind: str, source: str) -> None:
        if not _is_plausible_name(raw_name):
            return
        stem = strip_library_filename(raw_name)
        if stem.lower() in self.exclude:
            # A project referring to its own targets, not to a system library.
            return
        known = self._db.lookup(stem, kind)
        if known is None:
            return
        self.declare(
            SystemRequirement(
                name=known.name,
                kind=known.kind,
                pkgconfig=known.pkgconfig,
                debian=known.debian,
                fedora=known.fedora,
                found_in=(source,),
            )
        )

    def declare(self, requirement: SystemRequirement) -> None:
        existing = self.found.get(requirement.name)
        self.found[requirement.name] = (
            requirement.merged_with(existing) if existing else requirement
        )


def make_recorder(found: dict, exclude: Optional[set] = None) -> Recorder:
    """Build the recorder the scrapers write their findings into."""
    return Recorder(found, exclude)


def scan_members(
    members,
    result: SdistInspection,
    record,
    policy: ScanPolicy,
) -> None:
    """Apply every scraper to a sequence of ``(path, read)`` pairs.

    This is the shared core: :func:`inspect_sdist` feeds it an archive and
    :mod:`will_it_riscv.source` feeds it a directory walk. Neither the
    scrapers nor this loop know which.
    """
    profile = result.profile
    pkg_info: Optional[bytes] = None
    pyproject_raw: Optional[bytes] = None
    setup_cfg_raw: Optional[bytes] = None
    header_scans = 0

    for path, read in members:
        base = posixpath.basename(path).lower()
        depth = path.count("/")
        skipped = _is_skipped(path)
        vendored = policy.honour_vendored_dirs and _is_vendored(path)

        # -- Python metadata, root only
        if policy.read_python_metadata:
            if depth == 0:
                if base == "pkg-info" and pkg_info is None:
                    pkg_info = read()
                    continue
                if base == "pyproject.toml":
                    pyproject_raw = read()
                    continue
                if base == "setup.cfg":
                    setup_cfg_raw = read()
                    continue
            elif base == "pkg-info" and pkg_info is None and path.endswith(
                ".egg-info/PKG-INFO"
            ):
                pkg_info = read()
                continue

        # -- language detection from file extensions
        if not skipped and not vendored:
            ext = posixpath.splitext(path)[1].lower()
            lang = LANGUAGE_EXTENSIONS.get(ext)
            if lang:
                profile.languages.add(lang)
                profile.evidence.add(Evidence.SOURCE_FILES)
            # What the sources #include is the most reliable statement of what
            # they need to link against -- more so than setup.py, which often
            # assembles its library list at runtime.
            if ext in _INCLUDE_SCAN_EXTENSIONS and header_scans < policy.max_header_files:
                header_scans += 1
                _scan_includes(read()[:MAX_HEADER_BYTES], path, record)

        # -- build systems and their configuration
        deep_enough = (
            policy.build_config_depth is None or depth <= policy.build_config_depth
        )
        owns_build = not skipped and not vendored and deep_enough
        cargo_ok = (
            not skipped
            and not vendored
            and depth <= MAX_CARGO_DEPTH
            and "vendor/" not in path
        )

        if base == "cmakelists.txt" or base.endswith(".cmake"):
            if owns_build:
                profile.build_systems.add("cmake")
                profile.evidence.add(Evidence.BUILD_CONFIG)
                _scan_cmake(_decode(read()), path, record)
        elif base in ("meson.build", "meson_options.txt", "meson.options"):
            if owns_build:
                profile.build_systems.add("meson")
                profile.evidence.add(Evidence.BUILD_CONFIG)
                _scan_meson(_decode(read()), path, record, profile)
        elif base in ("configure.ac", "configure.in", "makefile.am"):
            if owns_build:
                profile.build_systems.add("autotools")
                profile.evidence.add(Evidence.BUILD_CONFIG)
                _scan_autoconf(_decode(read()), path, record)
        elif base == "cargo.toml":
            if cargo_ok:
                profile.build_systems.add("cargo")
                profile.languages.add("rust")
                profile.evidence.add(Evidence.BUILD_CONFIG)
                _scan_cargo(read(), path, record)
        elif base == "build.rs":
            if cargo_ok:
                profile.build_systems.add("cargo")
                profile.languages.add("rust")
                _scan_build_rs(_decode(read()), path, record)
        elif base == "sconstruct" and depth == 0:
            profile.build_systems.add("scons")
        elif base == "setup.py" and (depth == 0 or not policy.root_only_setup_py):
            if not vendored:
                _scan_setup_py(_decode(read()), path, record, profile)

    if policy.read_python_metadata:
        _apply_pyproject(pyproject_raw, result, record)
        _apply_setup_cfg(setup_cfg_raw, result, record)
        _apply_pkg_info(pkg_info, result)


def inspect_sdist(blob: bytes, filename: str) -> SdistInspection:
    """Read an sdist and work out how, and with what, it builds."""
    result = SdistInspection()
    result.profile.inspected = True
    found: dict[str, SystemRequirement] = {}
    record = make_recorder(found)

    try:
        members = list(_iter_archive(blob, filename))
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as exc:
        result.profile.inspected = False
        result.profile.notes.append(f"could not read sdist: {exc}")
        return result

    scan_members(members, result, record, ScanPolicy.for_sdist())
    _apply_implied_tools(result.profile, record)
    result.profile.system_requirements = sorted(
        found.values(), key=lambda r: (r.kind, r.name)
    )
    return result


# ------------------------------------------------------------- per-format


_SETUP_LIBRARIES = re.compile(r"\blibraries\s*=\s*\[([^\]]*)\]")
_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")
_DASH_L = re.compile(r"['\"]-l([A-Za-z0-9_+.-]+)['\"]")
_FIND_LIBRARY = re.compile(
    r"\bfind_library\w*\(\s*(?:self\s*,\s*)?['\"]([^'\"]+)['\"]", re.IGNORECASE
)
_PKGCONFIG_CALL = re.compile(
    r"\bpkgconfig\.(?:parse|libs|cflags|exists|installed)\(\s*['\"]([^'\"]+)['\"]"
)


def _scan_setup_py(text: str, path: str, record, profile: BuildProfile) -> None:
    if "Extension(" in text or "ext_modules" in text:
        profile.languages.add("c")
        profile.evidence.add(Evidence.BUILD_CONFIG)
    if "cythonize(" in text:
        profile.languages.add("cython")
        profile.evidence.add(Evidence.BUILD_CONFIG)
    if "setuptools_rust" in text or "RustExtension" in text:
        profile.languages.add("rust")
        profile.build_systems.add("cargo")
    for match in _SETUP_LIBRARIES.finditer(text):
        for name in _QUOTED.findall(match.group(1)):
            record(name, "library", path)
    for regex in (_DASH_L, _FIND_LIBRARY, _PKGCONFIG_CALL):
        for name in regex.findall(text):
            record(name, "library", path)


_CMAKE_FIND_PACKAGE = re.compile(r"\bfind_package\s*\(\s*([A-Za-z0-9_+.-]+)", re.IGNORECASE)
_CMAKE_PKG_CHECK = re.compile(
    r"\bpkg_(?:check|search)_modules?\s*\(\s*[A-Za-z0-9_]+([^)]*)\)", re.IGNORECASE
)
_CMAKE_FIND_LIBRARY = re.compile(
    r"\bfind_library\s*\(\s*[A-Za-z0-9_${}]+([^)]*)\)", re.IGNORECASE
)


#: A CMake comment runs from an unquoted # to end of line. Left in, the prose
#: inside a command body gets tokenised as library names.
_CMAKE_COMMENT = re.compile(r"(?<!\\)#[^\n]*")


def _scan_cmake(text: str, path: str, record) -> None:
    text = _CMAKE_COMMENT.sub("", text)
    for name in _CMAKE_FIND_PACKAGE.findall(text):
        record(name, "library", path)
    for regex in (_CMAKE_PKG_CHECK, _CMAKE_FIND_LIBRARY):
        for body in regex.findall(text):
            for token in _cmake_tokens(body):
                record(token, "library", path)


def _cmake_tokens(body: str) -> Iterator[str]:
    for token in re.split(r"[\s;]+", body.strip()):
        token = token.strip("\"'")
        if not token or token.startswith("$") or token.startswith("#"):
            continue
        if token.lower() in _CMAKE_KEYWORDS or token.isupper() and "_" in token:
            continue
        yield token


_MESON_DEPENDENCY = re.compile(r"\bdependency\s*\(\s*['\"]([^'\"]+)['\"]")
_MESON_FIND_LIBRARY = re.compile(r"\.find_library\s*\(\s*['\"]([^'\"]+)['\"]")
_MESON_PROJECT_LANGS = re.compile(r"\bproject\s*\([^)]*?\[([^\]]*)\]", re.DOTALL)
_MESON_ADD_LANGS = re.compile(r"\badd_languages\s*\(\s*([^)]*)\)")

_MESON_LANG_MAP = {"c": "c", "cpp": "c++", "c++": "c++", "fortran": "fortran", "rust": "rust",
                   "cython": "cython", "cuda": "cuda", "objc": "objective-c"}


def _scan_meson(text: str, path: str, record, profile: BuildProfile) -> None:
    for regex in (_MESON_DEPENDENCY, _MESON_FIND_LIBRARY):
        for name in regex.findall(text):
            record(name, "library", path)
    for regex in (_MESON_PROJECT_LANGS, _MESON_ADD_LANGS):
        for body in regex.findall(text):
            for name in _QUOTED.findall(body):
                lang = _MESON_LANG_MAP.get(name.strip().lower())
                if lang:
                    profile.languages.add(lang)


_AC_CHECK_LIB = re.compile(r"\bAC_CHECK_LIB\(\s*\[?([A-Za-z0-9_+.-]+)")
_AC_PKG_CHECK = re.compile(r"\bPKG_CHECK_MODULES\(\s*\[?[A-Za-z0-9_]+\]?\s*,\s*\[?([^\])]+)")
_AC_SEARCH_LIBS = re.compile(r"\bAC_SEARCH_LIBS\(\s*\[?[A-Za-z0-9_]+\]?\s*,\s*\[?([^\])]+)")


def _scan_autoconf(text: str, path: str, record) -> None:
    for name in _AC_CHECK_LIB.findall(text):
        record(name, "library", path)
    for regex in (_AC_PKG_CHECK, _AC_SEARCH_LIBS):
        for body in regex.findall(text):
            for token in re.split(r"[\s,]+", body.strip()):
                token = token.strip("[]\"'")
                if token and not token.startswith("$"):
                    record(token, "library", path)


def _scan_cargo(raw: bytes, path: str, record) -> None:
    try:
        data = tomllib.loads(_decode(raw))
    except (tomllib.TOMLDecodeError, ValueError):
        return
    for section in ("dependencies", "build-dependencies"):
        for crate in data.get(section, {}):
            name = crate.lower()
            if name == "pkg-config":
                record("pkg-config", "tool", path)
            elif name.endswith("-sys"):
                record(name[: -len("-sys")], "library", path)


_INCLUDE = re.compile(rb"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE)


def _scan_includes(raw: bytes, path: str, record) -> None:
    db = database()
    for match in _INCLUDE.findall(raw):
        library = db.header(match.decode("utf-8", errors="replace"))
        if library:
            record(library, "library", path)


_PKG_CONFIG_PROBE = re.compile(r"\bprobe(?:_library)?\s*\(\s*\"([^\"]+)\"")


def _scan_build_rs(text: str, path: str, record) -> None:
    for name in _PKG_CONFIG_PROBE.findall(text):
        record(name, "library", path)


_CFG_LIBRARIES = re.compile(r"^\s*libraries\s*=\s*(.*?)(?=^\S|\Z)", re.MULTILINE | re.DOTALL)


def _apply_setup_cfg(raw: Optional[bytes], result: SdistInspection, record) -> None:
    if raw is None:
        return
    text = _decode(raw)
    for body in _CFG_LIBRARIES.findall(text):
        for token in re.split(r"[\s,]+", body.strip()):
            if token and not token.startswith("#"):
                record(token, "library", "setup.cfg")


def _apply_pyproject(raw: Optional[bytes], result: SdistInspection, record) -> None:
    if raw is None:
        return
    profile = result.profile
    try:
        data = tomllib.loads(_decode(raw))
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        profile.notes.append(f"unparseable pyproject.toml: {exc}")
        return

    build_system = data.get("build-system", {})
    backend = build_system.get("build-backend")
    if backend:
        profile.build_backend = backend
        root = re.split(r"[.:]", backend)[0].replace("-", "_")
        langs, systems = BACKEND_SIGNALS.get(root, (set(), set()))
        if langs or systems:
            profile.languages |= langs
            profile.build_systems |= systems
            profile.evidence.add(Evidence.BACKEND)

    requires = [str(r) for r in build_system.get("requires", [])]
    profile.build_requires = requires
    for spec in requires:
        name = re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0].strip().lower()
        langs, systems, libs = BUILD_REQUIRE_SIGNALS.get(name, (set(), set(), ()))
        if langs or systems or libs:
            profile.languages |= langs
            profile.build_systems |= systems
            profile.evidence.add(Evidence.BACKEND)
            for lib in libs:
                record(lib, "library", "pyproject.toml:build-system.requires")

    # PEP 725: the package told us its external dependencies outright.
    external = data.get("external", {})
    for key in ("build-requires", "host-requires", "dependencies"):
        for purl in external.get(key, []):
            name = _purl_name(str(purl))
            if name:
                kind = "tool" if key == "build-requires" else "library"
                record(name, kind, f"pyproject.toml:[external].{key}")
                profile.evidence.add(Evidence.BUILD_CONFIG)

    project = data.get("project", {})
    if project and result.metadata_source == "none":
        deps = [str(d) for d in project.get("dependencies", [])]
        extras = project.get("optional-dependencies", {})
        for extra, specs in extras.items():
            for spec in specs:
                deps.append(f"{spec} ; extra == \"{extra}\"" if ";" not in str(spec)
                            else f"{spec} and extra == \"{extra}\"")
            result.provides_extra.append(str(extra))
        if deps or project.get("requires-python"):
            result.requires_dist = deps
            result.requires_python = project.get("requires-python")
            result.metadata_source = "pyproject.toml"


_PURL = re.compile(r"^pkg:(?:generic|github|cargo|deb|rpm)/(?:[^/]+/)?([^@?#]+)")


def _purl_name(purl: str) -> Optional[str]:
    match = _PURL.match(purl.strip())
    return match.group(1) if match else None


def _apply_pkg_info(raw: Optional[bytes], result: SdistInspection) -> None:
    if raw is None:
        return
    message = BytesParser().parsebytes(raw)
    requires = message.get_all("Requires-Dist") or []
    if requires:
        # PKG-INFO wins: it is what a real install would see.
        result.requires_dist = [str(r) for r in requires]
        result.provides_extra = [str(e) for e in (message.get_all("Provides-Extra") or [])]
        result.metadata_source = "PKG-INFO"
    elif result.metadata_source == "none":
        result.metadata_source = "PKG-INFO (no Requires-Dist)"
    if not result.requires_python:
        result.requires_python = message.get("Requires-Python")


def _apply_implied_tools(profile: BuildProfile, record) -> None:
    """Add the toolchain implied by the languages and build systems found."""
    for system in profile.build_systems:
        for tool in BUILD_SYSTEM_TOOLS.get(system, ()):
            record(tool, "tool", f"implied by {system}")
    for language in profile.languages:
        for tool in LANGUAGE_TOOLS.get(language, ()):
            record(tool, "tool", f"implied by {language} sources")
    if profile.is_native:
        record("python3-dev", "tool", "implied by building a C extension")
