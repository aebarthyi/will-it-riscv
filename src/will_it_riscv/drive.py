"""Run a project's own build driver, in the pretend environment, and watch.

MFC's ``./mfc.sh build`` installs 115 Python packages and then hands the
build to ``toolchain/main.py``, which loads its modules lazily -- only what
the command in hand needs. Whether jax is needed to *build* MFC is therefore
a question with an exact answer: run the driver, and see whether it imports
jax. Nothing is emulated, and nothing is compiled:

  the tree      the driver runs in a copy-on-write clone of the repository,
                with HOME pointed at scratch, so it can write what it likes
  build tools   cmake, make, ninja, the compilers and MPI wrappers are shims
                that record their arguments and report success
  imports       every module the driver imports is recorded; one it cannot
                import is installed for the host from what the plan resolved,
                or, if the plan does not install it, stubbed -- and the
                driver runs again, the same way a configure is unblocked

What the driver imports is what the build needs from the Python toolchain.
What it never imports came along for something else.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from packaging.utils import canonicalize_name

MAX_ROUNDS = 8

#: Build tools the driver may call, answered by a shim. None of them runs:
#: the point is what the driver asks for, not what it produces.
SHIMMED_TOOLS = (
    "cmake", "ctest", "cpack", "make", "gmake", "ninja", "meson",
    "cc", "c++", "gcc", "g++", "clang", "clang++", "gfortran", "f95", "flang",
    "mpicc", "mpicxx", "mpic++", "mpif90", "mpif77", "mpifort", "mpiexec", "mpirun",
    "nvcc", "nvfortran", "nvc", "nvc++", "ftn", "CC", "hipcc", "amdclang",
)

_VERSIONS = {
    "cmake": "cmake version 3.31.0",
    "ctest": "ctest version 3.31.0",
    "ninja": "1.12.1",
    "make": "GNU Make 4.4.1",
    "gmake": "GNU Make 4.4.1",
    "meson": "1.6.0",
}

#: Import names that are not their distribution's name.
MODULE_TO_DIST = {
    "yaml": "pyyaml", "PIL": "pillow", "skimage": "scikit-image",
    "sklearn": "scikit-learn", "cv2": "opencv-python", "dateutil": "python-dateutil",
    "attr": "attrs", "jwt": "pyjwt", "bs4": "beautifulsoup4", "Crypto": "pycryptodome",
    "OpenSSL": "pyopenssl", "git": "gitpython", "serial": "pyserial", "zmq": "pyzmq",
    "mpl_toolkits": "matplotlib", "pkg_resources": "setuptools",
    "markdown_it": "markdown-it-py", "google": "protobuf", "magic": "python-magic",
    "usb": "pyusb", "_cffi_backend": "cffi", "gi": "pygobject", "docx": "python-docx",
    "pptx": "python-pptx", "Levenshtein": "levenshtein", "mdurl": "mdurl",
}

_SHIM = '''#!{python}
import json, os, sys
tool = os.path.basename(sys.argv[0])
with open(os.environ["WIR_SHIM_LOG"], "a") as log:
    log.write(json.dumps({{"tool": tool, "argv": sys.argv[1:], "cwd": os.getcwd()}}) + "\\n")
versions = {versions!r}
if any(a in ("--version", "-version", "-v", "-V", "--showme:version") for a in sys.argv[1:]):
    print(versions.get(tool, tool + " (will-it-riscv shim) 99.9.9"))
'''

#: Runs the driver with every import recorded, and a permissive stub for any
#: module named in WIR_STUB_MODULES. Written as it goes: MFC ends by sending
#: itself SIGTERM, which no atexit handler survives.
_BOOTSTRAP = r'''
import atexit, importlib.abc, importlib.machinery, json, os, runpy, signal, sys

LOG = os.environ["WIR_IMPORT_LOG"]
STUBS = set(filter(None, os.environ.get("WIR_STUB_MODULES", "").split(",")))
missing = []


class _Anything:
    def __getattr__(self, name):
        return _Anything()
    def __call__(self, *args, **kwargs):
        return _Anything()
    def __iter__(self):
        return iter(())
    def __bool__(self):
        return False
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def __mro_entries__(self, bases):
        return (object,)


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__getattr__ = lambda name: _Anything()
        module.__path__ = []


class _Last(importlib.abc.MetaPathFinder):
    """Asked only after every real finder has failed."""
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in STUBS:
            return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
        if top not in missing:
            missing.append(top)
            _dump()
        return None


def _dump(*_):
    tops = sorted({m.split(".")[0] for m in list(sys.modules) if m})
    try:
        import importlib.metadata as md
        owners = md.packages_distributions()
    except Exception:
        owners = {}
    origins = {}
    for top in tops:
        module = sys.modules.get(top)
        origins[top] = getattr(module, "__file__", None) or ""
    with open(LOG, "w") as handle:
        json.dump({
            "modules": tops,
            "missing": missing,
            "owners": {t: owners.get(t, []) for t in tops},
            "origins": origins,
            "stdlib": sorted(getattr(sys, "stdlib_module_names", ())),
        }, handle)


def _on_term(signum, frame):
    _dump()
    os._exit(128 + signum)


sys.meta_path.append(_Last())
signal.signal(signal.SIGTERM, _on_term)
atexit.register(_dump)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
'''


@dataclass
class DriverTrace:
    """What running the build driver showed."""

    rounds: int = 0
    returncode: Optional[int] = None
    imported: set = field(default_factory=set)
    """Distributions the driver actually imported, canonically named."""
    installed: list = field(default_factory=list)
    """``name==version`` put in for the host because the driver asked for it."""
    stubbed: list = field(default_factory=list)
    """Modules the driver wanted that were stubbed, and why."""
    not_in_plan: list = field(default_factory=list)
    """Modules the driver imported that nothing the plan installs provides."""
    commands: list = field(default_factory=list)
    """Every shimmed build tool the driver called: ``(tool, argv)``."""
    made: list = field(default_factory=list)
    """Directories the driver complained were missing, made for it -- MFC's
    build/, which mfc.sh creates before handing over."""
    error: Optional[str] = None
    tail: str = ""
    duration: float = 0.0


def trace(
    root: Path,
    script: str,
    args: list[str],
    dists: dict[str, Optional[str]],
    *,
    timeout: int = 600,
    max_rounds: int = MAX_ROUNDS,
    pip_cache: Optional[Path] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> DriverTrace:
    """Run ``script args`` from a clone of ``root``, providing what it imports.

    ``dists`` maps each distribution the plan installs to the version it
    resolved to, so that what the driver imports can be put in for the host
    at the version the plan would have installed.
    """
    started = time.monotonic()
    result = DriverTrace()
    wanted: dict[str, Optional[str]] = {
        str(canonicalize_name(name)): version for name, version in dists.items()
    }
    stubs: set = set()
    handled: set = set()

    with tempfile.TemporaryDirectory(prefix="will-it-riscv-drive-") as scratch:
        scratch_path = Path(scratch)
        tree = scratch_path / "src"
        clone_tree(Path(root), tree)
        site = scratch_path / "site"
        shims = scratch_path / "shims"
        home = scratch_path / "home"
        for directory in (site, shims, home):
            directory.mkdir()
        shim_log = scratch_path / "shims.jsonl"
        import_log = scratch_path / "imports.json"
        _write_shims(shims)
        bootstrap = scratch_path / "bootstrap.py"
        bootstrap.write_text(_BOOTSTRAP)

        for attempt in range(1, max_rounds + 1):
            left = timeout - (time.monotonic() - started)
            if left <= 0:
                result.error = f"ran out of time after {attempt - 1} rounds"
                break
            result.rounds = attempt
            shim_log.write_text("")
            if import_log.exists():
                import_log.unlink()
            env = dict(os.environ)
            env.update({
                "PATH": f"{shims}{os.pathsep}{env.get('PATH', '')}",
                "PYTHONPATH": str(site),
                "HOME": str(home),
                "WIR_SHIM_LOG": str(shim_log),
                "WIR_IMPORT_LOG": str(import_log),
                "WIR_STUB_MODULES": ",".join(sorted(stubs)),
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            if progress is not None:
                progress(f"{script} {' '.join(args)} (round {attempt})")
            try:
                # -S: no site-packages. The driver must see the standard
                # library and what this loop installs for it -- not the
                # packages will-it-riscv itself runs on, which include rich.
                process = subprocess.run(
                    [sys.executable, "-S", str(bootstrap), str(tree / script), *args],
                    cwd=tree, env=env, capture_output=True, text=True,
                    timeout=max(1, int(left)), stdin=subprocess.DEVNULL,
                )
                result.returncode = process.returncode
                result.tail = "\n".join(
                    ((process.stdout or "") + (process.stderr or "")).strip().splitlines()[-12:]
                )
            except subprocess.TimeoutExpired:
                result.error = f"the driver did not finish within {int(left)}s"
            reading = _read_json(import_log)
            result.commands = _read_commands(shim_log, tree)
            result.imported = _imported(reading, tree, site)
            output = (process.stdout or "") + (process.stderr or "") if not result.error else ""
            made = _make_missing_directories(output, tree)
            result.made += [d for d in made if d not in result.made]

            fresh = [
                m for m in reading.get("missing", [])
                if m not in handled and m not in set(reading.get("stdlib", []))
            ]
            if (not fresh and not made) or result.error:
                break
            for module in fresh:
                handled.add(module)
                dist = dist_for_module(module, wanted)
                if dist is None:
                    # Imported, and nothing in the plan provides it. Could be
                    # an optional import in a try block; could be a gap in
                    # the plan. Stub it, say so, and let the driver carry on.
                    stubs.add(module)
                    result.not_in_plan.append(module)
                    continue
                spec = f"{dist}=={wanted[dist]}" if wanted.get(dist) else dist
                if _pip_install(spec, site, pip_cache):
                    result.installed.append(spec)
                else:
                    stubs.add(module)
                    result.stubbed.append(f"{module} ({spec} would not install for the host)")

    result.duration = time.monotonic() - started
    return result


def dist_for_module(module: str, wanted: dict) -> Optional[str]:
    """The distribution in the plan that provides an import name, if any."""
    candidates = [MODULE_TO_DIST.get(module), module, f"python-{module}", f"py{module}"]
    for candidate in candidates:
        if candidate and canonicalize_name(candidate) in wanted:
            return str(canonicalize_name(candidate))
    return None


def _imported(reading: dict, tree: Path, site: Path) -> set:
    """Distributions whose modules the driver loaded from what was installed."""
    found: set = set()
    site_text = str(site)
    for module, origin in reading.get("origins", {}).items():
        if not origin or not origin.startswith(site_text):
            continue   # the standard library, or the project's own code
        owners = reading.get("owners", {}).get(module) or [MODULE_TO_DIST.get(module, module)]
        found.update(canonicalize_name(o) for o in owners)
    return found


_NO_SUCH = re.compile(r"No such file or directory: '([^']+)'")


def _make_missing_directories(output: str, tree: Path) -> list[str]:
    """Make the directory a missing path was meant to be in, inside the tree.

    Only ever inside the clone: a path the driver wanted anywhere else is
    left alone and simply stays missing.
    """
    made: list[str] = []
    for text in _NO_SUCH.findall(output):
        # A relative path is relative to the driver's working directory,
        # which is the clone; an absolute one has to be inside it.
        relative = relativize(text, tree) if os.path.isabs(text) else text
        if os.path.isabs(relative) or relative.split("/")[0] == "..":
            continue   # not inside the clone
        path = tree / relative
        directory = path.parent if path.suffix else path
        if directory.exists():
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        made.append(relativize(str(directory), tree))
    return made


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _spellings(tree: Path) -> list[str]:
    """Every way a path to the clone can be written, longest first.

    On macOS the scratch directory is /var/..., which the driver is as likely
    to see as /private/var/... -- strip one and the other is left dangling.
    """
    return sorted({str(tree), str(tree.resolve())}, key=len, reverse=True)


def relativize(text: str, tree: Path) -> str:
    """A path, or a list of them, as it would read from the repository root."""
    for spelling in _spellings(tree):
        if text == spelling:
            return "."
        text = text.replace(spelling + os.sep, "").replace(spelling, ".")
    return text


def _read_commands(path: Path, tree: Path) -> list:
    commands: list = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return commands
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        argv = [relativize(a, tree) if isinstance(a, str) else a for a in event.get("argv", [])]
        commands.append((event.get("tool", "?"), argv))
    return commands


def _write_shims(directory: Path) -> None:
    body = _SHIM.format(python=sys.executable, versions=_VERSIONS)
    for tool in SHIMMED_TOOLS:
        path = directory / tool
        path.write_text(body)
        path.chmod(0o755)


def _pip_install(
    spec: str, site: Path, cache: Optional[Path], wheels_only: bool = False
) -> bool:
    command = [
        sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
        "--no-input", "--target", str(site), spec,
    ]
    if wheels_only:
        command.insert(-1, "--only-binary=:all:")
    env = dict(os.environ)
    if cache is not None:
        env["PIP_CACHE_DIR"] = str(cache)
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=600, env=env
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ------------------------------------------------------------------ cloning


_SKIP = {".git", "build", "__pycache__", ".venv", "venv", "node_modules"}


def clone_tree(source: Path, destination: Path) -> None:
    """Copy a source tree for the driver to write all over.

    Copy-on-write where the filesystem can -- APFS clonefile on macOS, a
    reflink on Linux -- so a 1.3 GB repository costs nothing until written.
    The .git directory and any earlier build output are left behind.
    """
    def ignore(directory: str, names: list[str]) -> list[str]:
        at_top = Path(directory) == source
        return [n for n in names if n in _SKIP and (n != "build" or at_top)]

    shutil.copytree(source, destination, ignore=ignore, copy_function=_clone_file, symlinks=True)


def _clone_file(src: str, dst: str) -> str:
    if _clonefile(src, dst) or _reflink(src, dst):
        return dst
    return shutil.copy2(src, dst)


def _clonefile(src: str, dst: str) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0
    except (OSError, AttributeError):
        return False


def _reflink(src: str, dst: str) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        import fcntl

        ficlone = 0x40049409
        with open(src, "rb") as source, open(dst, "wb") as destination:
            fcntl.ioctl(destination.fileno(), ficlone, source.fileno())
        shutil.copystat(src, dst)
        return True
    except OSError:
        try:
            os.unlink(dst)
        except OSError:
            pass
        return False
