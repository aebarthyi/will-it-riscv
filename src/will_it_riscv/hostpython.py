"""The Python a configure runs: this host's, holding what the build installs.

scikit-build-core pins the interpreter running the build, and pip's build
isolation has put the build requirements in front of it. A configure leans
on that: ml-dtypes asks it ``import numpy; print(numpy.get_include())`` and
dies if the answer does not come.

So the interpreter a pretend configure is given is this one, run without
its own site-packages, and with a directory of its own in front. Whatever a
configure imports and cannot find is put there -- for the host, at the
version the plan resolved, the same as a build driver's imports -- or, when
nothing says which version or it will not install, stubbed. Only an import
that killed the interpreter counts: one it caught and did without was
optional to begin with.

Nothing here runs anything for riscv64. Build requirements run on the build
host; that is the one place a host package is the right answer.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.utils import canonicalize_name

from .drive import _pip_install, dist_for_module

#: Run by this host's interpreter with -S, in place of ``python`` itself.
_RUNNER = r'''
import importlib.abc, importlib.machinery, json, os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
STUBSITE = os.path.join(HERE, "stubsite")
with open(os.path.join(HERE, "stubs.txt")) as handle:
    STUBS = set(handle.read().split())


@ANYTHING@
_ANSWERS = os.path.join(STUBSITE, "_answers")


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__getattr__ = _attribute
        module.__path__ = []
        # Asked where it is, or which it is -- neuron's FindPythonModule
        # prints numpy.__file__ -- a stub says something a string can hold.
        module.__file__ = os.path.join(STUBSITE, module.__name__.replace(".", os.sep),
                                       "__init__.py")
        module.__version__ = "99.9.9"


class _Stubs(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in STUBS:
            return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
        return None


def _missed(name):
    with open(os.path.join(HERE, "missing.jsonl"), "a") as log:
        log.write(json.dumps({"module": name.split(".")[0]}) + "\n")


sys.meta_path.append(_Stubs())
os.environ["PYTHONPATH"] = os.pathsep.join([SITE, os.environ.get("PYTHONPATH", ""), STUBSITE])
args = sys.argv[1:]
while args and args[0].startswith("-") and args[0] not in ("-c", "-m", "-"):
    flag = args.pop(0)
    if flag in ("-V", "-VV", "--version"):
        print("Python " + sys.version.split()[0])
        sys.exit(0)
    if flag in ("-W", "-X") and args:
        args.pop(0)
sys.path[0:1] = []
try:
    if args[:1] == ["-c"]:
        sys.argv = ["-c", *args[2:]]
        sys.path[0:0] = ["", SITE]
        exec(compile(args[1], "<string>", "exec"), {"__name__": "__main__"})
    elif args[:1] == ["-m"]:
        sys.argv = [args[1], *args[2:]]
        sys.path[0:0] = [os.getcwd(), SITE]
        import importlib.util
        if importlib.util.find_spec(args[1].split(".")[0]) is None:
            # runpy says so with a plain ImportError, which is not caught below.
            _missed(args[1])
            sys.stderr.write(f"{sys.executable}: No module named {args[1]}\n")
            sys.exit(1)
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    elif args and args[0] != "-":
        sys.argv = list(args)
        sys.path[0:0] = [os.path.dirname(os.path.abspath(args[0])), SITE]
        runpy.run_path(args[0], run_name="__main__")
    else:
        sys.path[0:0] = ["", SITE]
        exec(compile(sys.stdin.read(), "<stdin>", "exec"), {"__name__": "__main__"})
except ModuleNotFoundError as exc:
    if exc.name:
        _missed(exc.name)
    raise
'''

_WRAPPER = '#!/bin/sh\nexec "{python}" -S "{runner}" "$@"\n'

#: A stubbed module, as a file, so that any interpreter a build runs sees it:
#: Meson runs a ``.py`` script with its own Python, not the one it was given.
_STUB_MODULE = '''# stub emitted by will-it-riscv: nothing said which version to install
import os
@ANYTHING@
_ANSWERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_answers")
__version__ = "99.9.9"
__getattr__ = _attribute
'''

#: What a stubbed module's every attribute is. Most are only imported, or
#: called and ignored; but a configure asks some for real answers, by name --
#: numpy.get_include(), distutils.sysconfig.get_python_lib(), mpi4py's
#: __version__ and get_include() on two lines, ``setup.py --version`` with
#: setuptools stubbed -- and an empty string stops it (ascent, exago, dolfinx,
#: psi4, xrootd). So a version is one that passes, a path a directory that
#: exists, and setup() says the version it was given.
_ANYTHING = r'''
import re as _re
import sys as _sys

_VERSION = _re.compile(r"_*(get_)?(short_|full_)?version_*")
_PATHLIKE = {"include", "inc", "path", "dir", "directory", "lib", "libdir", "includedir",
             "prefix", "location", "root", "home", "site", "purelib", "platlib", "stdlib",
             "platstdlib", "scripts", "data"}
_PLURAL = ("dirs", "libs", "includes", "libraries", "directories")


class _Version(str):
    def __getattr__(self, name):
        return self if _VERSION.fullmatch(name.lower()) else _Anything(name)


def _answer(name, args=(), kwargs=None):
    lower = (name or "").lower()
    if lower in ("get_config_var", "get_path") and args and isinstance(args[0], str):
        lower = args[0].lower()   # sysconfig.get_config_var("LIBDIR")
    if lower == "setup" and "--version" in _sys.argv:
        print((kwargs or {}).get("version", "99.9.9"))
        return _Anything(name)
    if _VERSION.fullmatch(lower):
        return _Version("99.9.9")
    words = [w for w in _re.split(r"[^a-z0-9]+", lower) if w]
    if "paths" in words or not any(w in _PATHLIKE or w.rstrip("s") in _PATHLIKE for w in words):
        return _Anything(name)   # get_paths() is a mapping: asked by key, below
    where = _os.path.join(_ANSWERS, "_".join(words))
    _os.makedirs(where, exist_ok=True)
    return [where] if lower.endswith(_PLURAL) else where


def _attribute(name):
    return _Version("99.9.9") if _VERSION.fullmatch(name.lower()) else _Anything(name)


class _Anything:
    def __init__(self, name=""):
        self._name = name
    def __getattr__(self, name):
        return _attribute(name)
    def __call__(self, *args, **kwargs):
        return _answer(self._name, args, kwargs)
    def __getitem__(self, key):
        return _answer(key) if isinstance(key, str) else _Anything()
    def __iter__(self):
        return iter(())
    def __bool__(self):
        return False
    def __str__(self):
        return ""
    def __mro_entries__(self, bases):
        return (object,)
'''
_ANYTHING = "import os as _os\n" + _ANYTHING
_RUNNER = _RUNNER.replace("@ANYTHING@", _ANYTHING)
_STUB_MODULE = _STUB_MODULE.replace("@ANYTHING@", _ANYTHING)


@dataclass
class HostPython:
    """One configure's interpreter, and what it has been given so far."""

    home: Path
    dists: Optional[dict] = None
    """Distribution to version, for what the plan installs. None when there
    is no plan: then nothing is installed, and a missing import is stubbed."""
    pip_cache: Optional[Path] = None
    installed: list = field(default_factory=list)
    """``name==version`` put in for the host because the configure imported it."""
    stubbed: list = field(default_factory=list)
    """Modules stubbed, each with why."""
    not_in_plan: list = field(default_factory=list)
    """Modules the configure imported that nothing in the plan installs."""
    handled: set = field(default_factory=set)

    @classmethod
    def create(
        cls, home: Path, dists: Optional[dict] = None, pip_cache: Optional[Path] = None
    ) -> HostPython:
        home = Path(home)
        (home / "site").mkdir(parents=True, exist_ok=True)
        (home / "stubs.txt").write_text("")
        (home / "runner.py").write_text(_RUNNER)
        executable = home / f"python{sys.version_info[0]}.{sys.version_info[1]}"
        executable.write_text(_WRAPPER.format(python=sys.executable, runner=home / "runner.py"))
        executable.chmod(0o755)
        if dists is not None:
            dists = {str(canonicalize_name(k)): v for k, v in dists.items()}
        return cls(home=home, dists=dists, pip_cache=pip_cache)

    @property
    def executable(self) -> Path:
        return self.home / f"python{sys.version_info[0]}.{sys.version_info[1]}"

    @property
    def site(self) -> Path:
        return self.home / "site"

    @property
    def stubsite(self) -> Path:
        """Stubbed modules, as files: last on the path, so anything real wins."""
        return self.home / "stubsite"

    @property
    def path(self) -> str:
        """The PYTHONPATH for anything else the build runs Python with."""
        return os.pathsep.join([str(self.site), str(self.stubsite)])

    @property
    def scripts(self) -> Path:
        """Where installed packages put their programs: cython, meson, ninja."""
        return self.home / "site" / "bin"

    def version_of(self, dist: str) -> Optional[str]:
        """The version the plan resolved a distribution to, if it did."""
        return (self.dists or {}).get(str(canonicalize_name(dist)))

    def install(self, dist: str, version: Optional[str] = None) -> Optional[str]:
        """Put a distribution in for the host, from a wheel; the spec, or None."""
        spec = f"{dist}=={version}" if version else dist
        if spec in self.installed:
            return spec
        if not _pip_install(spec, self.site, self.pip_cache, wheels_only=True):
            return None
        self.installed.append(spec)
        return spec

    def missed(self, module: str) -> None:
        """Record an import that failed somewhere this interpreter did not see.

        Meson runs a ``.py`` script with the interpreter running Meson, not
        the one it was given: pandas' generate_version.py dies importing
        versioneer, and only Meson's log says so.
        """
        with open(self.home / "missing.jsonl", "a") as log:
            log.write(json.dumps({"module": module.split(".")[0]}) + "\n")

    def missing(self) -> list[str]:
        """Imports that killed the interpreter and have not been dealt with."""
        try:
            lines = (self.home / "missing.jsonl").read_text().splitlines()
        except OSError:
            return []
        names: list[str] = []
        for line in lines:
            try:
                name = json.loads(line).get("module")
            except ValueError:
                continue
            if name and name not in self.handled and name not in names:
                names.append(name)
        return names

    def provide(self) -> list[str]:
        """Install or stub every fresh miss; the modules dealt with."""
        fresh = self.missing()
        stubs = set((self.home / "stubs.txt").read_text().split())
        for module in fresh:
            self.handled.add(module)
            wanted = {} if self.dists is None else self.dists
            dist = dist_for_module(module, wanted)
            if dist is None:
                stubs.add(module)
                if self.dists is None:
                    self.stubbed.append(f"{module} (no plan says which version)")
                else:
                    self.not_in_plan.append(module)
                    self.stubbed.append(f"{module} (nothing in the plan installs it)")
                continue
            spec = f"{dist}=={wanted[dist]}" if wanted.get(dist) else dist
            # Wheels only: an sdist would mean running a fetched package's
            # setup.py, and nothing here builds anything.
            if _pip_install(spec, self.home / "site", self.pip_cache, wheels_only=True):
                self.installed.append(spec)
            else:
                stubs.add(module)
                self.stubbed.append(f"{module} ({spec} has no wheel for the host)")
        (self.home / "stubs.txt").write_text("\n".join(sorted(stubs)))
        for module in stubs:
            package = self.stubsite / module
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", module) and not package.exists():
                package.mkdir(parents=True)
                (package / "__init__.py").write_text(_STUB_MODULE)
        return fresh
