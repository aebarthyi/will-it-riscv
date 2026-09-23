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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .drive import _pip_install, dist_for_module

#: Run by this host's interpreter with -S, in place of ``python`` itself.
_RUNNER = r'''
import importlib.abc, importlib.machinery, json, os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
with open(os.path.join(HERE, "stubs.txt")) as handle:
    STUBS = set(handle.read().split())


class _Anything:
    def __getattr__(self, name):
        return _Anything()
    def __call__(self, *args, **kwargs):
        return _Anything()
    def __iter__(self):
        return iter(())
    def __bool__(self):
        return False
    def __str__(self):
        return ""
    def __mro_entries__(self, bases):
        return (object,)


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__getattr__ = lambda name: _Anything()
        module.__path__ = []


class _Stubs(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in STUBS:
            return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
        return None


def _missed(name):
    with open(os.path.join(HERE, "missing.jsonl"), "a") as log:
        log.write(json.dumps({"module": name.split(".")[0]}) + "\n")


sys.meta_path.append(_Stubs())
os.environ["PYTHONPATH"] = SITE + os.pathsep + os.environ.get("PYTHONPATH", "")
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
        return cls(home=home, dists=dists, pip_cache=pip_cache)

    @property
    def executable(self) -> Path:
        return self.home / f"python{sys.version_info[0]}.{sys.version_info[1]}"

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
        return fresh
