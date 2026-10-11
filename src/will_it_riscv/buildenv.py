"""The environment a configure runs in: this host's, less what would steer it.

A configure reads more than its command line. With /opt/rocm/bin on PATH,
PyTorch's says "ROCm enabled"; with a vendor toolchain ahead of /usr/bin, a
plan's clang++ is the vendor's wrapper, and saltfm stops on it. CC, CXX and
CFLAGS choose the compiler and its flags, CPATH and LIBRARY_PATH reach past
the empty sysroot, and CMAKE_PREFIX_PATH points find_package at what this
host has installed. None of that is the target's, and with it the answer
changes from one machine to the next.

So PATH keeps only this Python's directories and the system's, plus the
directory of any build tool that neither has, and the variables that choose a
compiler, its flags or its search paths are left out.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

SYSTEM = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")
if sys.platform == "darwin":
    SYSTEM = ("/opt/homebrew/bin", "/opt/homebrew/sbin", *SYSTEM)

#: What a configure expects to find by name. One found only outside the
#: system's directories (a compiler from an HPC module) brings its directory
#: along; one the system has never does, so a vendor's ninja stays out.
TOOLS = ("cc", "c++", "gfortran", "make", "ninja", "pkg-config", "git")

#: What chooses a compiler, its flags, or where it and CMake look.
STEERING = frozenset((
    "CC", "CXX", "FC", "F77", "F90", "CPP", "CXXCPP", "OBJC", "OBJCXX", "CUDACXX", "HIPCXX",
    "AR", "AS", "LD", "NM", "RANLIB", "STRIP",
    "CFLAGS", "CXXFLAGS", "CPPFLAGS", "FFLAGS", "FCFLAGS", "LDFLAGS", "LIBS",
    "CUDAFLAGS", "HIPFLAGS",
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH",
    "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR", "PKG_CONFIG_SYSROOT_DIR", "ACLOCAL_PATH",
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
    "BOOST_ROOT", "MKLROOT", "MAKEFLAGS", "MFLAGS",
))
STEERING_PREFIXES = ("CMAKE_", "CUDA", "ROCM_", "HIP_")


def environment(base: Optional[Mapping[str, str]] = None) -> dict:
    """``base`` (default: this process's environment) without what steers a
    configure, and with a PATH a configure can be trusted with."""
    env = {
        key: value for key, value in (os.environ if base is None else base).items()
        if key not in STEERING and not key.startswith(STEERING_PREFIXES)
    }
    env["PATH"] = search_path(env.get("PATH", ""))
    return env


def search_path(path: str) -> str:
    """``path`` cut down to this Python's directories and the system's, in the
    order ``path`` had them, then the rest of those, then any directory a tool
    in TOOLS was found in only outside them."""
    given = [os.path.normpath(d) for d in without_shims(path).split(os.pathsep) if d]
    own = [os.path.normpath(d) for d in (
        str(Path(sys.executable).parent), sysconfig.get_path("scripts"), *SYSTEM)]
    kept = [d for d in dict.fromkeys(given) if d in own]
    kept += [d for d in dict.fromkeys(own) if d not in kept and os.path.isdir(d)]
    for tool in TOOLS:
        if shutil.which(tool, path=os.pathsep.join(kept)) is None:
            found = shutil.which(tool, path=os.pathsep.join(given))
            if found and os.path.dirname(found) not in kept:
                kept.append(os.path.dirname(found))
    return os.pathsep.join(kept)


def without_shims(path: str) -> str:
    """PATH less pyenv's and asdf's shims: a shim is there for every program any
    installed version has, and fails when the active one has none -- so a
    configure finds a cython that says "pyenv: cython: command not found"
    where it should have found nothing, and been given a stub."""
    shim = re.compile(r"/\.(pyenv|asdf)/shims/?$")
    return os.pathsep.join(d for d in path.split(os.pathsep) if d and not shim.search(d))
