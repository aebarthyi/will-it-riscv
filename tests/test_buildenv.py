"""The environment a configure runs in: this host's, less what would steer it."""

import os
import sys
from pathlib import Path

from will_it_riscv import buildenv


def _program(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    program = directory / name
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    return program


def test_shims_that_only_fail_are_not_on_the_configures_path():
    path = "/home/u/.pyenv/shims:/usr/bin:/home/u/.asdf/shims/:/opt/bin"
    assert buildenv.without_shims(path) == "/usr/bin:/opt/bin"


def test_a_vendor_toolchain_ahead_of_the_system_is_left_off_the_path(tmp_path, monkeypatch):
    """The Vitis clang++ that stopped saltfm sat ahead of /usr/bin."""
    system, vendor = tmp_path / "usr-bin", tmp_path / "vendor-bin"
    for tool in buildenv.TOOLS:
        _program(system, tool)
    _program(vendor, "clang++")
    _program(vendor, "ninja")
    monkeypatch.setattr(buildenv, "SYSTEM", (str(system),))
    path = buildenv.search_path(os.pathsep.join([str(vendor), str(system)])).split(os.pathsep)
    assert str(vendor) not in path
    assert str(system) in path
    assert os.path.normpath(str(Path(sys.executable).parent)) in path


def test_a_build_tool_the_system_lacks_brings_its_directory(tmp_path, monkeypatch):
    """An HPC module's compiler is the one the build would use."""
    system, module = tmp_path / "usr-bin", tmp_path / "gcc-13" / "bin"
    system.mkdir()
    _program(module, "cc")
    monkeypatch.setattr(buildenv, "SYSTEM", (str(system),))
    path = buildenv.search_path(os.pathsep.join([str(module), str(system)])).split(os.pathsep)
    assert path[-1] == str(module)


def test_what_steers_a_configure_is_left_out_and_the_rest_kept():
    env = buildenv.environment({
        "CC": "clang", "CFLAGS": "-O3", "CPATH": "/opt/inc", "CMAKE_PREFIX_PATH": "/opt",
        "CUDA_HOME": "/usr/local/cuda", "ROCM_PATH": "/opt/rocm", "PYTHONPATH": "/x",
        "HOME": "/home/u", "https_proxy": "http://proxy:3128", "LANG": "C.UTF-8",
        "PATH": "/usr/bin",
    })
    assert set(env) == {"HOME", "https_proxy", "LANG", "PATH"}
