"""``will-it-riscv doctor``: what this machine has for ``plan``, and what it lacks.

``plan`` configures a project for real, with this machine's compilers, and has
the planner run by llama.cpp. Each of those can be missing, and finding out
half way through a run is the worst time. This says so first, with what to
do about each.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "[green]✓[/green]", WARN: "[yellow]![/yellow]", FAIL: "[red]✗[/red]"}

#: The planner on the CPU alone peaked at 11 GB (its weights, and a 40k
#: context's cache).
CPU_MEMORY = 12 * 1024 ** 3


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _first(*names: str) -> Optional[str]:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _memory() -> Optional[int]:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _free(path: Path) -> Optional[int]:
    for parent in (path, *path.parents):
        if parent.exists():
            try:
                return shutil.disk_usage(parent).free
            except OSError:
                return None
    return None


def _gpus(server: str) -> list[str]:
    """The devices llama.cpp would offload to, as it lists them."""
    try:
        listed = subprocess.run([server, "--list-devices"], capture_output=True, text=True,
                                timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    lines = listed.split("Available devices:", 1)[-1].splitlines()
    return [line.strip() for line in lines if ":" in line and "(none)" not in line]


def checks() -> list[Check]:
    from . import fetch, pseudobuild, pseudomeson, serve

    found: list[Check] = []
    system = platform.system()
    if system == "Linux":
        found.append(Check("platform", OK, f"Linux {platform.machine()}"))
    elif system == "Darwin":
        found.append(Check("platform", WARN, "macOS: the pretend configures are only "
                                             "tested on Linux"))
    else:
        found.append(Check("platform", FAIL, f"{system}: the pretend configures need Linux "
                                             "(or WSL)"))

    apt = "on Debian or Ubuntu: sudo apt install build-essential"
    for name, candidates, missing in (
        ("C compiler", ("cc", "gcc", "clang"), FAIL),
        ("C++ compiler", ("c++", "g++", "clang++"), FAIL),
        ("archiver", ("ar",), FAIL),
        ("Fortran compiler", ("gfortran",), WARN),
    ):
        path = _first(*candidates)
        if path:
            found.append(Check(name, OK, path))
        elif missing == WARN:
            found.append(Check(name, WARN, "none: a project with Fortran in it stops at its "
                                           "configure (sudo apt install gfortran)"))
        else:
            found.append(Check(name, FAIL, f"none: every configure needs one ({apt})"))

    cmake = pseudobuild.cmake()
    found.append(Check("cmake", OK, cmake) if cmake else
                 Check("cmake", FAIL, "none: reinstall will-it-riscv, which brings its own"))
    found.append(Check("meson", OK, "beside this Python, or from its wheel")
                 if pseudomeson.available() else
                 Check("meson", WARN, "none: meson projects cannot be configured"))

    server = serve.found()
    build, item = fetch.server_build()
    cached = fetch.home() / "llama.cpp" / f"{fetch.LLAMA_CPP}-{build}" / "llama-server"
    if server:
        found.append(Check("llama-server", OK, f"yours: {server}"))
    elif cached.is_file():
        server = str(cached)
        found.append(Check("llama-server", OK, f"llama.cpp {fetch.LLAMA_CPP} ({build}): "
                                               f"{cached}"))
    elif item is not None:
        found.append(Check("llama-server", OK, f"fetched on the first plan: llama.cpp "
                                               f"{fetch.LLAMA_CPP} ({build}), "
                                               f"{item.size / 1e6:.0f} MB"))
    else:
        found.append(Check("llama-server", FAIL, "no prebuilt one for this machine: build "
                                                 "llama.cpp and put llama-server on the PATH"))

    gpus = _gpus(server) if server else []
    memory = _memory()
    if gpus:
        found.append(Check("GPU", OK, "; ".join(gpus)))
    elif server:
        found.append(Check("GPU", WARN, "none llama.cpp can use: the planner runs on the "
                                        "CPU (about 2 minutes a plan, and 11 GB of memory)"))
    elif build == "vulkan":
        found.append(Check("GPU", OK, "a Vulkan loader: llama.cpp uses the GPU if it can"))
    else:
        found.append(Check("GPU", WARN, "no Vulkan loader: the planner runs on the CPU "
                                        "(sudo apt install libvulkan1 mesa-vulkan-drivers, "
                                        "for a GPU)"))
    if memory is not None:
        gigabytes = f"{memory / 1024 ** 3:.0f} GB"
        if memory >= CPU_MEMORY or gpus:
            found.append(Check("memory", OK, gigabytes))
        else:
            found.append(Check("memory", WARN, f"{gigabytes}: the planner on the CPU alone "
                                               "takes about 11 GB"))

    model = fetch.planner_path()
    if model.is_file() and model.stat().st_size == fetch.PLANNER.size:
        found.append(Check("planner", OK, str(model)))
    else:
        free = _free(model.parent)
        need = fetch.PLANNER.size
        if free is not None and free < need * 1.1:
            found.append(Check("planner", FAIL, f"{need / 1e9:.1f} GB to fetch, and "
                                                f"{free / 1e9:.1f} GB free at {model.parent} "
                                                "(set WILL_IT_RISCV_DATA elsewhere)"))
        else:
            found.append(Check("planner", OK, f"fetched on the first plan: "
                                              f"{need / 1e9:.1f} GB, into {model.parent}"))
    return found


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="will-it-riscv doctor",
        description="Say what this machine has for `will-it-riscv plan`, and what it lacks.",
    )
    parser.parse_args(argv)
    console = Console()
    with console.status("looking…"):
        found = checks()
    table = Table(show_header=False, box=None, pad_edge=False)
    for check in found:
        table.add_row(MARK[check.status], check.name, check.detail)
    console.print(table)
    failed = [c for c in found if c.status == FAIL]
    if failed:
        console.print(f"\n[red]{len(failed)} thing(s) `plan` cannot do without[/red]")
        return 1
    console.print(f"\nready: will-it-riscv plan PATH   (python {sys.version.split()[0]})")
    return 0
