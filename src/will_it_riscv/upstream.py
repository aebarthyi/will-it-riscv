"""Tools the archive has too old, followed to their own source.

jaxlib pins Bazel 8.7.0, and Debian 13 has 4.2.3. Bazel publishes no
riscv64 binaries, so a port starts by bootstrapping Bazel -- and whether
that can be done is the same question again, one level down. Fetch the
tool's source at the version the build wants, read what its bootstrap
needs, and check that against the archive.

Only what the bootstrap reads is fetched: a sparse checkout of the release
tag. A tool's bootstrap is a script -- Bazel's compile.sh builds Bazel with
itself -- so it is read, never run; what it needs from the system is the
part that can be checked.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .plan import SYSTEM_PACKAGES, Evidence, Plan, Step
from .sources import GIT_TIMEOUT, SourceTree


@dataclass(frozen=True)
class Upstream:
    package: str
    """The distro package it stands in for: ``bazel-bootstrap``."""
    name: str
    repository: str
    tag: str
    """The release tag, with ``{version}`` for the version wanted."""
    sparse: tuple[str, ...]
    """What the bootstrap reads -- all of the tree that is fetched."""
    note: str
    """Why it has to come from its source, for the answer."""
    planner: Callable[[Path, str], Plan]


def fetch(upstream: Upstream, version: str, cache_root: Path) -> SourceTree:
    """A sparse checkout of the tool's release tag, cached."""
    tag = upstream.tag.format(version=version)
    tree = SourceTree(
        name=upstream.name, version=version, kind="git", origin=f"{upstream.repository}@{tag}"
    )
    destination = Path(cache_root) / f"{upstream.name}-{version}"
    if not (destination / ".unpacked").exists():
        error = _sparse_clone(upstream.repository, tag, destination, upstream.sparse)
        if error:
            tree.error = error
            return tree
    tree.path = destination
    return tree


def _sparse_clone(url: str, tag: str, destination: Path, paths: tuple[str, ...]) -> Optional[str]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    commands = [
        ["git", "clone", "--quiet", "--depth", "1", "--branch", tag, "--filter=blob:none",
         "--sparse", url, str(destination)],
        ["git", "-C", str(destination), "sparse-checkout", "set", "--no-cone", *paths],
    ]
    for command in commands:
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=GIT_TIMEOUT, env=env
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"git: {exc}"
        if process.returncode != 0:
            tail = (process.stderr or "").strip().splitlines()
            return f"git clone {url} at {tag}: {tail[-1] if tail else 'failed'}"
    (destination / ".unpacked").touch()
    return None


# ---------------------------------------------------------------------- bazel


def _bazel_plan(tree: Path, version: str) -> Plan:
    """What Bazel's bootstrap needs, read from the scripts that run it.

    compile.sh builds a minimal Bazel with javac, then Bazel with that --
    using the JDK on the machine (``--java_runtime_version``, from
    buildenv.sh's JAVA_VERSION), rules_python's autodetecting toolchain for
    Python, and the system's C++ compiler for its native parts.
    """
    packages: list[str] = []
    evidence: list[Evidence] = []
    buildenv = "scripts/bootstrap/buildenv.sh"
    bootstrap = "scripts/bootstrap/bootstrap.sh"

    java = _find(tree, buildenv, r"JAVA_VERSION=\$\{JAVA_VERSION:-(\d+)\}")
    if java is not None:
        line, match = java
        packages.append(f"openjdk-{match.group(1)}-jdk-headless")
        evidence.append(Evidence(buildenv, line, line, match.group(0)))
    tools = _find(tree, buildenv, r"\buname unzip which\b")
    if tools is not None:
        packages.append("unzip")
        evidence.append(Evidence(buildenv, tools[0], tools[0], "unzip"))
    python = _find(tree, bootstrap, r"@rules_python//python:autodetecting_toolchain")
    if python is not None:
        packages.append("python3")
        evidence.append(Evidence(bootstrap, python[0], python[0], python[1].group(0)))
    cxx = _find(tree, bootstrap, r"--cxxopt=-std=c\+\+17")
    if cxx is not None:
        packages.append("g++")
        evidence.append(Evidence(bootstrap, cxx[0], cxx[0], cxx[1].group(0)))

    unsure = [
        "bootstraps with compile.sh, which builds Bazel with itself: read, not run",
    ]
    if not packages:
        unsure.append(f"its bootstrap scripts were not where Bazel {version} keeps them")
        return Plan(repo="bazel", steps=[], unsure=unsure)
    step = Step(
        id="bootstrap-tools",
        kind=SYSTEM_PACKAGES,
        packages=packages,
        note="what compile.sh needs from the system: a JDK, unzip, Python, a C++ compiler",
        evidence=evidence,
    )
    return Plan(repo="bazel", steps=[step], unsure=unsure)


def _find(tree: Path, relative: str, pattern: str) -> Optional[tuple[int, re.Match]]:
    try:
        lines = (tree / relative).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for number, line in enumerate(lines, 1):
        match = re.search(pattern, line)
        if match:
            return number, match
    return None


UPSTREAM = {
    "bazel-bootstrap": Upstream(
        package="bazel-bootstrap",
        name="bazel",
        repository="https://github.com/bazelbuild/bazel",
        tag="{version}",
        sparse=("/compile.sh", "/scripts/bootstrap/", "/MODULE.bazel", "/.bazelversion"),
        note=(
            "Bazel publishes no riscv64 binaries; it has to be bootstrapped from its "
            "source (github.com/bazelbuild/bazel) with a JDK"
        ),
        planner=_bazel_plan,
    ),
}
