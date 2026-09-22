"""Reading system dependencies out of CI configuration.

A repository usually states its system dependencies outright somewhere a
source distribution never ships: a workflow file, a Dockerfile, a nix
expression. Those lists are written and verified by people who actually build
the project, which makes them better evidence than anything inferred from a
build file.

The catch is that they are shell, not data. Pillow writes its list into a bash
array and installs ``"${packages[@]}"``; psycopg2 interpolates a version
variable and adds a third-party apt source first. Both are handled here, and
what cannot be resolved is reported rather than silently dropped.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import SystemRequirement
from .syslibs import database

MAX_CI_FILES = 400
MAX_CI_BYTES = 512 * 1024

#: Directory names whose contents are CI or container definitions.
CI_DIRS = frozenset({
    ".github", ".gitlab", ".circleci", ".devcontainer", ".ci", "ci",
    "containers", "container", "docker", "dockerfiles", "admin", "scripts",
    "tools", ".buildkite", ".azure", "packaging",
})

_DOCKERFILE = re.compile(r"^(dockerfile|containerfile)", re.IGNORECASE)


@dataclass
class CiFindings:
    packages: dict[str, str] = field(default_factory=dict)
    """distro package name -> the file it was declared in."""
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    files_read: int = 0


# ------------------------------------------------------------------ parsing

#: `apt-get -qq -y install --no-install-recommends foo bar` and friends.
_APT_INSTALL = re.compile(
    r"\bapt(?:-get)?\s+(?:-{1,2}[^\s]+\s+)*install\b([^\n;&|]*)", re.IGNORECASE
)
_DNF_INSTALL = re.compile(
    r"\b(?:dnf|yum|zypper)\s+(?:-{1,2}[^\s]+\s+)*install\b([^\n;&|]*)", re.IGNORECASE
)
_APK_ADD = re.compile(r"\bapk\s+add\b([^\n;&|]*)", re.IGNORECASE)

#: A bash array definition: `packages=(\n  cmake\n  ghostscript\n)`
_BASH_ARRAY = re.compile(
    r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)=\(\s*([^)]*?)\)", re.MULTILINE | re.DOTALL
)
#: `"${packages[@]}"` / `${packages[*]}` / `$packages`
_ARRAY_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)(?:\[[@*]\])?\}?")

#: HPC Container Maker, which GROMACS and many HPC projects use instead of
#: writing apt lines: `packages(ospackages=["git", "cmake"])`
_HPCCM_OSPACKAGES = re.compile(r"ospackages\s*=\s*\[([^\]]*)\]")

#: Nix derivations.
_NIX_INPUTS = re.compile(
    r"(?:nativeBuildInputs|buildInputs|propagatedBuildInputs)\s*=\s*"
    r"(?:with\s+[\w.]+\s*;\s*)?\[([^\]]*)\]"
)

#: Third-party package sources, which very often have no riscv64 builds.
_THIRD_PARTY_SOURCE = re.compile(
    r"(?:add-apt-repository\s+(?:-{1,2}\S+\s+)*(\S+)"
    r"|(ppa:[\w.+-]+/[\w.+-]+)"
    r"|deb\s+(?:\[[^\]]*\]\s*)?(https?://\S+))",
    re.IGNORECASE,
)

_RPM_VERSION_RELEASE = re.compile(r"-\d+\.[\d.]*\d(?:-[\w.+]+)?$")
#: A shell comment: a # that starts a word. Keeps URL fragments (http://x#y)
#: and shell expansions (${#arr}) intact.
_SHELL_COMMENT = re.compile(r"(?m)(?<![\S$])#.*$")
_LINE_CONTINUATION = re.compile(r"\\\s*\n")
_FLAG = re.compile(r"^-{1,2}")
_QUOTED_ITEM = re.compile(r"['\"]([^'\"]+)['\"]")


def _clean_package(token: str) -> Optional[str]:
    """Normalise one token from an install command into a package name."""
    token = token.strip().strip("'\"`,")
    if not token or _FLAG.match(token):
        return None
    # Strip a version pin: libpq-dev=16.1 or libpq-dev>=16
    token = token.split("/", 1)[0]  # apt's pkg/suite syntax
    token = re.split(r"[=<>]", token, maxsplit=1)[0].strip()
    # Strip an rpm-style version-release, which must carry a dot so that a
    # soname suffix (libpng16-16) survives: libcurl-devel-7.61.1-34.el8 .
    token = _RPM_VERSION_RELEASE.sub("", token)
    if not token or "$" in token or "{" in token or "`" in token:
        return None
    if token in ("&&", "||", "\\", "-", "install", "apt", "apt-get", "sudo"):
        return None
    if "//" in token or token.endswith(":"):
        return None  # a URL caught by an install command on the same line
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.+:-]*$", token):
        return None
    return token


def _bash_arrays(text: str) -> dict[str, list[str]]:
    arrays: dict[str, list[str]] = {}
    for name, body in _BASH_ARRAY.findall(text):
        items = []
        for raw in re.split(r"[\s,]+", body):
            cleaned = _clean_package(raw)
            if cleaned:
                items.append(cleaned)
        if items:
            arrays[name] = items
    return arrays


def _packages_from_install(tail: str, arrays: dict[str, list[str]]) -> list[str]:
    """Expand one install command's argument list, resolving array refs."""
    packages: list[str] = []
    unresolved = False
    for token in re.split(r"\s+", tail.strip()):
        if not token:
            continue
        reference = _ARRAY_REF.fullmatch(token.strip("'\""))
        if reference:
            name = reference.group(1)
            if name in arrays:
                packages.extend(arrays[name])
            else:
                unresolved = True
            continue
        cleaned = _clean_package(token)
        if cleaned:
            packages.append(cleaned)
        elif "$" in token:
            unresolved = True
    if unresolved and not packages:
        packages.append("\x00unresolved")
    return packages


def scan_text(text: str, path: str, findings: CiFindings) -> None:
    """Pull declared packages and third-party sources out of one file."""
    # Comments first, then continuations. A Dockerfile puts comments *between*
    # continued lines, so joining first folds the prose into the command:
    #     zstd-libs \
    #     # libturbojpeg.so is not used by GDAL. Only libjpeg.so*
    #     && rm -f ...
    # which is how "Only" becomes a package name.
    text = _LINE_CONTINUATION.sub(" ", _SHELL_COMMENT.sub("", text))
    arrays = _bash_arrays(text)

    for pattern in (_APT_INSTALL, _DNF_INSTALL, _APK_ADD):
        for tail in pattern.findall(text):
            for package in _packages_from_install(tail, arrays):
                if package == "\x00unresolved":
                    findings.notes.append(
                        f"{path}: an install command used a shell variable that "
                        "could not be resolved; its packages are missing here"
                    )
                    continue
                findings.packages.setdefault(package, path)

    for body in _HPCCM_OSPACKAGES.findall(text):
        for name in _QUOTED_ITEM.findall(body):
            cleaned = _clean_package(name)
            if cleaned:
                findings.packages.setdefault(cleaned, path)

    if path.endswith(".nix"):
        for body in _NIX_INPUTS.findall(text):
            for raw in re.split(r"[\s,]+", body):
                cleaned = _clean_package(raw)
                if cleaned and cleaned not in ("with", "pkgs"):
                    findings.packages.setdefault(cleaned, path)

    for match in _THIRD_PARTY_SOURCE.finditer(text):
        source = next((g for g in match.groups() if g), None)
        if not source or "archive.ubuntu.com" in source or "deb.debian.org" in source:
            continue
        findings.warnings.append(
            f"{path}: adds a third-party package source ({source}) -- check it "
            "publishes builds for your target architecture"
        )


def _is_ci_file(relative: str) -> bool:
    parts = relative.split("/")
    name = parts[-1].lower()
    in_ci_dir = any(part.lower() in CI_DIRS for part in parts[:-1])

    if _DOCKERFILE.match(name) or name.endswith(".dockerfile"):
        return True
    if name in ("apt.txt", "packages.txt", "aptfile"):
        return True
    if name.endswith(".nix"):
        return True
    if name.startswith(".gitlab-ci") or name.endswith(".gitlab-ci.yml"):
        return True
    if name in (".readthedocs.yaml", ".readthedocs.yml"):
        return True
    if in_ci_dir and name.endswith((".yml", ".yaml", ".sh", ".bash", ".py", ".json")):
        return True
    return False


def _apt_txt(text: str, path: str, findings: CiFindings) -> None:
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        cleaned = _clean_package(line)
        if cleaned:
            findings.packages.setdefault(cleaned, path)


def scan_ci_configuration(root: Path, record) -> CiFindings:
    """Walk a repository's CI and container definitions."""
    from .sdist import _is_vendored
    from .source import IGNORED_DIRS

    root = Path(root)
    findings = CiFindings()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d.lower() not in IGNORED_DIRS]
        for filename in sorted(filenames):
            absolute = Path(dirpath) / filename
            try:
                relative = absolute.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - defensive
                continue
            if not _is_ci_file(relative):
                continue
            if _is_vendored(relative):
                # Redis vendors hiredis; hiredis's workflow describes how to
                # build hiredis on its own, not how to build Redis.
                continue
            if findings.files_read >= MAX_CI_FILES:
                findings.notes.append(
                    f"stopped after {MAX_CI_FILES} CI files; some may be unread"
                )
                break
            try:
                text = absolute.read_bytes()[:MAX_CI_BYTES].decode(
                    "utf-8", errors="replace"
                )
            except OSError:
                continue
            findings.files_read += 1
            if filename.lower() in ("apt.txt", "packages.txt", "aptfile"):
                _apt_txt(text, relative, findings)
            else:
                scan_text(text, relative, findings)

    _record_packages(findings, record)
    return findings


#: Packages that say nothing about what a build links against.
_UNINTERESTING = frozenset({
    "ca-certificates", "curl", "wget", "git", "gnupg", "gnupg2", "lsb-release",
    "software-properties-common", "apt-transport-https", "sudo", "tzdata",
    "locales", "openssh-client", "unzip", "zip", "xz-utils", "bzip2", "less",
    "vim", "nano", "procps", "rsync", "jq", "tar", "gzip", "file", "patch",
    "pipx", "ghostscript", "netbase", "dirmngr", "bash", "coreutils",
    "findutils", "diffutils", "grep", "sed", "gawk", "which", "gettext",
    # C runtime and compiler support: pulled in by the compiler, never chosen
    "libc6", "libc-dev", "libc6-dev", "libc6-dbg", "libc-bin", "musl", "musl-dev",
    "libgcc", "libgcc1", "libgcc-s1", "libstdc++", "libstdc++6", "glibc",
    "libstdc++-dev", "linux-libc-dev", "libc6-dev-i386", "libc6-amd64",
    "libstdc++-devel", "libstdc++-static", "lib64stdc++6", "libstdc++5",
    "glibc-devel", "glibc-headers", "glibc-static", "musl-libc", "libc-devel",
    "dpkg-dev", "rpm-build", "epel-release", "build-base", "gcc-multilib",
    "g++-multilib", "python-is-python3", "python3-pip", "ca-certificates-bundle",
    "gcc", "g++", "clang", "make", "pkg-config", "pkgconf", "build-essential",
})


def _record_packages(findings: CiFindings, record) -> None:
    """Feed declared package names into the shared requirement table.

    These are distro package names, not library names, so they resolve
    backwards through the map -- ``libssl-dev`` converges on the same entry a
    scraped ``find_package(OpenSSL)`` produces, instead of being reported
    twice.
    """
    db = database()
    for package, path in sorted(findings.packages.items()):
        if package in _UNINTERESTING:
            continue
        known = db.by_distro_package(package)
        if known is not None:
            record(known.name, known.kind, f"declared in {path}")
            continue
        # -dev / -devel packages exist to be linked against, whatever they
        # are called; everything else unrecognised is treated as a tool.
        kind = (
            "library"
            if package.startswith("lib") or package.endswith(("-dev", "-devel"))
            else "tool"
        )
        record.declare(
            SystemRequirement(
                name=package,
                kind=kind,
                debian=(package,),
                found_in=(f"declared in {path}",),
                declared=True,
            )
        )
