"""Fetch the source of a dependency, at the version the plan resolved.

Recursion needs the thing itself, not its name. A package with no riscv64
wheel still has source: usually an sdist on the index, and failing that --
jaxlib publishes wheels and nothing else -- the repository its metadata
points at, checked out at the tag for that version. Either is unpacked
into the cache, once, and read from there on.

Nothing fetched is ever run here. Running a fetched package's configure is
the recursion's business, and it happens in the same pretend environment as
the root's: confined, in scratch.
"""

from __future__ import annotations

import email.parser
import io
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:  # pragma: no cover
    from .index import PackageIndex

#: A repository checkout is shallow, but some are still large.
GIT_TIMEOUT = 300


@dataclass
class SourceTree:
    name: str
    version: Optional[str]
    path: Optional[Path] = None
    kind: str = ""
    """``sdist`` or ``git``."""
    origin: str = ""
    """The sdist's filename, or ``url@tag``."""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.path is not None and self.error is None


def fetch(
    index: PackageIndex, name: str, version: Optional[str], cache_root: Path
) -> SourceTree:
    """The sdist if there is one, else the repository its metadata names."""
    canonical = canonicalize_name(name)
    tree = SourceTree(name=str(canonical), version=version)
    project = index.project(str(canonical))
    if project is None:
        tree.error = f"{canonical} is not on the index"
        return tree
    release = None
    if version is not None:
        try:
            release = project.releases.get(Version(version))
        except InvalidVersion:
            release = None
    if release is None:
        tree.error = f"{canonical} {version} is not on the index"
        return tree

    destination = cache_root / f"{canonical}-{version}"
    if release.sdists:
        sdist = release.sdists[0]
        tree.kind, tree.origin = "sdist", sdist.filename
        if not (destination / ".unpacked").exists():
            blob = index.download(sdist)
            if blob is None:
                tree.error = f"could not download {sdist.filename}"
                return tree
            error = unpack(blob, sdist.filename, destination)
            if error:
                tree.error = error
                return tree
        tree.path = _single_root(destination)
        return tree

    url = repository_url(index, release)
    if url is None:
        tree.error = "no sdist, and its metadata names no repository"
        return tree
    tag = matching_tag(url, str(canonical), version or "")
    if tag is None:
        tree.error = f"no tag for {version} in {url}"
        return tree
    tree.kind, tree.origin = "git", f"{url}@{tag}"
    if not (destination / ".unpacked").exists():
        error = _clone(url, tag, destination)
        if error:
            tree.error = error
            return tree
    tree.path = destination
    return tree


# --------------------------------------------------------------------- sdist


def unpack(blob: bytes, filename: str, destination: Path) -> Optional[str]:
    """Unpack an sdist, refusing anything that would land outside it."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
                for name in zipped.namelist():
                    if not (root / name).resolve().is_relative_to(root):
                        return f"{filename}: {name} would land outside the source tree"
                zipped.extractall(destination)
        else:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tarred:
                # Links are dropped: reading or configuring an sdist never
                # needs one, and a link is how an archive escapes its root.
                members = [m for m in tarred.getmembers() if not (m.issym() or m.islnk())]
                for info in members:
                    if not (root / info.name).resolve().is_relative_to(root):
                        return f"{filename}: {info.name} would land outside the source tree"
                tarred.extractall(destination, members=members)
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as exc:
        return f"{filename}: {exc}"
    (destination / ".unpacked").touch()
    return None


def _single_root(destination: Path) -> Path:
    """An sdist unpacks to one ``name-version/`` directory; step into it."""
    entries = [p for p in destination.iterdir() if p.name != ".unpacked"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return destination


# ---------------------------------------------------------------- repository

_URL_LABELS = ("source", "source code", "repository", "code", "github", "homepage", "home")


def repository_url(index: PackageIndex, release) -> Optional[str]:
    """Where the source lives, from any wheel's metadata."""
    for wheel in release.wheels:
        raw = index.metadata(wheel)
        if raw is None:
            continue
        message = email.parser.BytesParser().parsebytes(raw)
        candidates: list[tuple[int, str]] = []
        for value in message.get_all("Project-URL") or []:
            label, _, url = value.partition(",")
            label = label.strip().lower()
            rank = _URL_LABELS.index(label) if label in _URL_LABELS else len(_URL_LABELS)
            candidates.append((rank, url.strip()))
        home = message.get("Home-page")
        if home:
            candidates.append((len(_URL_LABELS), home.strip()))
        for _, url in sorted(candidates):
            repo = _git_url(url)
            if repo:
                return repo
    return None


def _git_url(url: str) -> Optional[str]:
    match = re.match(
        r"^https?://(github\.com|gitlab\.com|codeberg\.org)/([^/\s]+)/([^/\s#?]+)", url
    )
    if not match:
        return None
    host, owner, repo = match.groups()
    repo = repo.removesuffix(".git")
    return f"https://{host}/{owner}/{repo}"


def matching_tag(url: str, name: str, version: str) -> Optional[str]:
    """The tag a version was released as: ``jaxlib-v0.11.2``, ``v2.5.3``, ``2.5.3``."""
    try:
        process = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", url],
            capture_output=True, text=True, timeout=GIT_TIMEOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if process.returncode != 0:
        return None
    tags = [line.rsplit("refs/tags/", 1)[-1] for line in process.stdout.splitlines()]
    return choose_tag(tags, name, version)


def choose_tag(tags: list[str], name: str, version: str) -> Optional[str]:
    """Prefer the tag that names the package, then a bare version."""
    if not version:
        return None
    escaped = re.escape(version)
    exact = re.compile(rf"(?:^|[-_/v]){escaped}$")
    hits = [t for t in tags if exact.search(t)]
    if not hits:
        return None
    squashed = canonicalize_name(name).replace("-", "")
    named = [t for t in hits if squashed in t.lower().replace("-", "")]
    ranked = sorted(named or hits, key=lambda t: (len(t), t))
    return ranked[0]


def _clone(url: str, tag: str, destination: Path) -> Optional[str]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", tag, url, str(destination)],
            capture_output=True, text=True, timeout=GIT_TIMEOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git clone {url}: {exc}"
    if process.returncode != 0:
        tail = (process.stderr or "").strip().splitlines()
        return f"git clone {url} at {tag}: {tail[-1] if tail else 'failed'}"
    (destination / ".unpacked").touch()
    return None
