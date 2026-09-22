"""Client for a PEP 503 / PEP 691 package index.

Two things make this fast enough to walk a whole dependency tree:

* the JSON simple API (PEP 691) gives us every file for a project in one
  request, and
* PEP 658 ``core-metadata`` lets us fetch a wheel's ``METADATA`` as a separate
  small file instead of downloading the wheel itself.

Only the second is optional -- for sdist-only projects we fall back to
downloading the sdist, which :mod:`will_it_riscv.sdist` wants anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import httpx
from packaging.tags import Tag
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

from .cache import DEFAULT_TTL, Cache

SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"
PYPI_SIMPLE = "https://pypi.org/simple/"


@dataclass
class IndexFile:
    """One downloadable artifact from an index page."""

    filename: str
    url: str
    yanked: bool = False
    requires_python: Optional[str] = None
    core_metadata: bool = False
    size: Optional[int] = None

    @property
    def is_wheel(self) -> bool:
        return self.filename.endswith(".whl")

    @property
    def is_sdist(self) -> bool:
        return self.filename.endswith((".tar.gz", ".zip", ".tar.bz2", ".tgz"))


@dataclass
class Release:
    """Every file published under one version of a project."""

    version: Version
    wheels: list[IndexFile] = field(default_factory=list)
    sdists: list[IndexFile] = field(default_factory=list)
    _tags: dict[str, frozenset] = field(default_factory=dict, repr=False)

    def wheel_tags(self, wheel: IndexFile) -> frozenset:
        cached = self._tags.get(wheel.filename)
        if cached is None:
            try:
                cached = parse_wheel_filename(wheel.filename)[3]
            except InvalidWheelFilename:
                cached = frozenset()
            self._tags[wheel.filename] = cached
        return cached

    def pure_wheels(self) -> list[IndexFile]:
        """Wheels tagged ``any`` -- installable on every platform."""
        return [
            w
            for w in self.wheels
            if any(t.platform == "any" for t in self.wheel_tags(w))
        ]

    def matching_wheels(self, accepted: frozenset) -> list[IndexFile]:
        return [w for w in self.wheels if self.wheel_tags(w) & accepted]

    def platform_tags(self) -> list[str]:
        """Distinct platform tags across this release's wheels."""
        seen: dict[str, None] = {}
        for w in self.wheels:
            for tag in self.wheel_tags(w):
                seen.setdefault(tag.platform, None)
        return sorted(seen)

    @property
    def has_wheels(self) -> bool:
        return bool(self.wheels)

    @property
    def has_sdist(self) -> bool:
        return bool(self.sdists)


@dataclass
class Project:
    """An index page, parsed and grouped by version."""

    name: str
    releases: dict[Version, Release] = field(default_factory=dict)

    def versions(self, allow_yanked: bool = False) -> list[Version]:
        out = []
        for version, release in self.releases.items():
            files = release.wheels + release.sdists
            if not allow_yanked and files and all(f.yanked for f in files):
                continue
            out.append(version)
        return sorted(out, reverse=True)


class PackageIndex:
    """Fetches and caches index pages and wheel metadata."""

    def __init__(
        self,
        client: httpx.Client,
        cache: Cache,
        url: str = PYPI_SIMPLE,
        ttl: int = DEFAULT_TTL,
    ):
        self.client = client
        self.cache = cache
        self.url = url.rstrip("/") + "/"
        self.ttl = ttl
        self._projects: dict[str, Optional[Project]] = {}

    def project(self, name: str) -> Optional[Project]:
        """Return the index page for ``name``, or None if the index 404s."""
        canonical = canonicalize_name(name)
        if canonical in self._projects:
            return self._projects[canonical]
        project = self._fetch_project(canonical)
        self._projects[canonical] = project
        return project

    def _fetch_project(self, canonical: str) -> Optional[Project]:
        url = f"{self.url}{canonical}/"
        cached = self.cache.get_json("simple", url, self.ttl)
        if cached is None:
            try:
                response = self.client.get(url, headers={"Accept": SIMPLE_JSON})
            except httpx.HTTPError as exc:
                raise IndexError_(f"{canonical}: {exc}") from exc
            if response.status_code == 404:
                self.cache.put_json("simple", url, {"files": [], "_missing": True})
                return None
            response.raise_for_status()
            if SIMPLE_JSON not in response.headers.get("content-type", ""):
                raise IndexError_(
                    f"{url} did not serve the PEP 691 JSON API; "
                    "this tool requires a JSON-capable index"
                )
            cached = response.json()
            self.cache.put_json("simple", url, cached)
        if cached.get("_missing"):
            return None
        return _parse_project(canonical, cached)

    def metadata(self, file: IndexFile) -> Optional[bytes]:
        """Fetch a wheel's ``METADATA`` via PEP 658, without the wheel."""
        if not file.core_metadata:
            return None
        url = file.url + ".metadata"
        cached = self.cache.get("metadata", url, ttl=-1)
        if cached is not None:
            return cached
        try:
            response = self.client.get(url)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        self.cache.put("metadata", url, response.content)
        return response.content

    def download(self, file: IndexFile, max_bytes: int = 80 * 1024 * 1024) -> Optional[bytes]:
        """Download an artifact (used for sdists). Returns None if too large."""
        cached = self.cache.get("artifact", file.url, ttl=-1)
        if cached is not None:
            return cached
        if file.size is not None and file.size > max_bytes:
            return None
        try:
            with self.client.stream("GET", file.url) as response:
                if response.status_code != 200:
                    return None
                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        return None
                    chunks.append(chunk)
        except httpx.HTTPError:
            return None
        blob = b"".join(chunks)
        self.cache.put("artifact", file.url, blob)
        return blob


class IndexError_(RuntimeError):
    """Raised when the index itself is unusable (network, wrong API)."""


def _parse_project(canonical: str, payload: dict) -> Project:
    project = Project(name=canonical)
    for entry in payload.get("files", []):
        filename = entry.get("filename", "")
        if not filename:
            continue
        core_metadata = entry.get("core-metadata") or entry.get("data-dist-info-metadata")
        file = IndexFile(
            filename=filename,
            url=entry.get("url", ""),
            yanked=bool(entry.get("yanked")),
            requires_python=entry.get("requires-python"),
            core_metadata=bool(core_metadata),
            size=entry.get("size"),
        )
        version = _version_of(filename, canonical)
        if version is None:
            continue
        release = project.releases.setdefault(version, Release(version=version))
        if file.is_wheel:
            release.wheels.append(file)
        elif file.is_sdist:
            release.sdists.append(file)
    return project


def _version_of(filename: str, canonical: str) -> Optional[Version]:
    try:
        if filename.endswith(".whl"):
            return parse_wheel_filename(filename)[1]
        return parse_sdist_filename(filename)[1]
    except (InvalidWheelFilename, InvalidSdistFilename, InvalidVersion):
        return None


def tag_platforms(tags: frozenset) -> list[str]:
    return sorted({t.platform for t in tags if isinstance(t, Tag)})
