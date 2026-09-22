"""Shared fixtures: synthetic sdists and a fake index, so tests never hit PyPI."""

from __future__ import annotations

import io
import tarfile
from typing import Optional

import pytest

from will_it_riscv.index import IndexFile, PackageIndex, Project, Release, _parse_project
from will_it_riscv.target import Target


def make_sdist(name: str, version: str, files: dict[str, str]) -> bytes:
    """Build an in-memory .tar.gz laid out the way a real sdist is."""
    buffer = io.BytesIO()
    root = f"{name}-{version}"
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(f"{root}/{path}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeIndex(PackageIndex):
    """A PackageIndex backed by literal filename lists and metadata strings."""

    def __init__(
        self,
        projects: dict[str, list[str]],
        metadata: Optional[dict[str, str]] = None,
        sdists: Optional[dict[str, bytes]] = None,
    ):
        self._payloads = projects
        self._metadata = metadata or {}
        self._sdists = sdists or {}
        self._cached: dict[str, Optional[Project]] = {}
        self.downloads: list[str] = []

    def project(self, name: str) -> Optional[Project]:
        from packaging.utils import canonicalize_name

        canonical = canonicalize_name(name)
        if canonical in self._cached:
            return self._cached[canonical]
        filenames = self._payloads.get(canonical)
        if filenames is None:
            self._cached[canonical] = None
            return None
        payload = {
            "files": [
                {
                    "filename": filename,
                    "url": f"https://example.invalid/{filename}",
                    "core-metadata": filename.endswith(".whl"),
                }
                for filename in filenames
            ]
        }
        project = _parse_project(canonical, payload)
        self._cached[canonical] = project
        return project

    def metadata(self, file: IndexFile) -> Optional[bytes]:
        raw = self._metadata.get(file.filename)
        return raw.encode("utf-8") if raw else None

    def download(self, file: IndexFile, max_bytes: int = 0) -> Optional[bytes]:
        self.downloads.append(file.filename)
        return self._sdists.get(file.filename)


@pytest.fixture
def target() -> Target:
    return Target(arch="riscv64", libc="glibc", libc_version=(2, 39), python_version=(3, 12))


def metadata(*requires: str, name: str = "x", version: str = "1.0", extras: tuple = ()) -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
    ]
    lines += [f"Provides-Extra: {e}" for e in extras]
    lines += [f"Requires-Dist: {r}" for r in requires]
    return "\n".join(lines) + "\n\n"


def release_of(project: Project, version: str) -> Release:
    from packaging.version import Version

    return project.releases[Version(version)]
