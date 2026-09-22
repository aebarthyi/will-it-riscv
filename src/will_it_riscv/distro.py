"""Is a package available from the distro, already built for the target?

Checking this turns "you will have to build libopenblas" into "apt install
libopenblas-dev", which is usually the answer people actually want. It also
catches the case where the distro already ships the Python package itself, so
there is nothing to build at all.

Only Debian-family distros are checked for real. Fedora does not carry riscv64
in its primary repositories, so Fedora names are reported as suggestions
without an availability check.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import httpx
from packaging.utils import canonicalize_name

from .cache import LONG_TTL, Cache

try:  # pragma: no cover - trivial
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DATA = Path(__file__).parent / "data" / "syslibs.toml"


@dataclass(frozen=True)
class DistroSpec:
    id: str
    label: str
    base: str
    components: tuple[str, ...]
    family: str = "debian"


KNOWN_DISTROS = {
    "debian:trixie": DistroSpec(
        "debian:trixie", "Debian 13 (trixie)",
        "https://deb.debian.org/debian", ("main", "contrib"),
    ),
    "debian:sid": DistroSpec(
        "debian:sid", "Debian unstable (sid)",
        "https://deb.debian.org/debian", ("main", "contrib"),
    ),
    "debian:forky": DistroSpec(
        "debian:forky", "Debian 14 (forky)",
        "https://deb.debian.org/debian", ("main", "contrib"),
    ),
    "ubuntu:noble": DistroSpec(
        "ubuntu:noble", "Ubuntu 24.04 LTS (noble)",
        "http://ports.ubuntu.com/ubuntu-ports", ("main", "universe"),
    ),
    "ubuntu:plucky": DistroSpec(
        "ubuntu:plucky", "Ubuntu 25.04 (plucky)",
        "http://ports.ubuntu.com/ubuntu-ports", ("main", "universe"),
    ),
    "ubuntu:questing": DistroSpec(
        "ubuntu:questing", "Ubuntu 25.10 (questing)",
        "http://ports.ubuntu.com/ubuntu-ports", ("main", "universe"),
    ),
    "ubuntu:resolute": DistroSpec(
        "ubuntu:resolute", "Ubuntu 26.04 LTS (resolute)",
        "http://ports.ubuntu.com/ubuntu-ports", ("main", "universe"),
    ),
}

DEFAULT_DISTRO = "debian:trixie"

#: What a bare family name resolves to: the current stable for Debian, the
#: newest LTS for Ubuntu. Spelled out rather than left to dict order, so
#: adding a release cannot silently change what ``--distro ubuntu`` means.
FAMILY_DEFAULTS = {
    "debian": "debian:trixie",
    "ubuntu": "ubuntu:resolute",
}

_PACKAGE_LINE = re.compile(rb"^Package: (\S+)$", re.MULTILINE)
_PROVIDES_LINE = re.compile(rb"^Provides: (.+)$", re.MULTILINE)


@lru_cache(maxsize=1)
def _pymap() -> dict[str, str]:
    raw = tomllib.loads(DATA.read_text(encoding="utf-8"))
    return {canonicalize_name(k): v for k, v in raw.get("pymap", {}).items()}


class DistroIndex:
    """Lazily downloads and caches a distro's binary package list for one arch."""

    def __init__(self, client: httpx.Client, cache: Cache, spec: DistroSpec, arch: str):
        self.client = client
        self.cache = cache
        self.spec = spec
        self.arch = arch
        self._names: Optional[set[str]] = None
        self.error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.names() is not None

    def names(self) -> Optional[set[str]]:
        if self._names is not None:
            return self._names
        if self.error is not None:
            return None
        key = f"{self.spec.id}/{self.arch}/{','.join(self.spec.components)}"
        cached = self.cache.get("distro", key, ttl=LONG_TTL)
        if cached is not None:
            self._names = set(cached.decode("utf-8").split("\n"))
            return self._names

        collected: set[str] = set()
        suite = self.spec.id.split(":", 1)[1]
        for component in self.spec.components:
            url = (
                f"{self.spec.base}/dists/{suite}/{component}/"
                f"binary-{self.arch}/Packages.gz"
            )
            try:
                response = self.client.get(url, timeout=120.0, follow_redirects=True)
            except httpx.HTTPError as exc:
                self.error = f"{url}: {exc}"
                return None
            if response.status_code != 200:
                if component == self.spec.components[0]:
                    self.error = (
                        f"{self.spec.label} has no {self.arch} port "
                        f"(HTTP {response.status_code} for {component})"
                    )
                    return None
                continue
            try:
                raw = gzip.decompress(response.content)
            except (OSError, EOFError) as exc:
                self.error = f"{url}: {exc}"
                return None
            collected.update(m.decode("utf-8") for m in _PACKAGE_LINE.findall(raw))
            for line in _PROVIDES_LINE.findall(raw):
                for virtual in line.decode("utf-8").split(","):
                    name = virtual.strip().split(" ")[0]
                    if name:
                        collected.add(name)

        if not collected:
            self.error = f"{self.spec.label}: empty package index for {self.arch}"
            return None
        self.cache.put("distro", key, "\n".join(sorted(collected)).encode("utf-8"))
        self._names = collected
        return self._names

    def has(self, package: str) -> bool:
        names = self.names()
        return bool(names and package in names)

    def first_available(self, candidates) -> Optional[str]:
        names = self.names()
        if not names:
            return None
        for candidate in candidates:
            if candidate in names:
                return candidate
        return None

    def python_package(self, dist_name: str) -> Optional[str]:
        """Find a ``python3-*`` package providing this PyPI distribution."""
        names = self.names()
        if not names:
            return None
        canonical = canonicalize_name(dist_name)
        suffixes = []
        override = _pymap().get(canonical)
        if override:
            suffixes.append(override)
        suffixes.append(canonical)
        if canonical.startswith("python-"):
            suffixes.append(canonical[len("python-") :])
        if canonical.startswith("py") and len(canonical) > 3:
            suffixes.append(canonical[2:])
        for suffix in suffixes:
            for variant in {suffix, suffix.replace("-", "."), suffix.replace("-", "")}:
                candidate = f"python3-{variant}"
                if candidate in names:
                    return candidate
        return None


def resolve_spec(identifier: str) -> DistroSpec:
    key = identifier.strip().lower()
    if key in KNOWN_DISTROS:
        return KNOWN_DISTROS[key]
    if key in FAMILY_DEFAULTS:
        return KNOWN_DISTROS[FAMILY_DEFAULTS[key]]
    raise KeyError(
        f"unknown distro {identifier!r}; choose from {', '.join(sorted(KNOWN_DISTROS))}"
    )
