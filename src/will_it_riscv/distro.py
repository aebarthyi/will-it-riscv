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
import threading
import time
from dataclasses import dataclass, replace
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

    @property
    def vendor(self) -> str:
        """Whose archive: ``debian`` or ``ubuntu`` (both are of the Debian family)."""
        return self.id.split(":", 1)[0]


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

#: Where to look for "does this name exist at all": the fullest arch, with
#: the components a riscv64 check leaves out (CUDA is in Debian's non-free).
REFERENCE_ARCH = "amd64"
REFERENCE_COMPONENTS = {
    "debian": ("non-free", "non-free-firmware"),
    "ubuntu": ("restricted", "multiverse"),
}
REFERENCE_BASES = {"ubuntu": "http://archive.ubuntu.com/ubuntu"}


def _get(client: httpx.Client, url: str, tries: int = 3) -> httpx.Response:
    """A GET that survives an archive mirror dropping the connection once or twice."""
    for attempt in range(tries):
        try:
            return client.get(url, timeout=120.0, follow_redirects=True)
        except httpx.TransportError:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _core(package: str) -> str:
    """What a package name is about: libpcre3-dev -> pcre, postgresql-16 -> postgresql."""
    core = package.lower()
    core = re.sub(r"-(dev|devel|bin|tools|utils)$", "", core)
    if core.startswith("lib") and len(core) > 5:
        core = core[3:]
    return re.sub(r"[-.]?[\d.]+$", "", core) or core


#: Near names that are never what a build means: docs, cross toolchains,
#: another language's bindings.
_ASIDE = re.compile(
    r"-doc$|-cross$|-dbg$|^(lib)?(rust|ghc|ocaml|lua[\d.]*|perl|ruby|node|golang|r-cran|php)-|-ocaml-"
)


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
        self._versions: Optional[dict[str, str]] = None
        self.error: Optional[str] = None
        self._contents = None
        self._contents_tried = False
        # Jobs share one index: one loads it while the rest wait for it.
        self._loading = threading.RLock()

    @property
    def available(self) -> bool:
        return self.names() is not None

    def names(self) -> Optional[set[str]]:
        if self._names is not None:
            return self._names
        with self._loading:
            if self._names is not None or self.error is not None:
                return self._names
            cached = self.cache.get("distro", self._key, ttl=LONG_TTL)
            if cached is not None:
                self._names = set(cached.decode("utf-8").split("\n"))
                return self._names
            self._download()
            return self._names

    def version(self, package: str) -> Optional[str]:
        """The version the archive has, for a real package; None for a virtual one.

        A name being there is not always enough: jax pins Bazel 8.7.0, and
        Debian 13's bazel-bootstrap is 4.2.3.
        """
        if self._versions is None:
            with self._loading:
                if self._versions is None:
                    self._load_versions()
        return (self._versions or {}).get(package)

    def _load_versions(self) -> None:
        cached = self.cache.get("distro", self._key + "/versions", ttl=LONG_TTL)
        if cached is not None:
            self._versions = dict(
                line.split("\t", 1)
                for line in cached.decode("utf-8").split("\n") if "\t" in line
            )
        elif self.error is None:
            self._download()

    @property
    def _key(self) -> str:
        return f"{self.spec.id}/{self.arch}/{','.join(self.spec.components)}"

    def _download(self) -> None:
        collected: set[str] = set()
        versions: dict[str, str] = {}
        suite = self.spec.id.split(":", 1)[1]
        for component in self.spec.components:
            url = (
                f"{self.spec.base}/dists/{suite}/{component}/"
                f"binary-{self.arch}/Packages.gz"
            )
            try:
                response = _get(self.client, url)
            except httpx.HTTPError as exc:
                self.error = f"{url}: {exc}"
                return
            if response.status_code != 200:
                if component == self.spec.components[0]:
                    self.error = (
                        f"{self.spec.label} has no {self.arch} port "
                        f"(HTTP {response.status_code} for {component})"
                    )
                    return
                continue
            try:
                raw = gzip.decompress(response.content)
            except (OSError, EOFError) as exc:
                self.error = f"{url}: {exc}"
                return
            collected.update(m.decode("utf-8") for m in _PACKAGE_LINE.findall(raw))
            for line in _PROVIDES_LINE.findall(raw):
                for virtual in line.decode("utf-8").split(","):
                    name = virtual.strip().split(" ")[0]
                    if name:
                        collected.add(name)
            current = None
            for line in raw.split(b"\n"):
                if line.startswith(b"Package: "):
                    current = line[9:].strip().decode("utf-8")
                elif line.startswith(b"Version: ") and current is not None:
                    versions.setdefault(current, line[9:].strip().decode("utf-8"))
                    current = None

        if not collected:
            self.error = f"{self.spec.label}: empty package index for {self.arch}"
            return
        self.cache.put("distro", self._key, "\n".join(sorted(collected)).encode("utf-8"))
        self.cache.put(
            "distro", self._key + "/versions",
            "\n".join(f"{k}\t{v}" for k, v in sorted(versions.items())).encode("utf-8"),
        )
        self._names = collected
        self._versions = versions

    def contents(self):
        """Which package ships which file, for this distro and arch; None if unavailable."""
        # Tried only once loaded: a job asking while another downloads it
        # waits, rather than taking None for "no index" -- mosquitto's GTest
        # and CUnit went unchecked when six jobs found the cache expired.
        with self._loading:
            if not self._contents_tried:
                from .contents import ContentsIndex

                self._contents = ContentsIndex.load(
                    self.client, self.cache, self.spec, self.arch)
                self._contents_tried = True
        return self._contents

    def has(self, package: str) -> bool:
        names = self.names()
        return bool(names and package in names)

    def named(self, package: str) -> Optional[bool]:
        """Whether the distro has a package by this name at all, on any arch.

        Not having it for the target is an answer; not having it anywhere is
        a wrong name -- openmpi for libopenmpi-dev, Alpine's build-base, a
        package another release or distro has and this one does not. The same
        release's amd64, with every component, stands in for "anywhere" (for
        Ubuntu, from the main archive: ports has no amd64). None when that
        list cannot be had.
        """
        if self.has(package):
            return True
        reference = self._reference()
        names = reference.names() if reference is not None else None
        if names is None:
            return None
        return package in names

    def similar(self, package: str, limit: int = 3) -> list[str]:
        """Names for the target that look like a wrong one, the likeliest first."""
        import difflib

        core = _core(package)
        names = self.names()
        if not names or len(core) < 2:
            return []
        canonical = [f"lib{core}-dev", f"{core}-dev", core, f"{core}-bin", f"lib{core}"]

        def rank(name: str) -> tuple:
            return (
                canonical.index(name) if name in canonical else len(canonical),
                -difflib.SequenceMatcher(None, package, name).ratio(), name,
            )

        return sorted((n for n in names if core in n and not _ASIDE.search(n)), key=rank)[:limit]

    def _reference(self) -> Optional[DistroIndex]:
        vendor = self.spec.vendor
        if self.arch == REFERENCE_ARCH and not REFERENCE_COMPONENTS.get(vendor):
            return self
        if not hasattr(self, "_reference_index"):
            extra = REFERENCE_COMPONENTS.get(vendor, ())
            spec = replace(
                self.spec, base=REFERENCE_BASES.get(vendor, self.spec.base),
                components=self.spec.components + tuple(
                    c for c in extra if c not in self.spec.components),
            )
            self._reference_index = DistroIndex(self.client, self.cache, spec, REFERENCE_ARCH)
        return self._reference_index

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


def upstream_version(debian_version: str) -> Optional[str]:
    """The upstream part of a Debian version, comparable with PEP 440.

    ``1:4.2.3+ds-11`` is 4.2.3: drop the epoch, the Debian revision, and any
    repacking suffix after the leading dotted number.
    """
    text = debian_version.split(":", 1)[-1]
    text = text.rsplit("-", 1)[0] if "-" in text else text
    match = re.match(r"^(\d+(?:\.\d+)*)", text)
    return match.group(1) if match else None


def resolve_spec(identifier: str) -> DistroSpec:
    key = identifier.strip().lower()
    if key in KNOWN_DISTROS:
        return KNOWN_DISTROS[key]
    if key in FAMILY_DEFAULTS:
        return KNOWN_DISTROS[FAMILY_DEFAULTS[key]]
    raise KeyError(
        f"unknown distro {identifier!r}; choose from {', '.join(sorted(KNOWN_DISTROS))}"
    )
