"""Which distro package ships what a build looks for, from the archive's own file lists.

A configure never asks for ``libtbb-dev``. It asks for ``TBBConfig.cmake``,
``gmodule-2.0.pc``, a header its Find module searches for, or a program on
PATH. The archive's Contents index says which package ships each of those
files, for the target's architecture, so the answer is looked up rather
than remembered: ``find_package(TBB)`` is libtbb-dev because libtbb-dev
ships TBBConfig.cmake, and the curated map in syslibs.toml only has to
cover what no file answers.

Lookups go the way CMake goes: a config file for find_package, a ``.pc``
for pkg-config, then the headers and libraries the Find module searched
for (a ``lib<name>.so`` symlink lives in the -dev package), then programs.
When a split-out package ships the file -- Debian's libgio-2.0-dev ships
gmodule-2.0.pc -- the -dev package people know, from the same source and
depending on it, is accepted too.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from .cache import LONG_TTL, Cache

CONFIG = re.compile(
    r"^usr/(?:lib/[^/]+/|lib/|share/)cmake/[^/]+/([^/]+?)(?:-config|Config)\.cmake$"
)
PC = re.compile(r"^usr/(?:lib/[^/]+|share|lib)/pkgconfig/([^/]+)\.pc$")
HEADER = re.compile(r"^usr/include/(?:[a-z0-9_]+-linux-gnu[a-z0-9_]*/)?(.+)$")
LIBRARY = re.compile(r"^usr/lib/(?:[a-z0-9_]+-linux-gnu[a-z0-9_]*/)?lib([^/]+)\.so$")
PROGRAM = re.compile(r"^usr/s?bin/([^/]+)$")
#: More packages than this answering one lookup means the lookup says nothing.
AMBIGUOUS = 4


@dataclass
class ContentsIndex:
    configs: dict = field(default_factory=lambda: defaultdict(set))
    pcs: dict = field(default_factory=lambda: defaultdict(set))
    headers: dict = field(default_factory=lambda: defaultdict(set))
    libraries: dict = field(default_factory=lambda: defaultdict(set))
    programs: dict = field(default_factory=lambda: defaultdict(set))
    sources: dict = field(default_factory=dict)
    """Binary package to its source package."""
    depends: dict = field(default_factory=dict)

    TABLES = ("configs", "pcs", "headers", "libraries", "programs")

    # ------------------------------------------------------------ reading

    def read_contents(self, lines: Iterable[str]) -> None:
        for line in lines:
            path, _, owners = line.rstrip("\n").rpartition(" ")
            path = path.strip()
            if not path or not owners or not path.startswith("usr/"):
                continue
            packages = {o.rsplit("/", 1)[-1] for o in owners.split(",")}
            for pattern, table in (
                (CONFIG, self.configs),
                (PC, self.pcs),
                (LIBRARY, self.libraries),
                (HEADER, self.headers),
                (PROGRAM, self.programs),
            ):
                match = pattern.match(path)
                if match:
                    table[match.group(1).lower()] |= packages
                    break

    def read_packages(self, text: str) -> None:
        """Each binary's source and what it depends on, from a Packages index."""
        for stanza in text.split("\n\n"):
            fields = dict(re.findall(r"^([A-Za-z-]+): (.*)$", stanza, re.M))
            name = fields.get("Package")
            if not name:
                continue
            self.sources[name] = fields.get("Source", name).split(" ")[0]
            self.depends[name] = sorted(
                {
                    d.strip().split(" ")[0].split(":")[0]
                    for d in re.split(r"[,|]", fields.get("Depends", ""))
                    if d.strip()
                }
            )

    def to_json(self) -> bytes:
        raw = {name: {k: sorted(v) for k, v in getattr(self, name).items()} for name in self.TABLES}
        raw["sources"], raw["depends"] = self.sources, self.depends
        return gzip.compress(json.dumps(raw, separators=(",", ":")).encode())

    @classmethod
    def from_json(cls, blob: bytes) -> ContentsIndex:
        raw = json.loads(gzip.decompress(blob))
        index = cls(sources=raw["sources"], depends=raw["depends"])
        for name in cls.TABLES:
            table = getattr(index, name)
            for k, v in raw[name].items():
                table[k] = set(v)
        return index

    @classmethod
    def load(cls, client, cache: Cache, spec, arch: str) -> Optional[ContentsIndex]:
        """The index for one distro and architecture, downloaded once and cached.

        Debian keeps Contents under each component, Ubuntu's ports archive at
        the suite; both are tried. None when neither answers.
        """
        key = f"v2/{spec.id}/{arch}/{','.join(spec.components)}"
        blob = cache.get("contents", key, ttl=LONG_TTL)
        if blob is not None:
            try:
                return cls.from_json(blob)
            except (OSError, ValueError, KeyError):
                pass
        suite = spec.id.split(":", 1)[1]
        index = cls()
        got_contents = False
        # Architecture-independent packages -- header-only and CMake-only
        # ones like nlohmann-json3-dev -- are listed apart, under "all".
        for component in spec.components:
            for listing in (arch, "all"):
                raw = _fetch(client, f"{spec.base}/dists/{suite}/{component}/Contents-{listing}.gz")
                if raw is not None:
                    index.read_contents(
                        gzip.decompress(raw).decode("utf-8", "replace").splitlines()
                    )
                    got_contents = True
            raw = _fetch(client, f"{spec.base}/dists/{suite}/{component}/binary-{arch}/Packages.gz")
            if raw is not None:
                index.read_packages(gzip.decompress(raw).decode("utf-8", "replace"))
        if not got_contents:
            # Ubuntu's ports archive keeps one Contents for the whole suite.
            for listing in (arch, "all"):
                raw = _fetch(client, f"{spec.base}/dists/{suite}/Contents-{listing}.gz")
                if raw is not None:
                    index.read_contents(
                        gzip.decompress(raw).decode("utf-8", "replace").splitlines()
                    )
                    got_contents = True
        if not got_contents:
            return None
        cache.put("contents", key, index.to_json())
        return index

    # ------------------------------------------------------------ asking

    def providers(self, name: str, command: str = "find_package", lookups=()) -> tuple[tuple, str]:
        """The packages that satisfy one lookup, -dev first, and how that is known."""
        key = name.lower()
        steps = []
        if command == "find_package":
            steps.append((self.configs, [key], f"ships {name}Config.cmake"))
        if command in ("pkg_check_modules", "pkg_search_module", "dependency"):
            steps.append((self.pcs, [key], f"ships {name}.pc"))
        headers = [n for kind, _, names in lookups if kind == "find_path" for n in names]
        libraries = [n for kind, _, names in lookups if kind == "find_library" for n in names]
        programs = [n for kind, _, names in lookups if kind == "find_program" for n in names]
        if headers:
            steps.append(
                (self.headers, [h.lower() for h in headers], "ships " + ", ".join(headers[:3]))
            )
        if libraries:
            steps.append(
                (
                    self.libraries,
                    [_bare(n) for n in libraries],
                    "ships " + ", ".join(f"lib{_bare(n)}.so" for n in libraries[:3]),
                )
            )
        if programs:
            steps.append(
                (self.programs, [p.lower() for p in programs], "ships " + ", ".join(programs[:3]))
            )
        if command == "find_program":
            steps.append((self.programs, [key], f"ships {name}"))
        # Nothing the lookup itself named: the package's own spellings.
        steps.append((self.pcs, [key], f"ships {name}.pc"))
        steps.append((self.libraries, [_bare(key)], f"ships lib{_bare(key)}.so"))
        # Last, a program of that name: find_package(Doxygen) wants doxygen.
        if command != "find_program":
            steps.append((self.programs, [key], f"ships {key}"))
        for table, keys, how in steps:
            found: set = set()
            for k in keys:
                found |= table.get(k, set())
            chosen = _prefer_dev(found)
            if chosen:
                return self.rank(self.widen(chosen)), how
        return (), ""

    def rank(self, packages: set) -> tuple:
        """-dev first, then what the others depend on: qtbase5-dev before its
        add-ons, since each of them pulls it in."""
        pulled = {
            p: sum(p in self.depends.get(o, ()) for o in packages if o != p) for p in packages
        }
        return tuple(sorted(packages, key=lambda p: (not p.endswith("-dev"), -pulled[p], p)))

    def widen(self, providers: set) -> set:
        """Add the -dev packages of the same source that pull a provider in:
        libglib2.0-dev for gmodule-2.0, which Debian now ships in libgio-2.0-dev."""
        wider = set(providers)
        for name, needs in self.depends.items():
            if (
                name.endswith("-dev")
                and providers.intersection(needs)
                and any(self.sources.get(name) == self.sources.get(p) for p in providers)
            ):
                wider.add(name)
        return wider


def _bare(library: str) -> str:
    """``libz.so``, ``-lz``, ``z`` and ``libz`` are all the same library."""
    name = library.lower().removeprefix("-l")
    name = re.sub(r"\.(so|a|dylib|lib)(\.[0-9.]*)?$", "", name)
    return name.removeprefix("lib") if name.startswith("lib") and len(name) > 3 else name


def _prefer_dev(found: set) -> set:
    if not found:
        return set()
    dev = {p for p in found if p.endswith("-dev")}
    chosen = dev or found
    return chosen if len(chosen) <= AMBIGUOUS else set()


def _fetch(client, url: str) -> Optional[bytes]:
    try:
        response = client.get(url, timeout=300.0, follow_redirects=True)
    except Exception:  # noqa: BLE001 - any failure means this URL has nothing
        return None
    return response.content if response.status_code == 200 else None
