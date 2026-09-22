"""Turning raw names scraped out of build files into system packages."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .models import SystemRequirement

try:  # pragma: no cover - trivial
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]

DATA = Path(__file__).parent / "data" / "syslibs.toml"

#: Version constraints and decorations that ride along with a pkg-config or
#: CMake name: ``openssl >= 1.1``, ``libpng16``, ``Boost::python``.
_DECORATION = re.compile(r"\s*(?:[<>=!~]+\s*[\d.]+.*)$")
_NAMESPACE = re.compile(r"^.*::")
_NAMESPACE_PREFIX = re.compile(r"^([^:]+)::.*$")
_PKGCONFIG_VERSION = re.compile(r"[-_]\d+(?:\.\d+)*$")


class SysLibDatabase:
    """The curated name -> distro package map, plus a fallback guesser."""

    def __init__(self, path: Path = DATA):
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        # Normalized the same way lookups are, so an entry like "pybind11"
        # still matches the normalized candidate "pybind".
        self.ignore = {
            key
            for n in raw.get("ignore", {}).get("names", [])
            for key in (n.lower(), normalize(n))
            if key
        }
        headers = raw.get("header", {})
        self._header_exact = {k.lower(): v for k, v in headers.items() if not k.endswith("/")}
        self._header_prefix = tuple(
            sorted(
                ((k.lower(), v) for k, v in headers.items() if k.endswith("/")),
                key=lambda kv: -len(kv[0]),
            )
        )
        self._entries: dict[str, SystemRequirement] = {}
        self._by_package: dict[str, SystemRequirement] = {}
        for kind, section in (("library", "library"), ("tool", "buildtool")):
            for canonical, body in raw.get(section, {}).items():
                req = SystemRequirement(
                    name=canonical,
                    kind=kind,
                    pkgconfig=body.get("pkgconfig"),
                    debian=tuple(body.get("debian", ())),
                    fedora=tuple(body.get("fedora", ())),
                )
                # The pkg-config module name is itself a spelling we will
                # meet, in pkg_check_modules() and meson's dependency().
                keys = [canonical, *body.get("aliases", [])]
                if body.get("pkgconfig"):
                    keys.append(body["pkgconfig"])
                for key in keys:
                    for variant in _variants(normalize(key)):
                        self._entries.setdefault(variant, req)
                # CI files name distro packages, not libraries. Index them
                # backwards so "libssl-dev" and a scraped "OpenSSL" converge
                # on one entry instead of being reported twice.
                for package in (*req.debian, *req.fedora):
                    self._by_package.setdefault(package.lower(), req)

    def lookup(self, raw_name: str, kind: str = "library") -> Optional[SystemRequirement]:
        """Resolve a scraped name, or None if it should be ignored."""
        candidates = list(_candidates(raw_name))
        if not candidates:
            return None
        # The ignore list wins over the alias table, but only for libraries.
        # Stripping the "lib" prefix makes libgit2 answer to "git", which
        # would otherwise hijack CMake's find_package(Git) -- the tool, not a
        # library. The same list holds "gcc" (as in libgcc_s, never a thing
        # to install) while `gcc` the compiler is a perfectly real build tool,
        # so the precedence only applies on the library side.
        if kind == "library" and normalize(raw_name) in self.ignore:
            return None
        for key in candidates:
            hit = self._entries.get(key)
            if hit is not None:
                return hit
        if any(key in self.ignore for key in candidates):
            return None
        return self._guess(candidates[-1], kind)

    @staticmethod
    def _guess(name: str, kind: str) -> SystemRequirement:
        """Best-effort naming for something not in the curated map.

        Debian's convention is ``lib<name>-dev`` for libraries and the bare
        name for tools; Fedora's is ``<name>-devel``. Flagged as a guess by
        the report so nobody pastes it into a script unchecked.
        """
        if kind == "tool":
            return SystemRequirement(name=name, kind=kind, debian=(name,), fedora=(name,))
        return SystemRequirement(
            name=name,
            kind=kind,
            debian=(f"lib{name}-dev",),
            fedora=(f"{name}-devel",),
        )

    def by_distro_package(self, package: str) -> Optional[SystemRequirement]:
        """Resolve a distro package name (``libssl-dev``) to a known library.

        CI files install runtime packages as well as development ones, so
        ``libtiff6`` is tried again as ``libtiff-dev``: the same library,
        and worth converging rather than reporting twice.
        """
        key = package.strip().lower()
        hit = self._by_package.get(key)
        if hit is not None:
            return hit
        # A runtime package is the same library as its -dev counterpart, and
        # the soname may sit at the end (libtiff6, libpng16-16) or just
        # before the suffix (libtiff5-dev). Peel one soname at a time.
        embedded = re.match(r"^(?P<base>lib.+?)[0-9.]*(?:t64)?(?P<dev>-dev(?:el)?)$", key)
        if embedded:
            hit = self._by_package.get(embedded.group("base") + embedded.group("dev"))
            if hit is not None:
                return hit

        # Alpine and Fedora spell the development package differently
        # (brotli-dev, curl-devel). Strip the suffix and ask the library
        # alias table, so they converge with the Debian name.
        bare = re.sub(r"-(dev|devel|headers|static)$", "", key)
        if bare != key:
            for variant in _variants(bare):
                hit = self._entries.get(variant)
                if hit is not None:
                    return hit

        stem = key
        for _ in range(4):
            for suffix in ("-dev", "-devel", "t64-dev"):
                hit = self._by_package.get(stem + suffix)
                if hit is not None:
                    return hit
            peeled = re.sub(r"[-_]?[0-9.]*(t64)?$", "", stem).rstrip("-_.")
            if not peeled or peeled == stem:
                break
            stem = peeled
            hit = self._by_package.get(stem)
            if hit is not None:
                return hit
        return None

    def header(self, include_path: str) -> Optional[str]:
        """Map a ``#include`` path to a curated library name, if we know it.

        Matches the full path first (``openssl/ssl.h``), then a directory
        prefix (``webp/``), then the bare basename -- so a build that includes
        ``<freetype/ftglyph.h>`` and one that includes ``"ft2build.h"`` both
        land on freetype.
        """
        path = include_path.strip().lstrip("./").lower()
        if not path:
            return None
        hit = self._header_exact.get(path)
        if hit:
            return hit
        for prefix, name in self._header_prefix:
            if path.startswith(prefix):
                return name
        return self._header_exact.get(path.rsplit("/", 1)[-1])

    def is_guess(self, req: SystemRequirement) -> bool:
        if req.declared:
            return False
        return normalize(req.name) not in self._entries


def normalize(name: str) -> str:
    """Strip decoration so ``libpng16``, ``PNG`` and ``png >= 1.6`` agree."""
    name = _DECORATION.sub("", name.strip())
    name = name.strip().strip("\"'").lower()
    name = re.sub(r"^-l", "", name)
    return name.strip("-_. ")


#: A soname/ABI digit run, as in libpng16. Only stripped when it is short and
#: leaves a substantial stem behind -- otherwise dc1394 becomes "dc" and x264
#: becomes "x", which is how a camera library turns into nonsense.
_SONAME_DIGITS = re.compile(r"^(?P<stem>[a-z][a-z_+.-]{3,})\d{1,2}$")

#: Link-variant suffixes CMake enumerates alongside the real name.
_LINK_VARIANT = re.compile(r"[-_](static|shared|imp|import|mt|md)$")


def _variants(key: str) -> list[str]:
    """Alternative spellings of one candidate, most specific first."""
    out = [key]
    without_variant = _LINK_VARIANT.sub("", key)
    if without_variant != key and len(without_variant) > 2:
        out.append(without_variant)
    for candidate in list(out):
        match = _SONAME_DIGITS.match(candidate)
        if match:
            out.append(match.group("stem").rstrip("-_."))
        if candidate.startswith("lib") and len(candidate) > 4:
            out.append(candidate[3:])
    return out


def _candidates(raw_name: str) -> list[str]:
    """Spellings to try, most specific first.

    A CMake imported target like ``Boost::python`` should resolve to boost,
    not to its ``python`` component -- so the namespace is tried before the
    member, and the member last.
    """
    out: list[str] = []
    for part in (raw_name, _NAMESPACE_PREFIX.sub(r"\1", raw_name), _NAMESPACE.sub("", raw_name)):
        key = normalize(part)
        if not key:
            continue
        variants = _variants(key)
        # pkg-config names carry an API version: glib-2.0, libxml-2.0, dbus-1.
        bare = _PKGCONFIG_VERSION.sub("", key)
        if bare and bare != key:
            variants.extend(_variants(bare))
        for variant in variants:
            if variant and variant not in out:
                out.append(variant)
    return out


@lru_cache(maxsize=1)
def database() -> SysLibDatabase:
    return SysLibDatabase()
