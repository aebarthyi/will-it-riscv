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
                    self._entries.setdefault(normalize(key), req)

    def lookup(self, raw_name: str, kind: str = "library") -> Optional[SystemRequirement]:
        """Resolve a scraped name, or None if it should be ignored."""
        candidates = list(_candidates(raw_name))
        if not candidates:
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
        return normalize(req.name) not in self._entries


def normalize(name: str) -> str:
    """Strip decoration so ``libpng16``, ``PNG`` and ``png >= 1.6`` agree."""
    name = _DECORATION.sub("", name.strip())
    name = name.strip().strip("\"'").lower()
    name = re.sub(r"^-l", "", name)
    # Drop a trailing soname/ABI digit run: libpng16 -> libpng, icu-uc stays.
    name = re.sub(r"(?<=[a-z])\d+$", "", name)
    return name.strip("-_. ")


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
        variants = [key]
        # pkg-config names carry an API version: glib-2.0, libxml-2.0, dbus-1.
        bare = _PKGCONFIG_VERSION.sub("", key)
        if bare and bare != key:
            variants.append(bare)
        for variant in list(variants):
            if variant.startswith("lib") and len(variant) > 4:
                variants.append(variant[3:])
        for variant in variants:
            if variant and variant not in out:
                out.append(variant)
    return out


@lru_cache(maxsize=1)
def database() -> SysLibDatabase:
    return SysLibDatabase()
