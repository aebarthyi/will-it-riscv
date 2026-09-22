"""Describing a target platform we do not run on.

Everything downstream is parameterized by a :class:`Target`. riscv64 is the
default, but nothing here is riscv-specific -- the same machinery answers
"will it aarch64" or "will it loongarch64".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from packaging.tags import Tag, compatible_tags, cpython_tags, generic_tags

#: Architectures for which the legacy manylinux aliases were ever defined.
#: riscv64 is deliberately absent: it postdates PEP 600, so it only ever gets
#: ``manylinux_<glibcmajor>_<glibcminor>_riscv64`` tags.
_LEGACY_MANYLINUX = {
    "x86_64": [("manylinux2014", (2, 17)), ("manylinux2010", (2, 12)), ("manylinux1", (2, 5))],
    "i686": [("manylinux2014", (2, 17)), ("manylinux2010", (2, 12)), ("manylinux1", (2, 5))],
    "aarch64": [("manylinux2014", (2, 17))],
    "ppc64le": [("manylinux2014", (2, 17))],
    "ppc64": [("manylinux2014", (2, 17))],
    "s390x": [("manylinux2014", (2, 17))],
    "armv7l": [("manylinux2014", (2, 17))],
}

#: Oldest glibc a PEP 600 manylinux tag is allowed to name.
_MIN_GLIBC_MINOR = 17

#: Sensible glibc floor per architecture: no distro ever shipped riscv64 with
#: glibc older than 2.27, and manylinux only added riscv64 images at 2.39.
_DEFAULT_GLIBC = {
    "riscv64": (2, 39),
    "loongarch64": (2, 36),
}

_ALIASES = {
    "riscv64-unknown-linux": "riscv64",
    "riscv64-unknown-linux-gnu": "riscv64",
    "riscv64gc-unknown-linux-gnu": "riscv64",
    "riscv64-linux-gnu": "riscv64",
    "rv64gc": "riscv64",
    "rv64": "riscv64",
    "arm64": "aarch64",
    "amd64": "x86_64",
}

_MUSL_TRIPLE = re.compile(r"musl")

#: riscv64gc, riscv64imafdc, ... -- the ISA extension string is not part of the
#: platform tag, which only ever says "riscv64".
_RISCV_ISA = re.compile(r"^(riscv(?:32|64))[a-z]*$")


def _normalize_arch(spec: str) -> str:
    """Reduce an architecture, alias or target triple to a platform-tag arch."""
    if spec in _ALIASES:
        return _ALIASES[spec]
    # Strip a vendor/os/abi triple down to its first component.
    head = spec.split("-", 1)[0]
    head = _ALIASES.get(head, head)
    match = _RISCV_ISA.match(head)
    return match.group(1) if match else head


@dataclass(frozen=True)
class Target:
    """A platform to evaluate wheel compatibility against.

    :param arch: ``platform.machine()`` value, e.g. ``riscv64``.
    :param libc: ``glibc`` or ``musl``.
    :param libc_version: the *oldest* libc we are willing to require. A wheel
        tagged for a newer libc than this will not be considered compatible.
    :param python_version: interpreter version as ``(major, minor)``.
    """

    arch: str = "riscv64"
    libc: str = "glibc"
    libc_version: tuple[int, int] = (2, 39)
    python_version: tuple[int, int] = (3, 12)
    implementation: str = "cpython"
    free_threaded: bool = False
    _tags: frozenset = field(default=frozenset(), repr=False, compare=False)

    @classmethod
    def parse(cls, spec: str, python_version: tuple[int, int] = (3, 12)) -> Target:
        """Build a target from a CLI string.

        Accepts an architecture (``riscv64``), a Rust-style triple
        (``riscv64gc-unknown-linux-gnu``), or an explicit
        ``<arch>-<libc><version>`` form (``riscv64-glibc2.36``,
        ``riscv64-musl1.2``).
        """
        spec = spec.strip().lower()
        libc = "musl" if _MUSL_TRIPLE.search(spec) else "glibc"
        libc_version: Optional[tuple[int, int]] = None

        m = re.search(r"(?:glibc|musl)[-_]?(\d+)\.(\d+)", spec)
        if m:
            libc_version = (int(m.group(1)), int(m.group(2)))
            spec = spec[: m.start()].rstrip("-_")

        arch = _normalize_arch(spec)

        if libc_version is None:
            libc_version = (1, 2) if libc == "musl" else _DEFAULT_GLIBC.get(arch, (2, 17))

        return cls(
            arch=arch,
            libc=libc,
            libc_version=libc_version,
            python_version=python_version,
        )

    # -- naming -----------------------------------------------------------

    @property
    def slug(self) -> str:
        return f"{self.arch}-{self.libc}{self.libc_version[0]}.{self.libc_version[1]}"

    @property
    def python_tag(self) -> str:
        return f"cp{self.python_version[0]}{self.python_version[1]}"

    def __str__(self) -> str:
        py = f"{self.python_version[0]}.{self.python_version[1]}"
        libc = f"{self.libc} >= {self.libc_version[0]}.{self.libc_version[1]}"
        return f"{self.arch} linux ({libc}), CPython {py}"

    # -- tags -------------------------------------------------------------

    def platform_tags(self) -> list[str]:
        """Platform tags this target accepts, most specific first."""
        tags: list[str] = []
        major, minor = self.libc_version
        if self.libc == "musl":
            for m in range(minor, -1, -1):
                tags.append(f"musllinux_{major}_{m}_{self.arch}")
        else:
            for m in range(minor, _MIN_GLIBC_MINOR - 1, -1):
                tags.append(f"manylinux_{major}_{m}_{self.arch}")
            for alias, (amaj, amin) in _LEGACY_MANYLINUX.get(self.arch, []):
                if (amaj, amin) <= (major, minor):
                    tags.append(f"{alias}_{self.arch}")
        tags.append(f"linux_{self.arch}")
        return tags

    def tags(self) -> frozenset[Tag]:
        """Every (interpreter, abi, platform) tag triple this target can install."""
        if self._tags:
            return self._tags
        plats = self.platform_tags()
        collected: list[Tag] = []
        if self.implementation == "cpython":
            collected.extend(cpython_tags(python_version=self.python_version, platforms=plats))
            if self.free_threaded:
                # cpython_tags() derives the "t" ABI suffix from the running
                # interpreter, so add the free-threaded ABI explicitly.
                abi = f"{self.python_tag}t"
                collected.extend(
                    cpython_tags(
                        python_version=self.python_version, abis=[abi], platforms=plats
                    )
                )
        else:
            interp = f"{self.implementation}{self.python_version[0]}{self.python_version[1]}"
            collected.extend(generic_tags(interpreter=interp, abis=["none"], platforms=plats))
        collected.extend(compatible_tags(python_version=self.python_version, platforms=plats))
        return frozenset(collected)

    def accepts(self, wheel_tags) -> bool:
        """True if any of ``wheel_tags`` is installable on this target."""
        return bool(self.tags() & frozenset(wheel_tags))

    # -- PEP 508 marker environment ---------------------------------------

    def marker_environment(self) -> dict[str, str]:
        major, minor = self.python_version
        full = f"{major}.{minor}.0"
        impl = "CPython" if self.implementation == "cpython" else self.implementation
        return {
            "implementation_name": self.implementation,
            "implementation_version": full,
            "os_name": "posix",
            "platform_machine": self.arch,
            "platform_python_implementation": impl,
            "platform_release": "",
            "platform_system": "Linux",
            "platform_version": "",
            "python_full_version": full,
            "python_version": f"{major}.{minor}",
            "sys_platform": "linux",
        }
