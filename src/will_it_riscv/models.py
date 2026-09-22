"""Core data types shared across the analyzer."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class Verdict(enum.Enum):
    """How a single distribution fares on the target platform.

    Ordered roughly from best to worst; :meth:`severity` drives sorting and the
    process exit code.
    """

    PURE_PYTHON = "pure-python"
    """Ships a ``py3-none-any`` wheel. Installs anywhere, nothing to build."""

    WHEEL_AVAILABLE = "wheel-available"
    """Ships a wheel whose tags match the target. Binary, but already built."""

    NEEDS_BUILD = "needs-build"
    """No matching wheel, but an sdist exists and looks buildable on the target."""

    NEEDS_BUILD_BLOCKED = "needs-build-blocked"
    """An sdist exists but something in its build chain cannot be satisfied."""

    NO_DISTRIBUTION = "no-distribution"
    """Nothing installable at all: no usable wheel and no sdist."""

    UNRESOLVED = "unresolved"
    """Could not pick a version (not on the index, or conflicting constraints)."""

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    @property
    def ok(self) -> bool:
        """True when the distribution installs without compiling anything."""
        return self in (Verdict.PURE_PYTHON, Verdict.WHEEL_AVAILABLE)


_SEVERITY = {
    Verdict.PURE_PYTHON: 0,
    Verdict.WHEEL_AVAILABLE: 1,
    Verdict.NEEDS_BUILD: 2,
    Verdict.NEEDS_BUILD_BLOCKED: 3,
    Verdict.NO_DISTRIBUTION: 4,
    Verdict.UNRESOLVED: 5,
}


class Evidence(enum.Enum):
    """Why we believe a package is native. Ordered weakest to strongest."""

    BACKEND = "build-backend"
    SOURCE_FILES = "compiled-source-files"
    BUILD_CONFIG = "native-build-config"
    WHEEL_TAGS = "platform-specific-wheels"
    CURATED = "curated-override"


@dataclass(frozen=True)
class SystemRequirement:
    """An external, non-Python thing the build needs from the host or a distro.

    ``kind`` is ``"library"`` for things linked against (openssl, zlib) and
    ``"tool"`` for things invoked during the build (cmake, gfortran, cargo).
    """

    name: str
    kind: str = "library"
    pkgconfig: Optional[str] = None
    debian: tuple[str, ...] = ()
    fedora: tuple[str, ...] = ()
    found_in: tuple[str, ...] = ()
    """Files inside the sdist that mentioned it, for auditability."""

    def merged_with(self, other: SystemRequirement) -> SystemRequirement:
        return SystemRequirement(
            name=self.name,
            kind=self.kind,
            pkgconfig=self.pkgconfig or other.pkgconfig,
            debian=self.debian or other.debian,
            fedora=self.fedora or other.fedora,
            found_in=tuple(dict.fromkeys(self.found_in + other.found_in)),
        )


@dataclass
class BuildProfile:
    """What an sdist inspection learned about how a package builds."""

    build_backend: Optional[str] = None
    build_requires: list[str] = field(default_factory=list)
    languages: set[str] = field(default_factory=set)
    """e.g. {"c", "cython", "rust", "fortran", "c++"}"""
    build_systems: set[str] = field(default_factory=set)
    """e.g. {"cmake", "meson", "autotools", "cargo"}"""
    system_requirements: list[SystemRequirement] = field(default_factory=list)
    evidence: set[Evidence] = field(default_factory=set)
    inspected: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def is_native(self) -> bool:
        return bool(self.languages or self.build_systems)


@dataclass
class PackageReport:
    """The analyzer's conclusion about one resolved distribution."""

    name: str
    """Canonical (PEP 503 normalized) name."""
    version: Optional[str] = None
    verdict: Verdict = Verdict.UNRESOLVED
    required_by: list[str] = field(default_factory=list)
    extras: frozenset[str] = frozenset()
    depth: int = 0
    is_build_dependency: bool = False
    """True when this package is only needed to *build* something else."""
    matching_wheels: list[str] = field(default_factory=list)
    wheel_platform_tags: list[str] = field(default_factory=list)
    """Platform tags this release publishes, for explaining a near-miss."""
    has_sdist: bool = False
    build: BuildProfile = field(default_factory=BuildProfile)
    distro_packages: dict[str, str] = field(default_factory=dict)
    """distro id -> name of a system package providing this Python module."""
    reasons: list[str] = field(default_factory=list)

    def sort_key(self) -> tuple:
        return (-self.verdict.severity, self.name)


@dataclass
class Analysis:
    """The whole run: every package reached, plus the aggregate shopping lists."""

    target: str
    root: str
    python_version: str
    packages: dict[str, PackageReport] = field(default_factory=dict)
    system_requirements: dict[str, SystemRequirement] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def by_verdict(self, *verdicts: Verdict) -> list[PackageReport]:
        want = set(verdicts)
        return sorted(
            (p for p in self.packages.values() if p.verdict in want),
            key=lambda p: p.name,
        )

    @property
    def worst(self) -> Verdict:
        if not self.packages:
            return Verdict.PURE_PYTHON
        return max((p.verdict for p in self.packages.values()), key=lambda v: v.severity)

    def exit_code(self) -> int:
        """0 = installs clean, 1 = needs source builds, 2 = something is blocked."""
        worst = self.worst
        if worst.ok:
            return 0
        if worst is Verdict.NEEDS_BUILD:
            return 1
        return 2
