"""The recursive dependency walk and per-package verdict.

This is a resolver only in the loosest sense: for each package it takes the
highest version satisfying the constraints accumulated so far, and re-visits a
package when a later edge tightens its constraints. It does not backtrack.
That is enough to answer "what will not install", and it is deliberately not a
replacement for pip's or uv's resolver -- neither of which can target riscv64
today, which is why the walk lives here at all.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.parser import BytesParser
from typing import Callable, Optional

from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from .index import IndexFile, PackageIndex, Release
from .inputs import RootRequirements
from .models import Analysis, BuildProfile, Evidence, PackageReport, SystemRequirement, Verdict
from .sdist import SdistInspection, inspect_sdist
from .target import Target

Progress = Callable[[str, str], None]

MAX_ITERATIONS = 20000


#: Constraint scope for ordinary runtime dependencies: one environment, so
#: every requirement on a package has to agree.
RUNTIME_SCOPE = ""


@dataclass
class _Work:
    requirement: Requirement
    parent: str
    """Human-readable label of whatever pulled this in, for the report."""
    depth: int
    is_build: bool = False
    owner: str = ""
    """Canonical name of the package that declared this requirement.

    Distinct from :attr:`parent`, which is a display label like
    ``"uvicorn 0.53.0"``. Extras are keyed by canonical name, so evaluating a
    marker against the label silently loses every ``extra == ...`` dependency.
    """
    scope: str = RUNTIME_SCOPE
    """Which constraint namespace this edge belongs to.

    PEP 517 gives every package its own isolated build environment, so
    opencv pinning ``setuptools==59.2.0`` to build does not conflict with
    pillow requiring ``setuptools>=77`` to build. Merging those into one
    specifier invents an unsatisfiable constraint and reports both packages
    as blocked. Runtime dependencies do share one environment, and share
    :data:`RUNTIME_SCOPE`.
    """


@dataclass
class _State:
    constraints: dict[tuple[str, str], SpecifierSet] = field(default_factory=dict)
    extras: dict[str, set[str]] = field(default_factory=dict)
    walked: dict[tuple[str, str], tuple[Optional[str], frozenset]] = field(
        default_factory=dict
    )
    build_edges: dict[str, set[str]] = field(default_factory=dict)
    """canonical package name -> the packages needed to *build* it."""
    runtime_reached: set[str] = field(default_factory=set)
    """packages reachable without going through a build edge."""
    blocked_builders: dict[str, list[str]] = field(default_factory=dict)
    """package -> why its isolated build environment cannot be assembled."""


class Analyzer:
    def __init__(
        self,
        index: PackageIndex,
        target: Target,
        *,
        include_build_deps: bool = True,
        inspect_sdists: bool = True,
        allow_prereleases: bool = False,
        max_depth: Optional[int] = None,
        workers: int = 8,
        progress: Optional[Progress] = None,
    ):
        self.index = index
        self.target = target
        self.include_build_deps = include_build_deps
        self.inspect_sdists = inspect_sdists
        self.allow_prereleases = allow_prereleases
        self.max_depth = max_depth
        self.workers = max(1, workers)
        self.progress = progress or (lambda event, name: None)
        self.env = target.marker_environment()
        self.python = Version(
            f"{target.python_version[0]}.{target.python_version[1]}.0"
        )
        self._inspections: dict[tuple[str, str], SdistInspection] = {}

    # -------------------------------------------------------------- entry

    def run(self, roots: RootRequirements) -> Analysis:
        analysis = Analysis(
            target=str(self.target),
            root=roots.project_name or roots.source,
            python_version=f"{self.target.python_version[0]}.{self.target.python_version[1]}",
        )
        analysis.warnings.extend(roots.warnings)
        state = _State()

        frontier = [_Work(r, "(project)", 0, False) for r in roots.runtime]
        if self.include_build_deps:
            frontier += [
                _Work(r, "(project build-system)", 0, True, scope="(project)")
                for r in roots.build
            ]

        iterations = 0
        while frontier:
            if self.max_depth is not None and frontier[0].depth > self.max_depth:
                break
            self._prefetch(state, frontier)
            next_frontier: list[_Work] = []
            for item in frontier:
                iterations += 1
                if iterations > MAX_ITERATIONS:
                    analysis.warnings.append(
                        f"stopped after {MAX_ITERATIONS} edges; the graph may be incomplete"
                    )
                    frontier = []
                    break
                next_frontier.extend(self._visit(item, state, analysis))
            else:
                frontier = next_frontier
                continue
            break

        self._propagate_blocked(state, analysis)
        self._collect_system_requirements(analysis)
        return analysis

    # ------------------------------------------------------------ walking

    def _prefetch(self, state: _State, frontier: list[_Work]) -> None:
        """Warm the index cache for a whole BFS level at once."""
        names = []
        seen = set()
        for item in frontier:
            name = canonicalize_name(item.requirement.name)
            if name not in seen:
                seen.add(name)
                names.append(name)
        if len(names) < 2:
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self._safe_project, names))

    def _safe_project(self, name: str):
        try:
            return self.index.project(name)
        except Exception:  # noqa: BLE001 - prefetch is best-effort
            return None

    def _visit(self, item: _Work, state: _State, analysis: Analysis) -> list[_Work]:
        req = item.requirement
        name = canonicalize_name(req.name)

        if req.marker is not None and not self._marker_ok(req.marker, state, item.owner):
            return []

        key = (item.scope, name)
        previous = state.constraints.get(key)
        merged = (previous & req.specifier) if previous is not None else req.specifier
        state.constraints[key] = merged
        extras = state.extras.setdefault(name, set())
        extras |= {e.lower() for e in req.extras}

        report = analysis.packages.get(name)
        if report is None:
            report = PackageReport(name=name, depth=item.depth)
            analysis.packages[name] = report
        report.depth = min(report.depth, item.depth)
        report.extras = frozenset(extras)
        if item.parent not in report.required_by:
            report.required_by.append(item.parent)
        if not item.is_build:
            state.runtime_reached.add(name)

        version = self._pick(name, merged)
        signature = (str(version) if version else None, frozenset(extras))
        if state.walked.get(key) == signature:
            return []
        state.walked[key] = signature

        if version is None:
            self._record_unresolved(report, name, merged, item, state, analysis)
            return []

        project = self._safe_project(name)
        release = project.releases.get(version) if project else None
        if release is None:
            self._record_unresolved(report, name, merged, item, state, analysis)
            return []

        if report.version is not None and report.version != str(version):
            # The same package resolved differently in another scope -- almost
            # always a build environment pinning an older version.
            report.reasons.append(
                f"also resolved to {version} in a separate build environment"
            )
            if Version(report.version) > version:
                version = Version(report.version)
        report.version = str(version)
        self.progress("resolve", f"{name} {version}")

        requires_dist = self._classify(report, release, name, version, analysis)

        children: list[_Work] = []
        label = f"{name} {version}"
        for spec in requires_dist:
            try:
                dep = Requirement(spec)
            except Exception:  # noqa: BLE001 - malformed metadata in the wild
                analysis.warnings.append(f"{label}: unparseable requirement {spec!r}")
                continue
            if dep.marker is not None and not self._marker_ok(dep.marker, state, name):
                continue
            children.append(
                _Work(dep, label, item.depth + 1, item.is_build, owner=name, scope=item.scope)
            )

        if self.include_build_deps:
            for spec in report.build.build_requires:
                try:
                    dep = Requirement(spec)
                except Exception:  # noqa: BLE001
                    continue
                if dep.marker is not None and not self._marker_ok(dep.marker, state, name):
                    continue
                state.build_edges.setdefault(name, set()).add(canonicalize_name(dep.name))
                children.append(
                    _Work(
                        dep, f"{label} (build)", item.depth + 1, True,
                        owner=name, scope=name,
                    )
                )

        return children

    def _marker_ok(self, marker, state: _State, package: str) -> bool:
        """Evaluate a PEP 508 marker for the target, honouring requested extras.

        ``package`` is the canonical name of whoever declared the requirement;
        an empty string for a root requirement, which has no extras context.
        """
        wanted = state.extras.get(canonicalize_name(package), set()) if package else set()
        environments = [dict(self.env, extra="")]
        environments += [dict(self.env, extra=e) for e in sorted(wanted)]
        for env in environments:
            try:
                if marker.evaluate(env):
                    return True
            except Exception:  # noqa: BLE001 - undefined variables in odd markers
                return True
        return False

    # ----------------------------------------------------------- resolving

    def _pick(self, name: str, specifier: SpecifierSet) -> Optional[Version]:
        project = self._safe_project(name)
        if project is None:
            return None
        allow_pre = self.allow_prereleases or bool(specifier.prereleases)
        for version in project.versions():
            if version.is_prerelease and not allow_pre:
                continue
            if not specifier.contains(version, prereleases=allow_pre):
                continue
            release = project.releases[version]
            if any(self._python_ok(f) for f in release.wheels + release.sdists if not f.yanked):
                return version
        return None

    def _python_ok(self, file: IndexFile) -> bool:
        if not file.requires_python:
            return True
        try:
            return SpecifierSet(file.requires_python).contains(self.python, prereleases=True)
        except InvalidSpecifier:
            return True

    def _record_unresolved(
        self,
        report: PackageReport,
        name: str,
        specifier: SpecifierSet,
        item: _Work,
        state: _State,
        analysis: Analysis,
    ) -> None:
        """Nothing satisfies the constraints. Who that is a problem for depends.

        In the runtime scope there is one environment, so an unsatisfiable
        merged constraint is a genuine conflict and the package is unresolved.
        In an isolated build scope the package may be perfectly available to
        everyone else -- what is broken is the *builder's* environment.
        """
        scope = item.scope
        is_isolated_build = scope not in (RUNTIME_SCOPE, "(project)")

        if is_isolated_build:
            state.blocked_builders.setdefault(scope, []).append(
                f"build requirement {name}{specifier} cannot be satisfied for the target"
            )
            report.reasons.append(f"not satisfiable at {specifier} when building {scope}")
            if report.version is not None:
                return

        report.verdict = Verdict.UNRESOLVED
        project = self._safe_project(name)
        if project is None:
            report.reasons.append("not found on the index")
        elif not project.versions():
            report.reasons.append("index page has no usable files")
        else:
            report.reasons.append(
                f"no release satisfies {specifier or 'any version'} "
                f"for Python {self.python.major}.{self.python.minor}"
                + (f" (building {scope})" if is_isolated_build else "")
            )

    # --------------------------------------------------------- classifying

    def _classify(
        self,
        report: PackageReport,
        release: Release,
        name: str,
        version: Version,
        analysis: Analysis,
    ) -> list[str]:
        """Set the verdict and build profile; return the package's Requires-Dist."""
        accepted = self.target.tags()
        matching = release.matching_wheels(accepted)
        platform_specific = [
            w for w in release.wheels
            if not any(t.platform == "any" for t in release.wheel_tags(w))
        ]
        report.matching_wheels = [w.filename for w in matching]
        report.wheel_platform_tags = release.platform_tags()
        report.has_sdist = release.has_sdist

        pure_match = [
            w for w in matching if any(t.platform == "any" for t in release.wheel_tags(w))
        ]

        requires_dist: list[str] = []
        metadata_file = self._metadata_file(release, matching)
        if metadata_file is not None:
            raw = self.index.metadata(metadata_file)
            if raw:
                message = BytesParser().parsebytes(raw)
                requires_dist = [str(r) for r in (message.get_all("Requires-Dist") or [])]

        if pure_match and not platform_specific:
            report.verdict = Verdict.PURE_PYTHON
            return requires_dist

        if matching:
            report.verdict = Verdict.WHEEL_AVAILABLE
            report.reasons.append(
                f"{len(matching)} wheel(s) match the target, e.g. {matching[0].filename}"
            )
            return requires_dist

        # No usable wheel. Everything below is about the source build.
        if platform_specific:
            report.build.evidence.add(Evidence.WHEEL_TAGS)
            report.reasons.append(
                "publishes binary wheels, but none for "
                f"{self.target.arch} (has: {', '.join(report.wheel_platform_tags[:6])})"
            )

        if not release.has_sdist:
            report.verdict = Verdict.NO_DISTRIBUTION
            report.reasons.append("no source distribution to fall back to")
            return requires_dist

        inspection = self._inspect(release, name, version, analysis)
        if inspection is not None:
            report.build = _merge_profiles(report.build, inspection.profile)
            if not requires_dist and inspection.requires_dist:
                requires_dist = inspection.requires_dist
                report.reasons.append(
                    f"dependencies read from the sdist ({inspection.metadata_source})"
                )

        if report.build.is_native or Evidence.WHEEL_TAGS in report.build.evidence:
            report.verdict = Verdict.NEEDS_BUILD
            if report.build.inspected and report.build.is_native:
                what = ", ".join(sorted(report.build.languages | report.build.build_systems))
                report.reasons.append(f"source build compiles: {what}")
        elif report.build.inspected:
            report.verdict = Verdict.PURE_PYTHON
            report.reasons.append(
                "sdist only, but no compiled sources found -- installs as pure Python"
            )
        else:
            report.verdict = Verdict.NEEDS_BUILD
            report.reasons.append(
                "sdist was not inspected; treat as a source build until checked"
            )
        return requires_dist

    def _metadata_file(self, release: Release, matching: list[IndexFile]) -> Optional[IndexFile]:
        """Prefer a wheel we would actually install, then any wheel with PEP 658."""
        for candidate in (matching, release.wheels):
            for file in candidate:
                if file.core_metadata and not file.yanked:
                    return file
        return None

    def _inspect(self, release: Release, name: str, version: Version, analysis: Analysis):
        if not self.inspect_sdists:
            return None
        key = (name, str(version))
        cached = self._inspections.get(key)
        if cached is not None:
            return cached
        sdist = next((s for s in release.sdists if not s.yanked), None)
        if sdist is None:
            return None
        self.progress("inspect", f"{name} {version}")
        blob = self.index.download(sdist)
        if blob is None:
            analysis.warnings.append(f"{name} {version}: sdist too large or unreachable")
            return None
        inspection = inspect_sdist(blob, sdist.filename)
        self._inspections[key] = inspection
        return inspection

    # ------------------------------------------------------------- post

    def _propagate_blocked(self, state: _State, analysis: Analysis) -> None:
        """A package is blocked if anything in its build chain cannot be had."""
        for builder, reasons in state.blocked_builders.items():
            report = analysis.packages.get(builder)
            if report is not None and report.verdict is Verdict.NEEDS_BUILD:
                report.verdict = Verdict.NEEDS_BUILD_BLOCKED
                report.reasons.extend(reasons)

        for _ in range(len(analysis.packages) + 1):
            changed = False
            for name, needs in state.build_edges.items():
                report = analysis.packages.get(name)
                if report is None or report.verdict is not Verdict.NEEDS_BUILD:
                    continue
                for dep_name in needs:
                    dep = analysis.packages.get(dep_name)
                    if dep is None:
                        continue
                    if dep.verdict in (
                        Verdict.NO_DISTRIBUTION,
                        Verdict.UNRESOLVED,
                        Verdict.NEEDS_BUILD_BLOCKED,
                    ):
                        report.verdict = Verdict.NEEDS_BUILD_BLOCKED
                        report.reasons.append(
                            f"build requirement {dep_name} is {dep.verdict.value}"
                        )
                        changed = True
                        break
            if not changed:
                break

        # Anything never reached by a runtime edge exists only to build something.
        for name, report in analysis.packages.items():
            report.is_build_dependency = name not in state.runtime_reached

    def _collect_system_requirements(self, analysis: Analysis) -> None:
        merged: dict[str, SystemRequirement] = {}
        for report in analysis.packages.values():
            if report.verdict not in (Verdict.NEEDS_BUILD, Verdict.NEEDS_BUILD_BLOCKED):
                continue
            for req in report.build.system_requirements:
                tagged = SystemRequirement(
                    name=req.name,
                    kind=req.kind,
                    pkgconfig=req.pkgconfig,
                    debian=req.debian,
                    fedora=req.fedora,
                    found_in=(report.name,),
                )
                existing = merged.get(req.name)
                merged[req.name] = tagged.merged_with(existing) if existing else tagged
        analysis.system_requirements = dict(sorted(merged.items()))


def _merge_profiles(base: BuildProfile, other: BuildProfile) -> BuildProfile:
    base.build_backend = base.build_backend or other.build_backend
    base.build_requires = base.build_requires or other.build_requires
    base.languages |= other.languages
    base.build_systems |= other.build_systems
    base.evidence |= other.evidence
    base.notes += other.notes
    base.inspected = base.inspected or other.inspected
    base.system_requirements = other.system_requirements or base.system_requirements
    return base
