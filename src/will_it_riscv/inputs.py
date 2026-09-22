"""Reading the set of root requirements out of a project."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.requirements import InvalidRequirement, Requirement

try:  # pragma: no cover - trivial
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class RootRequirements:
    """What a project asks for, split by why it asks."""

    source: str
    runtime: list[Requirement] = field(default_factory=list)
    build: list[Requirement] = field(default_factory=list)
    project_name: Optional[str] = None
    requires_python: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.runtime or self.build)


def load(
    path: Path,
    extras: tuple[str, ...] = (),
    groups: tuple[str, ...] = (),
) -> RootRequirements:
    """Load requirements from a pyproject.toml or a requirements file."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "pyproject.toml"
        if not candidate.exists():
            raise FileNotFoundError(f"no pyproject.toml in {path}")
        path = candidate
    if path.name == "pyproject.toml":
        return _load_pyproject(path, extras, groups)
    return _load_requirements_txt(path)


def _parse(spec: str, into: list[Requirement], warnings: list[str], where: str) -> None:
    try:
        into.append(Requirement(spec))
    except InvalidRequirement as exc:
        warnings.append(f"{where}: skipping unparseable requirement {spec!r} ({exc})")


def _load_pyproject(
    path: Path, extras: tuple[str, ...], groups: tuple[str, ...]
) -> RootRequirements:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    result = RootRequirements(source=str(path))

    project = data.get("project", {})
    result.project_name = project.get("name")
    result.requires_python = project.get("requires-python")

    for spec in project.get("dependencies", []):
        _parse(str(spec), result.runtime, result.warnings, "[project].dependencies")

    optional = project.get("optional-dependencies", {})
    wanted_extras = set(optional) if "all" in extras else {e.lower() for e in extras}
    for extra, specs in optional.items():
        if extra.lower() not in wanted_extras:
            continue
        for spec in specs:
            _parse(str(spec), result.runtime, result.warnings, f"extra {extra!r}")

    # PEP 735 dependency groups.
    dep_groups = data.get("dependency-groups", {})
    wanted_groups = set(dep_groups) if "all" in groups else {g.lower() for g in groups}
    for group, specs in dep_groups.items():
        if group.lower() not in wanted_groups:
            continue
        for spec in specs:
            if isinstance(spec, dict):
                result.warnings.append(
                    f"dependency group {group!r}: group includes are not expanded"
                )
                continue
            _parse(str(spec), result.runtime, result.warnings, f"group {group!r}")

    for spec in data.get("build-system", {}).get("requires", []):
        _parse(str(spec), result.build, result.warnings, "[build-system].requires")

    if not result:
        poetry = data.get("tool", {}).get("poetry", {})
        if poetry:
            result.project_name = result.project_name or poetry.get("name")
            _load_poetry(poetry, extras, result)

    return result


_CARET = re.compile(r"^\^(\d+)(?:\.(\d+))?(?:\.(\d+))?$")
_TILDE = re.compile(r"^~(\d+)(?:\.(\d+))?(?:\.(\d+))?$")


def _poetry_constraint(raw: str) -> str:
    """Translate Poetry's caret/tilde shorthand into a PEP 440 specifier."""
    raw = raw.strip()
    if raw in ("*", ""):
        return ""
    m = _CARET.match(raw)
    if m:
        major, minor, patch = (int(g) if g else 0 for g in m.groups())
        lower = f"{major}.{minor}.{patch}"
        if major > 0:
            return f">={lower},<{major + 1}.0.0"
        if minor > 0:
            return f">={lower},<0.{minor + 1}.0"
        return f">={lower},<0.0.{patch + 1}"
    m = _TILDE.match(raw)
    if m:
        major, minor, patch = (int(g) if g else 0 for g in m.groups())
        upper = f"{major}.{minor + 1}.0" if m.group(2) else f"{major + 1}.0.0"
        return f">={major}.{minor}.{patch},<{upper}"
    return raw


def _load_poetry(poetry: dict, extras: tuple[str, ...], result: RootRequirements) -> None:
    result.warnings.append(
        "read Poetry-style dependencies; caret/tilde constraints were translated "
        "to PEP 440 and may differ slightly from Poetry's own resolution"
    )
    deps = poetry.get("dependencies", {})
    wanted = {e.lower() for e in extras}
    all_extras = "all" in wanted
    declared_extras = {k.lower() for k in poetry.get("extras", {})}
    for name, value in deps.items():
        if name.lower() == "python":
            result.requires_python = result.requires_python or _poetry_constraint(str(value))
            continue
        optional = isinstance(value, dict) and value.get("optional")
        if optional and not all_extras and not (declared_extras & wanted):
            continue
        constraint = value.get("version", "*") if isinstance(value, dict) else str(value)
        spec = f"{name}{_poetry_constraint(str(constraint))}"
        _parse(spec, result.runtime, result.warnings, "[tool.poetry.dependencies]")


_REQ_LINE_SKIP = re.compile(r"^\s*(?:#|-)")


def _load_requirements_txt(path: Path) -> RootRequirements:
    result = RootRequirements(source=str(path))
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split(" #", 1)[0].strip()
        if not line or _REQ_LINE_SKIP.match(line):
            if line.startswith("-r") or line.startswith("--requirement"):
                result.warnings.append(f"{path}:{lineno}: nested -r includes are not followed")
            continue
        if line.startswith(("http://", "https://", "git+", "file:", ".", "/")):
            result.warnings.append(f"{path}:{lineno}: skipping direct reference {line!r}")
            continue
        _parse(line, result.runtime, result.warnings, f"{path}:{lineno}")
    return result
