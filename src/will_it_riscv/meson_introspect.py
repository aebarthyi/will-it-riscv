"""Asking Meson what a project depends on, instead of guessing.

``meson introspect --scan-dependencies`` walks a project's ``meson.build``
files -- recursing through ``subdir()`` -- and reports every dependency with
whether it is required and whether it sits behind a condition. That is
precisely the classification this tool reconstructs by hand everywhere else,
produced by the parser that actually owns the language.

It is not free, and it is not purely static: Meson resolves the project's
languages first, so a project declaring Rust makes it run ``rustc --version``
and fail if there is no Rust toolchain. QEMU does exactly that. So this is
strictly an enhancement -- when it works its answers win, and when it does
not the regex scrapers carry on as before.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class MesonDependency:
    name: str
    required: Optional[bool]
    """True, False, or None where Meson itself reports ``unknown`` -- meaning
    the requirement is decided by a build option at configure time."""
    conditional: bool

    @property
    def optional(self) -> bool:
        """Whether a build with no options set would do without this.

        ``required: false`` says so outright. ``unknown`` means the project
        defers to an option, which for a default build on a fresh machine
        means it is used only if it happens to be there.
        """
        return self.required is not True

    @property
    def gate(self) -> Optional[str]:
        if self.required is False:
            return "meson: required: false"
        if self.required is None:
            return "meson: decided by a build option"
        return None


@dataclass
class MesonScan:
    dependencies: list[MesonDependency]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def available() -> bool:
    return shutil.which("meson") is not None


def _parse_required(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def scan_dependencies(root: Path) -> Optional[MesonScan]:
    """Run Meson's own dependency scan, or return None if it cannot be run."""
    meson_build = Path(root) / "meson.build"
    if not meson_build.exists():
        return None
    if not available():
        return MesonScan([], error="meson is not installed")

    try:
        process = subprocess.run(
            ["meson", "introspect", "--scan-dependencies", str(meson_build)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(root),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return MesonScan([], error=f"meson introspect failed to start: {exc}")

    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip().splitlines()
        return MesonScan([], error=detail[0] if detail else "meson introspect failed")

    try:
        payload = json.loads(process.stdout)
    except ValueError:
        return MesonScan([], error="meson introspect produced no usable JSON")
    if not isinstance(payload, list):
        return MesonScan([], error="unexpected meson introspect output")

    dependencies = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        dependencies.append(
            MesonDependency(
                name=name,
                required=_parse_required(entry.get("required")),
                conditional=bool(entry.get("conditional")),
            )
        )
    return MesonScan(dependencies)
