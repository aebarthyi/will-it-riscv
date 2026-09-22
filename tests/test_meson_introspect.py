"""Meson's own dependency scan, and the fallback when it cannot be used."""

import json
import subprocess

import pytest

from will_it_riscv import meson_introspect
from will_it_riscv.meson_introspect import (
    MesonDependency,
    available,
    scan_dependencies,
)

needs_meson = pytest.mark.skipif(not available(), reason="meson is not installed")


# -- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    "required, optional, gate_fragment",
    [
        (True, False, None),
        (False, True, "required: false"),
        (None, True, "build option"),
    ],
)
def test_required_maps_to_optionality(required, optional, gate_fragment):
    dependency = MesonDependency(name="x", required=required, conditional=False)
    assert dependency.optional is optional
    if gate_fragment is None:
        assert dependency.gate is None
    else:
        assert gate_fragment in dependency.gate


# -- the subprocess boundary ------------------------------------------------


def test_no_meson_build_means_nothing_to_do(tmp_path):
    assert scan_dependencies(tmp_path) is None


def test_missing_meson_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "meson.build").write_text("project('x', 'c')\n")
    monkeypatch.setattr(meson_introspect.shutil, "which", lambda _: None)
    scan = scan_dependencies(tmp_path)
    assert not scan.ok and "not installed" in scan.error


def test_a_failing_scan_reports_why(tmp_path, monkeypatch):
    """QEMU declares Rust, so Meson runs rustc and fails without it."""
    (tmp_path / "meson.build").write_text("project('x', 'c')\n")
    monkeypatch.setattr(meson_introspect.shutil, "which", lambda _: "/usr/bin/meson")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args, 1, stdout="", stderr="ERROR: Unknown compiler(s): [['rustc']]\n"
        )

    monkeypatch.setattr(meson_introspect.subprocess, "run", fake_run)
    scan = scan_dependencies(tmp_path)
    assert not scan.ok and "rustc" in scan.error


def test_unparseable_output_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "meson.build").write_text("project('x', 'c')\n")
    monkeypatch.setattr(meson_introspect.shutil, "which", lambda _: "/usr/bin/meson")
    monkeypatch.setattr(
        meson_introspect.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="not json", stderr=""),
    )
    assert "no usable JSON" in scan_dependencies(tmp_path).error


def test_timeout_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "meson.build").write_text("project('x', 'c')\n")
    monkeypatch.setattr(meson_introspect.shutil, "which", lambda _: "/usr/bin/meson")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="meson", timeout=1)

    monkeypatch.setattr(meson_introspect.subprocess, "run", timeout)
    assert not scan_dependencies(tmp_path).ok


def test_output_is_parsed(tmp_path, monkeypatch):
    (tmp_path / "meson.build").write_text("project('x', 'c')\n")
    payload = json.dumps([
        {"name": "zlib", "required": True, "conditional": False},
        {"name": "libcurl", "required": False, "conditional": True},
        {"name": "llvm", "required": "unknown", "conditional": True},
        {"name": "", "required": True, "conditional": False},
    ])
    monkeypatch.setattr(meson_introspect.shutil, "which", lambda _: "/usr/bin/meson")
    monkeypatch.setattr(
        meson_introspect.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=payload, stderr=""),
    )
    scan = scan_dependencies(tmp_path)
    assert scan.ok
    by_name = {d.name: d for d in scan.dependencies}
    assert set(by_name) == {"zlib", "libcurl", "llvm"}     # the blank is dropped
    assert not by_name["zlib"].optional
    assert by_name["libcurl"].optional
    assert by_name["llvm"].optional and by_name["llvm"].required is None


# -- against a real meson ---------------------------------------------------


@needs_meson
def test_real_scan_of_a_small_project(tmp_path):
    (tmp_path / "meson.build").write_text(
        "project('demo', 'c')\n"
        "zlib = dependency('zlib')\n"
        "curl = dependency('libcurl', required: false)\n"
    )
    scan = scan_dependencies(tmp_path)
    assert scan is not None and scan.ok, getattr(scan, "error", None)
    by_name = {d.name: d for d in scan.dependencies}
    assert not by_name["zlib"].optional
    assert by_name["libcurl"].optional


@needs_meson
def test_real_scan_recurses_through_subdir(tmp_path):
    """Meson follows subdir(); the regex reading of one file cannot."""
    (tmp_path / "meson.build").write_text("project('demo', 'c')\nsubdir('src')\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "meson.build").write_text(
        "deep = dependency('libxml-2.0', required: false)\n"
    )
    scan = scan_dependencies(tmp_path)
    assert scan.ok
    assert "libxml-2.0" in {d.name for d in scan.dependencies}
