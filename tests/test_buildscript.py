"""The build script: a plan's steps, as the build itself, on the target."""

import os
import platform
import subprocess

import pytest

from will_it_riscv import planrun
from will_it_riscv.buildscript import build_script, switch
from will_it_riscv.plan import parse_plan


def plan(*steps, **extra):
    return {"version": 1, "repo": "demo", "steps": list(steps), **extra}


def result_for(tmp_path, raw, statuses=None, packages=(), from_source=()):
    """A run's result, made by hand: every step done, every package in the archive."""
    result = planrun.PlanResult(plan=parse_plan(raw), root=tmp_path)
    for step in result.plan.steps:
        status = (statuses or {}).get(step.id, "done")
        result.steps.append(planrun.StepResult(step, status, "ok"))
    for name in packages:
        node = planrun._node(result, f"debian:{name}", "debian", name)
        node.required = True
        planrun._settle(node, planrun.BINARY, "in the archive")
        node.package = name
    for name, version in from_source:
        node = planrun._node(result, f"pypi:{name}", "pypi", name)
        node.required, node.version = True, version
        planrun._settle(node, planrun.SOURCE, "no wheel for riscv64")
    result.answer = planrun._answer(result)
    return result


def dry_run(tmp_path, script, **env):
    path = tmp_path / "build.sh"
    path.write_text(script)
    run = subprocess.run(
        ["bash", str(path)], capture_output=True, text=True, cwd=tmp_path, timeout=60,
        env={**os.environ, "DRY_RUN": "1", "ANY_ARCH": "1", "SRC": str(tmp_path), **env},
    )
    return run


def commands(run):
    return [line[2:] for line in run.stderr.splitlines() if line.startswith("+ ")]


CMAKE = {"id": "cfg", "kind": "cmake-configure", "after": ["deps"], "evidence": [],
         "defines": {"WITH_THING": True}}
DEPS = {"id": "deps", "kind": "system-packages", "packages": ["libfoo-dev"], "evidence": []}


def test_a_configure_is_built_and_installed_into_the_prefix(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("project(demo)\n")
    script = build_script(result_for(tmp_path, plan(DEPS, CMAKE), packages=["libfoo-dev"]))
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    run = dry_run(tmp_path, script)
    assert run.returncode == 0, run.stderr
    ran = commands(run)
    assert ran[0] == "sudo apt-get update" or ran[0] == "apt-get update"
    assert "libfoo-dev" in ran[1] and "build-essential" in ran[1] and "cmake" in ran[1]
    configure, build, install = ran[2:5]
    assert configure.startswith(f"cmake -S {tmp_path} -B {tmp_path}/_riscv/build/cfg")
    assert "-DWITH_THING=ON" in configure and "-DCMAKE_BUILD_TYPE=Release" in configure
    assert f"-DCMAKE_INSTALL_PREFIX={tmp_path}/_riscv/install" in configure
    assert build.startswith("cmake --build") and install.startswith("cmake --install")


def test_skip_apt_skips_the_system_packages(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("project(demo)\n")
    script = build_script(result_for(tmp_path, plan(DEPS, CMAKE), packages=["libfoo-dev"]))
    ran = commands(dry_run(tmp_path, script, SKIP_APT="1"))
    assert not any("apt-get" in c for c in ran)
    assert ran[0].startswith("cmake -S")


def test_nothing_a_plan_says_runs_as_a_command(tmp_path):
    """The plan is a model's: its values are data in the script, never shell."""
    odd = {"id": "cfg", "kind": "cmake-configure", "evidence": [], "source": "src dir",
           "defines": {"X": "$(touch pwned)", "Y": "a b; touch pwned2"}}
    (tmp_path / "src dir").mkdir()
    (tmp_path / "src dir" / "CMakeLists.txt").write_text("project(demo)\n")
    script = build_script(result_for(tmp_path, plan(odd)))
    run = dry_run(tmp_path, script, SKIP_APT="1")
    assert run.returncode == 0, run.stderr
    assert not (tmp_path / "pwned").exists() and not (tmp_path / "pwned2").exists()
    assert "-DX=\\$\\(touch\\ pwned\\)" in commands(run)[0]


def test_an_optional_step_runs_only_when_asked_for(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("project(demo)\n")
    gpu = {"id": "cfg-gpu", "kind": "cmake-configure", "evidence": [], "optional": True,
           "enabled_by": "--gpu", "defines": {"GPU": "ON"}}
    script = build_script(result_for(tmp_path, plan(CMAKE | {"after": []}, gpu)))
    assert switch("cfg-gpu") == "WITH_CFG_GPU" and "WITH_CFG_GPU=1" in script
    without = commands(dry_run(tmp_path, script, SKIP_APT="1"))
    with_gpu = commands(dry_run(tmp_path, script, SKIP_APT="1", WITH_CFG_GPU="1"))
    assert not any("-DGPU=ON" in c for c in without)
    assert any("-DGPU=ON" in c for c in with_gpu)


def test_a_driver_runs_its_own_configures(tmp_path):
    """MFC's ./mfc.sh build runs the CMake configures; the script runs the driver."""
    (tmp_path / "toolchain").mkdir()
    (tmp_path / "toolchain" / "main.py").write_text("")
    driver = {"id": "build", "kind": "python-run", "script": "toolchain/main.py",
              "args": ["build", "-j", "2"], "evidence": []}
    result = result_for(tmp_path, plan(CMAKE | {"after": []}, driver))
    next(s for s in result.steps if s.step.id == "build").driven = ["cfg"]
    script = build_script(result)
    assert "# cfg: the project's driver runs this configure itself" in script
    ran = commands(dry_run(tmp_path, script, SKIP_APT="1", JOBS="3"))
    assert not any(c.startswith("cmake") for c in ran)
    assert ran[0].startswith("python3 -m venv")
    # -j 2 suited the pretend run; the build takes the machine's job count.
    assert ran[-1] == f"(cd {tmp_path} && {tmp_path}/_riscv/venv/bin/python toolchain/main.py " \
                      "build -j 3)"


def test_pip_runs_the_configure_its_build_backend_owns(tmp_path):
    """numpy's meson-setup is what mesonpy runs: pip install . builds it once."""
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["meson-python"]\nbuild-backend = "mesonpy"\n'
        '[project]\nname = "demo"\nversion = "1"\n')
    (tmp_path / "meson.build").write_text("project('demo')\n")
    steps = plan(
        {"id": "setup", "kind": "meson-setup", "evidence": []},
        {"id": "install", "kind": "python-install", "manifest": "pyproject.toml",
         "evidence": []},
    )
    script = build_script(result_for(tmp_path, steps))
    assert "pip's build (mesonpy) runs this configure itself" in script
    ran = commands(dry_run(tmp_path, script, SKIP_APT="1"))
    assert not any(c.startswith("meson") for c in ran)
    assert ran[-1] == f"{tmp_path}/_riscv/venv/bin/python -m pip install {tmp_path}"


def test_what_has_only_source_is_built_first(tmp_path):
    (tmp_path / "requirements.txt").write_text("thing==1.0\n")
    steps = plan({"id": "py", "kind": "python-install", "manifest": "requirements.txt",
                  "evidence": []})
    script = build_script(result_for(tmp_path, steps, from_source=[("thing", "1.0")]))
    ran = commands(dry_run(tmp_path, script, SKIP_APT="1"))
    pip = [c for c in ran if " -m pip install " in c and "--upgrade" not in c]
    assert pip == [f"{tmp_path}/_riscv/venv/bin/python -m pip install thing==1.0",
                   f"{tmp_path}/_riscv/venv/bin/python -m pip install -r "
                   f"{tmp_path}/requirements.txt"]
    assert "python3-venv" in script and "python3-dev" in script


def test_rust_comes_from_rustup_when_the_archive_is_too_old(tmp_path):
    result = result_for(tmp_path, plan(DEPS))
    rustc = planrun._node(result, "debian:rustc", "debian", "rustc")
    rustc.required, rustc.wants = True, "1.95"
    planrun._settle(rustc, planrun.BINARY,
                    "too old; rustup ships riscv64gc-unknown-linux-gnu toolchains")
    script = build_script(result)
    assert "sh.rustup.rs" in script and 'export PATH="$HOME/.cargo/bin:$PATH"' in script
    assert "rustc >= 1.95" in script


def test_a_tool_built_from_its_source_is_looked_for_by_its_command(tmp_path):
    """jax wants Bazel 8.7.0: the package is bazel-bootstrap, and what it gives is bazel."""
    result = result_for(tmp_path, plan(DEPS))
    bazel = planrun._node(result, "debian:bazel-bootstrap", "debian", "bazel-bootstrap")
    bazel.required, bazel.wants = True, "8.7.0"
    planrun._settle(bazel, planrun.SOURCE, "too old; Bazel publishes no riscv64 binaries")
    result.answer = planrun._answer(result)
    script = build_script(result)
    assert "command -v bazel >/dev/null" in script
    assert "build bazel-bootstrap 8.7.0 from its source first" in script


def test_the_checkout_is_checked_before_anything_runs(tmp_path):
    script = build_script(result_for(tmp_path, plan(DEPS, CMAKE)))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    run = dry_run(tmp_path, script, SRC=str(elsewhere))
    assert run.returncode == 1 and "has no CMakeLists.txt" in run.stderr


@pytest.mark.skipif(platform.machine() == "riscv64", reason="this is the target")
def test_another_machine_is_refused_unless_asked(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("project(demo)\n")
    script = build_script(result_for(tmp_path, plan(CMAKE | {"after": []})))
    run = dry_run(tmp_path, script, ANY_ARCH="")
    assert run.returncode == 1 and "ANY_ARCH=1 runs it anyway" in run.stderr
