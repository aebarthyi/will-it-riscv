"""Bazel: bootstrapped from its source, and read for what a build downloads."""

from will_it_riscv import bazel, upstream
from will_it_riscv.plan import check_evidence


def write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


BAZEL_TREE = {
    "compile.sh": "#!/bin/bash\n",
    "scripts/bootstrap/buildenv.sh": (
        "for tool in basename cat chmod comm cp dirname find grep ln ls mkdir mktemp \\\n"
        "            readlink rm sed sort tail touch tr uname unzip which; do\n"
        "done\n"
        "JAVA_VERSION=${JAVA_VERSION:-21}\n"
    ),
    "scripts/bootstrap/bootstrap.sh": (
        '_BAZEL_ARGS="--spawn_strategy=standalone \\\n'
        "      --extra_toolchains=@rules_python//python:autodetecting_toolchain \\\n"
        "      --cxxopt=-std=c++17 \\\n"
    ),
}


def test_bazels_bootstrap_is_planned_from_its_own_scripts(tmp_path):
    write(tmp_path, BAZEL_TREE)
    plan = upstream.UPSTREAM["bazel-bootstrap"].planner(tmp_path, "8.7.0")
    assert plan.steps[0].packages == ["openjdk-21-jdk-headless", "unzip", "python3", "g++"]
    assert check_evidence(plan, tmp_path) == []
    assert "read, not run" in plan.unsure[0]


MODULE = """\
bazel_dep(name = "rules_python", version = "9.9.0")
DEFAULT_PYTHON_VERSION = "3.12"
python = use_extension("@rules_python//python/extensions:python.bzl", "python")
python.defaults(python_version = DEFAULT_PYTHON_VERSION)
python.toolchain(python_version = "3.12")
pip = use_extension("@rules_python//python/extensions:pip.bzl", "pip")
pip.parse(
    download_only = True,
    local_wheels = {
        "demo": "dist/demo-*.whl",
        "demo-cuda12-plugin": "dist/x.whl",
        "numpy": "dist/numpy-*.whl",
    },
    target_platforms = ["{os}_x86_64", "{os}_aarch64"],
)
register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")
register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")
"""


def rules_python(cache, riscv=True):
    """A rules_python checkout, already cached: nothing is fetched."""
    root = cache / "rules_python-9.9.0"
    triple = "riscv64-unknown-linux-gnu" if riscv else "s390x-unknown-linux-gnu"
    write(root, {
        ".unpacked": "",
        "python/versions.bzl": 'MINOR_MAPPING = {\n    "3.12": "3.12.13",\n}\n',
        "python/private/runtimes_manifest_workspace.bzl": (
            f"abc  20260414/cpython-3.12.13+20260414-{triple}-install_only.tar.gz\n"
        ),
    })


def test_a_bazel_build_is_read_for_what_it_downloads(tmp_path):
    tree = write(tmp_path / "demo", {
        "MODULE.bazel": MODULE,
        ".bazelrc": "common:clang_local --@rules_ml_toolchain//common:enable_hermetic_cc=False\n",
    })
    rules_python(tmp_path / "cache")
    reading = bazel.read(tree, "demo", "riscv64", tmp_path / "cache")
    assert reading.blocked == []
    assert reading.wheels == ["numpy"]      # its own wheel and the plugin left out
    assert reading.compiler == "clang"
    assert any("CPython 3.12.13 for riscv64-unknown-linux-gnu" in r for r in reading.read)
    assert any("--config=clang_local" in r for r in reading.read)
    from will_it_riscv.plan import Plan, Step

    cited = Plan(repo="demo", steps=[Step(
        id="x", kind="python-install",
        evidence=reading.wheel_evidence + reading.compiler_evidence,
    )])
    assert check_evidence(cited, tree) == []


def test_a_hermetic_python_with_no_build_for_the_target_blocks_it(tmp_path):
    tree = write(tmp_path / "demo", {"MODULE.bazel": MODULE})
    rules_python(tmp_path / "cache", riscv=False)
    reading = bazel.read(tree, "demo", "riscv64", tmp_path / "cache")
    assert reading.blocked and "no build of it for riscv64" in reading.blocked[0]
    assert reading.compiler is None   # and no .bazelrc config for a local compiler


def test_a_bazel_package_is_planned_with_what_its_build_reads(tmp_path):
    from will_it_riscv.autoplan import auto_plan

    tree = write(tmp_path / "demo", {
        "MODULE.bazel": MODULE, ".bazelversion": "8.7.0\n",
        ".bazelrc": "common:clang_local --repo_env USE_HERMETIC_CC_TOOLCHAIN=0\n",
    })
    rules_python(tmp_path / "cache")
    plan = auto_plan(tree, "demo", "riscv64", tmp_path / "cache")
    assert [s.id for s in plan.steps] == ["bazel-tools", "bazel-wheels", "bazel-cc"]
    assert plan.step("bazel-wheels").packages == ["numpy"]
    assert plan.step("bazel-cc").packages == ["clang"]
    assert plan.read and not plan.blocked
    assert check_evidence(plan, tree) == []
