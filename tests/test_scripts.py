"""What a project's own install scripts install before they build."""

from conftest import FakeIndex, metadata

from will_it_riscv import cli
from will_it_riscv.analyze import Analyzer
from will_it_riscv.inputs import RootRequirements
from will_it_riscv.models import Verdict
from will_it_riscv.scripts import entry_scripts, find_script_installs
from will_it_riscv.source import inspect_repository


def build_repo(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def installs(root):
    return {(i.kind, i.target, i.extras) for i in find_script_installs(root)}


TOOLCHAIN = (
    '[project]\nname = "mfc"\nversion = "1.0"\n'
    'dependencies = ["fypp", "jax"]\n'
)

#: MFC's shape: the entry script sources a bootstrap script, which installs
#: the toolchain through two layers of wrapper function -- or, without uv,
#: straight through pip3 with environment variables in front.
MFC = {
    "mfc.sh": (
        "#!/bin/bash\n"
        'if [ ! -f "$(pwd)/toolchain/util.sh" ]; then exit 1; fi\n'
        '. "$(pwd)/toolchain/util.sh"\n'
        'mkdir -p "$(pwd)/build"\n'
        '. "$(pwd)/toolchain/bootstrap/python.sh" "$@"\n'
        'python3 "$(pwd)/toolchain/main.py" "$@"\n'
    ),
    "toolchain/util.sh": 'log() { echo "mfc: $*"; }\n',
    "toolchain/bootstrap/python.sh": (
        'log "(venv) Installing uv with pip install uv..."\n'
        'if PIP_DISABLE_PIP_VERSION_CHECK=1 pip3 install uv > "$PIP_LOG" 2>&1; then\n'
        '    ok "installed"\n'
        "fi\n"
        'uv_install() { flock "$UV_INSTALL_LOCK" uv pip install "$@"; }\n'
        "uv_install_with_retry() {\n"
        '    if uv_install "$@"; then\n'
        "        return 0\n"
        "    fi\n"
        '    uv_install "$@"\n'
        "}\n"
        'uv_install_with_retry "$(pwd)/toolchain" > "$PIP_LOG" 2>&1\n'
        'PIP_DISABLE_PIP_VERSION_CHECK=1 MAKEFLAGS=$nthreads pip3 install "$(pwd)/toolchain" > "$PIP_LOG" 2>&1 &\n'
    ),
    "toolchain/pyproject.toml": TOOLCHAIN,
    "toolchain/main.py": "",
    "CMakeLists.txt": "project(mfc Fortran)\n",
}


def test_mfc_installs_its_toolchain_before_it_builds(tmp_path):
    build_repo(tmp_path, MFC)
    assert installs(tmp_path) == {
        ("manifest", "toolchain/pyproject.toml", ()),
        ("requirement", "uv", ()),
    }


def test_the_report_says_how_the_install_was_reached(tmp_path):
    build_repo(tmp_path, MFC)
    toolchain = next(i for i in find_script_installs(tmp_path) if i.kind == "manifest")
    # Through both wrappers, from the line that calls them with a path.
    assert toolchain.chain == ("mfc.sh:5", "toolchain/bootstrap/python.sh:12")


def test_mfc_is_flagged_blocked_by_jaxlib(tmp_path, target):
    """The regression this exists for: MFC's riscv64 blocker is not in CMake.

    jax is pure Python; jaxlib publishes wheels for everything but riscv64
    and no sdist, so on riscv64 ./mfc.sh build stops at the venv bootstrap.
    """
    build_repo(tmp_path, MFC)
    scan = inspect_repository(tmp_path, scan_ci=False, use_meson_introspect=False)
    roots = RootRequirements(source=str(tmp_path), project_name="MFC")
    adopted = cli._adopt_script_installs(roots, scan)
    assert {i.target for i in adopted} == {"toolchain/pyproject.toml", "uv"}
    assert {r.name for r in roots.runtime} == {"fypp", "jax", "uv"}

    index = FakeIndex(
        {
            "fypp": ["fypp-3.2-py3-none-any.whl"],
            "jax": ["jax-0.11.2-py3-none-any.whl"],
            "jaxlib": [
                "jaxlib-0.11.2-cp312-cp312-manylinux_2_27_x86_64.whl",
                "jaxlib-0.11.2-cp312-cp312-manylinux_2_27_aarch64.whl",
                "jaxlib-0.11.2-cp312-cp312-macosx_11_0_arm64.whl",
            ],
            "uv": ["uv-0.9.0-py3-none-manylinux_2_28_riscv64.whl"],
        },
        {
            "fypp-3.2-py3-none-any.whl": metadata(name="fypp", version="3.2"),
            "jax-0.11.2-py3-none-any.whl": metadata(
                "jaxlib<=0.11.2,>=0.11.2", name="jax", version="0.11.2"
            ),
            "jaxlib-0.11.2-cp312-cp312-manylinux_2_27_x86_64.whl": metadata(
                name="jaxlib", version="0.11.2"
            ),
            "uv-0.9.0-py3-none-manylinux_2_28_riscv64.whl": metadata(
                name="uv", version="0.9.0"
            ),
        },
    )
    result = Analyzer(index, target).run(roots)
    assert result.packages["jaxlib"].verdict is Verdict.NO_DISTRIBUTION
    assert result.packages["jaxlib"].required_by == ["jax 0.11.2"]
    assert result.packages["fypp"].verdict is Verdict.PURE_PYTHON
    assert result.exit_code() == 2


def test_an_adopted_manifest_is_not_warned_about(tmp_path):
    """It was analysed, so "not analysed: not at the root" would be false."""
    build_repo(tmp_path, {**MFC, "docs/requirements.txt": "sphinx\n"})
    scan = inspect_repository(tmp_path, scan_ci=False, use_meson_introspect=False)
    roots = RootRequirements(source=str(tmp_path), project_name="MFC")
    adopted = cli._adopt_script_installs(roots, scan)
    unanalysed = cli._unanalysed_manifests(scan, adopted)
    # The toolchain is analysed; the docs requirements nothing installs are
    # still only offered.
    assert [m.relative_to(scan.root).as_posix() for m in unanalysed] == [
        "docs/requirements.txt"
    ]


# -- reading shell ----------------------------------------------------------


def test_what_is_only_printed_is_not_installed(tmp_path):
    build_repo(tmp_path, {
        "setup.sh": (
            "#!/bin/sh\n"
            "echo pip install numpy\n"
            'printf "%s\\n" "then run: pip install scipy"\n'
            'log "pip install pandas"\n'
        ),
    })
    assert installs(tmp_path) == set()


def test_requirements_files_editables_and_extras(tmp_path):
    build_repo(tmp_path, {
        "pyproject.toml": '[project]\nname = "x"\nversion = "1"\n',
        "requirements/build.txt": "numpy\n",
        "bootstrap.sh": (
            "#!/usr/bin/env bash\n"
            "python3 -m pip install -r requirements/build.txt\n"
            'pip install -e ".[test]"\n'
            "pip install --upgrade --index-url https://example.invalid/simple cython\n"
        ),
    })
    assert installs(tmp_path) == {
        ("manifest", "requirements/build.txt", ()),
        ("manifest", "pyproject.toml", ("test",)),
        ("requirement", "cython", ()),
    }


def test_no_deps_installs_nothing_it_depends_on(tmp_path):
    build_repo(tmp_path, {
        "tools/pyproject.toml": '[project]\nname = "t"\nversion = "1"\n',
        "setup.sh": "pip install --no-deps ./tools\n",
    })
    assert installs(tmp_path) == set()


def test_the_root_is_found_however_it_is_spelled(tmp_path):
    build_repo(tmp_path, {
        "a/requirements.txt": "x\n",
        "b/requirements.txt": "y\n",
        "c/requirements.txt": "z\n",
        "build.sh": (
            "#!/bin/bash\n"
            'ROOT="$(cd "$(dirname "$0")" && pwd)"\n'
            'pip install -r "$ROOT/a/requirements.txt"\n'
            "TOP=$(git rev-parse --show-toplevel)\n"
            "pip install -r $TOP/b/requirements.txt\n"
            'pip install -r "${BASH_SOURCE%/*}/c/requirements.txt"\n'
        ),
    })
    assert {t for _, t, _ in installs(tmp_path)} == {
        "a/requirements.txt", "b/requirements.txt", "c/requirements.txt",
    }


def test_invoked_scripts_are_followed_too(tmp_path):
    build_repo(tmp_path, {
        "scripts/deps.sh": "pip install -r scripts/requirements.txt\n",
        "scripts/requirements.txt": "x\n",
        "install.sh": "#!/bin/sh\nbash scripts/deps.sh --quiet\n",
    })
    (only,) = find_script_installs(tmp_path)
    assert only.target == "scripts/requirements.txt"
    assert only.chain == ("install.sh:2", "scripts/deps.sh:1")


def test_a_source_cycle_terminates(tmp_path):
    build_repo(tmp_path, {
        "a.sh": ". ./b.sh\npip install -r req.txt\n",
        "b.sh": ". ./a.sh\n",
        "req.txt": "x\n",
    })
    assert installs(tmp_path) == {("manifest", "req.txt", ())}


def test_what_cannot_be_resolved_is_left_alone(tmp_path):
    build_repo(tmp_path, {
        "setup.sh": (
            'pip install -r "$REQS"\n'          # set somewhere this cannot see
            "pip install -r /etc/requirements.txt\n"   # outside the repository
            "pip install -r missing.txt\n"      # not there
            'pip install "$@"\n'                # a bare forward, no caller
        ),
    })
    assert installs(tmp_path) == set()


def test_a_generated_configure_is_not_an_install_recipe(tmp_path):
    build_repo(tmp_path, {
        "configure": "#!/bin/sh\npip install -r requirements.txt\n",
        "requirements.txt": "x\n",
    })
    assert entry_scripts(tmp_path) == []
    assert installs(tmp_path) == set()


def test_only_root_level_shell_scripts_are_entry_points(tmp_path):
    build_repo(tmp_path, {
        "run": "#!/usr/bin/env bash\ntrue\n",
        "tool.py": "#!/usr/bin/env python3\n",
        "notes": "plain text\n",
        "sub/deep.sh": "pip install -r req.txt\n",
    })
    assert [p.name for p in entry_scripts(tmp_path)] == ["run"]
