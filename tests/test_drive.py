"""Running a project's own build driver in the pretend environment."""

import json

from conftest import FakeIndex, metadata

from will_it_riscv import drive, planrun
from will_it_riscv.plan import parse_plan


def fake_install(installed):
    """Stand in for pip: write a package, with its dist-info, into the site."""
    def install(spec, site, cache):
        name = spec.split("==")[0]
        module = name.replace("-", "_")
        (site / module).mkdir()
        (site / module / "__init__.py").write_text(f"NAME = {name!r}\n")
        info = site / f"{module}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
        (info / "top_level.txt").write_text(module + "\n")
        installed.append(spec)
        return True
    return install


DRIVER = '''\
import os, signal, subprocess
import usedpkg                                  # the plan installs this
try:
    import optionalthing                        # nothing provides this
except ImportError:
    pass
import json                                     # the standard library
os.makedirs("out", exist_ok=True)
open("build/state.yaml", "w").write("x")        # build/ does not exist yet
subprocess.run(["cmake", "-DDEMO=ON", "-S", os.getcwd(), "-B", "build/b"], check=True)
subprocess.run(["cmake", "--build", "build/b"], check=True)
os.kill(os.getpid(), signal.SIGTERM)             # MFC's way of leaving
'''


def repo_with_driver(tmp_path, body=DRIVER):
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "tools" / "main.py").write_text(body)
    (tmp_path / "CMakeLists.txt").write_text("project(demo NONE)\n")
    return tmp_path


def test_the_driver_is_watched_not_trusted(tmp_path, monkeypatch):
    installed = []
    monkeypatch.setattr(drive, "_pip_install", fake_install(installed))
    root = repo_with_driver(tmp_path / "repo")
    trace = drive.trace(root, "tools/main.py", [], {"usedpkg": "1.0", "neverpkg": "2.0"},
                        timeout=120)
    # It asked for usedpkg, which the plan installs: put in, at the plan's version.
    assert installed == ["usedpkg==1.0"]
    assert trace.imported == {"usedpkg"}
    # It tried optionalthing, which nothing provides: stubbed, and said so.
    assert trace.not_in_plan == ["optionalthing"]
    # It wanted build/ to exist, as mfc.sh would have made it.
    assert trace.made == ["build"]
    # Its build tools were answered by shims, with paths as the repo sees them.
    assert ("cmake", ["-DDEMO=ON", "-S", ".", "-B", "build/b"]) in trace.commands
    assert ("cmake", ["--build", "build/b"]) in trace.commands


def test_the_driver_sees_none_of_will_it_riscvs_own_packages(tmp_path, monkeypatch):
    """rich is installed alongside this tool; MFC importing it must still count."""
    installed = []
    monkeypatch.setattr(drive, "_pip_install", fake_install(installed))
    root = repo_with_driver(tmp_path / "repo", "import rich\n")
    trace = drive.trace(root, "tools/main.py", [], {"rich": "15.0.0"}, timeout=120)
    assert installed == ["rich==15.0.0"]
    assert trace.imported == {"rich"}


def test_the_source_tree_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "_pip_install", fake_install([]))
    root = repo_with_driver(tmp_path / "repo")
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    drive.trace(root, "tools/main.py", [], {"usedpkg": "1.0"}, timeout=120)
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before


def test_module_names_are_mapped_to_their_distributions():
    wanted = {"pyyaml": "6.0", "scikit-image": "0.2", "rich": "15"}
    assert drive.dist_for_module("yaml", wanted) == "pyyaml"
    assert drive.dist_for_module("skimage", wanted) == "scikit-image"
    assert drive.dist_for_module("rich", wanted) == "rich"
    assert drive.dist_for_module("jax", wanted) is None


# -- in a plan ---------------------------------------------------------------


def test_the_default_install_is_required_whatever_the_build_imports(
    tmp_path, monkeypatch, target
):
    """MFC's shape: the build step imports pyrometheus, which declares jaxlib,
    which has nothing for riscv64. The build step never loads jaxlib -- but
    the default install brings it, and the default build is the minimal spec.
    That the build step never loads it is information, not a pass."""
    monkeypatch.setattr(drive, "_pip_install", fake_install([]))
    root = repo_with_driver(tmp_path / "repo", "import pyrometheus\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "t"\nversion = "1"\ndependencies = ["pyrometheus", "extras"]\n'
    )
    index = FakeIndex(
        {
            "pyrometheus": ["pyrometheus-1.0-py3-none-any.whl"],
            "jaxlib": ["jaxlib-1.0-cp312-cp312-manylinux_2_27_x86_64.whl"],
            "extras": ["extras-1.0.tar.gz"],
        },
        {
            "pyrometheus-1.0-py3-none-any.whl": metadata("jaxlib", name="pyrometheus"),
            "jaxlib-1.0-cp312-cp312-manylinux_2_27_x86_64.whl": metadata(name="jaxlib"),
        },
    )
    plan = parse_plan({"repo": "demo", "steps": [
        {"id": "py", "kind": "python-install", "manifest": "pyproject.toml", "evidence": []},
        {"id": "build", "kind": "python-run", "script": "tools/main.py", "after": ["py"],
         "evidence": []},
    ]})
    result = planrun.execute(plan, root, index=index, target=target, timeout=120)
    jaxlib = result.nodes["pypi:jaxlib"]
    assert jaxlib.required
    assert (jaxlib.usage, jaxlib.declared_by) == ("declared", ["pyrometheus"])
    assert result.nodes["pypi:pyrometheus"].usage == "imported"
    assert result.answer.verdict == "no"
    assert result.answer.blockers == ["pypi:jaxlib"]
    json.dumps(planrun.to_dict(result))


def test_a_driver_that_did_not_finish_proves_nothing_unused(tmp_path, monkeypatch, target):
    monkeypatch.setattr(drive, "_pip_install", fake_install([]))
    root = repo_with_driver(tmp_path / "repo", "import sys\nsys.exit(3)\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "t"\nversion = "1"\ndependencies = ["lonely"]\n'
    )
    index = FakeIndex(
        {"lonely": ["lonely-1.0-py3-none-any.whl"]},
        {"lonely-1.0-py3-none-any.whl": metadata(name="lonely")},
    )
    plan = parse_plan({"repo": "demo", "steps": [
        {"id": "py", "kind": "python-install", "manifest": "pyproject.toml", "evidence": []},
        {"id": "build", "kind": "python-run", "script": "tools/main.py", "after": ["py"],
         "evidence": []},
    ]})
    result = planrun.execute(plan, root, index=index, target=target, timeout=120)
    assert result.nodes["pypi:lonely"].usage is None


def test_the_plan_check_compares_configures_with_the_plan(tmp_path, monkeypatch, target):
    monkeypatch.setattr(drive, "_pip_install", fake_install([]))
    root = repo_with_driver(tmp_path / "repo", (
        "import os, subprocess\n"
        "subprocess.run(['cmake', '-DDEMO=ON', '-S', os.getcwd(), '-B', 'b'])\n"
        "subprocess.run(['cmake', '-DOTHER=ON', '-S', os.getcwd() + '/sub', '-B', 'c'])\n"
    ))
    plan = parse_plan({"repo": "demo", "steps": [
        {"id": "cfg", "kind": "cmake-configure", "defines": {"DEMO": True}, "evidence": []},
        {"id": "missing", "kind": "cmake-configure", "source": "nowhere", "evidence": []},
        {"id": "build", "kind": "python-run", "script": "tools/main.py", "evidence": []},
    ]})
    result = planrun.execute(plan, root, index=FakeIndex({}), target=target, timeout=120)
    checks = next(s for s in result.steps if s.step.id == "build").plan_check
    assert checks[0] == "ran 2 configures; 1 of the plan's 2 cmake steps match"
    assert "the plan has 'missing', which the driver never configured" in checks
    assert any("the driver configured sub" in c for c in checks)
