"""Recursion: fetch what has no riscv64 build, configure it, build back up."""

import io
import json
import tarfile

import pytest
from conftest import FakeIndex, make_sdist, metadata

from will_it_riscv import planrun, recurse
from will_it_riscv.autoplan import auto_plan, configured
from will_it_riscv.plan import check_evidence, parse_plan
from will_it_riscv.pseudobuild import available
from will_it_riscv.sources import choose_tag, fetch, repository_url, unpack

needs_cmake = pytest.mark.skipif(not available(), reason="cmake is not installed")


# -- fetching ---------------------------------------------------------------


def test_the_tag_that_names_the_package_wins():
    tags = ["jax-v0.11.2", "jaxlib-v0.11.2", "jaxlib-v0.11.20", "v0.11.2"]
    assert choose_tag(tags, "jaxlib", "0.11.2") == "jaxlib-v0.11.2"
    assert choose_tag(["v2.5.3", "v2.5.30", "2.5.3rc1"], "numpy", "2.5.3") == "v2.5.3"
    assert choose_tag(["v1.0"], "x", "2.0") is None


def test_an_archive_cannot_write_outside_its_source_tree(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b"x"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    error = unpack(buffer.getvalue(), "evil-1.0.tar.gz", tmp_path / "dest")
    assert error and "outside the source tree" in error
    assert not (tmp_path / "escape.txt").exists()


def test_links_in_an_sdist_are_left_behind(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        link = tarfile.TarInfo("pkg-1.0/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
        data = b"[project]\n"
        info = tarfile.TarInfo("pkg-1.0/pyproject.toml")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    assert unpack(buffer.getvalue(), "pkg-1.0.tar.gz", tmp_path / "dest") is None
    assert (tmp_path / "dest" / "pkg-1.0" / "pyproject.toml").exists()
    assert not (tmp_path / "dest" / "pkg-1.0" / "link").exists()


def test_the_sdist_is_fetched_at_the_version_resolved(tmp_path):
    sdist = make_sdist("pkg", "1.0", {"pyproject.toml": "[project]\nname = 'pkg'\n"})
    index = FakeIndex({"pkg": ["pkg-1.0.tar.gz", "pkg-2.0.tar.gz"]}, {},
                      {"pkg-1.0.tar.gz": sdist})
    tree = fetch(index, "pkg", "1.0", tmp_path)
    assert tree.ok and tree.kind == "sdist" and tree.origin == "pkg-1.0.tar.gz"
    assert (tree.path / "pyproject.toml").exists()


def test_without_an_sdist_the_repository_its_metadata_names_is_found():
    wheel = "jaxlib-0.11.2-cp312-cp312-manylinux_2_27_x86_64.whl"
    meta = metadata(name="jaxlib", version="0.11.2").replace(
        "\n\n", "\nProject-URL: Source Code, https://github.com/jax-ml/jax\n\n"
    )
    index = FakeIndex({"jaxlib": [wheel]}, {wheel: meta})
    release = next(iter(index.project("jaxlib").releases.values()))
    assert repository_url(index, release) == "https://github.com/jax-ml/jax"


def test_nothing_public_is_said_plainly(tmp_path):
    wheel = "closed-1.0-cp312-cp312-manylinux_2_27_x86_64.whl"
    index = FakeIndex({"closed": [wheel]}, {wheel: metadata(name="closed")})
    tree = fetch(index, "closed", "1.0", tmp_path)
    assert not tree.ok and "names no repository" in tree.error


# -- planning what was fetched ----------------------------------------------


def write(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def test_a_cmake_package_is_planned_to_be_configured(tmp_path):
    write(tmp_path, {
        "pyproject.toml": '[build-system]\nrequires = ["scikit-build-core"]\n'
                          'build-backend = "scikit_build_core.build"\n',
        "CMakeLists.txt": "cmake_minimum_required(VERSION 3.18)\nproject(pkg C)\n",
    })
    plan = auto_plan(tmp_path, "pkg")
    assert [(s.id, s.kind) for s in plan.steps] == [
        ("build-requires", "python-install"), ("configure", "cmake-configure"),
    ]
    assert plan.steps[0].section == "build-system"
    assert configured(plan)
    assert check_evidence(plan, tmp_path) == []   # its citations hold, like any plan's


def test_a_meson_python_package_is_planned_to_be_set_up(tmp_path):
    """meson and ninja come from pip, as build requirements: nothing to ask the distro.

    numpy ships its own Meson, and says so in [tool.meson-python].
    """
    write(tmp_path, {
        "pyproject.toml": '[build-system]\nrequires = ["meson-python"]\n'
                          'build-backend = "mesonpy"\n'
                          "[tool.meson-python]\nmeson = 'vendored-meson/meson/meson.py'\n"
                          "[tool.meson-python.args]\nsetup = ['-Dblas=openblas', '--vsenv']\n",
        "meson.build": "project('pkg', 'c')\n",
        "CMakeLists.txt": "project(unused)\n",
    })
    plan = auto_plan(tmp_path, "pkg")
    assert [(s.id, s.kind) for s in plan.steps] == [
        ("build-requires", "python-install"), ("setup", "meson-setup"),
    ]
    setup = plan.steps[1]
    assert setup.meson == "vendored-meson/meson/meson.py"
    assert setup.defines == {"blas": "openblas"}
    assert configured(plan)
    assert check_evidence(plan, tmp_path) == []


def test_a_plain_meson_build_needs_meson_from_the_distro(tmp_path):
    write(tmp_path, {"meson.build": "project('pkg', 'c')\n"})
    plan = auto_plan(tmp_path, "pkg")
    assert next(s for s in plan.steps if s.id == "meson-tools").packages == [
        "meson", "ninja-build",
    ]
    assert any(s.kind == "meson-setup" for s in plan.steps) and configured(plan)


def test_a_pinned_tool_is_asked_for_at_its_pin(tmp_path):
    """jax pins Bazel 8.7.0; a crate names the oldest rustc that builds it."""
    write(tmp_path, {".bazelversion": "8.7.0\n", "MODULE.bazel": "module(name = 'x')\n"})
    assert auto_plan(tmp_path, "x").steps[-1].packages == ["bazel-bootstrap>=8.7.0"]
    other = tmp_path / "rust"
    write(other, {"Cargo.toml": '[package]\nname = "r"\nrust-version = "1.88"\n'})
    assert auto_plan(other, "r").steps[-1].packages == ["rustc>=1.88", "cargo"]


def test_a_setuptools_build_says_what_it_is(tmp_path):
    write(tmp_path, {"setup.py": "from setuptools import setup\nsetup()\n", "ext.c": "int x;\n"})
    plan = auto_plan(tmp_path, "pkg")
    assert any("builds with setuptools" in u for u in plan.unsure)


def test_a_rust_backend_needs_the_rust_toolchain(tmp_path):
    write(tmp_path, {
        "pyproject.toml": '[build-system]\nrequires = ["maturin"]\nbuild-backend = "maturin"\n',
    })
    plan = auto_plan(tmp_path, "pkg")
    assert next(s for s in plan.steps if s.id == "cargo-tools").packages == ["rustc", "cargo"]


def test_a_repository_of_several_projects_plans_the_one_asked_for(tmp_path):
    """The jax repository is jax at the top and jaxlib in jaxlib/, built by Bazel."""
    write(tmp_path, {
        "pyproject.toml": '[project]\nname = "jax"\n[build-system]\nrequires = ["setuptools"]\n',
        "jaxlib/pyproject.toml": '[project]\nname = "jaxlib"\n'
                                 '[build-system]\nrequires = ["wheel"]\n',
        ".bazelversion": "7.4.1\n",
    })
    plan = auto_plan(tmp_path, "jaxlib")
    install = next(s for s in plan.steps if s.id == "build-requires")
    assert install.manifest == "jaxlib/pyproject.toml"
    assert next(s for s in plan.steps if s.id == "bazel-tools").packages == [
        "bazel-bootstrap>=7.4.1"
    ]


# -- recursing --------------------------------------------------------------


class FakeDistro:
    def __init__(self, names):
        from types import SimpleNamespace

        self.names_ = set(names)
        self.versions_ = dict(names) if isinstance(names, dict) else {}
        self.spec = SimpleNamespace(label="Debian 13 (trixie)", id="debian:trixie")
        self.arch = "riscv64"
        self.available = True

    def first_available(self, candidates):
        return next((c for c in candidates if c in self.names_), None)

    def has(self, package):
        return package in self.names_

    def version(self, package):
        return self.versions_.get(package)

    def python_package(self, name):
        return None


def sdist_of(name, build_requires=(), files=None):
    requires = ", ".join(f'"{r}"' for r in build_requires)
    content = {
        "pyproject.toml": f'[project]\nname = "{name}"\nversion = "1.0"\n'
                          f"[build-system]\nrequires = [{requires}]\n",
        "PKG-INFO": f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n\n",
        "src.c": "int x;\n",
        **(files or {}),
    }
    return make_sdist(name, "1.0", content)


def run(tmp_path, target, root_deps, sdists, wheels=None, distro=None, **options):
    """A root plan installing root_deps, recursed into against a fake index."""
    projects = {name: [f"{name}-1.0.tar.gz"] for name in sdists}
    meta = {}
    for name, (wheel, requires) in (wheels or {}).items():
        projects.setdefault(name, []).append(wheel)
        meta[wheel] = metadata(*requires, name=name)
    index = FakeIndex(projects, meta, {f"{n}-1.0.tar.gz": b for n, b in sdists.items()})
    repo = tmp_path / "repo"
    repo.mkdir()
    deps = ", ".join(f'"{d}"' for d in root_deps)
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "root"\nversion = "1"\ndependencies = [{deps}]\n'
    )
    plan = parse_plan({"repo": "root", "steps": [
        {"id": "py", "kind": "python-install", "manifest": "pyproject.toml", "evidence": []},
    ]})
    result = planrun.execute(plan, repo, index=index, target=target, distro=distro, timeout=120)
    return recurse.recurse(
        result, index=index, target=target, distro=distro, cache_root=tmp_path / "src",
        timeout=120, **options,
    )


@needs_cmake
def test_the_answer_is_built_back_up_from_the_leaves(tmp_path, target):
    """a configures with CMake; it builds with b, which builds with SCons."""
    sdists = {
        "a": sdist_of("a", ["b"], {"CMakeLists.txt": "cmake_minimum_required(VERSION 3.18)\n"
                                                     "project(a NONE)\n"}),
        "b": sdist_of("b", [], {"SConstruct": "Program('b.c')\n"}),
    }
    walk = run(tmp_path, target, ["a"], sdists, distro=FakeDistro({"scons"}))
    a, b = walk.nodes["pypi:a@1.0"], walk.nodes["pypi:b@1.0"]
    assert b.status == recurse.PROBABLY          # read, not configured
    assert a.status == recurse.PROBABLY          # configured, but b could only be read
    assert "b could only be read" in a.reason
    assert walk.order.index("pypi:b@1.0") < walk.order.index("pypi:a@1.0")
    assert walk.verdict == "probably"
    json.dumps(recurse.to_dict(walk))
    assert '"root" -> "pypi:a@1.0"' in recurse.to_dot(walk)


@needs_cmake
def test_a_configured_package_with_nothing_under_it_is_buildable(tmp_path, target):
    sdists = {"a": sdist_of("a", [], {"CMakeLists.txt": "cmake_minimum_required(VERSION 3.18)\n"
                                                         "project(a NONE)\n"})}
    walk = run(tmp_path, target, ["a"], sdists)
    assert walk.nodes["pypi:a@1.0"].status == recurse.BUILDABLE
    assert walk.verdict == "yes" and walk.order == ["pypi:a@1.0"]


def test_what_has_nothing_public_blocks_everything_above_it(tmp_path, target):
    wheels = {"closed": ("closed-1.0-cp312-cp312-manylinux_2_27_x86_64.whl", [])}
    sdists = {"a": sdist_of("a", ["closed"])}
    walk = run(tmp_path, target, ["a"], sdists, wheels=wheels)
    closed = next(n for n in walk.nodes.values() if n.name == "closed")
    assert closed.status == recurse.BLOCKED
    assert walk.nodes["pypi:a@1.0"].status == recurse.BLOCKED
    assert walk.verdict == "no"
    assert recurse.chain_to_leaf(walk, "pypi:a@1.0")[-1] == closed.key


def test_a_bootstrap_cycle_is_named_not_followed_forever(tmp_path, target):
    sdists = {"a": sdist_of("a", ["b"]), "b": sdist_of("b", ["a"])}
    walk = run(tmp_path, target, ["a"], sdists)
    assert any("bootstrap cycle" in n.reason for n in walk.nodes.values())
    assert walk.verdict == "unknown"


def test_the_budget_is_kept(tmp_path, target):
    sdists = {"a": sdist_of("a", ["b"]), "b": sdist_of("b", [])}
    walk = run(tmp_path, target, ["a"], sdists, max_packages=1)
    assert walk.fetched == 1 and walk.budget_hit
    assert any("already fetched" in n.reason for n in walk.nodes.values())


@needs_cmake
def test_the_tree_and_the_build_order_are_reported(tmp_path, target):
    from io import StringIO

    from rich.console import Console

    sdists = {
        "a": sdist_of("a", ["b"], {"CMakeLists.txt": "cmake_minimum_required(VERSION 3.18)\n"
                                                     "project(a NONE)\n"}),
        "b": sdist_of("b", [], {"CMakeLists.txt": "cmake_minimum_required(VERSION 3.18)\n"
                                                  "project(b NONE)\n"}),
    }
    walk = run(tmp_path, target, ["a"], sdists)
    out = StringIO()
    recurse.render_text(walk, Console(file=out, width=200))
    text = out.getvalue()
    assert "✓ a 1.0" in text and "✓ b 1.0" in text
    assert "build order, dependencies first:" in text and "b 1.0 → a 1.0" in text
    assert "Will it riscv, all the way down?  YES" in text


@needs_cmake
def test_what_has_to_be_installed_first_is_listed_under_a_package(tmp_path, target):
    """cantera needs Boost, BLAS and HDF5 installed before it can be built."""
    sdists = {"a": sdist_of("a", [], {"CMakeLists.txt": "cmake_minimum_required(VERSION 3.18)\n"
                                                         "project(a NONE)\n"
                                                         "find_package(ZLIB REQUIRED)\n"})}
    walk = run(tmp_path, target, ["a"], sdists, distro=FakeDistro({"zlib1g-dev"}))
    a = walk.nodes["pypi:a@1.0"]
    assert a.status == recurse.BUILDABLE, a.reason
    assert recurse.installed(a) == ["zlib1g-dev"]
    assert recurse.to_dict(walk)["nodes"][0]["installed_first"] == ["zlib1g-dev"]


def test_a_tool_the_archive_has_too_old_is_installed_from_upstream(tmp_path, target):
    sdists = {"a": sdist_of("a", ["maturin"], {
        "Cargo.toml": '[package]\nname = "a"\nrust-version = "1.95"\n',
    })}
    distro = FakeDistro({"rustc": "1.85.0+dfsg1-1", "cargo": "1.85.0+dfsg1-1"})
    walk = run(tmp_path, target, ["a"], sdists, distro=distro)
    assert recurse.installed(walk.nodes["pypi:a@1.0"]) == ["cargo", "rustc (rustup)"]


def test_a_tool_the_archive_has_too_old_is_followed_to_its_source(tmp_path, target, monkeypatch):
    """jax pins Bazel 8.7.0; Debian has 4.2.3. So: Bazel's own bootstrap, one level down."""
    from test_bazel import BAZEL_TREE, write

    from will_it_riscv.sources import SourceTree

    bazel_tree = write(tmp_path / "bazel-src", BAZEL_TREE)
    fetched = []

    def fetch_upstream(tool, version, cache_root):
        fetched.append((tool.name, version))
        return SourceTree(name=tool.name, version=version, path=bazel_tree, kind="git",
                          origin=f"{tool.repository}@{version}")

    monkeypatch.setattr(recurse, "fetch_upstream", fetch_upstream)
    sdists = {"a": sdist_of("a", [], {".bazelversion": "8.7.0\n", "MODULE.bazel": ""})}
    distro = FakeDistro({
        "bazel-bootstrap": "4.2.3+ds-11", "openjdk-21-jdk-headless": "21.0.12",
        "unzip": "6.0", "python3": "3.13.5", "g++": "14.2.0",
    })
    walk = run(tmp_path, target, ["a"], sdists, distro=distro)
    assert fetched == [("bazel", "8.7.0")]
    tool = walk.nodes["debian:bazel-bootstrap@8.7.0"]
    assert tool.status == recurse.PROBABLY, tool.reason
    assert "read, not run" in tool.reason
    assert walk.order.index(tool.key) < walk.order.index("pypi:a@1.0")
    assert walk.nodes["pypi:a@1.0"].status == recurse.PROBABLY
    assert recurse.installed(tool) == [
        "g++", "openjdk-21-jdk-headless", "python3", "unzip",
    ]
