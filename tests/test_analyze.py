from conftest import FakeIndex, make_sdist, metadata
from packaging.requirements import Requirement

from will_it_riscv.analyze import Analyzer
from will_it_riscv.inputs import RootRequirements
from will_it_riscv.models import Verdict


def roots(*specs, build=()):
    return RootRequirements(
        source="test",
        project_name="test",
        runtime=[Requirement(s) for s in specs],
        build=[Requirement(s) for s in build],
    )


def analyze(target, index, *specs, build=(), **kwargs):
    return Analyzer(index, target, **kwargs).run(roots(*specs, build=build))


def test_pure_python_package(target):
    index = FakeIndex(
        {"click": ["click-8.0-py3-none-any.whl"]},
        {"click-8.0-py3-none-any.whl": metadata(name="click", version="8.0")},
    )
    result = analyze(target, index, "click")
    assert result.packages["click"].verdict is Verdict.PURE_PYTHON
    assert result.exit_code() == 0


def test_riscv64_wheel_is_available(target):
    index = FakeIndex(
        {"fast": [
            "fast-1.0-cp312-cp312-manylinux_2_39_x86_64.whl",
            "fast-1.0-cp312-cp312-manylinux_2_39_riscv64.whl",
        ]},
        {"fast-1.0-cp312-cp312-manylinux_2_39_riscv64.whl": metadata(name="fast")},
    )
    report = analyze(target, index, "fast").packages["fast"]
    assert report.verdict is Verdict.WHEEL_AVAILABLE
    assert report.matching_wheels == ["fast-1.0-cp312-cp312-manylinux_2_39_riscv64.whl"]


def test_binary_wheels_but_none_for_the_target(target):
    sdist = make_sdist("slow", "1.0", {
        "setup.py": 'from setuptools import setup, Extension\n'
                    'setup(ext_modules=[Extension("s", ["s.c"], libraries=["pq"])])\n',
        "s.c": "#include <libpq-fe.h>\n",
    })
    index = FakeIndex(
        {"slow": [
            "slow-1.0-cp312-cp312-manylinux_2_39_x86_64.whl",
            "slow-1.0.tar.gz",
        ]},
        {"slow-1.0-cp312-cp312-manylinux_2_39_x86_64.whl": metadata(name="slow")},
        {"slow-1.0.tar.gz": sdist},
    )
    result = analyze(target, index, "slow")
    report = result.packages["slow"]
    assert report.verdict is Verdict.NEEDS_BUILD
    assert "x86_64" in report.reasons[0]
    assert "libpq" in result.system_requirements
    assert result.system_requirements["libpq"].debian == ("libpq-dev",)
    assert result.exit_code() == 1


def test_sdist_only_but_pure_is_still_pure(target):
    """An sdist with no compiled sources installs as pure Python."""
    sdist = make_sdist("old", "1.0", {
        "setup.py": "from setuptools import setup\nsetup()\n",
        "old/__init__.py": "",
        "PKG-INFO": "Metadata-Version: 2.1\nName: old\nVersion: 1.0\n\n",
    })
    index = FakeIndex({"old": ["old-1.0.tar.gz"]}, {}, {"old-1.0.tar.gz": sdist})
    report = analyze(target, index, "old").packages["old"]
    assert report.verdict is Verdict.PURE_PYTHON
    assert "no compiled sources" in report.reasons[-1]


def test_no_sdist_and_no_matching_wheel_is_unbuildable(target):
    index = FakeIndex(
        {"binonly": ["binonly-1.0-cp312-cp312-manylinux_2_39_x86_64.whl"]},
        {"binonly-1.0-cp312-cp312-manylinux_2_39_x86_64.whl": metadata(name="binonly")},
    )
    result = analyze(target, index, "binonly")
    assert result.packages["binonly"].verdict is Verdict.NO_DISTRIBUTION
    assert result.exit_code() == 2


def test_missing_from_index_is_unresolved(target):
    result = analyze(target, FakeIndex({}), "nope")
    assert result.packages["nope"].verdict is Verdict.UNRESOLVED
    assert "not found on the index" in result.packages["nope"].reasons[0]


def test_transitive_dependencies_are_followed(target):
    index = FakeIndex(
        {
            "a": ["a-1.0-py3-none-any.whl"],
            "b": ["b-2.0-py3-none-any.whl"],
            "c": ["c-3.0-py3-none-any.whl"],
        },
        {
            "a-1.0-py3-none-any.whl": metadata("b", name="a"),
            "b-2.0-py3-none-any.whl": metadata("c>=3", name="b"),
            "c-3.0-py3-none-any.whl": metadata(name="c"),
        },
    )
    result = analyze(target, index, "a")
    assert set(result.packages) == {"a", "b", "c"}
    assert result.packages["c"].depth == 2


def test_extras_are_followed(target):
    index = FakeIndex(
        {"srv": ["srv-1.0-py3-none-any.whl"],
         "base": ["base-1.0-py3-none-any.whl"],
         "turbo": ["turbo-1.0-py3-none-any.whl"]},
        {
            "srv-1.0-py3-none-any.whl": metadata(
                "base", 'turbo; extra == "fast"', name="srv", extras=("fast",)
            ),
            "base-1.0-py3-none-any.whl": metadata(name="base"),
            "turbo-1.0-py3-none-any.whl": metadata(name="turbo"),
        },
    )
    assert "turbo" not in analyze(target, index, "srv").packages
    assert "turbo" in analyze(target, index, "srv[fast]").packages


def test_markers_are_evaluated_for_the_target_not_the_host(target):
    index = FakeIndex(
        {"x": ["x-1.0-py3-none-any.whl"],
         "wintool": ["wintool-1.0-py3-none-any.whl"],
         "lintool": ["lintool-1.0-py3-none-any.whl"]},
        {
            "x-1.0-py3-none-any.whl": metadata(
                'wintool; sys_platform == "win32"',
                'lintool; platform_machine == "riscv64"',
                name="x",
            ),
            "wintool-1.0-py3-none-any.whl": metadata(name="wintool"),
            "lintool-1.0-py3-none-any.whl": metadata(name="lintool"),
        },
    )
    packages = analyze(target, index, "x").packages
    assert "wintool" not in packages
    assert "lintool" in packages


def test_build_requirements_are_followed(target):
    sdist = make_sdist("needsc", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["setuptools","Cython>=3"]\n',
        "s.c": "int main(){}\n",
    })
    index = FakeIndex(
        {
            "needsc": ["needsc-1.0.tar.gz"],
            "setuptools": ["setuptools-70.0-py3-none-any.whl"],
            "cython": ["cython-3.0-cp312-cp312-manylinux_2_39_riscv64.whl"],
        },
        {
            "setuptools-70.0-py3-none-any.whl": metadata(name="setuptools"),
            "cython-3.0-cp312-cp312-manylinux_2_39_riscv64.whl": metadata(name="cython"),
        },
        {"needsc-1.0.tar.gz": sdist},
    )
    result = analyze(target, index, "needsc")
    assert result.packages["cython"].is_build_dependency
    assert result.packages["needsc"].verdict is Verdict.NEEDS_BUILD


def test_unbuildable_build_backend_blocks_the_package(target):
    sdist = make_sdist("blocked", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["weirdbackend"]\n'
                          'build-backend="weirdbackend"\n',
        "s.c": "int main(){}\n",
    })
    index = FakeIndex(
        {
            "blocked": ["blocked-1.0.tar.gz"],
            "weirdbackend": ["weirdbackend-1.0-cp312-cp312-manylinux_2_39_x86_64.whl"],
        },
        {"weirdbackend-1.0-cp312-cp312-manylinux_2_39_x86_64.whl":
            metadata(name="weirdbackend")},
        {"blocked-1.0.tar.gz": sdist},
    )
    result = analyze(target, index, "blocked")
    assert result.packages["weirdbackend"].verdict is Verdict.NO_DISTRIBUTION
    assert result.packages["blocked"].verdict is Verdict.NEEDS_BUILD_BLOCKED
    assert result.exit_code() == 2


def test_build_environments_do_not_share_constraints(target):
    """PEP 517 isolates build environments, so these pins do not conflict."""
    old = make_sdist("old", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["setuptools==59.2.0"]\n',
        "s.c": "int main(){}\n",
    })
    new = make_sdist("new", "1.0", {
        "pyproject.toml": '[build-system]\nrequires=["setuptools>=70"]\n',
        "s.c": "int main(){}\n",
    })
    index = FakeIndex(
        {
            "old": ["old-1.0.tar.gz"],
            "new": ["new-1.0.tar.gz"],
            "setuptools": [
                "setuptools-59.2.0-py3-none-any.whl",
                "setuptools-70.0-py3-none-any.whl",
            ],
        },
        {
            "setuptools-59.2.0-py3-none-any.whl": metadata(name="setuptools"),
            "setuptools-70.0-py3-none-any.whl": metadata(name="setuptools"),
        },
        {"old-1.0.tar.gz": old, "new-1.0.tar.gz": new},
    )
    result = analyze(target, index, "old", "new")
    assert result.packages["setuptools"].verdict is Verdict.PURE_PYTHON
    assert result.packages["old"].verdict is Verdict.NEEDS_BUILD
    assert result.packages["new"].verdict is Verdict.NEEDS_BUILD


def test_runtime_constraints_do_share(target):
    """One environment at runtime: an impossible pin really is impossible."""
    index = FakeIndex(
        {
            "a": ["a-1.0-py3-none-any.whl"],
            "b": ["b-1.0-py3-none-any.whl"],
            "shared": ["shared-1.0-py3-none-any.whl", "shared-2.0-py3-none-any.whl"],
        },
        {
            "a-1.0-py3-none-any.whl": metadata("shared==1.0", name="a"),
            "b-1.0-py3-none-any.whl": metadata("shared>=2.0", name="b"),
            "shared-1.0-py3-none-any.whl": metadata(name="shared"),
            "shared-2.0-py3-none-any.whl": metadata(name="shared"),
        },
    )
    result = analyze(target, index, "a", "b")
    assert result.packages["shared"].verdict is Verdict.UNRESOLVED


def test_requires_python_excludes_a_release(target):
    index = FakeIndex({"x": ["x-1.0-py3-none-any.whl"]},
                      {"x-1.0-py3-none-any.whl": metadata(name="x")})
    index._payloads["x"] = ["x-1.0-py3-none-any.whl"]
    project = index.project("x")
    for release in project.releases.values():
        for file in release.wheels:
            file.requires_python = ">=3.13"
    result = analyze(target, index, "x")
    assert result.packages["x"].verdict is Verdict.UNRESOLVED


def test_no_inspect_sdists_still_reports_a_build(target):
    index = FakeIndex({"x": ["x-1.0.tar.gz"]}, {}, {})
    result = analyze(target, index, "x", inspect_sdists=False)
    assert result.packages["x"].verdict is Verdict.NEEDS_BUILD
    assert index.downloads == []


def test_prereleases_are_excluded_by_default(target):
    index = FakeIndex(
        {"x": ["x-1.0-py3-none-any.whl", "x-2.0b1-py3-none-any.whl"]},
        {"x-1.0-py3-none-any.whl": metadata(name="x"),
         "x-2.0b1-py3-none-any.whl": metadata(name="x")},
    )
    assert analyze(target, index, "x").packages["x"].version == "1.0"
    assert analyze(target, index, "x", allow_prereleases=True).packages["x"].version == "2.0b1"
