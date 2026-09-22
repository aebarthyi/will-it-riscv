import pytest

from will_it_riscv.inputs import _poetry_constraint, load


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def names(reqs):
    return sorted(r.name for r in reqs)


def test_pep621_dependencies(tmp_path):
    write(tmp_path, "pyproject.toml", """
[build-system]
requires = ["setuptools>=68", "wheel"]

[project]
name = "demo"
requires-python = ">=3.10"
dependencies = ["requests>=2", "click"]

[project.optional-dependencies]
fast = ["orjson"]
dev = ["pytest"]
""")
    result = load(tmp_path / "pyproject.toml")
    assert names(result.runtime) == ["click", "requests"]
    assert names(result.build) == ["setuptools", "wheel"]
    assert result.project_name == "demo"
    assert result.requires_python == ">=3.10"


def test_extras_are_opt_in(tmp_path):
    path = write(tmp_path, "pyproject.toml", """
[project]
name = "demo"
dependencies = ["click"]
[project.optional-dependencies]
fast = ["orjson"]
dev = ["pytest"]
""")
    assert names(load(path, extras=("fast",)).runtime) == ["click", "orjson"]
    assert names(load(path, extras=("all",)).runtime) == ["click", "orjson", "pytest"]


def test_dependency_groups(tmp_path):
    path = write(tmp_path, "pyproject.toml", """
[project]
name = "demo"
dependencies = ["click"]
[dependency-groups]
test = ["pytest", "coverage"]
""")
    assert names(load(path).runtime) == ["click"]
    assert names(load(path, groups=("test",)).runtime) == ["click", "coverage", "pytest"]


def test_directory_finds_pyproject(tmp_path):
    write(tmp_path, "pyproject.toml", '[project]\nname="d"\ndependencies=["click"]\n')
    assert names(load(tmp_path).runtime) == ["click"]


def test_missing_pyproject(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_requirements_txt(tmp_path):
    path = write(tmp_path, "requirements.txt", """
# a comment
requests==2.31.0
click>=8  # trailing comment

--index-url https://example.invalid/simple
-r other.txt
./local-package
https://example.invalid/pkg.whl
numpy ; python_version >= "3.9"
""")
    result = load(path)
    assert names(result.runtime) == ["click", "numpy", "requests"]
    assert any("nested -r" in w for w in result.warnings)
    assert any("direct reference" in w for w in result.warnings)


def test_unparseable_requirement_warns_but_does_not_raise(tmp_path):
    path = write(tmp_path, "pyproject.toml",
                 '[project]\nname="d"\ndependencies=["click", "=== broken ==="]\n')
    result = load(path)
    assert names(result.runtime) == ["click"]
    assert any("unparseable" in w for w in result.warnings)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("^1.2.3", ">=1.2.3,<2.0.0"),
        ("^0.3.1", ">=0.3.1,<0.4.0"),
        ("^0.0.4", ">=0.0.4,<0.0.5"),
        ("~1.2.3", ">=1.2.3,<1.3.0"),
        ("~1", ">=1.0.0,<2.0.0"),
        ("*", ""),
        (">=1.0,<2.0", ">=1.0,<2.0"),
    ],
)
def test_poetry_constraint_translation(raw, expected):
    assert _poetry_constraint(raw) == expected


def test_poetry_dependencies(tmp_path):
    path = write(tmp_path, "pyproject.toml", """
[tool.poetry]
name = "poetry-demo"

[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.31"
numpy = { version = "^1.26" }
extra-thing = { version = "*", optional = true }
""")
    result = load(path)
    assert "requests" in names(result.runtime)
    assert "numpy" in names(result.runtime)
    assert "extra-thing" not in names(result.runtime)
    assert result.requires_python == ">=3.10.0,<4.0.0"
    assert any("Poetry" in w for w in result.warnings)


def test_empty_project_is_falsy(tmp_path):
    path = write(tmp_path, "pyproject.toml", '[project]\nname="d"\n')
    assert not load(path)
