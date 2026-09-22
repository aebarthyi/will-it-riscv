import json

import pytest

from will_it_riscv import cli
from will_it_riscv.models import Analysis, PackageReport, SystemRequirement, Verdict
from will_it_riscv.report import render_json, render_list, render_markdown


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.target == "riscv64"
    assert args.path == "."
    assert args.format == "text"
    assert not args.no_distro


def test_python_version_parsing():
    assert cli._python_version("3.11") == (3, 11)
    assert cli._python_version("3") == (3, 0)
    with pytest.raises(SystemExit):
        cli._python_version("nonsense")


def test_bad_package_spec_exits():
    args = cli.build_parser().parse_args(["-p", "=== nope ==="])
    with pytest.raises(SystemExit):
        cli._roots(args)


def test_package_specs_bypass_the_filesystem():
    args = cli.build_parser().parse_args(["-p", "numpy>=2", "-p", "pandas"])
    roots = cli._roots(args)
    assert sorted(r.name for r in roots.runtime) == ["numpy", "pandas"]


def test_missing_path_exits():
    args = cli.build_parser().parse_args(["/definitely/not/here/pyproject.toml"])
    with pytest.raises(SystemExit):
        cli._roots(args)


@pytest.fixture
def analysis():
    a = Analysis(target="riscv64 linux", root="demo", python_version="3.12")
    a.packages["pure"] = PackageReport(name="pure", version="1.0",
                                       verdict=Verdict.PURE_PYTHON)
    a.packages["wheeled"] = PackageReport(name="wheeled", version="2.0",
                                          verdict=Verdict.WHEEL_AVAILABLE)
    built = PackageReport(name="built", version="3.0", verdict=Verdict.NEEDS_BUILD)
    built.build.languages = {"c"}
    built.build.system_requirements = [
        SystemRequirement(name="openssl", debian=("libssl-dev",), found_in=("s.c",))
    ]
    built.reasons = ["publishes binary wheels, but none for riscv64"]
    a.packages["built"] = built
    a.system_requirements["openssl"] = SystemRequirement(
        name="openssl", debian=("libssl-dev",), found_in=("built",)
    )
    return a


def test_exit_codes(analysis):
    assert analysis.exit_code() == 1
    analysis.packages["built"].verdict = Verdict.WHEEL_AVAILABLE
    assert analysis.exit_code() == 0
    analysis.packages["built"].verdict = Verdict.NO_DISTRIBUTION
    assert analysis.exit_code() == 2


def test_json_round_trips(analysis):
    payload = json.loads(render_json(analysis))
    assert payload["summary"] == {"pure-python": 1, "wheel-available": 1, "needs-build": 1}
    assert payload["exit_code"] == 1
    built = next(p for p in payload["packages"] if p["name"] == "built")
    assert built["verdict"] == "needs-build"
    assert built["pure"] is False
    assert built["build"]["system_requirements"][0]["debian"] == ["libssl-dev"]
    assert payload["system_requirements"][0]["required_by"] == ["built"]


def test_json_orders_worst_first(analysis):
    payload = json.loads(render_json(analysis))
    assert payload["packages"][0]["name"] == "built"


def test_list_format_omits_pure_python(analysis):
    assert render_list(analysis).splitlines() == ["built==3.0", "wheeled==2.0"]


def test_markdown_has_the_apt_line(analysis):
    text = render_markdown(analysis)
    assert "sudo apt install libssl-dev" in text
    assert "| `built` |" in text
    assert "# will-it-riscv: demo" in text
