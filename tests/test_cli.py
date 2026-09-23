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


# -- the pseudobuild's answer, in every format -------------------------------


@pytest.fixture
def configured(analysis):
    from will_it_riscv.pseudobuild import Probe, PseudoBuild

    analysis.pseudobuild = PseudoBuild(
        completed=True,
        platform="linux/riscv64",
        rounds=2,
        blockers=["PROJ"],
        round_blockers=["PROJ", None],
        probes={
            "proj": Probe(name="PROJ", command="find_package", site="CMakeLists.txt:9"),
            "geos": Probe(name="GEOS", command="find_package", round=2),
        },
        soft_misses={"GEOS"},
        host_gaps=["linux/fs.h"],
    )
    return analysis


def test_the_text_report_answers_the_question(configured):
    from io import StringIO

    from rich.console import Console

    from will_it_riscv.report import render_text

    out = StringIO()
    render_text(configured, Console(file=out, width=160))
    text = out.getvalue()
    assert "Will it riscv?" in text
    assert "PROJ" in text and "CMakeLists.txt:9" in text
    assert "rounds: PROJ → completed" in text
    assert "linux/fs.h" in text


def test_json_carries_the_graph(configured):
    payload = json.loads(render_json(configured))
    pseudo = payload["pseudobuild"]
    assert pseudo["platform"] == "linux/riscv64"
    assert pseudo["graph"]["hard_requirements"] == ["PROJ"]
    assert pseudo["graph"]["answer"]["verdict"] == "probably"   # no archive checked


def test_dot_format_is_offered_and_renders(configured):
    from will_it_riscv.report import render_dot

    assert cli.build_parser().parse_args(["-f", "dot"]).format == "dot"
    assert render_dot(configured).startswith('digraph "demo"')


def test_dot_without_a_pseudobuild_draws_what_static_reading_found(analysis):
    from will_it_riscv.report import render_dot

    analysis.project_requirements["zlib"] = SystemRequirement(
        name="zlib", debian=("zlib1g-dev",), found_in=("CMakeLists.txt",)
    )
    dot = render_dot(analysis)
    assert '"__project__" -> "zlib"' in dot


def test_markdown_leads_with_the_answer(configured):
    text = render_markdown(configured)
    assert "## Will it riscv? **probably**" in text
    assert "| 1 | `PROJ` |" in text
