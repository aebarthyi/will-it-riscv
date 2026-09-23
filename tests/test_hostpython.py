"""The configure's interpreter: this host's, holding what the build installs."""

import subprocess
import sys

from will_it_riscv.hostpython import HostPython


def _run(python, *args, **kwargs):
    return subprocess.run(
        [str(python.executable), *args], capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=60, **kwargs,
    )


def test_it_answers_as_this_interpreter(tmp_path):
    python = HostPython.create(tmp_path)
    version = ".".join(str(p) for p in sys.version_info[:3])
    assert _run(python, "-V").stdout.strip() == f"Python {version}"
    out = _run(python, "-E", "-c", "import sys; print(sys.argv[1:])", "a", "b")
    assert out.stdout.strip() == "['a', 'b']"


def test_it_sees_the_standard_library_and_nothing_this_tool_runs_on(tmp_path):
    """rich is will-it-riscv's; a configure's interpreter must not see it."""
    python = HostPython.create(tmp_path)
    assert _run(python, "-c", "import sysconfig, json").returncode == 0
    assert _run(python, "-c", "import rich").returncode != 0


def test_a_fatal_import_is_recorded_and_a_caught_one_is_not(tmp_path):
    python = HostPython.create(tmp_path)
    _run(python, "-c", "try:\n    import wir_optional\nexcept ImportError:\n    pass")
    assert python.missing() == []
    _run(python, "-c", "import wir_needed.sub")
    assert python.missing() == ["wir_needed"]


def test_a_module_nothing_installs_is_stubbed(tmp_path):
    python = HostPython.create(tmp_path, dists={"numpy": "2.5.3"})
    _run(python, "-c", "import wir_needed")
    assert python.provide() == ["wir_needed"]
    assert python.not_in_plan == ["wir_needed"]
    assert _run(python, "-c", "import wir_needed; print(wir_needed.get_include())").returncode == 0
    assert python.provide() == []   # dealt with once


def test_a_script_and_a_module_run_with_their_own_directory_first(tmp_path):
    python = HostPython.create(tmp_path / "python")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "helper.py").write_text("VALUE = 7\n")
    (tmp_path / "tools" / "main.py").write_text("import helper\nprint(helper.VALUE)\n")
    assert _run(python, str(tmp_path / "tools" / "main.py")).stdout.strip() == "7"
    assert _run(python, "-m", "helper", cwd=tmp_path / "tools").returncode == 0
