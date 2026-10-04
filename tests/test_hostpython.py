"""The configure's interpreter: this host's, holding what the build installs."""

import subprocess
import sys
from pathlib import Path

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


def test_a_stubbed_module_says_where_it_is_and_which_it_is(tmp_path):
    """neuron's FindPythonModule prints numpy.__file__, then numpy.__version__."""
    python = HostPython.create(tmp_path, dists={})
    _run(python, "-c", "import wir_stubbed")
    python.provide()
    where = _run(python, "-c", "import re, wir_stubbed; "
                 "print(re.compile('/__init__.py.*').sub('', wir_stubbed.__file__))")
    assert where.returncode == 0 and where.stdout.strip().endswith("wir_stubbed")
    which = _run(python, "-c", "import wir_stubbed; print(wir_stubbed.__version__)")
    assert which.stdout.strip() == "99.9.9"


def test_a_script_and_a_module_run_with_their_own_directory_first(tmp_path):
    python = HostPython.create(tmp_path / "python")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "helper.py").write_text("VALUE = 7\n")
    (tmp_path / "tools" / "main.py").write_text("import helper\nprint(helper.VALUE)\n")
    assert _run(python, str(tmp_path / "tools" / "main.py")).stdout.strip() == "7"
    assert _run(python, "-m", "helper", cwd=tmp_path / "tools").returncode == 0


def test_a_module_run_with_dash_m_that_is_not_there_is_recorded(tmp_path):
    """matplotlib's version comes from `python -m setuptools_scm`."""
    python = HostPython.create(tmp_path)
    assert _run(python, "-m", "wir_scm").returncode == 1
    assert python.missing() == ["wir_scm"]


def test_a_stub_answers_what_a_configure_asks_it_by_name(tmp_path):
    """exago reads mpi4py's version and include directory, two lines; ascent
    asks distutils for the site directory; FindPython asks numpy its version."""
    python = HostPython.create(tmp_path, dists={})
    for module in ("wir_mpi", "wir_utils.sysconfig"):
        _run(python, "-c", f"import {module}")
    python.provide()
    two = _run(python, "-c", "import wir_mpi as m; print(m.__version__); print(m.get_include())")
    version, include = two.stdout.split()
    assert version == "99.9.9" and Path(include).is_dir()
    site = _run(python, "-c", "from wir_utils.sysconfig import get_python_lib, get_paths\n"
                "print(get_python_lib()); print(get_paths()['include'])").stdout.split()
    assert len(site) == 2 and all(Path(p).is_dir() for p in site)
    nested = _run(python, "-c", "import wir_mpi; print(wir_mpi.version.version)")
    assert nested.stdout.strip() == "99.9.9"


def test_a_stubbed_setup_says_the_version_it_was_given(tmp_path):
    """xrootd runs its setup.py --version for the version, setuptools stubbed."""
    python = HostPython.create(tmp_path, dists={})
    _run(python, "-c", "import wir_setuptools")
    python.provide()
    script = tmp_path / "setup.py"
    script.write_text(
        "from wir_setuptools import setup, Extension\n"
        "class CMakeExtension(Extension):\n"
        "    def __init__(self, name):\n"
        "        Extension.__init__(self, name, sources=[])\n"
        "setup(name='x', version='5.6.7', ext_modules=[CMakeExtension('x')])\n"
    )
    assert _run(python, str(script), "--version").stdout.strip() == "5.6.7"
