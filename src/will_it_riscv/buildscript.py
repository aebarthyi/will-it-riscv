"""A build script from a plan: the build itself, on the target, start to finish.

The plan was written for the pretend environment, which configures and never
compiles. But every step in it is typed -- install these packages, configure
this directory with these flags, run this driver -- so the same steps, each
followed by the compile and install its build system does after a configure,
are the build. The script is written from the plan the run held up, by this
code, not by a model; and every value the plan carries (a path, a flag, a
package name) goes into it quoted, so no plan can put a command in it.

It runs on the target machine itself, from the repository's checkout:

  1. checks the machine and the distro are the ones planned for;
  2. installs everything the build needs from the archive, in one go;
  3. installs Rust with rustup, where the archive's is too old;
  4. makes a virtual environment for the Python steps -- Debian and Ubuntu
     refuse pip installs into the system's Python;
  5. builds what the target has only as source, in the order the recursion
     found: dependencies first;
  6. runs the plan's steps in order. pip installs; CMake and Meson
     configures, each built and installed into $PREFIX; the project's own
     driver, which runs the configures it was seen to run, so the script
     leaves those to it -- as it leaves to pip a configure that the build
     backend of a package it installs runs.

Optional steps run only when asked for (``WITH_<STEP>=1``), and
``DRY_RUN=1`` prints every command and runs none.
"""

from __future__ import annotations

import re
import shlex
from datetime import date
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Optional

from . import __version__, planrun
from .autoplan import tomllib
from .plan import CMAKE_CONFIGURE, MESON_SETUP, PYTHON_INSTALL, PYTHON_RUN, SYSTEM_PACKAGES
from .upstream import UPSTREAM

if TYPE_CHECKING:  # pragma: no cover
    from .plan import Step
    from .planrun import PlanResult
    from .recurse import Recursion

#: Build backends that run a configure of their own: pip's build of the
#: package runs the CMake or Meson step the plan lists for it.
NATIVE_BACKENDS = {
    "mesonpy", "scikit_build_core.build", "py_build_cmake.build", "cmeel.build",
    "scikit_build_core.setuptools.build_meta",
}
_REQUIREMENTS = re.compile(r"(requirements|constraints)[^/]*\.(txt|in)$|\.txt$", re.I)


def _q(value: str) -> str:
    return shlex.quote(str(value))


def _under_src(path: str) -> str:
    """A path in the repository, for the script: ``"$SRC"/path``, quoted."""
    clean = str(PurePosixPath(path or "."))
    return '"$SRC"' if clean in (".", "") else f'"$SRC"/{_q(clean)}'


def switch(step_id: str) -> str:
    """The variable that turns an optional step on: ``WITH_SIMULATION_GPU_ACC``."""
    return "WITH_" + re.sub(r"[^A-Za-z0-9]+", "_", step_id).strip("_").upper()


def driven(result: PlanResult) -> set:
    """Configures the project's own driver was seen to run: left to it."""
    return {step_id for outcome in result.steps for step_id in outcome.driven}


def backend_owned(result: PlanResult) -> dict[str, str]:
    """Configures that pip's build of a package the plan installs runs itself,
    with the backend that runs them."""
    owned: dict[str, str] = {}
    for step in result.plan.steps:
        if step.kind != PYTHON_INSTALL or not step.manifest or step.section != "dependencies":
            continue
        directory = PurePosixPath(step.manifest).parent
        backend = _backend(result.root / str(directory))
        if backend is None:
            continue
        for other in result.plan.steps:
            if other.kind not in (CMAKE_CONFIGURE, MESON_SETUP):
                continue
            source = PurePosixPath(other.source or ".")
            if source == directory or directory in source.parents or str(directory) == ".":
                owned[other.id] = backend
    return owned


def _backend(directory: Path) -> Optional[str]:
    """The package's build backend, when it is one that configures a build."""
    pyproject = directory / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
        except (tomllib.TOMLDecodeError, OSError):
            data = {}
        backend = (data.get("build-system") or {}).get("build-backend")
        if backend in NATIVE_BACKENDS:
            return backend
    setup = directory / "setup.py"
    if setup.is_file() and re.search(r"\bcmake\b|\bmeson\b",
                                     setup.read_text(encoding="utf-8", errors="replace")):
        return "setup.py"
    return None


def build_script(
    result: PlanResult, walk: Optional[Recursion] = None, *, distro: str = "ubuntu:resolute",
    name: str = "build-riscv.sh",
) -> str:
    """The script, as text; ``name`` is what it will be called."""
    plan = result.plan
    answer = result.answer
    steps = plan.ordered()
    by_driver = driven(result)
    by_backend = backend_owned(result)

    def runs(step: Step) -> bool:
        return step.id not in by_driver and step.id not in by_backend

    builds = [s for s in steps if runs(s)]
    python = any(s.kind in (PYTHON_INSTALL, PYTHON_RUN) for s in builds) or any(
        result.nodes[n].ecosystem == "pypi" for n in (answer.from_source if answer else [])
    )
    meson_steps = [s for s in builds if s.kind == MESON_SETUP]

    # -- what to install from the archive, and why
    needs = list(answer.install) if answer else []
    tools = ["build-essential"]
    if any(s.kind == CMAKE_CONFIGURE for s in builds):
        tools += ["cmake", "pkg-config"]
    if meson_steps:
        tools += ["ninja-build", "pkg-config"]
        if any(not s.meson for s in meson_steps):
            tools.append("meson")
    if python:
        tools += ["python3-venv", "python3-dev"]
    sources = _from_source(result, walk)
    if any(kind == "git" for _, kind, _, _ in sources):
        tools.append("git")
    rust = [n for n in result.nodes.values()
            if n.required and n.tier == planrun.BINARY and "rustup ships" in n.detail]
    if rust:
        tools.append("curl")
    for_sources: list[tuple[str, list[str]]] = []
    for source, _, _, tree in sources:
        if tree is not None and tree.result is not None:
            archive = sorted({d.package for d in tree.result.nodes.values()
                              if d.required and d.tier == planrun.BINARY and d.package})
            if archive:
                for_sources.append((source, archive))
    listed: set = set()

    def fresh(names: list[str]) -> list[str]:
        out = [n for n in dict.fromkeys(names) if n not in listed]
        listed.update(out)
        return out

    package_lines: list[str] = []
    for comment, names in (
        ("what the build needs (will-it-riscv's answer)", needs),
        ("what this script runs", tools),
        *((f"what building {source} from source needs", names)
          for source, names in for_sources),
    ):
        names = fresh(names)
        if names:
            package_lines.append(f"  # {comment}")
            package_lines.append("  " + " ".join(_q(n) for n in names))

    # -- the header
    verdict = (walk.verdict if walk is not None else answer.verdict if answer else "unknown")
    headline = (walk.headline if walk is not None else answer.headline if answer else "")
    optional = [s for s in steps if s.optional and runs(s)]
    arch = result.arch
    os_id, _, codename = distro.partition(":")
    out: list[str] = [
        "#!/usr/bin/env bash",
        f"# Build {plan.repo} on {arch} ({distro}), as will-it-riscv planned it.",
        f"# Written by will-it-riscv {__version__} on {date.today().isoformat()} from the steps",
        "# of the plan its run held up. Run it on the target, from the repository's checkout:",
        "#",
        f"#   bash {name}",
        "#",
        f"# will-it-riscv's answer then: {verdict.upper()} — {headline}",
    ]
    if plan.entry:
        out.append(f"# the project's own instructions: {_entry(plan.entry)}")
    if answer and answer.blockers:
        out.append("# nothing public for the target, which the build requires: "
                   + ", ".join(result.nodes[n].name for n in answer.blockers))
    out += [
        "#",
        "# Settings, from the environment:",
        "#   SRC=dir        the checkout (default: the current directory)",
        "#   WORK=dir       build directories (default: $SRC/_riscv)",
        "#   PREFIX=dir     where everything is installed (default: $WORK/install)",
        "#   JOBS=n         parallel jobs (default: nproc)",
        "#   SKIP_APT=1     the system packages are installed already",
        "#   DRY_RUN=1      print every command, run none",
        f"#   ANY_ARCH=1     run on a machine that is not {arch}",
    ]
    for step in optional:
        out.append(f"#   {switch(step.id)}=1  also {step.id} ({step.enabled_by or 'optional'})")
    out += [
        "set -euo pipefail",
        "",
        'SRC=${SRC:-$PWD}',
        'WORK=${WORK:-$SRC/_riscv}',
        'PREFIX=${PREFIX:-$WORK/install}',
        'JOBS=${JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}',
        'VENV=${VENV:-$WORK/venv}',
        "",
        "run() {",
        "  printf '+' >&2; printf ' %q' \"$@\" >&2; printf '\\n' >&2",
        '  if [ -z "${DRY_RUN:-}" ]; then "$@"; fi',
        "}",
        "run_in() {",
        "  local dir=$1; shift",
        "  printf '+ (cd %q &&' \"$dir\" >&2; printf ' %q' \"$@\" >&2; printf ')\\n' >&2",
        '  if [ -z "${DRY_RUN:-}" ]; then (cd "$dir" && "$@"); fi',
        "}",
        "say() { printf '\\n== %s\\n' \"$*\" >&2; }",
        "",
        f'if [ "$(uname -m)" != {_q(arch)} ] && [ -z "${{ANY_ARCH:-}}" ]; then',
        f'  echo "this builds {plan.repo} for {arch}, and this machine is $(uname -m)'
        ' (ANY_ARCH=1 runs it anyway)" >&2',
        "  exit 1",
        "fi",
        "here=$(. /etc/os-release 2>/dev/null && echo \"${ID:-}:${VERSION_CODENAME:-}\" || true)",
        f'if [ "$here" != {_q(f"{os_id}:{codename}")} ]; then',
        f'  echo "note: planned for {distro}, and this is ${{here:-an unknown system}}:'
        ' package names may differ" >&2',
        "fi",
    ]
    marker = _marker(builds)
    if marker:
        out += [
            f"if [ ! -e {_under_src(marker)} ]; then",
            f'  echo "$SRC has no {marker}: run this from {plan.repo}\'s checkout, or set SRC"'
            " >&2",
            "  exit 1",
            "fi",
        ]
    if verdict in ("no", "unknown"):
        out.append(f'echo "will-it-riscv expected this build to stop: {verdict}" >&2')
    out += [
        'mkdir -p "$WORK" "$PREFIX"',
        'multiarch=$(gcc -print-multiarch 2>/dev/null || true)',
        'export PATH="$PREFIX/bin:$PATH"',
        'export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib/$multiarch/pkgconfig:'
        '$PREFIX/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"',
        'export CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"',
        'export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/$multiarch'
        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        "",
        'SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo',
        "packages=(",
        *package_lines,
        ")",
        'if [ -z "${SKIP_APT:-}" ]; then',
        '  say "system packages: ${#packages[@]}"',
        "  run $SUDO apt-get update",
        "  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "
        '--no-install-recommends "${packages[@]}"',
        "fi",
    ]

    if rust:
        wanted = max((n.wants for n in rust if n.wants), default=None)
        out += [
            "",
            f"# {rust[0].name}{' >= ' + wanted if wanted else ''}: {rust[0].detail}",
            'say "Rust, from rustup"',
            'if [ -x "$HOME/.cargo/bin/rustup" ]; then',
            '  run "$HOME/.cargo/bin/rustup" update stable',
            "else",
            "  run sh -c 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs"
            " | sh -s -- -y --profile minimal'",
            "fi",
            'export PATH="$HOME/.cargo/bin:$PATH"',
        ]

    if python:
        out += [
            "",
            'say "Python: a virtual environment in $VENV"',
            'run python3 -m venv "$VENV"',
            'PY="$VENV/bin/python"',
            'run "$PY" -m pip install --upgrade pip',
        ]

    if sources:
        out += ["", f"# Built from source first, dependencies first ({len(sources)})."]
        for source, _, line, tree in sources:
            status = f" — {tree.status}: {tree.reason}" if tree is not None else ""
            out.append(f"say {_q(f'from source: {source}')}"
                       + (f"  # {status[3:]}" if status else ""))
            out.extend(line)

    for step in steps:
        lines = _step(step, result, by_driver, by_backend)
        if not lines:
            continue
        out.append("")
        if step.optional and runs(step):
            out.append(f'if [ -n "${{{switch(step.id)}:-}}" ]; then')
            out.extend("  " + line for line in lines)
            out.append("fi")
        else:
            out.extend(lines)

    out += [
        "",
        f'say "done: {plan.repo}"',
        'echo "installed into $PREFIX; to use it:" >&2',
        'echo "  export PATH=\\"$PREFIX/bin:\\$PATH\\"" >&2',
    ]
    if python:
        out.append('echo "  source \\"$VENV/bin/activate\\"" >&2')
    return "\n".join(out) + "\n"


def _entry(entry) -> str:
    if isinstance(entry, dict):
        return str(entry.get("command", ""))
    return str(entry)


def _marker(builds: list[Step]) -> Optional[str]:
    """A file the checkout must have: the first one a step reads."""
    for step in builds:
        if step.optional:
            continue
        if step.kind == CMAKE_CONFIGURE:
            return str(PurePosixPath(step.source or ".") / "CMakeLists.txt")
        if step.kind == MESON_SETUP:
            return str(PurePosixPath(step.source or ".") / "meson.build")
        if step.kind == PYTHON_INSTALL and step.manifest:
            return step.manifest
        if step.kind == PYTHON_RUN and step.script:
            return step.script
    return None


def _from_source(result: PlanResult, walk: Optional[Recursion]) -> list[tuple]:
    """``(name, kind, lines, tree)`` for each package built from source, in order."""
    entries: list[tuple] = []
    if walk is not None:
        for key in walk.order:
            tree = walk.nodes[key]
            kind = tree.source.kind if tree.source is not None else ""
            origin = tree.source.origin if tree.source is not None else ""
            entries.append((tree.name, kind, _source_lines(
                tree.name, tree.version, tree.ecosystem, kind, origin), tree))
        return entries
    for node_id in (result.answer.from_source if result.answer else []):
        node = result.nodes[node_id]
        entries.append((node.name, "", _source_lines(
            node.name, node.version or node.wants, node.ecosystem, "", ""), None))
    return entries


def _source_lines(name: str, version: Optional[str], ecosystem: str, kind: str,
                  origin: str) -> list[str]:
    if ecosystem == "pypi":
        if kind == "git" and "@" in origin:
            url, _, tag = origin.rpartition("@")
            spec = f"{name} @ git+{url}@{tag}"
        else:
            spec = f"{name}=={version}" if version else name
        # No wheel for the target: pip builds it from its source.
        return [f'run "$PY" -m pip install {_q(spec)}']
    wanted = f"{name} {version}" if version else name
    where = f": {origin}" if origin else ""
    # The distro package names what it installs: bazel-bootstrap is bazel.
    tool = UPSTREAM.get(name)
    command = tool.name if tool is not None else name
    return [
        f"if ! command -v {_q(command)} >/dev/null 2>&1; then",
        f"  echo {_q(f'build {wanted} from its source first{where}; will-it-riscv read its build but has no recipe to run it')} >&2",  # noqa: E501
        '  [ -n "${DRY_RUN:-}" ] || exit 1',
        "fi",
    ]


def _step(step: Step, result: PlanResult, by_driver: set, by_backend: dict) -> list[str]:
    """What one step of the plan becomes."""
    title = f"say {_q(f'{step.id}: {step.kind}')}"
    if step.id in by_driver:
        return [f"# {step.id}: the project's driver runs this configure itself"]
    if step.id in by_backend:
        return [f"# {step.id}: pip's build ({by_backend[step.id]}) runs this configure itself"]
    if step.kind == SYSTEM_PACKAGES:
        names = [result.nodes[f"debian:{_bare(p)}"].package for p in step.packages
                 if f"debian:{_bare(p)}" in result.nodes]
        names = [n for n in names if n]
        if not step.optional:
            return [f"# {step.id}: its packages are in the install above"]
        if not names:
            return [f"# {step.id}: none of its packages is in the archive"]
        return [title, 'if [ -z "${SKIP_APT:-}" ]; then',
                "  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "--no-install-recommends " + " ".join(_q(n) for n in names),
                "fi"]
    if step.kind == PYTHON_INSTALL:
        return [title, *_pip(step, result)]
    if step.kind == CMAKE_CONFIGURE:
        build = f'"$WORK"/build/{_q(step.id)}'
        defines = dict(step.defines)
        defines.pop("CMAKE_INSTALL_PREFIX", None)
        flags = [f"-D{k}={v}" for k, v in defines.items()]
        first = ["-DCMAKE_BUILD_TYPE=Release"] if "CMAKE_BUILD_TYPE" not in defines else []
        return [
            title,
            f"run cmake -S {_under_src(step.source)} -B {build}"
            + "".join(" " + _q(f) for f in first)
            + ' -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX"'
            + "".join(" " + _q(f) for f in flags),
            f'run cmake --build {build} --parallel "$JOBS"',
            f"run cmake --install {build}",
        ]
    if step.kind == MESON_SETUP:
        build = f'"$WORK"/build/{_q(step.id)}'
        meson = (f'"$PY" {_under_src(str(PurePosixPath(step.source or ".") / step.meson))}'
                 if step.meson else "meson")
        options = [f"-D{k}={v}" for k, v in step.defines.items()]
        buildtype = [] if "buildtype" in step.defines else ["--buildtype=release"]
        return [
            title,
            f"run {meson} setup {build} {_under_src(step.source)} "
            '--prefix "$PREFIX"' + "".join(" " + _q(o) for o in buildtype + options),
            f"run {meson} compile -C {build}",
            f"run {meson} install -C {build}",
        ]
    if step.kind == PYTHON_RUN:
        return [title, f'run_in "$SRC" "$PY" {_driver_args(step)}']
    return [f"# {step.id}: {step.kind} is not something this script runs"]


def _driver_args(step: Step) -> str:
    """The driver's command line; a job count the plan fixed (``-j 1``, what
    suited the pretend run) becomes the machine's, $JOBS."""
    words = [_q(a) for a in [step.script or "", *step.args]]
    for i, arg in enumerate(step.args, start=1):
        if arg in ("-j", "--jobs") and i < len(step.args) and step.args[i].isdigit():
            words[i + 1] = '"$JOBS"'
        elif re.fullmatch(r"(-j|--jobs=)\d+", arg):
            words[i] = f'{arg.rstrip("0123456789")}"$JOBS"'
    return " ".join(words)


def _bare(spec: str) -> str:
    """``cmake>=3.20`` -> ``cmake``."""
    return re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]


def _pip(step: Step, result: PlanResult) -> list[str]:
    pip = 'run "$PY" -m pip install'
    if step.manifest is None:
        return [pip + "".join(" " + _q(p) for p in step.packages)] if step.packages else []
    manifest = PurePosixPath(step.manifest)
    if _REQUIREMENTS.search(manifest.name):
        return [f"{pip} -r {_under_src(str(manifest))}"]
    directory = str(manifest.parent)
    if step.section == "build-system":
        from .inputs import load

        try:
            build = [str(r) for r in load(result.root / str(manifest)).build]
        except (OSError, ValueError):
            build = []
        if not build:
            return [f"# {step.id}: {manifest} declares no build requirements"]
        return [pip + "".join(" " + _q(r) for r in build)]
    extras = f"[{','.join(step.extras)}]" if step.extras else ""
    if extras:
        return [f'{pip} {_under_src(directory)}{_q(extras)}']
    return [f"{pip} {_under_src(directory)}"]
