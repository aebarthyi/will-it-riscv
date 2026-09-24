"""Reading a Bazel build for what it downloads prebuilt, and for which platforms.

Bazel is not run in the pretend environment. Analysing jaxlib's build would
mean fetching XLA and LLVM -- gigabytes -- to answer questions its
MODULE.bazel already answers. Everything a Bazel build fetches as source,
Bazel builds, for any platform with a C++ compiler. What can be missing on
riscv64 is what it downloads *prebuilt*:

  hermetic Python   python.toolchain(), from python-build-standalone. Is there
                    a CPython at that version for riscv64-unknown-linux-gnu?
                    Read from rules_python's own manifest, at the version the
                    build pins -- a sparse checkout of that tag.
  hermetic C++      register_toolchains() for linux_x86_64 and linux_aarch64
                    means no C++ toolchain for riscv64 -- unless a .bazelrc
                    config builds with the machine's own compiler.
  Python wheels     pip.parse(download_only = True) fetches wheels for the
                    target_platforms it lists. local_wheels names those that
                    dist/ can supply instead: a riscv64 build has to build
                    them first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.utils import canonicalize_name

from .plan import Evidence

#: python-build-standalone's name for each target.
PBS_TRIPLES = {
    "riscv64": "riscv64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "x86_64": "x86_64-unknown-linux-gnu",
    "ppc64le": "ppc64le-unknown-linux-gnu",
    "s390x": "s390x-unknown-linux-gnu",
}

_ARCHES = r"x86_64|aarch64|arm64|riscv64|ppc64le|s390x|armv7"
_ACCELERATOR = re.compile(r"cuda|rocm|tpu|pjrt|sycl|oneapi", re.IGNORECASE)


@dataclass
class Reading:
    read: list[str] = field(default_factory=list)
    """One sentence per thing found, for the report."""
    blocked: list[str] = field(default_factory=list)
    wheels: list[str] = field(default_factory=list)
    """The wheels dist/ must supply: numpy, scipy, ml-dtypes."""
    wheel_evidence: list[Evidence] = field(default_factory=list)
    compiler: Optional[str] = None
    """The distro package a local C++ toolchain needs, when one is needed."""
    compiler_evidence: list[Evidence] = field(default_factory=list)


def read(
    tree: Path, name: str, arch: str = "riscv64", cache_root: Optional[Path] = None
) -> Optional[Reading]:
    """What a Bazel build downloads prebuilt, and whether ``arch`` has it."""
    module = Path(tree) / "MODULE.bazel"
    try:
        text = module.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    reading = Reading()
    python = _default_python(text)
    _hermetic_python(text, python, arch, cache_root, reading)
    _hermetic_cc(Path(tree), text, arch, reading)
    _wheels(Path(tree), text, name, arch, reading)
    return reading


# ------------------------------------------------------------------ helpers


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _cite(text: str, match: re.Match, path: str = "MODULE.bazel") -> Evidence:
    line = _line(text, match.start())
    return Evidence(path, line, line, match.group(0).strip()[:80])


def _arguments(text: str, start: int) -> str:
    depth = 1
    for index in range(start, min(len(text), start + 20000)):
        if text[index] in "([{":
            depth += 1
        elif text[index] in ")]}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return text[start:start + 20000]


def _default_python(text: str) -> Optional[str]:
    """The Python version the build defaults to: 3.12, for jax."""
    defaults = re.search(r"python\.defaults\((.*?)\)", text, re.DOTALL)
    value = None
    if defaults:
        match = re.search(r"python_version\s*=\s*(\"[^\"]+\"|[A-Z_][A-Z0-9_]*)", defaults.group(1))
        value = match.group(1) if match else None
    if value is None:
        default = re.search(
            r"python\.toolchain\([^)]*?python_version\s*=\s*\"([^\"]+)\"[^)]*is_default\s*=\s*True",
            text, re.DOTALL,
        )
        first = re.search(r"python\.toolchain\([^)]*?python_version\s*=\s*\"([^\"]+)\"", text)
        return default.group(1) if default else first.group(1) if first else None
    if value.startswith('"'):
        return value.strip('"')
    constant = re.search(rf"^{re.escape(value)}\s*=\s*\"([^\"]+)\"", text, re.MULTILINE)
    return constant.group(1) if constant else None


# --------------------------------------------------------- hermetic Python


def _hermetic_python(
    text: str, python: Optional[str], arch: str, cache_root: Optional[Path], reading: Reading
) -> None:
    toolchain = re.search(r"python\.toolchain\(", text)
    dep = re.search(r'bazel_dep\(\s*name\s*=\s*"rules_python"\s*,\s*version\s*=\s*"([^"]+)"', text)
    if toolchain is None or python is None:
        return
    triple = PBS_TRIPLES.get(arch)
    if dep is None or triple is None or cache_root is None:
        reading.read.append(
            f"hermetic Python {python} comes from rules_python, whose builds for {arch} "
            "were not checked"
        )
        return
    version = dep.group(1)
    from .upstream import _sparse_clone

    destination = cache_root / f"rules_python-{version}"
    if not (destination / ".unpacked").exists():
        error = _sparse_clone(
            "https://github.com/bazel-contrib/rules_python", version, destination,
            ("/python/versions.bzl", "/python/private/runtimes_manifest_workspace.bzl"),
        )
        if error:
            reading.read.append(
                f"hermetic Python {python} comes from rules_python {version}, which "
                f"could not be read: {error}"
            )
            return
    full = _full_version(text, destination, python)
    if _has_build(destination, full, triple):
        reading.read.append(
            f"hermetic Python: rules_python {version} has CPython {full} for {triple} ✓"
        )
        return
    reading.blocked.append(
        f"its hermetic Python {full} comes from rules_python {version}, which has no "
        f"build of it for {triple}"
    )
    reading.read.append(reading.blocked[-1])


def _full_version(text: str, rules_python: Path, python: str) -> str:
    """3.12 is 3.12.13 to rules_python: its MINOR_MAPPING, or the build's own override."""
    override = re.search(rf'"{re.escape(python)}"\s*:\s*"({re.escape(python)}[^"]*)"', text)
    if override:
        return override.group(1)
    try:
        versions = (rules_python / "python" / "versions.bzl").read_text(errors="replace")
    except OSError:
        return python
    mapped = re.search(rf'"{re.escape(python)}"\s*:\s*"({re.escape(python)}[^"]*)"', versions)
    return mapped.group(1) if mapped else python


def _has_build(rules_python: Path, full: str, triple: str) -> bool:
    texts = []
    for relative in ("python/private/runtimes_manifest_workspace.bzl", "python/versions.bzl"):
        try:
            texts.append((rules_python / relative).read_text(errors="replace"))
        except OSError:
            continue
    combined = "\n".join(texts)
    # The manifest: "<sha>  20260414/cpython-3.12.13+20260414-riscv64-unknown-linux-gnu-..."
    if re.search(rf"cpython-{re.escape(full)}\+\d+-{re.escape(triple)}-", combined):
        return True
    # Older rules_python: TOOL_VERSIONS = {"3.12.4": {"sha256": {"<triple>": ...}}}
    block = re.search(rf'"{re.escape(full)}"\s*:\s*\{{', combined)
    return block is not None and f'"{triple}"' in _arguments(combined, block.end())


# ------------------------------------------------------------- hermetic C++


def _hermetic_cc(tree: Path, text: str, arch: str, reading: Reading) -> None:
    registered = list(re.finditer(r'register_toolchains\(\s*"([^"]+)"', text))
    linux: dict[str, re.Match] = {}
    for match in registered:
        for host in re.findall(rf"linux_({_ARCHES})", match.group(1)):
            linux.setdefault(host, match)
    if not linux or arch in linux:
        return
    arches = " and ".join(sorted(linux))
    config = _local_cc_config(tree)
    if config is None:
        reading.read.append(
            f"its hermetic C++ toolchains are for linux {arches} only, and no config in "
            ".bazelrc builds with the machine's own compiler"
        )
        return
    name, evidence = config
    reading.compiler = "clang" if "clang" in name else "g++"
    reading.compiler_evidence = [evidence]
    reading.read.append(
        f"its hermetic C++ toolchains are for linux {arches} only; --config={name} builds "
        f"with the machine's own compiler instead ({reading.compiler})"
    )


def _local_cc_config(tree: Path) -> Optional[tuple[str, Evidence]]:
    try:
        text = (tree / ".bazelrc").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    pattern = re.compile(
        r"^\s*(?:common|build):([\w-]+)\s+.*(?:enable_hermetic_cc=False|USE_HERMETIC_CC_TOOLCHAIN=0)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1), _cite(text, match, ".bazelrc")


# ----------------------------------------------------------------- wheels


def _wheels(tree: Path, text: str, name: str, arch: str, reading: Reading) -> None:
    for parse in re.finditer(r"pip\.parse\(", text):
        arguments = _arguments(text, parse.end())
        if not re.search(r"download_only\s*=\s*True", arguments):
            continue
        platforms = re.search(r"target_platforms\s*=\s*\[([^\]]*)\]", arguments)
        listed = set(re.findall(_ARCHES, platforms.group(1))) if platforms else set()
        if not listed or arch in listed:
            continue
        local = re.search(r"local_wheels\s*=\s*\{([^}]*)\}", arguments)
        names = re.findall(r'"([A-Za-z0-9_.-]+)"\s*:', local.group(1)) if local else []
        # Not the wheels this repository builds itself -- jax and jaxlib
        # both come out of the jax tree -- nor accelerator plugins.
        own = _built_here(tree) | {canonicalize_name(name)}
        wanted = [
            n for n in names
            if canonicalize_name(n) not in own and not _ACCELERATOR.search(n)
        ]
        # local_wheels matches dist/numpy-*.whl: whatever version is built.
        specs = [str(canonicalize_name(n)) for n in wanted]
        at = parse.start() + len(text[parse.start():parse.end()])
        line = _line(text, at)
        evidence = [Evidence("MODULE.bazel", line, line, "pip.parse(")]
        if local:
            local_line = _line(text, parse.end() + local.start())
            evidence.append(Evidence("MODULE.bazel", local_line, local_line, "local_wheels"))
        arches = " and ".join(sorted(listed))
        if not specs:
            reading.blocked.append(
                f"its Python packages are downloaded as wheels for {arches} only, and "
                "nothing lets them come from elsewhere"
            )
            reading.read.append(reading.blocked[-1])
            return
        reading.wheels = specs
        reading.wheel_evidence = evidence
        reading.read.append(
            f"its Python packages are downloaded as wheels for {arches} only; for {arch}, "
            f"local_wheels takes {', '.join(sorted(wanted))} from dist/, built first"
        )
        return


def _built_here(tree: Path) -> set:
    """The distributions a repository's own pyproject.toml and setup.py files build."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    names: set = set()
    for pattern in ("pyproject.toml", "*/pyproject.toml", "*/*/pyproject.toml"):
        for path in tree.glob(pattern):
            try:
                declared = tomllib.loads(path.read_text(errors="replace")).get("project", {})
            except (OSError, ValueError):
                continue
            if isinstance(declared.get("name"), str):
                names.add(canonicalize_name(declared["name"]))
    for pattern in ("setup.py", "*/setup.py", "*/*/setup.py"):
        for path in tree.glob(pattern):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            match = re.search(r"\bname\s*=\s*(?:(['\"])([\w.-]+)\1|([A-Za-z_]\w*))", text)
            if match is None:
                continue
            value = match.group(2)
            if value is None:
                constant = re.search(
                    rf"^{re.escape(match.group(3))}\s*=\s*['\"]([\w.-]+)['\"]", text, re.MULTILINE
                )
                value = constant.group(1) if constant else None
            if value:
                names.add(canonicalize_name(value))
    return names
