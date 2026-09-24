"""The evidence pack: what a planner is shown of a repository.

A small model should not go looking through a repository for how it builds.
It gets a pack: the parts of the tree that say so, chosen deterministically
and shown with line numbers, so that every step it writes can cite
``path:line`` from what it was shown.

In order, until the budget runs out:

  docs          README, INSTALL, BUILDING: the sections about installing and
                building, or else the shell blocks in them
  scripts       the shell scripts at the top of the repository, what they
                source and run, and the Python drivers they hand over to --
                mfc.sh, toolchain/bootstrap/python.sh, toolchain/main.py
  manifests     pyproject.toml, setup.py, requirements files, Cargo.toml,
                MODULE.bazel, meson.options
  build files   the top of CMakeLists.txt and meson.build: project(), its
                options, what it looks for
  CI            the workflows that build it, and the Dockerfiles

A file that does not fit is named as omitted, so the planner knows it is
there. The same tree always gives the same pack: a training example has to
be reproducible from the repository alone.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: Characters, not tokens: about 20k tokens of a typical repository's files.
BUDGET = 80_000
#: A line longer than this is cut, and says so.
LINE_WIDTH = 240

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", "third_party",
    "3rdparty", "external", "extern", "vendor", "subprojects", ".tox", ".mypy_cache",
    "site-packages", "tests", "test", "examples", "benchmarks", "docs/_build",
}
_DOC = re.compile(
    r"^(README|INSTALL|INSTALLING|INSTALLATION|BUILDING|BUILD|COMPILING|GETTING[_-]?STARTED)"
    r"(\.(md|rst|txt|markdown))?$",
    re.I,
)
_INSTALL_HEADING = re.compile(
    r"install|build|compil|getting[ _-]?started|quick[ _-]?start|requirement|dependenc|"
    r"prerequisite|from source|setup|toolchain|packaging|development",
    re.I,
)
#: A page under docs/ worth showing, by its name alone.
_DOC_PAGE = re.compile(
    r"install|building|^build|getting[ _-]?started|quick[ _-]?start|developer|from[ _-]source|"
    r"compiling|prerequisite",
    re.I,
)
_SHELL_WORDS = re.compile(
    r"^\s*(\$\s+)?(cmake|make|ninja|meson|pip3?|python3?|uv|conda|mamba|apt(-get)?|yum|dnf|"
    r"brew|spack|cargo|bazel(isk)?|\./|sudo|git clone|source|export|module load|scons)\b"
)
_CMAKE_LINES = re.compile(
    r"^\s*(cmake_minimum_required|project|option|find_package|pkg_check_modules|"
    r"add_subdirectory|include|FetchContent_Declare|ExternalProject_Add|set\(CMAKE_[A-Z_]+_STANDARD|"
    r"enable_language|check_language|message\(FATAL_ERROR)\b",
    re.I,
)
_MESON_LINES = re.compile(
    r"^\s*(project|dependency|.*=\s*dependency|find_program|.*find_library|subdir|"
    r"import\('python'\)|.*get_option|option)\b"
)
_CI_RUN = re.compile(r"^\s*(-\s+)?(run|script|name|uses|runs-on|image|container|RUN|FROM)\b")
_LOW_VALUE_SCRIPT = re.compile(r"lint|format|spell|completion|precheck|clean|docs?|release", re.I)
_MANIFESTS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "requirements-build.txt",
    "requirements_build.txt", "environment.yml", "environment.yaml", "Cargo.toml",
    "MODULE.bazel", "WORKSPACE", "WORKSPACE.bazel", ".bazelversion", ".bazelrc",
    "meson.options", "meson_options.txt", "conda/meta.yaml", "recipe/meta.yaml",
    "conda-recipe/meta.yaml", "package.json",
)
_BUILD_FILES = (
    "CMakeLists.txt", "meson.build", "configure.ac", "configure.in", "Makefile",
    "GNUmakefile", "SConstruct", "build.rs",
)


@dataclass
class Excerpt:
    path: str
    why: str
    total: int
    """Lines in the whole file."""
    lines: list = field(default_factory=list)
    """``(number, text)`` for the lines shown, in order."""

    @property
    def size(self) -> int:
        return sum(len(text) + 8 for _, text in self.lines) + len(self.path) + 40

    def shows(self, start: int, end: int) -> bool:
        numbers = {n for n, _ in self.lines}
        return all(n in numbers for n in range(start, end + 1))


@dataclass
class EvidencePack:
    repo: str
    tree: list = field(default_factory=list)
    excerpts: list = field(default_factory=list)
    omitted: list = field(default_factory=list)
    """Files that would have been shown, had there been room."""

    def excerpt(self, path: str) -> Optional[Excerpt]:
        return next((e for e in self.excerpts if e.path == path), None)

    def shows(self, path: str, start: int, end: int) -> bool:
        excerpt = self.excerpt(path)
        return excerpt is not None and excerpt.shows(start, end)

    def render(self) -> str:
        out = [f"REPOSITORY {self.repo}", "", "TOP OF THE TREE"]
        out += [f"  {entry}" for entry in self.tree] or ["  (empty)"]
        for excerpt in self.excerpts:
            shown = len(excerpt.lines)
            extent = (
                f"all {excerpt.total} lines" if shown == excerpt.total
                else f"{shown} of {excerpt.total} lines"
            )
            out += ["", f"=== {excerpt.path}  ({excerpt.why}; {extent})"]
            previous = 0
            for number, text in excerpt.lines:
                if previous and number > previous + 1:
                    out.append("      …")
                out.append(f"{number:>5}| {text}")
                previous = number
        if self.omitted:
            out += ["", "NOT SHOWN, FOR LACK OF ROOM"]
            out += [f"  {path}  ({why})" for path, why in self.omitted]
        return "\n".join(out) + "\n"

    def digest(self) -> str:
        return hashlib.sha256(self.render().encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "tree": list(self.tree),
            "excerpts": [
                {
                    "path": e.path, "why": e.why, "total": e.total,
                    "lines": [list(x) for x in e.lines],
                }
                for e in self.excerpts
            ],
            "omitted": [list(x) for x in self.omitted],
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidencePack:
        return cls(
            repo=data["repo"],
            tree=list(data.get("tree", [])),
            excerpts=[
                Excerpt(e["path"], e["why"], e["total"], [tuple(x) for x in e["lines"]])
                for e in data.get("excerpts", [])
            ],
            omitted=[tuple(x) for x in data.get("omitted", [])],
        )


def build(root: Path, name: Optional[str] = None, budget: int = BUDGET) -> EvidencePack:
    """The pack for a repository, the same every time for the same tree."""
    root = Path(root).resolve()
    pack = EvidencePack(repo=name or root.name, tree=_tree(root))
    seen: set = set()
    remaining = budget - len(pack.render())
    for excerpt in _candidates(root, name):
        if excerpt.path in seen or not excerpt.lines:
            continue
        seen.add(excerpt.path)
        if excerpt.size <= remaining:
            pack.excerpts.append(excerpt)
            remaining -= excerpt.size
            continue
        # The head of what was chosen, if a useful part of it fits.
        if remaining > 2000:
            kept: list = []
            used = len(excerpt.path) + 40
            for line in excerpt.lines:
                if used + len(line[1]) + 8 > remaining:
                    break
                kept.append(line)
                used += len(line[1]) + 8
            if len(kept) >= 10:
                pack.excerpts.append(Excerpt(excerpt.path, excerpt.why, excerpt.total, kept))
                remaining -= used
                continue
        pack.omitted.append((excerpt.path, excerpt.why))
    return pack


# ----------------------------------------------------------------- choosing


def _candidates(root: Path, name: Optional[str] = None):
    """Excerpts in the order they deserve room."""
    yield from _docs(root)
    yield from _scripts(root, minor=False)
    # A repository of several projects: the one asked for has its own files.
    # jaxlib's setup.py is in jaxlib/, beside jax's at the top.
    tops = [root]
    if name:
        for candidate in (name, name.replace("-", "_")):
            if (root / candidate).is_dir() and root / candidate not in tops:
                tops.insert(0, root / candidate)
    for top in tops:
        for relative in _MANIFESTS:
            path = top / relative
            if not path.is_file():
                continue
            if path.name == "pyproject.toml":
                yield _pyproject(root, path)
            else:
                yield _whole(root, path, "how it installs and what it declares", cap=220)
    for pattern in ("requirements/*.txt", "requirements/*.in", "*/pyproject.toml"):
        for path in sorted(root.glob(pattern))[:6]:
            if _skipped(path.relative_to(root)):
                continue
            yield _whole(root, path, "a manifest", cap=120)
    for relative in _BUILD_FILES:
        path = root / relative
        if path.is_file():
            yield _build_file(root, path)
    yield from _ci(root)
    # Linting, formatting, completions: reached from the entry script, but
    # the last things a build needs to be shown.
    yield from _scripts(root, minor=True)


def _docs(root: Path):
    paths = [p for p in sorted(root.iterdir()) if p.is_file() and _DOC.match(p.name)]
    docs = root / "docs"
    if docs.is_dir():
        paths += [
            p for p in sorted(docs.rglob("*"))
            if p.is_file() and p.suffix.lower() in (".md", ".rst", ".txt")
            and _DOC_PAGE.search(p.stem) and len(p.relative_to(docs).parts) <= 3
        ][:3]
    for path in paths:
        lines = _read(path)
        if lines is None:
            continue
        chosen = _doc_sections(lines) or _shell_blocks(lines)
        if chosen:
            yield _excerpt(root, path, "install and build instructions", lines, chosen, cap=160)


def _scripts(root: Path, minor: bool):
    from .scripts import reached_files

    reached = reached_files(root)

    def is_minor(relative: str, chain: tuple) -> bool:
        names = [Path(relative).stem] + [Path(c.split(":")[0]).stem for c in chain[1:]]
        return any(_LOW_VALUE_SCRIPT.search(n) for n in names)

    wanted = [
        (relative, chain) for relative, chain in reached.items()
        if is_minor(relative, chain) == minor
    ]
    for relative, chain in sorted(wanted, key=lambda item: (len(item[1]), item[0])):
        path = root / relative
        lines = _read(path)
        if lines is None:
            continue
        why = "the script a person runs" if not chain else f"reached from {chain[0].split(':')[0]}"
        cap = 200 if not chain else 140
        if relative.endswith(".py"):
            why = f"a Python driver, run from {chain[-1] if chain else 'a script'}"
            yield _excerpt(root, path, why, lines, _python_driver(lines), cap=120)
        else:
            yield _excerpt(root, path, why, lines, set(range(1, len(lines) + 1)), cap=cap)


_WORKFLOW_GOOD = ("build", "test", "ci", "linux", "ubuntu", "wheel", "main", "check")
_WORKFLOW_BAD = (
    "doc", "lint", "format", "release", "deploy", "bench", "coverage", "claude", "stale",
    "label", "spell", "codeql", "dependabot", "pages", "publish", "cleanliness", "tap",
)


def _ci(root: Path):
    """The workflows that build it, its Dockerfiles, and the scripts they call."""
    workflows = []
    for pattern in (".github/workflows/*.yml", ".github/workflows/*.yaml"):
        workflows += sorted(root.glob(pattern))
    workflows += [
        root / name for name in (".gitlab-ci.yml", ".circleci/config.yml", "azure-pipelines.yml",
                                 ".travis.yml", ".cirrus.yml")
        if (root / name).is_file()
    ]

    def score(path: Path) -> tuple:
        name = path.name.lower()
        return (
            -sum(word in name for word in _WORKFLOW_GOOD)
            + 3 * sum(word in name for word in _WORKFLOW_BAD),
            name,
        )

    chosen = sorted(workflows, key=score)[:3]
    for path in chosen:
        lines = _read(path)
        if lines is None:
            continue
        picked = {n for n, text in enumerate(lines, 1) if _CI_RUN.search(text)}
        picked |= _following_blocks(lines, picked)
        yield _excerpt(root, path, "how CI builds it", lines, picked, cap=110)
    dockerfiles = [
        p for p in sorted(root.glob("*")) + sorted(root.glob("*/*"))
        if p.is_file() and (p.name.lower().startswith("dockerfile") or p.suffix == ".dockerfile")
        and not _skipped(p.relative_to(root))
    ]
    for path in dockerfiles[:2]:
        yield _whole(root, path, "a container it builds in", cap=80)
    # Scripts the chosen workflows call by path: MFC's build-and-test.sh.
    called = set()
    for path in chosen:
        for text in _read(path) or []:
            for match in re.finditer(r"([\w./-]+\.(?:sh|bash|py))\b", text):
                target = (root / match.group(1).lstrip("./")).resolve()
                # Named by CI, so wanted even from build/ -- jax's build.py
                # is there -- but not an example CI happens to run.
                if (
                    target.is_file() and target.is_relative_to(root)
                    and not _skipped(target.relative_to(root), _EXAMPLE_DIRS)
                ):
                    called.add(target)
    for path in sorted(called)[:3]:
        yield _whole(root, path, "a script CI runs", cap=80)


# ------------------------------------------------------------------ helpers


_EXAMPLE_DIRS = {"examples", "example", "tests", "test", "benchmarks", "samples"}


def _skipped(relative: Path, dirs: Optional[set] = None) -> bool:
    parts = [p.lower() for p in relative.parts[:-1]]
    return any(part in (_SKIP_DIRS if dirs is None else dirs) for part in parts)


def _read(path: Path) -> Optional[list[str]]:
    try:
        data = path.read_bytes()[:400_000]
    except OSError:
        return None
    if b"\0" in data[:2000]:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def _excerpt(
    root: Path, path: Path, why: str, lines: list[str], chosen: set, cap: int
) -> Excerpt:
    numbers = sorted(n for n in chosen if 1 <= n <= len(lines))[:cap]
    shown = []
    for number in numbers:
        text = lines[number - 1].rstrip()
        if len(text) > LINE_WIDTH:
            text = text[:LINE_WIDTH] + " …(cut)"
        shown.append((number, text))
    return Excerpt(path.relative_to(root).as_posix(), why, len(lines), shown)


def _whole(root: Path, path: Path, why: str, cap: int) -> Excerpt:
    lines = _read(path) or []
    return _excerpt(root, path, why, lines, set(range(1, len(lines) + 1)), cap)


def _around(numbers: set, lines: list[str], before: int = 1, after: int = 2) -> set:
    wider: set = set()
    for n in numbers:
        wider |= set(range(max(1, n - before), min(len(lines), n + after) + 1))
    return wider


def _doc_sections(lines: list[str]) -> set:
    """Lines of the sections whose headings are about installing or building."""
    headings = []
    for index, text in enumerate(lines):
        level = _heading_level(lines, index)
        if level:
            headings.append((index + 1, level, text))
    chosen: set = set()
    for position, (number, level, text) in enumerate(headings):
        if not _INSTALL_HEADING.search(text):
            continue
        end = len(lines)
        for later, later_level, _ in headings[position + 1:]:
            if later_level <= level:
                end = later - 1
                break
        chosen |= set(range(number, min(end, number + 80) + 1))
    return chosen


def _heading_level(lines: list[str], index: int) -> int:
    text = lines[index]
    markdown = re.match(r"^(#{1,6})\s+\S", text)
    if markdown:
        return len(markdown.group(1))
    if index + 1 < len(lines) and text.strip() and re.match(r"^(=+|-+|~+)\s*$", lines[index + 1]):
        return {"=": 1, "-": 2, "~": 3}[lines[index + 1].strip()[0]]
    return 0


def _shell_blocks(lines: list[str]) -> set:
    """Command lines in the docs, with a little context, when no heading says install."""
    return _around({n for n, text in enumerate(lines, 1) if _SHELL_WORDS.match(text)}, lines, 2, 2)


def _python_driver(lines: list[str]) -> set:
    """The head of a driver, and the lines where it runs things or takes options."""
    interesting = re.compile(
        r"subprocess|os\.system|check_call|run\(|add_argument|argparse|cmake|pip|"
        r"def main|if __name__"
    )
    chosen = set(range(1, min(len(lines), 25) + 1))
    hits = {n for n, text in enumerate(lines, 1) if interesting.search(text)}
    chosen |= _around(hits, lines, 0, 1)
    return chosen


#: pyproject.toml tables a build reads. The rest -- linters, test runners,
#: coverage -- only take room: numpy's [tool.meson-python] is at line 242.
_PYPROJECT_TABLES = re.compile(
    r"^(build-system|project|project\.optional-dependencies|dependency-groups|"
    r"tool\.(meson-python|scikit-build|maturin|setuptools|setuptools_scm|poetry|pdm|hatch|"
    r"flit|cibuildwheel|spin|py-build-cmake|sip|pybind11|nanobind|conda))(\..*)?$"
)


def _pyproject(root: Path, path: Path) -> Excerpt:
    lines = _read(path) or []
    chosen: set = set()
    keep = True   # before the first table: comments, top-level keys
    for number, text in enumerate(lines, 1):
        header = re.match(r"^\s*\[\[?([^\]]+)\]\]?\s*(#.*)?$", text)
        if header:
            keep = bool(_PYPROJECT_TABLES.match(header.group(1).strip()))
        if keep:
            chosen.add(number)
    return _excerpt(root, path, "how it installs and what it declares", lines, chosen, cap=260)


def _build_file(root: Path, path: Path) -> Excerpt:
    lines = _read(path) or []
    pattern = _MESON_LINES if path.name == "meson.build" else _CMAKE_LINES
    chosen = set(range(1, min(len(lines), 30) + 1))
    chosen |= _around({n for n, text in enumerate(lines, 1) if pattern.search(text)}, lines, 0, 1)
    return _excerpt(root, path, "the build's own configuration", lines, chosen, cap=150)


def _following_blocks(lines: list[str], starts: set) -> set:
    """The indented lines under a CI ``run: |`` block."""
    chosen: set = set()
    for number in starts:
        text = lines[number - 1]
        if not re.search(r"(run|script):\s*[|>]", text):
            continue
        indent = len(text) - len(text.lstrip())
        for later in range(number + 1, min(len(lines), number + 25) + 1):
            line = lines[later - 1]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            chosen.add(later)
    return chosen


#: Directories whose contents are shown in the tree: where build machinery lives.
_BUILD_DIRS = {
    "toolchain", "build", "scripts", "script", "ci", ".ci", "tools", "cmake", "packaging",
    "conda", "recipe", "conda-recipe", "docker", ".github", "bootstrap", "config", "misc",
    "dev", "devtools", "buildscripts", "build_tools", "infra",
}


def _tree(root: Path) -> list[str]:
    """The top level, and one level into the directories build machinery lives in."""
    entries: list[str] = []
    try:
        top = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return entries
    for path in top:
        if path.name in (".git", "__pycache__", ".DS_Store"):
            continue
        if not path.is_dir():
            entries.append(path.name)
            continue
        entries.append(f"{path.name}/")
        if path.name.lower() not in _BUILD_DIRS:
            continue
        try:
            children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            continue
        children = [c for c in children if c.name not in ("__pycache__", ".DS_Store")]
        shown = children[:10]
        entries += [f"{path.name}/{c.name}{'/' if c.is_dir() else ''}" for c in shown]
        if len(children) > len(shown):
            entries.append(f"{path.name}/… ({len(children) - len(shown)} more)")
    return entries[:100]
