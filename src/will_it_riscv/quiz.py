"""A quiz on how real builds find their dependencies, answered by the builds themselves.

Two kinds of question, and neither answer comes from a model:

  trace    Given a CMake project and a configure command, is a dependency
           ``required`` (the configure stops without it), ``optional`` (it
           looks, fails to find it, and carries on) or ``not-asked`` (the
           name is in the project's CMake, but this configure never reaches
           it)? The answer is what :mod:`pseudobuild` saw real CMake do.
           Each project is configured plainly, then once per option that
           guards a lookup, turned ON -- the questions a flag changes are the
           hard ones.
  pkgmap   Which Debian package makes ``find_package(TBB)`` succeed? The
           answer is every package whose files the lookup would find -- its
           config file, its ``.pc`` file, or the header its Find module
           searches for -- read from Debian's Contents index.

A trace question carries the lines of CMake it can be answered from, the way
an evidence pack does; a pkgmap question carries none, since it asks what a
model knows. Questions inherit the train/test split of the repository they
come from, so the test half can score a model trained on the other.

  build     trace the dataset's CMake projects and write the questions
  ask       put the questions to a model and score it
  sample    a small balanced test set, for people to take
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REQUIRED = "required"
OPTIONAL = "optional"
NOT_ASKED = "not-asked"
CLASSES = (REQUIRED, OPTIONAL, NOT_ASKED)

#: Directories whose CMake belongs to someone else's project.
VENDORED = re.compile(
    r"(^|/)(third[_-]?party|3rd[_-]?party|extern(al)?|vendor(ed)?|_?deps|submodules|"
    r"build|\.git)(/|$)",
    re.I,
)
FIND_PACKAGE = re.compile(r"\bfind_package\s*\(\s*([A-Za-z0-9_.+-]+)", re.I)
OPTION = re.compile(
    r"^\s*(?:option|cmake_dependent_option)\s*\(\s*([A-Za-z0-9_]+)\s+\"[^\"]*\"\s*([A-Za-z0-9_${}]*)",
    re.I | re.M,
)
IF_LINE = re.compile(r"^\s*(?:else)?if\s*\((.*)", re.I)
#: Options that decide how, not what: flipping them rarely asks for anything.
DULL = re.compile(
    r"TEST|DOC|EXAMPLE|BENCH|WERROR|SANITI|COVERAGE|INSTALL|SHARED|STATIC|WARN|"
    r"VERBOSE|DEBUG|PIC|LTO|CCACHE|FORMAT|TIDY|PEDANTIC",
    re.I,
)
PKG_COMMANDS = {"find_package", "pkg_check_modules", "pkg_search_module"}
MAX_CONTEXT = 14000


@dataclass
class Question:
    id: str
    family: str
    split: str
    subject: str
    question: str
    answer: list[str]
    """Every accepted answer; for trace, the one class."""
    choices: Optional[list[str]] = None
    repo: Optional[str] = None
    origin: Optional[str] = None
    command: Optional[str] = None
    context: list[dict] = field(default_factory=list)
    """Excerpts, ``{"path", "lines": [[number, text], ...]}``."""
    why: str = ""
    """How the answer is known, for whoever doubts it."""
    tags: list[str] = field(default_factory=list)


def _id(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ the project's CMake


@dataclass
class Source:
    """The project's own CMake: its files, where it names packages and options."""

    root: Path
    files: dict[str, list[str]] = field(default_factory=dict)
    named: dict[str, tuple[str, int]] = field(default_factory=dict)
    """Package (lower case) to the first ``path, line`` a find_package names it."""
    options: dict[str, tuple[str, int, str]] = field(default_factory=dict)
    """Option to where it is declared and its default."""

    @classmethod
    def read(cls, root: Path, limit: int = 3000) -> Source:
        source = cls(root)
        paths = sorted(
            p
            for pattern in ("CMakeLists.txt", "*.cmake")
            for p in root.rglob(pattern)
            if p.is_file() and not VENDORED.search(str(p.relative_to(root).parent))
        )
        for path in paths[:limit]:
            relative = str(path.relative_to(root))
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            source.files[relative] = lines
            for number, line in enumerate(lines, 1):
                code = line.split("#", 1)[0]
                for match in FIND_PACKAGE.finditer(code):
                    source.named.setdefault(match.group(1).lower(), (relative, number))
            for match in OPTION.finditer("\n".join(lines)):
                number = "\n".join(lines)[: match.start()].count("\n") + 1
                default = match.group(2).upper() or "OFF"
                source.options.setdefault(match.group(1), (relative, number, default))
        return source

    def says_required(self, where: str) -> bool:
        """Does the call at ``path:line`` say REQUIRED, within its next few lines?"""
        path, line = _site(where)
        lines = self.files.get(path, [])
        call = " ".join(lines[line - 1 : line + 3]).split(")", 1)[0]
        return bool(re.search(r"\bREQUIRED\b", call))

    def standalone(self, path: str) -> bool:
        """A file no plain build reads: a test, an example, or a project of its own."""
        if re.search(
            r"(^|/)(tests?|examples?|samples?|docs?|benchmarks?|tutorials?|"
            r"autotest|demos?)(/|$)",
            path,
            re.I,
        ):
            return True
        return path != "CMakeLists.txt" and any(
            re.match(r"\s*project\s*\(", line, re.I) for line in self.files.get(path, [])
        )

    def guarding(self, limit: int = 8) -> list[str]:
        """Options off by default that an if() in front of a find_package names."""
        score: Counter = Counter()
        off = {
            name
            for name, (_, _, default) in self.options.items()
            if default in ("OFF", "FALSE", "0", "NO")
        }
        for lines in self.files.values():
            for number, line in enumerate(lines):
                if not FIND_PACKAGE.search(line.split("#", 1)[0]):
                    continue
                for before in lines[max(0, number - 15) : number]:
                    condition = IF_LINE.match(before)
                    if condition:
                        for name in re.findall(r"[A-Za-z0-9_]+", condition.group(1)):
                            if name in off:
                                score[name] += 1
        ranked = sorted(score, key=lambda n: (bool(DULL.search(n)), -score[n], n))
        return ranked[:limit]

    def excerpt(self, wanted: dict[str, list[int]], budget: int = MAX_CONTEXT) -> list[dict]:
        """The lines around each wanted line, whole files when they are short."""
        out = []
        spent = 0
        for path, centres in wanted.items():
            lines = self.files.get(path)
            if not lines:
                continue
            if len(lines) <= 250:
                keep = set(range(1, len(lines) + 1))
            else:
                keep = set()
                for centre in centres:
                    radius = 60 if centre == centres[0] else 12
                    keep |= set(
                        range(max(1, centre - radius), min(len(lines), centre + radius) + 1)
                    )
            numbered = [[n, lines[n - 1]] for n in sorted(keep)]
            size = sum(len(t) + 8 for _, t in numbered)
            if spent + size > budget:
                # Trim to the nearest lines around the first centre.
                centre = centres[0]
                numbered.sort(key=lambda item: abs(item[0] - centre))
                room = []
                for item in numbered:
                    if spent + len(item[1]) + 8 > budget:
                        break
                    room.append(item)
                    spent += len(item[1]) + 8
                numbered = sorted(room)
            else:
                spent += size
            if numbered:
                out.append({"path": path, "lines": numbered})
        return out


# ------------------------------------------------------------------ what a configure did


def classify(build, source: Source) -> dict[str, tuple[str, str, str]]:
    """Package to ``(class, why, where)`` for everything this configure settles.

    Only what the project's own CMake asked for counts -- its CMakeLists and
    its own Find modules, as cp2k's FindLapack asks for Blas; what CMake's or
    a package's module asked is that package's business. A found package
    that was not asked for REQUIRED settles nothing: nobody saw the
    configure without it.
    """
    blockers = {b.lower() for b in build.blockers}
    soft = {s.lower() for s in build.soft_misses}
    found = {f.lower() for f in build.found}
    settled: dict[str, tuple[str, str, str]] = {}
    for key, probe in build.probes.items():
        if probe.command not in PKG_COMMANDS:
            continue
        if probe.parent is not None and _site(probe.site or "")[0] not in source.files:
            continue
        where = probe.site or ""
        if key in blockers:
            settled[probe.name] = (REQUIRED, f"stopped the configure (round {probe.round})", where)
        elif probe.required and key in found and source.says_required(where):
            settled[probe.name] = (REQUIRED, "asked for REQUIRED, and found", where)
        elif key in soft:
            settled[probe.name] = (OPTIONAL, "looked for, not found, configure carried on", where)
    if build.completed:
        for key, (path, line) in source.named.items():
            if key not in build.probes and not source.standalone(path):
                name = source.files[path][line - 1]
                match = FIND_PACKAGE.search(name)
                settled[match.group(1) if match else key] = (
                    NOT_ASKED,
                    "the configure completed without reaching it",
                    f"{path}:{line}",
                )
    return settled


def _site(where: str) -> tuple[str, int]:
    path, _, line = where.rpartition(":")
    return (path, int(line)) if line.isdigit() else (where, 1)


def trace_questions(
    key: str,
    meta: dict,
    root: Path,
    runs: list[tuple[dict, object]],
    source: Source,
    rng: random.Random,
) -> list[Question]:
    """Questions from one project's configures: the plain one, then each flag."""
    base: dict = {}
    questions = []
    top = "CMakeLists.txt"
    for defines, build in runs:
        settled = classify(build, source)
        flag = next(iter(defines), None)
        if flag is None:
            base = settled
            chosen = [n for n, (c, _, _) in settled.items() if c != NOT_ASKED]
            absent = sorted(n for n, (c, _, _) in settled.items() if c == NOT_ASKED)
            rng.shuffle(absent)
            chosen += absent[: max(3, len(chosen))]
        else:
            changed = [n for n, v in settled.items() if base.get(n, (None,))[0] != v[0]]
            same = [n for n in settled if n not in changed]
            chosen = changed + rng.sample(same, min(1, len(same))) if changed else []
        command = "cmake -S . -B build" + "".join(f" -D{k}={v}" for k, v in defines.items())
        for name in chosen:
            answer, why, where = settled[name]
            path, line = _site(where)
            wanted: dict[str, list[int]] = defaultdict(list)
            wanted[path].append(line)
            if flag and flag in source.options:
                declared, at, _ = source.options[flag]
                wanted[declared].append(at)
            if top in source.files and top not in wanted:
                wanted[top].append(1)
            tags = []
            if flag:
                tags.append("flag")
            if path != top:
                tags.append("nested-file")
            text = source.files.get(path, [""] * line)[line - 1] if path in source.files else ""
            if re.search(r"\bREQUIRED\b", text):
                tags.append("keyword-required")
            probe = build.probes.get(name.lower())
            if probe is not None and probe.via:
                tags.append("via-macro")
            if probe is not None and probe.round > 1:
                tags.append("behind-blocker")
            questions.append(
                Question(
                    id=_id("trace", key, command, name),
                    family="trace",
                    split=meta.get("split") or "train",
                    subject=name,
                    question=(
                        f"In {key}, the build is configured with `{command}`. "
                        f"What does this configure do about `{name}`? "
                        f"required: it stops without {name}; "
                        f"optional: it looks for {name}, and carries on without it; "
                        f"not-asked: it never looks for {name}."
                    ),
                    answer=[answer],
                    choices=list(CLASSES),
                    repo=key,
                    origin=(meta.get("origin") or "").removeprefix("git "),
                    command=command,
                    context=source.excerpt(dict(wanted)),
                    why=f"{why}; {where}",
                    tags=tags,
                )
            )
    return questions


# ------------------------------------------------------------------ Debian's Contents


def contents_index(cache: Path):
    """Debian sid's riscv64 file lists: the same index the resolver answers from."""
    import httpx

    from .cache import Cache
    from .contents import ContentsIndex
    from .distro import resolve_spec

    with httpx.Client() as client:
        return ContentsIndex.load(client, Cache(cache), resolve_spec("debian:sid"), "riscv64")


def pkgmap_questions(asked: dict[str, tuple[str, list]], contents) -> list[Question]:
    """One question per package the dataset's builds looked for."""
    questions = []
    for name, (command, lookups) in sorted(asked.items()):
        packages, why = contents.providers(name, command, lookups)
        if not packages:
            continue
        call = (
            f"find_package({name})" if command == "find_package" else f"pkg_check_modules(X {name})"
        )
        split = "test" if int(_id("pkgmap", name.lower()), 16) % 5 == 0 else "train"
        tags = ["pkg-config"] if command != "find_package" else []
        if not any(name.lower().replace("_", "") in p.replace("-", "") for p in packages):
            tags.append("name-differs")
        questions.append(
            Question(
                id=_id("pkgmap", command, name),
                family="pkgmap",
                split=split,
                subject=name,
                question=(
                    f"On Debian (sid), which binary package do you install so that `{call}` "
                    f"in a CMake build finds {name}? Answer with one package name."
                ),
                answer=sorted(packages),
                why=why,
                tags=tags,
            )
        )
    return questions


# ------------------------------------------------------------------ building


def build(
    out: Path, dataset_dir: Path, cache: Path, flags: int = 8, progress=print
) -> list[Question]:
    """Trace every verified CMake project in the dataset; each trace is cached."""
    from . import dataset, pseudobuild

    traces = cache / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    questions: list[Question] = []
    asked: dict[str, tuple[str, list]] = {}
    for directory in sorted(p for p in dataset_dir.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("tier") != dataset.VERIFIED:
            continue
        root = dataset.tree(meta).root
        if root is None or not (root / "CMakeLists.txt").is_file():
            continue
        key = directory.name
        source = Source.read(root)
        configs = [{}] + [{name: "ON"} for name in source.guarding(flags)]
        runs = []
        for defines in configs:
            saved = traces / f"{key}.{_id(json.dumps(defines, sort_keys=True))}.json"
            if saved.is_file():
                result = _load_build(json.loads(saved.read_text()))
            else:
                result = pseudobuild.run(root, defines=defines, max_rounds=8, timeout=300)
                if result is None:
                    continue
                saved.write_text(json.dumps(_dump_build(result)))
            if result.error and not result.probes:
                progress(f"{key} {defines or 'plain'}: {result.error[:120]}")
                continue
            runs.append((defines, result))
            for probe in result.probes.values():
                if probe.command in PKG_COMMANDS and _site(probe.site or "")[0] in source.files:
                    asked.setdefault(
                        probe.name, (probe.command, result.lookups.get(probe.name.lower(), []))
                    )
        made = trace_questions(key, meta, root, runs, source, random.Random(key))
        questions += made
        progress(f"{key}: {len(runs)} configures, {len(made)} questions")
    contents = contents_index(cache)
    questions += pkgmap_questions(asked, contents)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(asdict(q)) + "\n" for q in questions))
    return questions


def _dump_build(result) -> dict:
    return {
        "probes": [asdict(p) for p in result.probes.values()],
        "completed": result.completed,
        "error": result.error,
        "found": sorted(result.found),
        "soft_misses": sorted(result.soft_misses),
        "blockers": list(result.blockers),
        "rounds": result.rounds,
        "lookups": {k: [list(x) for x in v] for k, v in result.lookups.items()},
    }


def _load_build(raw: dict):
    from .pseudobuild import Probe, PseudoBuild

    result = PseudoBuild(
        completed=raw["completed"],
        error=raw["error"],
        found=set(raw["found"]),
        soft_misses=set(raw["soft_misses"]),
        blockers=raw["blockers"],
        rounds=raw["rounds"],
        lookups={k: [tuple(x) for x in v] for k, v in raw["lookups"].items()},
    )
    for p in raw["probes"]:
        result.probes[p["name"].lower()] = Probe(**p)
    return result


def load(path: Path) -> list[Question]:
    return [Question(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ asking


SYSTEM = (
    "You answer questions about how C and C++ builds find their dependencies: CMake's "
    "find_package, pkg-config, options, and the Debian packages that satisfy them. Reason "
    "briefly from the CMake you are shown, then answer."
)


def render(q: Question) -> str:
    out = [q.question]
    for excerpt in q.context:
        out += ["", f"--- {excerpt['path']}"]
        out += [f"{n:5}| {text}" for n, text in excerpt["lines"]]
    return "\n".join(out)


def schema(q: Question) -> dict:
    answer = {"type": "string", "enum": q.choices} if q.choices else {"type": "string"}
    return {
        "type": "object",
        "properties": {"reasoning": {"type": "string", "maxLength": 1500}, "answer": answer},
        "required": ["reasoning", "answer"],
        "additionalProperties": False,
    }


def grade(q: Question, given: Optional[str]) -> bool:
    if not given:
        return False
    normal = given.strip().strip("`'\"").split()[0].lower() if given.strip() else ""
    return normal in {a.lower() for a in q.answer}


def ask(questions: list[Question], backend, progress=print) -> list[dict]:
    results = []
    for number, q in enumerate(questions, 1):
        given, error = None, None
        try:
            completion = backend.complete(
                SYSTEM, [{"role": "user", "content": render(q)}], schema(q)
            )
            given = json.loads(completion.text).get("answer")
        except Exception as exc:  # noqa: BLE001 - an unanswered question is a wrong one
            error = str(exc)[:200]
        results.append(
            {
                "id": q.id,
                "family": q.family,
                "tags": q.tags,
                "answer": q.answer,
                "given": given,
                "correct": grade(q, given),
                "error": error,
            }
        )
        if number % 50 == 0:
            right = sum(r["correct"] for r in results)
            progress(f"{number}/{len(questions)}: {right / number:.2f}")
    return results


def summary(questions: list[Question], results: list[dict]) -> list[tuple[str, int, float]]:
    """Accuracy overall, by family, by trace class (and their mean) and by tag."""
    by_id = {q.id: q for q in questions}
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        q = by_id[r["id"]]
        groups["all"].append(r["correct"])
        groups[q.family].append(r["correct"])
        if q.family == "trace":
            groups[f"trace:{q.answer[0]}"].append(r["correct"])
        for tag in q.tags:
            groups[f"{q.family}+{tag}"].append(r["correct"])
    rows = [(name, len(v), sum(v) / len(v)) for name, v in groups.items()]
    classes = [groups[f"trace:{c}"] for c in CLASSES if groups.get(f"trace:{c}")]
    if classes:
        rows.append(
            (
                "trace:balanced",
                sum(map(len, classes)),
                sum(sum(c) / len(c) for c in classes) / len(classes),
            )
        )
    return sorted(rows)


def sample(
    questions: list[Question], trace: int = 24, pkgmap: int = 12, seed: int = 0
) -> list[Question]:
    """A small test set, balanced across classes, leaning on the hard ones."""
    rng = random.Random(seed)
    test = [q for q in questions if q.split == "test"]
    chosen: list[Question] = []
    per_repo: Counter = Counter()
    for answer in CLASSES:
        pool = [q for q in test if q.family == "trace" and q.answer == [answer]]
        # The hard half first, then the rest, at most two a repository.
        pool.sort(key=lambda q: (-len(q.tags), q.id))
        half = len(pool) // 2
        hard, easy = pool[:half], pool[half:]
        rng.shuffle(hard)
        rng.shuffle(easy)
        taken = 0
        for q in hard + easy:
            if taken >= trace // 3:
                break
            if per_repo[q.repo] < 2:
                chosen.append(q)
                per_repo[q.repo] += 1
                taken += 1
    pool = [q for q in test if q.family == "pkgmap"]
    pool.sort(key=lambda q: ("name-differs" not in q.tags, q.id))
    chosen += pool[:pkgmap]
    rng.shuffle(chosen)
    return chosen


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="will-it-riscv-quiz", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    making = sub.add_parser("build", help="trace the dataset's CMake projects into questions")
    making.add_argument("--data", required=True, type=Path, help="a labelled dataset directory")
    making.add_argument("--out", required=True, type=Path, help="the questions, as JSONL")
    making.add_argument(
        "--cache", type=Path, help="traces and Debian's Contents (default: the cache)"
    )
    making.add_argument("--flags", type=int, default=8, help="options to turn on, per project")
    asking = sub.add_parser("ask", help="put the questions to a model and score it")
    asking.add_argument("--quiz", required=True, type=Path)
    asking.add_argument("--split", default="test")
    asking.add_argument("--family", choices=("trace", "pkgmap"))
    asking.add_argument("--model", required=True)
    asking.add_argument("--url", required=True, help="an OpenAI-compatible endpoint")
    asking.add_argument("--json", type=Path, help="write every answer here")
    sampling = sub.add_parser("sample", help="a small balanced test set, for people")
    sampling.add_argument("--quiz", required=True, type=Path)
    sampling.add_argument("--out", required=True, type=Path)
    sampling.add_argument("--trace", type=int, default=24)
    sampling.add_argument("--pkgmap", type=int, default=12)
    args = parser.parse_args(argv)
    err = lambda m: print(m, file=sys.stderr)  # noqa: E731

    if args.command == "build":
        if args.cache is None:
            from .cache import Cache

            args.cache = Cache().root / "quiz"
        made = build(args.out, args.data, args.cache, args.flags, progress=err)
        counts = Counter(
            (q.family, q.split, q.answer[0] if q.family == "trace" else "") for q in made
        )
        for key, n in sorted(counts.items()):
            print(*key, n)
        return 0
    questions = load(args.quiz)
    if args.command == "sample":
        chosen = sample(questions, args.trace, args.pkgmap)
        args.out.write_text("".join(json.dumps(asdict(q)) + "\n" for q in chosen))
        print(f"{len(chosen)} questions written to {args.out}")
        return 0
    from . import modelplan

    chosen = [
        q
        for q in questions
        if q.split == args.split and (args.family is None or q.family == args.family)
    ]
    backend = modelplan.OpenAICompatible(args.url, args.model)
    backend.max_tokens = 1024
    results = ask(chosen, backend, progress=err)
    rows = summary(chosen, results)
    for name, n, accuracy in rows:
        print(f"{name:32} {n:5} {accuracy:.3f}")
    if args.json:
        args.json.write_text(json.dumps({"summary": rows, "answers": results}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
