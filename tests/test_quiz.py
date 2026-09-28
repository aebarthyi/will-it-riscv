"""The quiz: questions whose answers come from real configures and Debian's index."""

import json

from will_it_riscv import quiz
from will_it_riscv.pseudobuild import Probe, PseudoBuild


def write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


PROJECT = {
    "CMakeLists.txt": (
        "project(demo C)\n"
        'option(WITH_TBB "use TBB" OFF)\n'
        'option(BUILD_DOCS "docs" OFF)\n'
        "find_package(ZLIB\n  REQUIRED)\n"
        "find_package(PNG)\n"
        "if(WITH_TBB)\n  find_package(TBB REQUIRED)\nendif()\n"
        "if(BUILD_DOCS)\n  find_package(Doxygen)\nendif()\n"
    ),
    "examples/CMakeLists.txt": "find_package(Demo REQUIRED)\n",
    "third_party/x/CMakeLists.txt": "find_package(Vendored)\n",
}


def test_source_reads_the_project_and_not_its_vendored_code(tmp_path):
    source = quiz.Source.read(write(tmp_path, PROJECT))
    assert set(source.named) == {"zlib", "png", "tbb", "doxygen", "demo"}
    assert source.options["WITH_TBB"] == ("CMakeLists.txt", 2, "OFF")
    assert source.guarding() == ["WITH_TBB", "BUILD_DOCS"]  # the dull one last
    assert source.says_required("CMakeLists.txt:4")
    assert not source.says_required("CMakeLists.txt:6")
    assert source.standalone("examples/CMakeLists.txt")


def test_classify_settles_only_what_the_configure_proved(tmp_path):
    source = quiz.Source.read(write(tmp_path, PROJECT))
    build = PseudoBuild(completed=True, blockers=[], soft_misses={"PNG"}, found={"ZLIB", "Threads"})
    build.probes = {
        "zlib": Probe("ZLIB", "find_package", required=True, site="CMakeLists.txt:4"),
        "png": Probe("PNG", "find_package", site="CMakeLists.txt:6"),
        "threads": Probe("Threads", "find_package", site="CMakeLists.txt:1"),  # found, not REQUIRED
        "pkgconfig": Probe("PkgConfig", "find_package", parent="ZLIB"),
    }
    settled = quiz.classify(build, source)
    assert {name: c for name, (c, _, _) in settled.items()} == {
        "ZLIB": quiz.REQUIRED, "PNG": quiz.OPTIONAL, "TBB": quiz.NOT_ASKED, "Doxygen": quiz.NOT_ASKED,
    }
    build.completed = False  # absence proves nothing then
    assert quiz.NOT_ASKED not in {c for c, _, _ in quiz.classify(build, source).values()}


def test_grade_accepts_any_listed_answer():
    q = quiz.Question("x", "pkgmap", "test", "TBB", "?", ["libtbb-dev", "libonetbb-dev"])
    assert quiz.grade(q, " `libtbb-dev` ")
    assert not quiz.grade(q, "tbb")
    assert not quiz.grade(q, None)



def test_a_question_becomes_a_training_example_that_explains_itself():
    q = quiz.Question(
        "x", "trace", "train", "MPI", "?", ["not-asked"], choices=list(quiz.CLASSES),
        repo="hdf5", command="cmake -S . -B build",
        why="the configure completed without reaching it; CMakeLists.txt:833",
    )
    example = quiz.to_chat(q)
    answer = json.loads(example["messages"][-1]["content"])
    assert answer["answer"] == "not-asked"
    assert "CMakeLists.txt:833" in answer["reasoning"]
    assert example["meta"]["split"] == "train"
