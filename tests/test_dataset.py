"""Finding a labelled example's tree again, on a machine it was not labelled on."""

import json
import subprocess
from dataclasses import asdict

from will_it_riscv import dataset, evidence
from will_it_riscv.dataset import Entry


def _meta(root, entry, origin="", digest=None):
    return {
        "entry": asdict(entry), "split": "train", "root": str(root), "tier": dataset.VERIFIED,
        "origin": origin, "pack_digest": digest,
    }


def test_the_recorded_root_is_used_when_it_is_there(tmp_path):
    found = dataset.tree(_meta(tmp_path, Entry(name="x", path=str(tmp_path))))
    assert found.root == tmp_path


def test_a_root_from_another_machine_is_found_in_this_cache(tmp_path):
    cache = tmp_path / "cache" / "dataset"
    (cache / "sources" / "foo-1.0" / "foo-1.0").mkdir(parents=True)
    recorded = "/Users/someone/Library/Caches/will-it-riscv/dataset/sources/foo-1.0/foo-1.0"
    found = dataset.tree(_meta(recorded, Entry(name="foo", pypi="foo", version="1.0")), cache)
    assert found.root == cache / "sources" / "foo-1.0" / "foo-1.0"


def test_a_missing_tree_is_not_fetched_unless_asked(tmp_path):
    recorded = "/Users/someone/Library/Caches/will-it-riscv/dataset/git/gone"
    found = dataset.tree(_meta(recorded, Entry(name="gone", git="https://example.invalid/g")),
                         tmp_path)
    assert found.root is None
    assert str(tmp_path / "git" / "gone") in found.error


def _git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                   check=True, capture_output=True)


def test_a_git_tree_is_fetched_again_at_the_commit_it_was_labelled_at(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "-q", "-b", "main", cwd=upstream)
    (upstream / "CMakeLists.txt").write_text("project(first)\n")
    _git("add", ".", cwd=upstream)
    _git("commit", "-qm", "first", cwd=upstream)
    labelled = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True,
                              text=True, check=True).stdout.strip()
    (upstream / "CMakeLists.txt").write_text("project(second)\n")
    _git("commit", "-qam", "second", cwd=upstream)

    entry = Entry(name="proj", git=upstream.as_uri())
    cache = tmp_path / "cache"
    pack = evidence.build(upstream, "proj")      # at "second": not what was labelled
    meta = _meta("/Users/someone/Library/Caches/will-it-riscv/dataset/git/proj", entry,
                 origin=f"git {entry.git}@{labelled}", digest=pack.digest())
    found = dataset.tree(meta, cache, fetch_missing=True)
    assert found.root == cache / "git" / "proj"
    assert (found.root / "CMakeLists.txt").read_text() == "project(first)\n"
    assert found.origin.endswith(labelled)


def test_check_tells_a_tree_that_rebuilds_its_pack_from_one_that_does_not(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache" / "dataset"
    good = cache / "git" / "good"
    good.mkdir(parents=True)
    (good / "CMakeLists.txt").write_text("project(good)\n")
    out = tmp_path / "out"
    for name, digest in (("good", evidence.build(good, "good").digest()),
                         ("changed", "0000000000000000"), ("gone", "0000000000000000")):
        if name == "changed":
            (cache / "git" / name).mkdir()
        (out / name).mkdir(parents=True)
        recorded = f"/Users/someone/Library/Caches/will-it-riscv/dataset/git/{name}"
        meta = _meta(recorded, Entry(name=name, git=f"https://example.invalid/{name}"),
                     digest=digest)
        (out / name / "meta.json").write_text(json.dumps(meta))

    monkeypatch.setenv("WILL_IT_RISCV_CACHE", str(tmp_path / "cache"))
    outcomes = dataset.check(out, progress=lambda m: None)
    assert outcomes == {"ok": ["good"], "differs": ["changed"], "missing": ["gone"]}


def test_a_clone_gets_its_submodules(tmp_path, monkeypatch):
    """A build that vendors through submodules stops at add_subdirectory without them."""
    import subprocess

    from will_it_riscv.sources import init_submodules

    def git(*args, cwd):
        subprocess.run(["git", "-c", "protocol.file.allow=always", *args], cwd=cwd, check=True,
                       capture_output=True)

    inner, outer = tmp_path / "inner", tmp_path / "outer"
    for repo in (inner, outer):
        repo.mkdir()
        git("init", "-q", cwd=repo)
        git("config", "user.email", "t@t", cwd=repo)
        git("config", "user.name", "t", cwd=repo)
    (inner / "CMakeLists.txt").write_text("# vendored\n")
    git("add", ".", cwd=inner)
    git("commit", "-qm", "inner", cwd=inner)
    git("submodule", "add", "-q", str(inner), "third_party/inner", cwd=outer)
    git("commit", "-qm", "outer", cwd=outer)
    clone = tmp_path / "clone"
    git("clone", "-q", str(outer), str(clone), cwd=tmp_path)
    assert not (clone / "third_party" / "inner" / "CMakeLists.txt").exists()
    # Local paths stand in for remotes here; git refuses them unless told.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    assert init_submodules(clone) is None
    assert (clone / "third_party" / "inner" / "CMakeLists.txt").is_file()


def _ran(*steps, unknown=()):
    from will_it_riscv import planrun
    from will_it_riscv.plan import parse_plan

    plan = parse_plan({"version": 1, "repo": "demo", "steps": [raw for raw, _, _ in steps]})
    result = planrun.PlanResult(plan, None)
    for step, (_, status, detail) in zip(plan.steps, steps):
        result.steps.append(planrun.StepResult(step, status, detail))
    result.unknown_names = list(unknown)
    return result


def test_a_name_debian_does_not_have_goes_back_with_the_near_names():
    result = _ran(({"id": "apt", "kind": "system-packages", "packages": ["openmpi"],
                    "evidence": []}, "done", "1 package"),
                  unknown=[("apt", "openmpi", ["libopenmpi-dev", "openmpi-bin"],
                            "Ubuntu 26.04 LTS (resolute)")])
    (problem,) = dataset.plan_faults(result)
    assert "Ubuntu 26.04 LTS (resolute) has no package named 'openmpi'" in problem
    assert "libopenmpi-dev, openmpi-bin" in problem


def test_a_placeholder_compiler_that_stops_the_configure_goes_back():
    """pflogger's CI passes -DCMAKE_Fortran_COMPILER=${FC}; bolt's README <C compiler>."""
    result = _ran(
        ({"id": "configure", "kind": "cmake-configure", "evidence": [],
          "defines": {"CMAKE_Fortran_COMPILER": "${FC}", "CMAKE_INSTALL_PREFIX": "${HOME}/x"}},
         "failed", "CMake Error at CMakeLists.txt:22 (project): No CMAKE_Fortran_COMPILER "
                   "could be found."),
        ({"id": "fine", "kind": "cmake-configure", "evidence": [], "source": "sub",
          "defines": {"CMAKE_C_COMPILER": "<C compiler>"}}, "completed", "1 round"),
    )
    (problem,) = dataset.plan_faults(result)
    assert "'configure' sets CMAKE_Fortran_COMPILER to '${FC}'" in problem
