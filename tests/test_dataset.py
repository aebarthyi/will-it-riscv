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
