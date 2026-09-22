from packaging.version import Version

from will_it_riscv.index import _parse_project, tag_platforms
from will_it_riscv.target import Target

PAYLOAD = {
    "files": [
        {"filename": "demo-1.0.tar.gz", "url": "u1", "core-metadata": False},
        {"filename": "demo-1.0-py3-none-any.whl", "url": "u2",
         "core-metadata": {"sha256": "abc"}},
        {"filename": "demo-2.0-cp312-cp312-manylinux_2_39_riscv64.whl", "url": "u3",
         "core-metadata": True, "requires-python": ">=3.9"},
        {"filename": "demo-2.0-cp312-cp312-manylinux_2_39_x86_64.whl", "url": "u4"},
        {"filename": "demo-3.0-cp312-cp312-win_amd64.whl", "url": "u5", "yanked": True},
        {"filename": "not-a-package.txt", "url": "u6"},
        {"filename": "demo-garbage-!!.whl", "url": "u7"},
    ]
}


def test_parses_and_groups_by_version():
    project = _parse_project("demo", PAYLOAD)
    assert set(project.releases) == {Version("1.0"), Version("2.0"), Version("3.0")}
    assert len(project.releases[Version("2.0")].wheels) == 2
    assert project.releases[Version("1.0")].has_sdist


def test_unparseable_filenames_are_dropped():
    project = _parse_project("demo", PAYLOAD)
    everything = [
        f.filename
        for r in project.releases.values()
        for f in r.wheels + r.sdists
    ]
    assert "not-a-package.txt" not in everything
    assert "demo-garbage-!!.whl" not in everything


def test_core_metadata_flag_handles_both_spellings():
    project = _parse_project("demo", PAYLOAD)
    wheels = {f.filename: f for r in project.releases.values() for f in r.wheels}
    assert wheels["demo-1.0-py3-none-any.whl"].core_metadata
    assert wheels["demo-2.0-cp312-cp312-manylinux_2_39_riscv64.whl"].core_metadata
    assert not wheels["demo-2.0-cp312-cp312-manylinux_2_39_x86_64.whl"].core_metadata


def test_fully_yanked_releases_are_excluded_from_versions():
    project = _parse_project("demo", PAYLOAD)
    assert Version("3.0") not in project.versions()
    assert Version("3.0") in project.versions(allow_yanked=True)


def test_matching_and_pure_wheels():
    project = _parse_project("demo", PAYLOAD)
    target = Target(arch="riscv64", libc_version=(2, 39), python_version=(3, 12))
    v1 = project.releases[Version("1.0")]
    v2 = project.releases[Version("2.0")]
    assert [w.filename for w in v1.pure_wheels()] == ["demo-1.0-py3-none-any.whl"]
    assert v2.pure_wheels() == []
    assert [w.filename for w in v2.matching_wheels(target.tags())] == [
        "demo-2.0-cp312-cp312-manylinux_2_39_riscv64.whl"
    ]


def test_platform_tags_listing():
    project = _parse_project("demo", PAYLOAD)
    assert project.releases[Version("2.0")].platform_tags() == [
        "manylinux_2_39_riscv64",
        "manylinux_2_39_x86_64",
    ]


def test_tag_platforms_helper():
    from packaging.utils import parse_wheel_filename

    tags = parse_wheel_filename("d-1.0-cp312-cp312-manylinux_2_39_riscv64.whl")[3]
    assert tag_platforms(tags) == ["manylinux_2_39_riscv64"]
