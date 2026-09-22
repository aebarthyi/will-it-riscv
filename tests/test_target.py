import pytest
from packaging.utils import parse_wheel_filename

from will_it_riscv.target import Target


@pytest.mark.parametrize(
    "spec, arch, libc, libc_version",
    [
        ("riscv64", "riscv64", "glibc", (2, 39)),
        ("riscv64-unknown-linux-gnu", "riscv64", "glibc", (2, 39)),
        ("riscv64gc-unknown-linux-gnu", "riscv64", "glibc", (2, 39)),
        ("riscv64-musl1.2", "riscv64", "musl", (1, 2)),
        ("riscv64gc-unknown-linux-musl", "riscv64", "musl", (1, 2)),
        ("riscv64-glibc2.36", "riscv64", "glibc", (2, 36)),
        ("aarch64", "aarch64", "glibc", (2, 17)),
        ("arm64", "aarch64", "glibc", (2, 17)),
    ],
)
def test_parse(spec, arch, libc, libc_version):
    t = Target.parse(spec)
    assert (t.arch, t.libc, t.libc_version) == (arch, libc, libc_version)


def test_riscv64_has_no_legacy_manylinux_aliases():
    """manylinux2014 et al. predate riscv64 and were never defined for it."""
    tags = Target.parse("riscv64").platform_tags()
    assert "manylinux_2_39_riscv64" in tags
    assert "linux_riscv64" in tags
    assert not any(t.startswith(("manylinux1", "manylinux2010", "manylinux2014")) for t in tags)


def test_aarch64_keeps_manylinux2014_alias():
    tags = Target(arch="aarch64", libc_version=(2, 28)).platform_tags()
    assert "manylinux2014_aarch64" in tags
    assert "manylinux_2_28_aarch64" in tags


def test_glibc_floor_is_respected():
    """A wheel needing newer glibc than the target promises is not compatible."""
    t = Target(arch="riscv64", libc_version=(2, 36))
    assert "manylinux_2_36_riscv64" in t.platform_tags()
    assert "manylinux_2_39_riscv64" not in t.platform_tags()


@pytest.mark.parametrize(
    "filename, accepted",
    [
        ("foo-1.0-py3-none-any.whl", True),
        ("foo-1.0-py2.py3-none-any.whl", True),
        ("foo-1.0-cp312-cp312-manylinux_2_39_riscv64.whl", True),
        ("foo-1.0-cp312-abi3-manylinux_2_39_riscv64.whl", True),
        ("foo-1.0-cp39-abi3-manylinux_2_39_riscv64.whl", True),
        ("foo-1.0-cp312-cp312-linux_riscv64.whl", True),
        # wrong architecture
        ("foo-1.0-cp312-cp312-manylinux_2_39_x86_64.whl", False),
        ("foo-1.0-cp312-cp312-manylinux_2_28_aarch64.whl", False),
        # newer glibc than the target offers
        ("foo-1.0-cp312-cp312-manylinux_2_41_riscv64.whl", False),
        # wrong interpreter
        ("foo-1.0-cp311-cp311-manylinux_2_39_riscv64.whl", False),
        ("foo-1.0-cp313-cp313-manylinux_2_39_riscv64.whl", False),
        # musl is not glibc
        ("foo-1.0-cp312-cp312-musllinux_1_2_riscv64.whl", False),
        # a real-world compressed tag set: one of the two matches
        ("foo-1.0-cp312-cp312-manylinux_2_38_riscv64.manylinux_2_39_riscv64.whl", True),
    ],
)
def test_wheel_acceptance(target, filename, accepted):
    tags = parse_wheel_filename(filename)[3]
    assert target.accepts(tags) is accepted


def test_musl_target_accepts_musllinux_only(target):
    musl = Target(arch="riscv64", libc="musl", libc_version=(1, 2), python_version=(3, 12))
    musl_wheel = parse_wheel_filename("foo-1.0-cp312-cp312-musllinux_1_2_riscv64.whl")[3]
    gnu_wheel = parse_wheel_filename("foo-1.0-cp312-cp312-manylinux_2_39_riscv64.whl")[3]
    assert musl.accepts(musl_wheel)
    assert not musl.accepts(gnu_wheel)
    assert not target.accepts(musl_wheel)


def test_marker_environment(target):
    env = target.marker_environment()
    assert env["platform_machine"] == "riscv64"
    assert env["sys_platform"] == "linux"
    assert env["python_version"] == "3.12"
