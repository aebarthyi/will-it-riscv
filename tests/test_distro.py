import gzip

import httpx
import pytest

from will_it_riscv.cache import Cache
from will_it_riscv.distro import (
    FAMILY_DEFAULTS,
    KNOWN_DISTROS,
    DistroIndex,
    resolve_spec,
)


def test_every_family_default_names_a_known_distro():
    for family, target in FAMILY_DEFAULTS.items():
        assert target in KNOWN_DISTROS
        assert target.startswith(family + ":")


@pytest.mark.parametrize("identifier", sorted(KNOWN_DISTROS))
def test_exact_identifiers_resolve(identifier):
    assert resolve_spec(identifier).id == identifier


def test_bare_family_resolves_to_the_documented_default():
    """Spelled out, so adding a release cannot silently change this."""
    assert resolve_spec("debian").id == "debian:trixie"
    assert resolve_spec("ubuntu").id == "ubuntu:resolute"


def test_identifiers_are_case_and_space_insensitive():
    assert resolve_spec("  Ubuntu:Resolute ").id == "ubuntu:resolute"


@pytest.mark.parametrize("identifier", ["fedora:41", "resolute", "", "ubuntu:nonesuch"])
def test_unknown_identifiers_raise_with_the_choices_listed(identifier):
    with pytest.raises(KeyError) as excinfo:
        resolve_spec(identifier)
    assert "ubuntu:resolute" in str(excinfo.value)


def test_ubuntu_releases_use_ports_and_include_universe():
    """riscv64 lives on ports, and most of what we look up is in universe."""
    for spec in KNOWN_DISTROS.values():
        if spec.id.startswith("ubuntu:"):
            assert spec.base == "http://ports.ubuntu.com/ubuntu-ports"
            assert "universe" in spec.components


class _FakeTransport(httpx.BaseTransport):
    """Serves a canned Packages.gz for main and 404s everything else."""

    def __init__(self, payload: bytes, only: str = "main"):
        self.payload = payload
        self.only = only
        self.requested: list[str] = []

    def handle_request(self, request):
        self.requested.append(str(request.url))
        if f"/{self.only}/" in str(request.url):
            return httpx.Response(200, content=gzip.compress(self.payload))
        return httpx.Response(404)


def _index(payload: bytes, tmp_path, spec_id="ubuntu:resolute", arch="riscv64"):
    transport = _FakeTransport(payload)
    client = httpx.Client(transport=transport)
    cache = Cache(root=tmp_path / "cache")
    return DistroIndex(client, cache, KNOWN_DISTROS[spec_id], arch), transport


PACKAGES = b"""Package: libssl-dev
Version: 3.5.0-1
Architecture: riscv64

Package: python3-numpy
Version: 2.2.0-1
Architecture: riscv64

Package: libblas-dev
Provides: libblas.so-dev, virtual-blas
Architecture: riscv64
"""


def test_parses_package_and_provides_names(tmp_path):
    index, _ = _index(PACKAGES, tmp_path)
    assert index.available
    assert index.has("libssl-dev")
    assert index.has("virtual-blas")          # from Provides:
    assert not index.has("intel-mkl")


def test_python_package_lookup(tmp_path):
    index, _ = _index(PACKAGES, tmp_path)
    assert index.python_package("numpy") == "python3-numpy"
    assert index.python_package("NumPy") == "python3-numpy"
    assert index.python_package("scipy") is None


def test_python_package_uses_the_name_overrides(tmp_path):
    index, _ = _index(b"Package: python3-yaml\nArchitecture: riscv64\n", tmp_path)
    assert index.python_package("PyYAML") == "python3-yaml"


def test_first_available_picks_a_present_candidate(tmp_path):
    index, _ = _index(PACKAGES, tmp_path)
    assert index.first_available(["nope-dev", "libssl-dev"]) == "libssl-dev"
    assert index.first_available(["nope-dev"]) is None


def test_results_are_cached_between_instances(tmp_path):
    index, transport = _index(PACKAGES, tmp_path)
    assert index.available
    first = len(transport.requested)
    again, transport2 = _index(PACKAGES, tmp_path)
    assert again.available
    assert transport2.requested == []      # served from cache
    assert first > 0


def test_missing_port_is_reported_not_raised(tmp_path):
    transport = _FakeTransport(PACKAGES, only="nothing-matches")
    index = DistroIndex(
        httpx.Client(transport=transport),
        Cache(root=tmp_path / "cache"),
        KNOWN_DISTROS["ubuntu:resolute"],
        "sparc64",
    )
    assert not index.available
    assert "no sparc64 port" in index.error
    assert index.has("libssl-dev") is False
