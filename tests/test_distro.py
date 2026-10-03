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


class _ByArch(httpx.BaseTransport):
    """A Packages.gz for each (arch, component) given; 404 for the rest."""

    def __init__(self, lists: dict):
        self.lists = lists

    def handle_request(self, request):
        for (arch, component), names in self.lists.items():
            if f"/{component}/binary-{arch}/" in str(request.url):
                body = "".join(f"Package: {n}\n\n" for n in names).encode()
                return httpx.Response(200, content=gzip.compress(body))
        return httpx.Response(404)


def _debian(tmp_path, lists):
    client = httpx.Client(transport=_ByArch(lists))
    return DistroIndex(client, Cache(root=tmp_path / "cache"), KNOWN_DISTROS["debian:trixie"],
                       "riscv64")


def test_a_name_no_arch_has_is_not_named_and_one_amd64_has_is(tmp_path):
    """openmpi is nobody's package; CUDA is amd64's, and in non-free."""
    index = _debian(tmp_path, {
        ("riscv64", "main"): ["libopenmpi-dev", "openmpi-bin"],
        ("amd64", "main"): ["libopenmpi-dev", "openmpi-bin", "libnuma-dev"],
        ("amd64", "non-free"): ["nvidia-cuda-toolkit"],
    })
    assert index.named("libopenmpi-dev") is True
    assert index.named("libnuma-dev") is True           # just not for riscv64
    assert index.named("nvidia-cuda-toolkit") is True
    assert index.named("openmpi") is False


def test_named_is_unsure_when_the_reference_list_cannot_be_had(tmp_path):
    index = _debian(tmp_path, {("riscv64", "main"): ["gcc"]})
    assert index.named("gcc") is True
    assert index.named("openmpi") is None


def test_similar_puts_the_dev_package_first_and_leaves_out_docs_and_bindings(tmp_path):
    index = _debian(tmp_path, {("riscv64", "main"): [
        "openmpi-doc", "openmpi-bin", "libopenmpi-dev", "libopenmpi40", "zlib1g-dev",
        "lua-zlib-dev", "libpcre2-dev", "libghc-zlib-dev",
    ]})
    assert index.similar("openmpi") == ["libopenmpi-dev", "openmpi-bin", "libopenmpi40"]
    assert index.similar("zlib-dev") == ["zlib1g-dev"]
    assert index.similar("libpcre3-dev") == ["libpcre2-dev"]
    assert index.similar("build-base") == []


def test_ubuntu_looks_for_a_name_in_the_main_archive_since_ports_has_no_amd64(tmp_path):
    seen = []

    class Archive(httpx.BaseTransport):
        def handle_request(self, request):
            url = str(request.url)
            seen.append(url)
            names = {
                ("ports.ubuntu.com", "main"): ["libopenmpi-dev"],
                ("archive.ubuntu.com", "main"): ["libopenmpi-dev"],
                ("archive.ubuntu.com", "multiverse"): ["nvidia-cuda-toolkit"],
            }
            for (host, component), listed in names.items():
                if request.url.host == host and f"/{component}/" in url:
                    body = "".join(f"Package: {n}\n\n" for n in listed).encode()
                    return httpx.Response(200, content=gzip.compress(body))
            return httpx.Response(404)

    index = DistroIndex(httpx.Client(transport=Archive()), Cache(root=tmp_path / "cache"),
                        KNOWN_DISTROS["ubuntu:resolute"], "riscv64")
    assert index.named("nvidia-cuda-toolkit") is True
    assert index.named("openmpi") is False
    assert any("archive.ubuntu.com/ubuntu/dists/resolute/multiverse/binary-amd64" in u
               for u in seen)
