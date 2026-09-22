
import pytest

from will_it_riscv.ci import CiFindings, _is_ci_file, scan_ci_configuration, scan_text
from will_it_riscv.sdist import make_recorder


def scan(text: str, path: str = ".github/workflows/ci.yml") -> CiFindings:
    findings = CiFindings()
    scan_text(text, path, findings)
    return findings


def test_plain_apt_install():
    f = scan("RUN apt-get install -y libssl-dev zlib1g-dev")
    assert sorted(f.packages) == ["libssl-dev", "zlib1g-dev"]


def test_flags_are_not_packages():
    f = scan("sudo apt-get -qq install --no-install-recommends -y libpq-dev")
    assert sorted(f.packages) == ["libpq-dev"]


def test_bash_array_indirection():
    """Pillow writes its list into an array and installs "${packages[@]}"."""
    f = scan(
        "packages=(\n"
        "    cmake\n"
        "    libfreetype6-dev\n"
        "    libjpeg-turbo8-dev\n"
        ")\n"
        'sudo apt-get -qq install --no-install-recommends "${packages[@]}"\n',
        ".ci/install.sh",
    )
    assert sorted(f.packages) == ["cmake", "libfreetype6-dev", "libjpeg-turbo8-dev"]


def test_version_pins_are_stripped():
    """psycopg2 pins to an interpolated version variable."""
    f = scan('sudo apt-get -qq -y install "libpq-dev=${pqver}" "libpq5=${pqver}"')
    assert sorted(f.packages) == ["libpq-dev", "libpq5"]


def test_unresolvable_variable_is_reported_not_dropped():
    f = scan("apt-get install -y ${MYSTERY_PACKAGES}")
    assert f.packages == {}
    assert any("could not be resolved" in n for n in f.notes)


def test_third_party_apt_source_is_flagged():
    f = scan(
        'echo "deb http://apt.postgresql.org/pub/repos/apt jammy-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
    )
    assert any("apt.postgresql.org" in w for w in f.warnings)


def test_ppa_is_flagged():
    f = scan("add-apt-repository ppa:intel-opencl/intel-opencl")
    assert any("ppa:intel-opencl/intel-opencl" in w for w in f.warnings)


def test_official_archives_are_not_flagged():
    f = scan("deb http://deb.debian.org/debian trixie main")
    assert f.warnings == []


def test_line_continuations_are_joined():
    f = scan("RUN apt-get install -y \\\n    libssl-dev \\\n    libffi-dev")
    assert sorted(f.packages) == ["libffi-dev", "libssl-dev"]


def test_hpccm_ospackages():
    """GROMACS declares packages through HPC Container Maker, not shell."""
    f = scan(
        'packages(ospackages=["git", "cmake", "libhwloc-dev"])',
        "admin/containers/build.py",
    )
    assert sorted(f.packages) == ["cmake", "git", "libhwloc-dev"]


def test_dnf_and_apk():
    assert "openssl-devel" in scan("dnf install -y openssl-devel").packages
    assert "openssl-dev" in scan("apk add --no-cache openssl-dev").packages


def test_nix_build_inputs():
    f = scan("buildInputs = [ openssl zlib libffi ];", "shell.nix")
    assert sorted(f.packages) == ["libffi", "openssl", "zlib"]


def test_nix_inputs_ignored_outside_nix_files():
    f = scan("buildInputs = [ openssl ];", ".github/workflows/ci.yml")
    assert f.packages == {}


@pytest.mark.parametrize(
    "path, expected",
    [
        (".github/workflows/ci.yml", True),
        (".gitlab-ci.yml", True),
        ("admin/gitlab-ci/gromacs.gitlab-ci.yml", True),
        ("Dockerfile", True),
        ("docker/Dockerfile.ci", True),
        ("ci/setup.sh", True),
        ("admin/containers/build.py", True),
        ("shell.nix", True),
        ("apt.txt", True),
        ("src/main.c", False),
        ("README.md", False),
        ("setup.py", False),
    ],
)
def test_ci_file_classification(path, expected):
    assert _is_ci_file(path) is expected


def test_end_to_end_records_into_the_shared_table(tmp_path):
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "steps:\n  - run: sudo apt-get install -y libssl-dev libhwloc-dev cowsay\n"
    )
    found: dict = {}
    findings = scan_ci_configuration(tmp_path, make_recorder(found))
    assert findings.files_read == 1
    # Known packages resolve backwards onto their canonical library...
    assert "openssl" in found and found["openssl"].debian == ("libssl-dev",)
    assert "hwloc" in found
    # ...and an unrecognised one is kept verbatim, marked as declared.
    assert "cowsay" in found
    assert found["cowsay"].declared
    assert found["cowsay"].debian == ("cowsay",)


def test_boilerplate_packages_are_dropped(tmp_path):
    (tmp_path / "Dockerfile").write_text(
        "RUN apt-get install -y ca-certificates curl git wget libssl-dev\n"
    )
    found: dict = {}
    scan_ci_configuration(tmp_path, make_recorder(found))
    assert set(found) == {"openssl"}


def test_declared_packages_are_never_guesses(tmp_path):
    from will_it_riscv.syslibs import database

    (tmp_path / "apt.txt").write_text("some-obscure-lib-dev\n# a comment\n")
    found: dict = {}
    scan_ci_configuration(tmp_path, make_recorder(found))
    assert not database().is_guess(found["some-obscure-lib-dev"])


def test_rpm_version_release_is_stripped():
    f = scan("dnf install -y libcurl-devel-7.61.1-34.el8_10.11")
    assert sorted(f.packages) == ["libcurl-devel"]


def test_soname_suffix_survives_version_stripping():
    """libpng16-16 is a package name; the 16 is not a version to strip."""
    f = scan("apt-get install -y libpng16-16")
    assert sorted(f.packages) == ["libpng16-16"]


def test_urls_on_an_install_line_are_not_packages():
    f = scan("apt-get install -y https://example.invalid/x.deb libssl-dev")
    assert sorted(f.packages) == ["libssl-dev"]


def test_runtime_packages_resolve_to_their_library(tmp_path):
    from will_it_riscv.sdist import make_recorder

    (tmp_path / "Dockerfile").write_text(
        "RUN apt-get install -y libtiff6 libtiff5-dev libpng16-16 libssl3 libcurl4\n"
    )
    found: dict = {}
    scan_ci_configuration(tmp_path, make_recorder(found))
    assert set(found) == {"libtiff", "libpng", "openssl", "libcurl"}


def test_alpine_and_fedora_dev_suffixes_converge(tmp_path):
    from will_it_riscv.sdist import make_recorder

    (tmp_path / "Dockerfile").write_text(
        "RUN apk add brotli-dev openssl-dev\nRUN dnf install -y curl-devel\n"
    )
    found: dict = {}
    scan_ci_configuration(tmp_path, make_recorder(found))
    assert set(found) == {"brotli", "openssl", "libcurl"}


def test_comments_between_line_continuations_are_not_packages():
    """GDAL's Dockerfile comments sit inside a continued RUN command."""
    f = scan(
        "RUN apk add \\\n"
        "        zstd-libs \\\n"
        "    # libturbojpeg.so is not used by GDAL. Only libjpeg.so*\n"
        "    && rm -f /usr/lib/libturbojpeg.so*\n",
        "docker/Dockerfile",
    )
    assert sorted(f.packages) == ["zstd-libs"]


def test_url_fragments_and_shell_expansions_survive_comment_stripping():
    f = scan('RUN apt-get install -y libssl-dev  # see http://example.invalid#notes')
    assert sorted(f.packages) == ["libssl-dev"]


# -- build / test / docs separation -----------------------------------------


def purposes(text: str, path: str = ".github/workflows/ci.yml") -> dict:
    f = scan(text, path)
    return {p: f.purposes.get(p, "build") for p in f.packages}


def test_one_command_can_mix_purposes():
    """git installs compiler, libraries, a web server and a VCS together."""
    got = purposes(
        "apt-get install -y gcc libcurl4-openssl-dev zlib1g-dev "
        "apache2 apache2-http2 cvs subversion valgrind asciidoc"
    )
    assert got["gcc"] == "build"
    assert got["libcurl4-openssl-dev"] == "build"
    assert got["zlib1g-dev"] == "build"
    assert got["apache2"] == "test"
    assert got["apache2-http2"] == "test"      # prefix match
    assert got["cvs"] == "test"
    assert got["subversion"] == "test"
    assert got["valgrind"] == "test"
    assert got["asciidoc"] == "docs"


def test_step_name_classifies_otherwise_unknown_packages():
    """Redis names the step 'testprep' and installs tcl there."""
    got = purposes(
        "steps:\n"
        "  - name: build\n"
        "    run: sudo apt-get install -y libssl-dev\n"
        "  - name: testprep\n"
        "    run: sudo apt-get install -y some-harness-thing\n"
    )
    assert got["libssl-dev"] == "build"
    assert got["some-harness-thing"] == "test"


def test_step_name_cannot_demote_a_known_build_tool():
    """git installs cmake from a step whose name matches 'test'."""
    got = purposes(
        "steps:\n"
        "  - name: run tests\n"
        "    run: sudo apt-get install -y cmake libssl-dev ninja-build\n"
    )
    assert got["cmake"] == "build"
    assert got["libssl-dev"] == "build"
    assert got["ninja-build"] == "build"


def test_documentation_step_context():
    got = purposes(
        "steps:\n"
        "  - name: Build documentation\n"
        "    run: apt-get install -y some-doc-generator\n"
    )
    assert got["some-doc-generator"] == "docs"


def test_filename_is_a_weak_fallback_context():
    assert purposes("run: apt-get install -y mystery-tool",
                    ".github/workflows/codecov.yml")["mystery-tool"] == "test"
    assert purposes("run: apt-get install -y mystery-tool",
                    ".github/workflows/build.yml")["mystery-tool"] == "build"


def test_unknown_package_with_no_context_defaults_to_build():
    """Dropping a real build dependency is worse than keeping a test one."""
    assert purposes("run: apt-get install -y mystery-lib")["mystery-lib"] == "build"


def test_build_wins_when_a_package_appears_in_both(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("find_package(OpenSSL)\n")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "test.yml").write_text(
        "- name: test\n  run: apt-get install -y libssl-dev\n"
    )
    from will_it_riscv.source import inspect_repository

    inspection = inspect_repository(tmp_path, scan_ci=True)
    openssl = next(r for r in inspection.profile.system_requirements
                   if r.name == "openssl")
    assert openssl.purpose == "build"
