"""The dependency graph a pseudobuild observed, and the answer it gives."""

from types import SimpleNamespace

from will_it_riscv import graph
from will_it_riscv.pseudobuild import Probe, PseudoBuild


class FakeDistro:
    """Just enough of DistroIndex: a label, an arch and a set of names."""

    def __init__(self, names):
        self.names_ = set(names)
        self.spec = SimpleNamespace(label="Debian 13 (trixie)", id="debian:trixie")
        self.arch = "riscv64"
        self.available = True

    def first_available(self, candidates):
        return next((c for c in candidates if c in self.names_), None)

    def has(self, package):
        return package in self.names_


def probe(name, command="find_package", **kwargs):
    return name.lower(), Probe(name=name, command=command, **kwargs)


def pseudobuild(**kwargs):
    kwargs.setdefault("platform", "linux/riscv64")
    kwargs.setdefault("completed", True)
    probes = dict(kwargs.pop("probes", []))
    return PseudoBuild(probes=probes, **kwargs)


def test_a_blocker_is_required_and_says_how_it_was_shown():
    result = pseudobuild(
        blockers=["PROJ", "OpenMP"],
        experiments=[("OpenMP", True)],
        probes=[probe("PROJ", site="CMakeLists.txt:122"), probe("OpenMP")],
    )
    g = graph.build(result, "gdal")
    proj, openmp = g.required()
    assert (proj.name, openmp.name) == ("PROJ", "OpenMP")   # in the order demanded
    assert proj.proof == "stopped the configure"
    assert "experiment" in openmp.proof
    assert proj.site == "CMakeLists.txt:122"


def test_confined_and_completed_every_other_probe_is_optional():
    """Nothing real was in the sysroot, so whatever it asked for was absent."""
    result = pseudobuild(
        soft_misses={"MySQL"},
        probes=[probe("MySQL"), probe("libcurl", command="pkg_check_modules")],
    )
    g = graph.build(result, "x")
    statuses = {n.name: (n.status, n.proof) for n in g.nodes.values()}
    assert statuses["MySQL"] == (graph.OPTIONAL, "absent, and the configure carried on")
    assert statuses["libcurl"][0] == graph.OPTIONAL
    assert "completed anyway" in statuses["libcurl"][1]


def test_an_unfinished_configure_leaves_its_probes_open():
    result = pseudobuild(completed=False, probes=[probe("GEOS")])
    assert graph.build(result, "x").nodes["geos"].status == graph.UNKNOWN


def test_a_found_program_is_a_host_tool_not_a_library():
    result = pseudobuild(
        found={"Perl"}, found_at={"Perl": "/usr/bin/perl"}, probes=[probe("Perl")]
    )
    perl = graph.build(result, "curl").nodes["perl"]
    assert (perl.status, perl.kind) == (graph.PRESENT, "tool")
    assert perl.debian == ("perl",)       # the program's package, not libperl-dev


def test_a_found_library_under_confinement_is_unverified():
    """A compile-only check claims BLAS; nothing real was there to find."""
    result = pseudobuild(found={"BLAS"}, probes=[probe("BLAS")])
    assert graph.build(result, "x").nodes["openblas"].status == graph.UNVERIFIED


def test_required_without_being_a_blocker_is_only_a_claim():
    """REQUIRED on a tool the host happened to have demonstrates nothing."""
    result = pseudobuild(
        found={"SWIG"}, found_at={"SWIG": "/opt/homebrew/bin/swig"},
        probes=[probe("SWIG", required=True)],
    )
    swig = graph.build(result, "gdal").nodes["swig"]
    assert swig.status == graph.PRESENT and swig.asked_required


def test_a_find_module_asking_for_another_package_is_an_edge():
    result = pseudobuild(
        probes=[probe("CURL"), probe("ZLIB", parent="CURL")],
    )
    g = graph.build(result, "x")
    assert ("libcurl", "zlib") in g.edges
    assert (graph.ROOT, "libcurl") in g.edges


def test_alternative_spellings_fold_into_the_package_searching_for_them():
    """ssleay32MD inside FindOpenSSL is OpenSSL, not a dependency of its own."""
    result = pseudobuild(
        probes=[
            probe("OpenSSL"),
            probe("ssleay32MD", command="find_library", parent="OpenSSL"),
            probe("ptcblas_r", command="find_library"),
            probe("gstreamer-app-1.0", command="pkg_check_modules"),
        ],
    )
    names = {n.name for n in graph.build(result, "x").nodes.values()}
    assert "OpenSSL" in names
    assert "ssleay32MD" not in names
    assert "ptcblas_r" not in names          # an unknown file name, not a package
    assert "gstreamer-app-1.0" in names      # a pkg-config module is a package's own name


def test_a_language_component_folds_into_its_package():
    result = pseudobuild(
        blockers=["OpenMP"], probes=[probe("OpenMP"), probe("OpenMP_CXX")]
    )
    assert list(graph.build(result, "x").nodes) == ["openmp"]


def test_what_a_probe_was_hidden_behind_is_recorded():
    result = pseudobuild(
        blockers=["PROJ"],
        round_blockers=["PROJ", None],
        probes=[probe("PROJ"), probe("GEOS", round=2)],
    )
    assert graph.build(result, "gdal").nodes["geos"].behind == "PROJ"


# -- will it riscv? ---------------------------------------------------------


def test_yes_when_the_archive_has_every_hard_requirement():
    result = pseudobuild(blockers=["PROJ"], probes=[probe("PROJ")])
    answer = graph.build(result, "gdal", FakeDistro({"libproj-dev"})).answer
    assert answer.verdict == "yes"
    assert answer.install == ["libproj-dev"]


def test_no_when_a_hard_requirement_has_no_package_for_the_target():
    result = pseudobuild(blockers=["PROJ"], probes=[probe("PROJ")])
    answer = graph.build(result, "gdal", FakeDistro(set())).answer
    assert answer.verdict == "no"
    assert answer.missing == ["PROJ"]


def test_openmp_comes_with_the_compiler():
    result = pseudobuild(blockers=["OpenMP"], probes=[probe("OpenMP")])
    answer = graph.build(result, "gromacs", FakeDistro(set())).answer
    assert answer.verdict == "yes"
    assert any("compiler" in line for line in answer.toolchain)


def test_a_guessed_name_the_archive_has_is_installed_and_owned_up_to():
    result = pseudobuild(blockers=["Wirtest"], probes=[probe("Wirtest")])
    answer = graph.build(result, "x", FakeDistro({"libwirtest-dev"})).answer
    assert answer.verdict == "yes"
    assert answer.install == ["libwirtest-dev"]
    assert any("matched by name" in note for note in answer.notes)


def test_a_guessed_name_the_archive_lacks_is_not_called_missing():
    """The guess being wrong says nothing about the target."""
    result = pseudobuild(blockers=["Wirtest"], probes=[probe("Wirtest")])
    g = graph.build(result, "x", FakeDistro(set()))
    assert g.answer.verdict == "probably"
    assert g.answer.missing == [] and g.answer.unmapped == ["Wirtest"]


def test_unknown_when_the_configure_never_finished():
    result = pseudobuild(completed=False, rounds=2, error="CMake Error: boom")
    answer = graph.build(result, "x", FakeDistro(set())).answer
    assert answer.verdict == "unknown"
    assert "boom" in answer.headline


def test_only_probably_when_the_host_answered_instead_of_the_target():
    result = pseudobuild(platform="host", blockers=["PROJ"], probes=[probe("PROJ")])
    answer = graph.build(result, "x", FakeDistro({"libproj-dev"})).answer
    assert answer.verdict == "probably"
    assert "host" in answer.headline


def test_required_tools_go_on_the_install_line():
    result = pseudobuild(
        found={"SWIG"}, found_at={"SWIG": "/usr/bin/swig"},
        probes=[probe("SWIG", required=True)],
    )
    answer = graph.build(result, "gdal", FakeDistro({"swig"})).answer
    assert answer.install == ["swig"]


# -- output -----------------------------------------------------------------


def test_dot_output_is_valid_and_shows_the_rounds():
    result = pseudobuild(
        blockers=["PROJ"],
        round_blockers=["PROJ", None],
        probes=[probe("PROJ"), probe("GEOS", round=2)],
    )
    dot = graph.to_dot(graph.build(result, "gdal", FakeDistro({"libproj-dev"})))
    assert dot.startswith('digraph "gdal" {') and dot.rstrip().endswith("}")
    assert "cluster_round_2" in dot and "reached once PROJ existed" in dot
    assert '"PROJ\\nlibproj-dev"' in dot         # a real DOT line break, not "\\\\n"
    assert '"__project__" -> "proj"' in dot


def test_the_graph_serialises():
    result = pseudobuild(blockers=["PROJ"], probes=[probe("PROJ"), probe("GEOS")])
    data = graph.to_dict(graph.build(result, "gdal", FakeDistro({"libproj-dev"})))
    assert data["answer"]["verdict"] == "yes"
    assert data["hard_requirements"] == ["PROJ"]
    assert {"from": None, "to": "proj"} in data["edges"]
    assert {n["id"] for n in data["nodes"]} == {"proj", "geos"}


def test_a_miss_before_a_configure_that_died_is_left_open():
    result = pseudobuild(
        completed=False, soft_misses={"LLVM"}, probes=[probe("LLVM")],
        experiments=[("LLVM", False)],
    )
    llvm = graph.build(result, "adaptivecpp").nodes["llvm"]
    assert llvm.status == graph.UNKNOWN
