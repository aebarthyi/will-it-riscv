"""The CMake condition walk: what a build with no -D flags actually reaches."""

import pytest

from will_it_riscv.cmake_conditions import (
    Symbols,
    Tri,
    analyze,
    collect_symbols,
    evaluate_condition,
    iter_commands,
)


def findings(text, symbols=None):
    return {f.name: f for f in analyze(text, symbols).findings}


# -- the tokenizer ----------------------------------------------------------


def test_commands_may_span_lines():
    text = "find_package(\n  LLVM\n  CONFIG\n)\nfind_package(Boost)\n"
    assert [c.name for c in iter_commands(text)] == ["find_package", "find_package"]
    assert "LLVM" in list(iter_commands(text))[0].args


def test_nested_parentheses_are_balanced():
    text = 'if(NOT (A AND B))\nfind_package(Zlib)\nendif()\n'
    assert [c.name for c in iter_commands(text)] == ["if", "find_package", "endif"]


def test_comments_are_removed_before_parsing():
    text = "# find_package(NotReal)\nfind_package(Real)\n"
    assert [c.args.strip() for c in iter_commands(text)] == ["Real"]


# -- three-valued logic -----------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expect_and, expect_or",
    [
        (Tri.TRUE, Tri.TRUE, Tri.TRUE, Tri.TRUE),
        (Tri.TRUE, Tri.FALSE, Tri.FALSE, Tri.TRUE),
        (Tri.FALSE, Tri.FALSE, Tri.FALSE, Tri.FALSE),
        (Tri.UNKNOWN, Tri.TRUE, Tri.UNKNOWN, Tri.TRUE),
        (Tri.UNKNOWN, Tri.FALSE, Tri.FALSE, Tri.UNKNOWN),
    ],
)
def test_tri_logic(a, b, expect_and, expect_or):
    assert (a & b) is expect_and
    assert (a | b) is expect_or


def test_tri_negation():
    assert ~Tri.TRUE is Tri.FALSE
    assert ~Tri.UNKNOWN is Tri.UNKNOWN


# -- option defaults --------------------------------------------------------


def test_option_default_is_off_when_unstated():
    """CMake's option() defaults to OFF."""
    symbols = collect_symbols(iter(['option(WITH_FOO "docs")\n']))
    assert symbols.evaluate("WITH_FOO")[0] is Tri.FALSE


def test_option_explicit_default():
    symbols = collect_symbols(iter(['option(WITH_FOO "docs" ON)\n']))
    assert symbols.evaluate("WITH_FOO")[0] is Tri.TRUE


def test_set_cache_default():
    symbols = collect_symbols(iter(['set(WITH_BAR OFF CACHE BOOL "docs")\n']))
    assert symbols.evaluate("WITH_BAR")[0] is Tri.FALSE


def test_autodetected_default_is_off():
    """AdaptiveCpp: set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL ...).

    The switch defaults to whatever autodetection found. A default build on
    a machine without CUDA finds nothing, so it is off.
    """
    symbols = collect_symbols(
        iter(['set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "docs")\n'])
    )
    value, gate = symbols.evaluate("WITH_CUDA_BACKEND")
    assert value is Tri.FALSE
    assert "autodetected" in gate


def test_unknown_variable_stays_unknown():
    """Guessing a real dependency away is the error that matters."""
    assert Symbols().evaluate("SOMETHING_ELSE")[0] is Tri.UNKNOWN


def test_found_variable_for_an_impossible_package_is_false():
    """if(CUDA_FOUND) on a target with no CUDA build at all."""
    value, gate = Symbols().evaluate("CUDA_FOUND")
    assert value is Tri.FALSE
    assert "architecture" in gate


# -- conditions -------------------------------------------------------------


def test_not_and_or():
    symbols = collect_symbols(
        iter(['option(A "" ON)\noption(B "" OFF)\n'])
    )
    assert evaluate_condition("A", symbols)[0] is Tri.TRUE
    assert evaluate_condition("NOT A", symbols)[0] is Tri.FALSE
    assert evaluate_condition("A AND B", symbols)[0] is Tri.FALSE
    assert evaluate_condition("A OR B", symbols)[0] is Tri.TRUE
    assert evaluate_condition("NOT B", symbols)[0] is Tri.TRUE


def test_platform_variables_resolve_for_a_linux_target():
    assert evaluate_condition("WIN32", Symbols())[0] is Tri.FALSE
    assert evaluate_condition("APPLE", Symbols())[0] is Tri.FALSE
    assert evaluate_condition("MSVC", Symbols())[0] is Tri.FALSE
    assert evaluate_condition("UNIX", Symbols())[0] is Tri.TRUE


def test_comparisons_are_not_guessed_at():
    for condition in (
        'CMAKE_SYSTEM_NAME STREQUAL "Linux"',
        "DEFINED WITH_FOO",
        "EXISTS /usr/lib",
        "LLVM_VERSION VERSION_GREATER 14",
    ):
        assert evaluate_condition(condition, Symbols())[0] is Tri.UNKNOWN


# -- reachability -----------------------------------------------------------


def test_top_level_find_is_required():
    found = findings("find_package(ZLIB REQUIRED)\n")
    assert found["ZLIB"].reachable is Tri.TRUE
    assert not found["ZLIB"].optional


def test_find_behind_an_off_option_is_optional():
    text = (
        'option(WITH_CUDA "Build CUDA support" OFF)\n'
        "if(WITH_CUDA)\n"
        "  find_package(CUDAToolkit REQUIRED)\n"
        "endif()\n"
        "find_package(ZLIB REQUIRED)\n"
    )
    found = findings(text, collect_symbols(iter([text])))
    assert found["CUDAToolkit"].optional
    assert found["CUDAToolkit"].gate == "WITH_CUDA"
    assert not found["ZLIB"].optional


def test_find_behind_an_on_option_is_required():
    text = (
        'option(WITH_CORE "" ON)\n'
        "if(WITH_CORE)\n  find_package(Boost)\nendif()\n"
    )
    assert not findings(text, collect_symbols(iter([text])))["Boost"].optional


def test_else_branch_of_a_false_condition_is_reachable():
    text = (
        'option(WITH_BUNDLED "" OFF)\n'
        "if(WITH_BUNDLED)\n  find_package(Vendored)\n"
        "else()\n  find_package(SystemLib)\nendif()\n"
    )
    found = findings(text, collect_symbols(iter([text])))
    assert found["Vendored"].optional
    assert not found["SystemLib"].optional


def test_elseif_chain():
    text = (
        'option(A "" OFF)\noption(B "" ON)\n'
        "if(A)\n  find_package(First)\n"
        "elseif(B)\n  find_package(Second)\n"
        "else()\n  find_package(Third)\nendif()\n"
    )
    found = findings(text, collect_symbols(iter([text])))
    assert found["First"].optional
    assert not found["Second"].optional
    assert found["Third"].optional


def test_nested_conditions_multiply():
    text = (
        'option(OUTER "" ON)\noption(INNER "" OFF)\n'
        "if(OUTER)\n  if(INNER)\n    find_package(Deep)\n  endif()\n"
        "  find_package(Shallow)\nendif()\n"
    )
    found = findings(text, collect_symbols(iter([text])))
    assert found["Deep"].optional
    assert not found["Shallow"].optional


def test_unknown_condition_keeps_the_dependency():
    text = 'if(SOMETHING_WE_CANNOT_EVALUATE)\n  find_package(Maybe)\nendif()\n'
    found = findings(text)
    assert found["Maybe"].reachable is Tri.UNKNOWN
    assert not found["Maybe"].optional     # kept, because we do not know


def test_windows_only_branch_is_optional_on_a_linux_target():
    text = "if(WIN32)\n  find_package(WindowsThing)\nendif()\n"
    found = findings(text)
    assert found["WindowsThing"].optional
    assert "not this platform" in found["WindowsThing"].gate


def test_quiet_without_required_is_a_probe():
    """find_package(CUDA QUIET) asks whether CUDA is there, it does not need it."""
    found = findings("find_package(CUDA QUIET)\n")
    assert found["CUDA"].optional


def test_find_library_names_keyword_is_not_a_library():
    found = findings('find_library(ZLIB_LIB NAMES z zlib PATHS /usr/lib)\n')
    assert "NAMES" not in found
    assert {"z", "zlib"} <= set(found)


def test_pkg_check_modules_names_the_modules_not_the_variable():
    found = findings("pkg_check_modules(PC_FOO REQUIRED libxml-2.0)\n")
    assert "PC_FOO" not in found
    assert "libxml-2.0" in found


def test_options_are_collected_across_files():
    """A switch declared in one file routinely gates a find in another."""
    declaring = 'option(WITH_EXTRA "" OFF)\n'
    using = "if(WITH_EXTRA)\n  find_package(Extra)\nendif()\n"
    symbols = collect_symbols(iter([declaring, using]))
    assert findings(using, symbols)["Extra"].optional
