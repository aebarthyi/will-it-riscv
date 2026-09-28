"""The executor sets each stated package's required/optional mark, where it settled it."""

from will_it_riscv.plan import parse_plan
from will_it_riscv.relabel import OPTIONAL_BECAUSE, Verdicts, relabel


def plan(steps):
    return parse_plan({"version": 1, "repo": "demo", "steps": steps})


DEPS = {"id": "deps", "kind": "system-packages", "evidence": ["README.md:3"],
        "packages": ["libssl-dev", "libfoo-dev", "zip"]}
EXTRAS = {"id": "extras", "kind": "system-packages", "evidence": ["README.md:5"],
          "optional": True, "enabled_by": "-DWITH_BAR=ON", "packages": ["libbar-dev"]}
CONFIGURE = {"id": "configure", "kind": "cmake-configure", "evidence": ["README.md:7"],
             "after": ["deps"]}


def test_marks_follow_what_the_configure_proved_and_nothing_else_moves():
    found = Verdicts(
        optional={"libssl-dev": ("OpenSSL", "absent, and the configure carried on")},
        required={"libbar-dev": ("Bar", "stopped the configure")},
    )
    relabelled, changes = relabel(plan([DEPS, EXTRAS, CONFIGURE]), found)
    steps = {s.id: s for s in relabelled.steps}
    assert steps["deps"].packages == ["libfoo-dev", "zip", "libbar-dev"]
    assert not steps["deps"].optional
    moved = steps["deps-optional"]
    assert moved.optional and moved.packages == ["libssl-dev"]
    assert moved.enabled_by == OPTIONAL_BECAUSE and moved.evidence == steps["deps"].evidence[:1]
    assert "extras" not in steps  # its only package became required
    assert {(c["package"], c["to"]) for c in changes} == {
        ("libssl-dev", "optional"), ("libbar-dev", "required")}


def test_a_step_emptied_hands_its_place_in_the_order_on():
    found = Verdicts(optional={p: ("X", "carried on") for p in DEPS["packages"]})
    relabelled, _ = relabel(plan([DEPS, CONFIGURE]), found)
    steps = {s.id: s for s in relabelled.steps}
    assert "deps" not in steps and steps["deps-optional"].optional
    assert steps["configure"].after == []


def test_a_package_the_configure_never_asked_about_keeps_its_mark():
    relabelled, changes = relabel(plan([DEPS, EXTRAS, CONFIGURE]), Verdicts())
    assert changes == []
    assert [s.packages for s in relabelled.steps if s.kind == "system-packages"] == [
        DEPS["packages"], EXTRAS["packages"]]
