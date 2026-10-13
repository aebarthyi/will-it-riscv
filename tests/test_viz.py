"""The dependency graph, drawn."""

import json
import re
import xml.etree.ElementTree as ET

from will_it_riscv import planrun, viz
from will_it_riscv.plan import parse_plan


def graph(*edges, optional=()):
    g = viz.Graph(title="demo", verdict="yes", headline="fine")
    for a, b in edges:
        for key in (a, b):
            g.add(viz.Node(id=key, title=key, subtitle="debian", optional=key in optional))
        g.link(a, b)
    return g


def test_every_edge_runs_left_to_right_and_nodes_never_overlap():
    g = graph(("step", "a"), ("a", "b"), ("b", "c"), ("step", "c"), ("step", "d"))
    viz.layout(g)
    for edge in g.edges:
        source, target = g.nodes[edge.source], g.nodes[edge.target]
        assert source.x + source.w < target.x
        assert edge.points[0] == (source.x + source.w, source.y + source.h / 2)
        assert edge.points[-1] == (target.x, target.y + target.h / 2)
    # step -> c spans two ranks: it passes a waypoint between them.
    assert len(next(e for e in g.edges if (e.source, e.target) == ("step", "c")).points) == 4
    columns: dict = {}
    for node in g.nodes.values():
        columns.setdefault(node.x, []).append(node)
    for column in columns.values():
        column.sort(key=lambda n: n.y)
        for upper, lower in zip(column, column[1:]):
            assert upper.y + upper.h <= lower.y
    assert all(0 <= n.y and n.y + n.h <= g.height for n in g.nodes.values())


def test_a_bootstrap_cycle_is_still_drawn():
    g = graph(("a", "b"), ("b", "a"))
    viz.layout(g)
    assert all(len(e.points) >= 2 for e in g.edges)


def test_the_picture_is_well_formed_svg():
    g = graph(("step", "a & b"), ("step", "<c>"))
    viz.layout(g)
    root = ET.fromstring(viz.render_svg(g))
    assert root.tag.endswith("svg")
    titles = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "a & b" in titles and "<c>" in titles


def test_the_page_is_self_contained_and_its_data_cannot_close_its_script():
    nasty = "x</script><img src=x onerror=alert(1)>"
    g = graph(("step", nasty), ("step", "opt"), optional={"opt"})
    viz.layout(g)
    page = viz.render_html(g)
    assert "<img src=x" not in page
    assert not re.search(r'(src|href)="https?://', page)
    blob = re.search(r'<script type="application/json" id="wir-data">(.*?)</script>', page,
                     re.S).group(1)
    data = json.loads(blob)
    assert data["nodes"][nasty]["title"] == nasty
    assert data["nodes"]["opt"]["optional"] is True
    assert "Optional (1)" in page


def plan(*steps):
    return {"version": 1, "repo": "demo", "steps": list(steps)}


def test_a_plan_draws_its_steps_and_what_each_needs(tmp_path):
    raw = plan(
        {"id": "deps", "kind": "system-packages", "packages": ["libfoo-dev"],
         "evidence": [{"at": "README.md:3", "quote": "apt install libfoo-dev"}]},
        {"id": "cfg", "kind": "cmake-configure", "after": ["deps"], "evidence": []},
    )
    result = planrun.PlanResult(plan=parse_plan(raw), root=tmp_path)
    for step in result.plan.steps:
        result.steps.append(planrun.StepResult(step, "done", "ok"))
    foo = planrun._node(result, "debian:libfoo-dev", "debian", "libfoo-dev")
    foo.required, foo.basis = True, planrun.BY_LIST
    planrun._settle(foo, planrun.BINARY, "in the archive")
    planrun._edge(result, "step:deps", foo.id)
    gone = planrun._node(result, "lib:guile", "library", "Guile")
    gone.carried_on = True
    planrun._settle(gone, planrun.UNKNOWN, "absent")
    planrun._edge(result, "step:cfg", gone.id)
    result.answer = planrun._answer(result)

    g = viz.from_plan(result)
    assert set(g.nodes) == {"step:deps", "step:cfg", "debian:libfoo-dev", "lib:guile"}
    assert g.nodes["debian:libfoo-dev"].tone == "archive"
    assert g.nodes["lib:guile"].optional          # looked for, and done without
    assert {(e.source, e.target, e.kind) for e in g.edges} >= {
        ("step:deps", "step:cfg", "after"), ("step:deps", "debian:libfoo-dev", "needs")}
    chain = g.nodes["debian:libfoo-dev"].info["chain"]
    assert [c[0] for c in chain] == ["step:deps", "debian:libfoo-dev"]
    assert g.nodes["step:deps"].info["evidence"] == [["README.md:3", "apt install libfoo-dev"]]
    path = tmp_path / "deps.html"
    viz.write(g, path)
    assert "libfoo-dev" in path.read_text()


def test_python_packages_that_are_fine_fold_into_one_node(tmp_path):
    """MFC's toolchain brings in a hundred wheels; the one with only source shows."""
    raw = plan({"id": "py", "kind": "python-install", "packages": ["app"], "evidence": []})
    result = planrun.PlanResult(plan=parse_plan(raw), root=tmp_path)
    result.steps.append(planrun.StepResult(result.plan.steps[0], "done", "ok"))
    for name, tier in (("app", planrun.BINARY), ("idna", planrun.BINARY),
                       ("certifi", planrun.BINARY), ("scipy", planrun.SOURCE)):
        node = planrun._node(result, f"pypi:{name}", "pypi", name)
        node.required, node.steps = True, ["py"]
        planrun._settle(node, tier, "")
    planrun._edge(result, "step:py", "pypi:app")
    planrun._edge(result, "pypi:app", "pypi:idna")
    planrun._edge(result, "pypi:app", "pypi:scipy")
    planrun._edge(result, "step:py", "pypi:certifi")
    result.answer = planrun._answer(result)

    g = viz.from_plan(result)
    assert "pypi:app" in g.nodes and "pypi:scipy" in g.nodes     # the way to scipy
    assert "pypi:idna" not in g.nodes and "pypi:certifi" not in g.nodes
    summary = g.nodes["wheels:step:py"]
    assert summary.title == "2 Python packages"
    assert [row[0] for row in summary.info["rows"]] == ["certifi", "idna"]
    assert ("step:py", "wheels:step:py") in {(e.source, e.target) for e in g.edges}
