"""interactive.html: payload content, proposal overlay, trace alignment, escaping.

CPU only, on the synthetic ``archfixture`` package of ``conftest.py`` (and two
small extra trees); the analysed code is never imported.
"""
from __future__ import annotations

import json
import re

import pytest

from tools.architecture import generate, interactive
from tools.architecture.coverage import CoverageError
from tools.architecture.generate import Options, render

FIXTURE_ROOTS = {"entry": ["archfixture.a.make_child"], "uses": ["archfixture.a.uses"]}
PAYLOAD = re.compile(r'<script id="architecture-data" type="application/json">(.*?)</script>',
                     re.S)


def payload(html: str) -> dict:
    return json.loads(PAYLOAD.search(html).group(1))


def _opts(**kw):
    base = dict(src=".", package="archfixture", call_roots=FIXTURE_ROOTS, title="fixture")
    base.update(kw)
    return Options(**base)


@pytest.fixture(scope="module")
def atlas(fixture_src):
    return payload(render(_opts(), root=fixture_src)["interactive.html"])


def _edge(data, source, target, kind):
    of = {n: i for i, n in enumerate(data["ids"])}
    return data["calls"].index([of[source], of[target], kind])


def test_page_is_generated_offline_and_deterministic(fixture_src):
    first = render(_opts(), root=fixture_src)["interactive.html"]
    assert first == render(_opts(), root=fixture_src)["interactive.html"]
    assert "<script src" not in first and "<link" not in first
    assert "http://" not in first.split("<script id=")[0]  # no remote resource in the shell
    assert payload(first)["schema"] == interactive.ATLAS_SCHEMA


def test_every_static_relationship_is_present(atlas, project):
    from tools.architecture.calls import build_call_graph
    from tools.architecture.imports import build_import_graph
    calls, _ = build_call_graph(project)
    modules = build_import_graph(project)
    ids = atlas["ids"]
    assert {(ids[s], ids[t], k) for s, t, k in atlas["calls"]} == set(calls.edges)
    assert {(ids[s], ids[t]) for s, t, _ in atlas["imports"]} == {
        (s, t) for s, t, _ in modules.edges}


def test_constructor_dispatch_and_super_are_classified(atlas):
    ids = atlas["ids"]
    kinds = {(ids[s], ids[t]): k for s, t, k in atlas["calls"]}
    assert kinds[("archfixture.a.make_child", "archfixture.child.Child")] == "instantiate"
    assert kinds[("archfixture.child.Child", "archfixture.base.Base.__init__")] == "constructor"
    assert kinds[("archfixture.child.Child.step", "archfixture.base.Base.step")] == "super"
    # self.step() in Base.run: lookup target and the subclass redefinition.
    i = _edge(atlas, "archfixture.base.Base.run", "archfixture.base.Base.step", "self")
    j = _edge(atlas, "archfixture.base.Base.run", "archfixture.child.Child.step", "override")
    (site,) = atlas["call_sites"][i]
    assert site[4] == [ids.index("archfixture.child.Child.step")]  # possible targets at the site
    assert atlas["call_sites"][j]  # the override edge points at the same call expression
    assert atlas["meta"]["evidence"]["override"][0] == "possible-dispatch"
    assert atlas["meta"]["evidence"]["self"][0] == "receiver-lookup"
    assert atlas["meta"]["evidence"]["constructor"][0] == "structural"


def test_call_sites_bind_arguments_and_show_annotations(make_tree, tmp_path):
    src = make_tree(tmp_path, {
        "pk/__init__.py": "",
        "pk/m.py": """
            class Grid:
                def __init__(self, nlat: int, nlon: int = 4):
                    self.n = nlat * nlon


            def area(radius: float, *, scale: float = 1.0) -> float:
                return radius * scale


            def caller(r):
                g = Grid(8, nlon=16)
                area(r, scale=2.0)
                area(r)
                area(r)
                return g
        """})
    data = payload(render(Options(src=".", package="pk", call_roots={"c": ["pk.m.caller"]},
                                  max_sites=2), root=src)["interactive.html"])
    i = _edge(data, "pk.m.caller", "pk.m.Grid", "instantiate")
    (site,) = data["call_sites"][i]
    assert site[3] == [["nlat", "8"], ["nlon", "16"]]  # self skipped for the constructor
    j = _edge(data, "pk.m.caller", "pk.m.area", "call")
    assert len(data["call_sites"][j]) == 2 and data["call_omitted"][j] == 1
    assert data["call_site_totals"][j] == 3
    assert data["call_sites"][j][0][3] == [["radius", "r"], ["scale", "2.0"]]
    sig = data["signatures"][str(data["ids"].index("pk.m.area"))]
    assert sig[1] == [["pos", "radius", "float", None], ["kw", "scale", "float", "1.0"]]
    assert sig[2] == "float"
    path = "pk/m.py"
    assert data["lines"][path][str(site[0])].strip() == "g = Grid(8, nlon=16)"


def test_import_sites_carry_symbols_aliases_and_scopes(atlas):
    ids = atlas["ids"]

    def sites(source, target):
        k = [i for i, (s, t, _) in enumerate(atlas["imports"])
             if ids[s] == source and ids[t] == target][0]
        return atlas["import_sites"][k], atlas["imports"][k][2]
    alias, _ = sites("archfixture.a", "archfixture.ns.tool")
    assert alias[0][2] == ["helper as h"] and alias[0][1] == "module"
    relative, _ = sites("archfixture.a", "archfixture.c")
    assert relative[0][2] == ["c"]
    child, scopes = sites("archfixture.a", "archfixture.child")
    assert {s[1] for s in child} == {"type_checking", "function"}
    assert "type_checking" in scopes
    local, _ = sites("archfixture.a", "archfixture.d")
    assert local[0][1] == "function" and local[0][2] == ["deep"]


def test_unresolved_external_and_builtin_sites_stay_visible(atlas):
    i = str(atlas["ids"].index("archfixture.a.uses"))
    counts, examples = atlas["unresolved"][i]
    assert counts["builtin"] >= 1 and counts["external"] >= 1 and counts["unresolved"] >= 1
    assert any("obj.anything" in e[2] for e in examples)


def test_fields_keep_source_comments_verbatim(make_tree, tmp_path):
    src = make_tree(tmp_path, {
        "pk/__init__.py": "",
        "pk/body.py": """
            from dataclasses import dataclass


            @dataclass(frozen=True)
            class Body:
                radius: float            # m
                gravity: float           # m s^-2 at the reference level
                label: str = "a # b"
        """})
    data = payload(render(Options(src=".", package="pk", call_roots={}),
                          root=src)["interactive.html"])
    rows = data["fields"][str(data["ids"].index("pk.body.Body"))]
    assert [r[0] for r in rows] == ["radius", "gravity", "label"]
    assert rows[0][3] == "m" and rows[1][3] == "m s^-2 at the reference level"
    assert rows[2][3] is None  # '#' inside a string is not a comment


def test_source_text_cannot_escape_the_payload(make_tree, tmp_path):
    src = make_tree(tmp_path, {
        "pk/__init__.py": "",
        "pk/evil.py": '''
            def f():
                """</script><img src=x onerror=alert(1)> &  """
                return "</script>"
        '''})
    html = render(Options(src=".", package="pk", call_roots={}), root=src)["interactive.html"]
    body = PAYLOAD.search(html).group(1)
    assert "</script" not in body and "<img" not in body
    assert " " not in body
    data = json.loads(body)
    assert any("</script>" in t for t in data["lines"]["pk/evil.py"].values())


def test_proposal_assignment_most_specific_exact_and_symbol_rules():
    ids = ["tropoi", "tropoi.physics", "tropoi.physics.planets",
           "tropoi.physics.planets.Body", "tropoi.physics.planets.GridSpec", "tropoi.run.x"]
    proposal = {"groups": {"physics": {}, "world": {}, "setup": {}, "cli": {}},
                "rules": [{"prefix": "tropoi.physics", "group": "physics"},
                          {"prefix": "tropoi.physics.planets", "group": "world"},
                          {"prefix": "tropoi.physics.planets.GridSpec", "group": "setup"},
                          {"prefix": "tropoi", "exact": True, "group": "cli"}]}
    groups, used = interactive.assign(ids, proposal)
    assert groups == ["cli", "physics", "world", "world", "setup", "unassigned"]
    assert used == [3, 0, 1, 1, 2, -1]


def test_proposal_directions_exceptions_and_cycles():
    ids = ["a.x", "b.y", "c.z"]
    groups = ["top", "mid", "low"]
    proposal = {"layers": [["top"], ["mid"], ["low"]],
                "exceptions": [{"source": "c.z", "target": "b.y", "reason": "adapter"}]}
    imports = [[0, 1, "module"], [1, 2, "module"], [2, 1, "module"], [2, 0, "type_checking"]]
    calls = [[0, 2, "call"], [2, 0, "constructor"]]
    a = interactive.analyse_proposal(ids, groups, imports, calls, proposal)
    cells = a["imports"]["cells"]
    assert cells["top|mid"]["direction"] == "downward"
    assert cells["low|mid"]["direction"] == "upward" and cells["low|mid"]["excepted"] == 1
    assert cells["low|top"]["runtime_edges"] == 0  # TYPE_CHECKING only
    assert a["imports"]["group_cycles"] == [["low", "mid"]]
    assert a["imports"]["by_direction"] == {"downward": 2, "upward": 1}
    assert a["exceptions_used"] == [1]
    assert "low|top" not in a["calls"]["cells"]  # constructor relations are structural


def test_proposal_file_is_validated_and_recorded(fixture_src, tmp_path):
    good = {"schema": 2, "status": "draft", "groups": {"core": {"role": "r"}, "app": {}},
            "layers": [["app"], ["core"]],
            "rules": [{"prefix": "archfixture", "group": "core"},
                      {"prefix": "archfixture.a", "group": "app"},
                      {"prefix": "archfixture.gone", "group": "app"}]}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    files = render(_opts(proposal=str(path)), root=fixture_src)
    data = payload(files["interactive.html"])
    assert data["proposal"]["source"]["sha256"] == interactive._sha256_text(path)
    summary = json.loads(files["metrics.json"])["atlas"]["proposal"]
    assert summary["rules_matching_nothing"] == ["archfixture.gone"]
    assert summary["members"]["app"] > 0
    for bad in ({**good, "schema": 1}, {**good, "layers": [["nope"]]},
                {**good, "rules": [{"prefix": "x", "group": "undeclared"}]}):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(CoverageError, match="atlas input"):
            render(_opts(proposal=str(path)), root=fixture_src)


def _trace(edges, **kw):
    return {"schema": 2, "id": "t", "scenario": "t", "edges": edges,
            "provenance": {"code": {"commit": "abc", "source_sha256": "x"}}, **kw}


def test_runtime_observations_are_aligned_with_static_edges(fixture_src, tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    rows = [
        {"source": "archfixture.a.uses", "target": "archfixture.c.cfun", "count": 2},
        {"source": "archfixture.a.make_child", "target": "archfixture.base.Base.__init__",
         "count": 1},
        {"source": "archfixture.base.Base.run", "target": "archfixture.child.Child.step",
         "count": 1},
        {"source": "archfixture.a.uses", "target": "archfixture.d.deep", "count": 1},
        {"source": "archfixture.a.make_child.<locals>.inner", "target": "archfixture.c.cfun",
         "count": 1},
        {"source": "archfixture.gone.f", "target": "archfixture.c.cfun", "count": 1},
    ]
    (evidence / "t.json").write_text(json.dumps(_trace(rows)), encoding="utf-8")
    files = render(_opts(evidence=str(evidence)), root=fixture_src)
    (trace,) = payload(files["interactive.html"])["traces"]
    cat = [e["category"] for e in trace["edges"]]
    assert cat == ["corroborated", "corroborated", "corroborated", "runtime-only",
                   "runtime-only", "unmapped"]
    assert trace["edges"][1]["static"][0][0] == "instantiate+constructor"
    assert trace["edges"][2]["static"][0][0] == "override"
    assert trace["edges"][4]["map"][0] == "enclosing-function"
    assert trace["file"]["path"].endswith("t.json")
    meta = json.loads(files["metrics.json"])["atlas"]["runtime_traces"][0]
    assert meta["commit"] == "abc" and meta["alignment"]["corroborated"] == 3
    (evidence / "t.json").write_text(json.dumps({**_trace(rows), "schema": 1}),
                                     encoding="utf-8")
    with pytest.raises(CoverageError, match="trace schema"):
        render(_opts(evidence=str(evidence)), root=fixture_src)


def test_generator_records_the_atlas_and_links_it(fixture_src):
    files = render(_opts(), root=fixture_src)
    assert "interactive.html" in files
    assert "interactive.html" in files["index.md"]
    assert "`interactive.html`" in files["README.md"]
    summary = json.loads(files["metrics.json"])["atlas"]
    assert summary["call_edges"] > 0 and summary["max_sites_per_edge"] == 8
    assert summary["proposal"] is None and summary["runtime_traces"] == []


def test_cli_optional_inputs(monkeypatch):
    with pytest.raises(SystemExit):
        generate.main(["--proposal", "does/not/exist.json", "--check"])
    with pytest.raises(SystemExit):
        generate.main(["--max-sites", "0", "--check"])
