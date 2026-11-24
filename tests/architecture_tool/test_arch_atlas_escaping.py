"""Issue #62: data-derived values reach ``innerHTML`` only through escaping.

Three layers:

* a static scan of the template: no interpolation prints a raw embedded-data
  path unless it is a reviewed JavaScript-computed value (CPU);
* the generator rejects runtime traces whose printed fields have the wrong
  type (CPU);
* a hostile payload, with markup in every numeric field the page prints, is
  rendered in a real browser across every tab and inspector: nothing executes,
  no element is injected, the markup is shown as text (skipped without a
  browser, like ``test_arch_atlas_browser.py``).
"""
from __future__ import annotations

import json
from collections import Counter

import pytest

from tools.architecture import interactive
from tools.architecture.generate import Options, render
from tools.browser_cdp import OfflinePage, find_browser

from _template_scan import interpolations, raw_data_reads

#: Helpers whose output is escaped HTML (or built only from escaped parts).
SAFE = {"esc", "num", "ev", "nodeBtn", "srcLines", "describe", "boundaryHtml", "crossBadge",
        "legend", "intro", "relRow", "egoSvg", "row", "runtimeForNode"}

#: Interpolations that read a data-named value without ``num``/``esc``, each
#: reviewed, with its occurrence count so that a new raw read of the same
#: shape still fails. Keep the reason next to every entry.
REVIEWED = Counter({
    # JavaScript-computed integers: loop counters, indices, SVG geometry, tallies.
    "n": 2,              # srcLines line counter; <option>.textContent label
    "x": 1,              # egoSvg box x (8 or W-BW-8)
    "x+6": 1,            # egoSvg text x
    "c": 3,              # child index from the loop; per-module tally; constant category name
    "c.n": 3,            # boundary-matrix cell tally
    "c.excepted?', '+c.excepted+' under declared exceptions':''": 1,  # same tally
    "c.excepted&&c.excepted<c.n?'<sup>'+c.excepted+'x</sup>':''": 1,  # same tally
    "R.leaving": 1,      # reach(): sum of array lengths
    "r.rel": 1,          # constant 'call' | 'import' | 'obs'
    "r.rel==='obs'?r.ti+':'+r.ei:r.k": 1,  # forEach indices
    # Label text handed to ev(), which escapes it.
    "b": 1,              # crossBadge label `${a}→${b}`
    "v": 3,              # `${k} ${v}` labels of unresolved counts and by-direction tallies
    "e.count": 2,        # `observed ×${e.count}` labels, rendered via ev()
    # Plain text escaped by the caller.
    "s[0]": 1,           # sigText(), always passed through esc()
    # Not HTML: S.cell comparison, URL hash and textContent.
    "S.cell===a+'|'+b?'sel':''": 1,
    "s.ti": 1, "s.ei": 1, "s.t": 1, "s.k": 1,   # writeHash (history.replaceState)
    "e.kind": 1, "e.nodes": 1,                  # entry <option>.textContent
    "D.meta.tool_version": 1,                   # header textContent
    "P.source?P.source.path+' ('+(P.status||'')+')':'none'": 1,  # header textContent
})


def _script() -> str:
    text = interactive.TEMPLATE.read_text(encoding="utf-8")
    return text[text.index("<script>\n'use strict'"):]


def test_template_has_no_unreviewed_raw_data_interpolation():
    found = Counter(e for e in interpolations(_script()) if raw_data_reads(e, SAFE))
    assert found == REVIEWED, ("new or removed raw interpolation of embedded data; wrap it "
                               f"in num() or esc(), or review it here: {found - REVIEWED} "
                               f"/ {REVIEWED - found}")


@pytest.mark.parametrize("expr", [
    "e.nodes", "D.meta.max_sites", "st[0]", "b.call_index", "D.import_omitted[k]",
    "v.omitted", "e.truncated_nodes?' · '+e.truncated_nodes+' cut':''",
])
# Known blind spot: a data field *named* ``length`` (describe()'s ``v.length``)
# looks like Array.length to the scan; the browser test below poisons it.
def test_scan_flags_the_issue_62_shapes(expr):
    assert raw_data_reads(expr, SAFE)
    assert not raw_data_reads(f"num({expr})", SAFE)


def test_num_helper_preserves_raw_rendering_and_escapes():
    script = _script()
    assert "const num=x=>esc(String(x));" in script
    assert "const esc=s=>String(s??'')" in script


# ---------------------------------------------------------------------------
# Generator: trace types
# ---------------------------------------------------------------------------

def _write_trace(tmp_path, edge=None, boundary=None):
    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok=True)
    trace = {"schema": 2, "id": "t", "scenario": "t",
             "provenance": {"code": {"commit": "abc"}},
             "edges": [{"source": "a.f", "target": "a.g", "count": 1,
                        "phases": {"default": 1}, "wall_ns_inclusive": 10,
                        "lines": [[3, "direct"]], **(edge or {})}],
             "boundaries": [{"function": "a.g", "phase": "default", "call_index": 1,
                             **(boundary or {})}]}
    (evidence / "t.json").write_text(json.dumps(trace), encoding="utf-8")
    return "evidence"


HOSTILE = '<img src=x onerror="window.__xss=1">'


def test_well_typed_trace_loads(tmp_path):
    (trace,) = interactive.load_traces(tmp_path, _write_trace(tmp_path))
    assert trace["edges"][0]["count"] == 1


@pytest.mark.parametrize("edge,boundary", [
    ({"count": HOSTILE}, None),
    ({"count": "3"}, None),
    ({"count": True}, None),
    ({"count": 1.5}, None),
    ({"phases": {"default": HOSTILE}}, None),
    ({"phases": [1]}, None),
    ({"lines": [[HOSTILE, "direct"]]}, None),
    ({"lines": [[3, 4]]}, None),
    ({"lines": "3"}, None),
    ({"wall_ns_inclusive": HOSTILE}, None),
    ({"wall_ns_inclusive": float("nan")}, None),
    ({"source": 7}, None),
    (None, {"call_index": HOSTILE}),
])
def test_malformed_trace_fields_are_rejected(tmp_path, edge, boundary):
    rel = _write_trace(tmp_path, edge, boundary)
    with pytest.raises(interactive.AtlasInputError, match="unexpected type"):
        interactive.load_traces(tmp_path, rel)


# ---------------------------------------------------------------------------
# Browser: a hostile payload renders as text
# ---------------------------------------------------------------------------

BROWSER = find_browser()
ROOTS = {"entry": ["archfixture.a.make_child"], "uses": ["archfixture.a.uses"]}


def _poison(data: dict) -> dict:
    """Markup in every numeric field the page prints (bypassing the generator)."""
    data["meta"]["max_sites"] = HOSTILE
    for node in data["nodes"]:
        node[2] = HOSTILE                       # line number
        if len(node) > 6:
            node[6] = HOSTILE                   # dynamic import sites
    data["import_omitted"] = [HOSTILE] * len(data["import_omitted"])
    data["call_omitted"] = [HOSTILE] * len(data["call_omitted"])
    data["call_site_totals"] = [HOSTILE] * len(data["call_site_totals"])
    for sites in data["call_sites"]:
        for st in sites:
            st[0], st[1] = HOSTILE, HOSTILE
    for sites in data["import_sites"]:
        for st in sites:
            st[0] = HOSTILE
    for u in data["unresolved"].values():
        for example in u[1]:
            example[0] = HOSTILE
    for e in data["entries"].values():
        for k in ("nodes", "max_depth", "truncated_nodes", "reachable_nodes_unbounded"):
            e[k] = HOSTILE
    for t in data["traces"]:
        t["alignment"] = {k: HOSTILE for k in t["alignment"]}
        for e in t["edges"]:
            e["count"] = HOSTILE
            e["phases"] = {p: HOSTILE for p in e.get("phases", {"p": 0})}
            e["lines"] = [[HOSTILE, "direct"]]
            e["wall_ns_inclusive"] = HOSTILE
        for b in t["boundaries"]:
            b["call_index"] = HOSTILE
            b["args"] = {"x": {"type": "list", "length": HOSTILE, "omitted": HOSTILE,
                               "fields": {}}}
    analysis = data["proposal"]["analysis"]
    analysis["exceptions_used"] = [HOSTILE] * len(analysis["exceptions_used"])
    analysis["members"] = {g: HOSTILE for g in analysis["members"]}
    for rel in ("imports", "calls"):
        analysis[rel]["by_direction"] = {k: HOSTILE for k in analysis[rel]["by_direction"]}
    return data


@pytest.fixture(scope="module")
def hostile_page(fixture_src, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hostile")
    proposal = {"schema": 2, "status": "test", "groups": {"app": {"role": "a"},
                                                           "core": {"role": "c"}},
                "layers": [["app"], ["core"]],
                "rules": [{"prefix": "archfixture", "group": "core"},
                          {"prefix": "archfixture.a", "group": "app"}],
                "exceptions": [{"source": "archfixture.b", "target": "archfixture.a",
                                "reason": "test"}]}
    (tmp / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    evidence = tmp / "evidence"
    evidence.mkdir()
    trace = {"schema": 2, "id": "t", "scenario": "fixture-run",
             "provenance": {"code": {"commit": "abc"}},
             "boundaries": [{"function": "archfixture.c.cfun", "phase": "stepping",
                             "call_index": 1, "args": {}, "mutations": {}, "unverified": []}],
             "edges": [{"source": "archfixture.a.uses", "target": "archfixture.c.cfun",
                        "count": 2, "phases": {"stepping": 2}, "wall_ns_inclusive": 10,
                        "lines": [[21, "direct"]]},
                       {"source": "archfixture.a.shadowed", "target": "archfixture.c.cfun",
                        "count": 1, "phases": {"stepping": 1}}]}
    (evidence / "t.json").write_text(json.dumps(trace), encoding="utf-8")
    files = render(Options(src=".", package="archfixture", call_roots=ROOTS,
                           proposal=str(tmp / "proposal.json"), evidence=str(evidence)),
                   root=fixture_src)
    html = files["interactive.html"]
    start = html.index('type="application/json">') + len('type="application/json">')
    data = json.loads(html[start:html.index("</script>", start)])
    path = tmp / "interactive.html"
    path.write_text(interactive.render_html(_poison(data)), encoding="utf-8", newline="\n")
    return path


#: Visit every tab under both groupings, every entry, and every inspector
#: (each node, call, import and observation). Counts injected elements after
#: each render and whether the payload was ever shown as text.
VISIT_ALL = r"""(()=>{const A=window.__atlas,D=A.D,S=A.S,$=id=>document.getElementById(id);
 let injected=0,shown=0,states=0;
 const check=()=>{states++;for(const id of ['nav','main','inspector','tabs'])
   {const el=$(id);injected+=el.querySelectorAll('img,[onerror]').length;
    if(el.textContent.includes('<img src=x'))shown++}};
 const tabs=['neighborhood','boundaries','runtime','entry','legend'];
 for(const g of ['package','proposal'])for(const entry of ['',...Object.keys(D.entries)])
  for(const tab of tabs){S.grouping=g;S.entry=entry;S.tab=tab;S.sel=null;A.render();check();
   for(const rel of ['imports','imports-type-only','calls']){S.rel=rel;A.render();check()}}
 S.entry='';S.grouping='package';
 for(const tab of ['neighborhood','runtime']){S.tab=tab;
  for(let i=0;i<D.ids.length;i++){A.selectNode(i);S.tab=tab;A.render();check()}
  D.calls.forEach((_,k)=>{A.selectEdge('call:'+k);check()});
  D.imports.forEach((_,k)=>{A.selectEdge('import:'+k);check()});
  D.traces.forEach((t,ti)=>t.edges.forEach((_,ei)=>{A.selectEdge('obs:'+ti+':'+ei);check()}))}
 return {injected,shown,states,xss:window.__xss===undefined?null:window.__xss}})()"""


@pytest.mark.skipif(BROWSER is None, reason="needs Chrome, Chromium or Edge")
def test_hostile_numeric_fields_render_as_text(hostile_page):
    with OfflinePage(BROWSER) as page:
        url = page.open(hostile_page)
        page.wait_for("window.__atlas")
        result = page.eval(VISIT_ALL)
        page.eval("new Promise(r=>setTimeout(r,300))")
        assert result["states"] > 50, result
        assert result["injected"] == 0, result
        assert result["shown"] > 0, result      # displayed literally, not dropped
        assert page.eval("window.__xss === undefined")
        assert page.eval("document.querySelectorAll('img').length") == 0
        assert page.requests == [url], page.requests  # an <img src=x> would fetch
        assert page.exceptions == [], page.exceptions
