"""interactive.html in a real headless browser (Chrome, Chromium or Edge).

The page is opened from ``file://`` with networking blocked (tools/browser_cdp):
it must issue exactly one request (itself), raise no console error, and its
interactions must agree with the generator's own numbers:

* search by name and by source path; selecting nodes and edges by click;
* constructor, possible-dispatch, import-symbol and call-site rendering;
* entry-path reach recomputed in JavaScript equals ``calls.scoped``;
* the proposed-owner matrix equals ``interactive.analyse_proposal``;
* a runtime observation links to its static counterpart, and back;
* deep links (``#node=``) restore the selection;
* hostile source text is shown as text, never as markup.

Skipped when no browser is installed.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tools.architecture.generate import REPO_ROOT, Options, render
from tools.browser_cdp import OfflinePage, find_browser

BROWSER = find_browser()
pytestmark = pytest.mark.skipif(BROWSER is None, reason="needs Chrome, Chromium or Edge")
ROOTS = {"entry": ["archfixture.a.make_child"], "uses": ["archfixture.a.uses"]}
EVIL = '</script><img src=x onerror="window.__pwned=1"> & <b>bold</b>'


@pytest.fixture(scope="module")
def page_file(fixture_src, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("atlas")
    (fixture_src / "archfixture" / "evil.py").write_text(
        f'def hostile():\n    """{EVIL}"""\n    return {EVIL!r}\n', encoding="utf-8")
    proposal = {"schema": 2, "status": "test", "groups": {"app": {"role": "a"},
                                                           "core": {"role": "c"}},
                "layers": [["app"], ["core"]],
                "rules": [{"prefix": "archfixture", "group": "core"},
                          {"prefix": "archfixture.a", "group": "app"},
                          {"prefix": "archfixture.b", "group": "app"}],
                # A declared exception makes every crossing badge consult
                # exceptionFor (a selected edge once crashed it).
                "exceptions": [{"source": "archfixture.b", "target": "archfixture.a",
                                "reason": "test"}]}
    (tmp / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    evidence = tmp / "evidence"
    evidence.mkdir()
    trace = {"schema": 2, "id": "t", "scenario": "fixture-run", "description": "test",
             "provenance": {"code": {"commit": "abc", "source_sha256": None}},
             "functions": {"archfixture.c.cfun": {"stepping": 2}},
             "boundaries": [{"function": "archfixture.c.cfun", "phase": "stepping",
                             "call_index": 1, "args": {},
                             "return": {"type": "int", "value": 2}, "mutations": {},
                             "unverified": []}],
             "edges": [{"source": "archfixture.a.uses", "target": "archfixture.c.cfun",
                        "count": 2, "phases": {"stepping": 2}, "wall_ns_inclusive": 10,
                        "lines": [[21, "direct"]]},
                       {"source": "archfixture.a.shadowed", "target": "archfixture.c.cfun",
                        "count": 1, "phases": {"stepping": 1}}]}
    (evidence / "t.json").write_text(json.dumps(trace), encoding="utf-8")
    try:
        files = render(Options(src=".", package="archfixture", call_roots=ROOTS,
                               proposal=str(tmp / "proposal.json"), evidence=str(evidence)),
                       root=fixture_src)
    finally:
        (fixture_src / "archfixture" / "evil.py").unlink()
    path = tmp / "interactive.html"
    path.write_text(files["interactive.html"], encoding="utf-8", newline="\n")
    return path, json.loads(files["metrics.json"])


def _clean(page, url):
    assert page.requests == [url], page.requests
    assert page.console_errors == [] and page.exceptions == [], (page.console_errors,
                                                                  page.exceptions)


def _js(page, body):
    return page.eval(f"(()=>{{const A=window.__atlas;{body}}})()")


def test_navigation_search_and_inspection(page_file):
    path, _ = page_file
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        page.wait_for("window.__atlas")
        names = _js(page, "return A.search('make_child').map(i=>A.ID[i])")
        assert names[0] == "archfixture.a.make_child"
        assert "archfixture.child" in _js(page, "return A.search('child.py').map(i=>A.ID[i])")
        # select by clicking a search result
        _js(page, "const q=document.getElementById('q');q.value='make_child';"
                  "q.dispatchEvent(new Event('input'));return 1")
        page.eval("document.querySelector('#nav [data-node]').click()")
        assert _js(page, "return A.ID[A.S.sel.i]") == "archfixture.a.make_child"
        text = page.eval("document.getElementById('inspector').innerText")
        assert "def make_child()" in text and "Proposed owner" in text and "app" in text
        main = page.eval("document.getElementById('main').innerText")
        assert "archfixture.child.Child" in main
        assert page.eval("document.querySelectorAll('#main svg .nb').length") >= 1
        # neighbourhood box click moves the centre
        page.eval("document.querySelector('#main svg .nb').dispatchEvent("
                  "new MouseEvent('click',{bubbles:true}))")
        assert _js(page, "return A.S.sel.t") == "node"
        _clean(page, url)


def test_edges_show_kinds_sites_and_possible_dispatch(page_file):
    path, _ = page_file
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        page.wait_for("window.__atlas")

        def edge(s, t, k):
            return _js(page, f"return A.D.calls.findIndex(c=>A.ID[c[0]]==={json.dumps(s)}&&"
                             f"A.ID[c[1]]==={json.dumps(t)}&&c[2]==={json.dumps(k)})")
        k = edge("archfixture.a.make_child", "archfixture.child.Child", "instantiate")
        _js(page, f"A.selectEdge('call:{k}');return 1")
        # the Neighbourhood tab re-centres on the edge's source
        assert "archfixture.a.make_child" in page.eval(
            "document.getElementById('main').innerText")
        text = page.eval("document.getElementById('inspector').innerText")
        assert "instantiate" in text and "call sites (1)" in text.lower()
        assert "def __init__(self, n)" in text  # constructor signature behind the class
        k = edge("archfixture.base.Base.run", "archfixture.base.Base.step", "self")
        _js(page, f"A.selectEdge('call:{k}');return 1")
        text = page.eval("document.getElementById('inspector').innerText")
        assert "Possible dispatch targets at this site" in text
        assert "archfixture.child.Child.step" in text
        k = edge("archfixture.base.Base.run", "archfixture.child.Child.step", "override")
        _js(page, f"A.selectEdge('call:{k}');return 1")
        assert page.eval("document.querySelector('#inspector .ev.possible-dispatch')"
                         " !== null")
        k = _js(page, "return A.D.imports.findIndex(e=>A.ID[e[0]]==='archfixture.a'&&"
                      "A.ID[e[1]]==='archfixture.ns.tool')")
        _js(page, f"A.selectEdge('import:{k}');return 1")
        assert "helper as h" in page.eval("document.getElementById('inspector').innerText")
        _clean(page, url)


def test_entry_reach_and_proposal_matrix_match_the_generator(page_file):
    path, metrics = page_file
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        page.wait_for("window.__atlas")
        got = _js(page, "return Object.fromEntries(Object.keys(A.D.entries).map(n=>"
                        "{const r=A.reach(n);return [n,[r.depth.size,r.truncated.size]]}))")
        want = {n: [s["nodes"], s["truncated_nodes"]]
                for n, s in metrics["call_graph"]["scoped"].items()}
        assert got == want
        cells = _js(page, "const M=A.matrix('proposal','imports');return Object.fromEntries("
                          "[...M.cells].filter(([k])=>k.split('|')[0]!==k.split('|')[1])"
                          ".map(([k,v])=>[k,v.n]))")
        analysis = _js(page, "return A.D.proposal.analysis.imports.cells")
        assert cells == {k: v["runtime_edges"] for k, v in analysis.items()
                         if v["runtime_edges"]}
        _js(page, "document.getElementById('grouping').value='proposal';document.getElementById"
                  "('grouping').dispatchEvent(new Event('change'));A.S.tab='boundaries';"
                  "A.render();return 1")
        page.eval("document.querySelector('#main td[data-cell]').click()")
        assert "edge(s)" in page.eval("document.getElementById('main').innerText").lower()
        _clean(page, url)


def test_runtime_observation_links_to_static_edge(page_file):
    path, _ = page_file
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        page.wait_for("window.__atlas")
        _js(page, "A.selectEdge('obs:0:0');return 1")
        text = page.eval("document.getElementById('inspector').innerText")
        assert "corroborated" in text and "static counterpart" in text.lower()
        page.eval("document.querySelector('#inspector [data-edge^=\"call:\"]').click()")
        assert _js(page, "return A.S.sel.t") == "call"
        text = page.eval("document.getElementById('inspector').innerText")
        assert "fixture-run" in text and "×2" in text  # observed back on the static edge
        _js(page, "A.selectEdge('obs:0:1');return 1")
        assert "runtime-only" in page.eval("document.getElementById('inspector').innerText")
        _js(page, "A.selectNode(A.of.get('archfixture.c.cfun'));return 1")
        text = page.eval("document.getElementById('inspector').innerText")
        assert "data at this boundary" in text.lower() and "stepping ×2" in text
        _js(page, "A.selectNode(A.of.get('archfixture.d.deep'));return 1")
        assert "Unobserved is not unused" in page.eval(
            "document.getElementById('inspector').innerText")
        _clean(page, url)


def test_deep_link_and_hostile_text(page_file):
    path, _ = page_file
    with OfflinePage(BROWSER) as page:
        url = page.open(path, "node=archfixture.evil.hostile&tab=neighborhood")
        page.wait_for("window.__atlas")
        assert _js(page, "return A.ID[A.S.sel.i]") == "archfixture.evil.hostile"
        text = page.eval("document.getElementById('inspector').innerText")
        assert "</script>" in text and "<b>bold</b>" in text
        assert page.eval("document.querySelectorAll('img').length") == 0
        assert page.eval("window.__pwned === undefined")
        assert page.eval("[...document.querySelectorAll('b')].every(b=>b.textContent!=='bold')")
        _clean(page, url)


def test_repository_atlas_in_the_browser(tmp_path):
    """The real snapshot: offline, consistent entry reach, every trace aligned."""
    out = tmp_path / "current"
    subprocess.run([sys.executable, "-m", "tools.architecture.generate", "--out", str(out)],
                   cwd=REPO_ROOT, check=True, capture_output=True)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    with OfflinePage(BROWSER) as page:
        url = page.open(out / "interactive.html")
        page.wait_for("window.__atlas")
        got = _js(page, "return Object.fromEntries(Object.keys(A.D.entries).map(n=>"
                        "{const r=A.reach(n);return [n,[r.depth.size,r.truncated.size]]}))")
        assert got == {n: [s["nodes"], s["truncated_nodes"]]
                       for n, s in metrics["call_graph"]["scoped"].items()}
        for tab in ("neighborhood", "boundaries", "runtime", "entry", "legend"):
            _js(page, f"A.S.tab='{tab}';A.render();return 1")
        _js(page, "A.selectNode(A.of.get('tropoi.spatial.planet.Planet'));return 1")
        assert "Planet" in page.eval("document.getElementById('inspector').innerText")
        for spec in ("call:0", "import:0", "obs:0:0"):   # edge selections, default tab
            _js(page, f"A.S.tab='neighborhood';A.selectEdge('{spec}');return 1")
        n = _js(page, "return A.D.traces.reduce((a,t)=>a+t.edges.length,0)")
        assert n == sum(t["observed_edges"] for t in metrics["atlas"]["runtime_traces"])
        _clean(page, url)
