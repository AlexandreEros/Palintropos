"""Tier-A evidence JSON for the Held–Suarez report, from a JUnit XML.

Inputs: a full-suite JUnit XML (``pytest tests --junitxml=...``) produced at
the solver commit, the base ref the branch's tests are compared with, and
the solver commit the production run used. Output: the JSON that
``python -m tropoi.run.held_suarez report --tier-a`` reads
(``tropoi.run.held_suarez.report.tier_a``):

* ``pytest``: passed / failed / skipped / errors of the whole suite;
* ``tests_diff_additions_only``: ``git diff <base> <commit> -- tests/`` deletes
  no line;
* ``new_tests``: every test case in a test file added since ``base``, plus
  the test functions added to pre-existing files -> PASS / FAIL / SKIP;
* ``new_skips``: the skipped new tests with their reasons;
* ``smoke_resume_bit_identical``: the A3 test's outcome
  (``test_smoke_forced_stop_and_resume_is_bit_identical``);
* ``solver_commit`` and ``solver_commit_pushed_ancestor``: whether the solver
  commit is an ancestor of ``origin/<branch>`` (fetch first).

    python tools/held_suarez/tier_a_evidence.py --junit suite.xml --base main \
        --commit 206ef25 --branch feat/held-suarez --out tier_a.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import xml.etree.ElementTree as ET

A3_TEST = "test_smoke_forced_stop_and_resume_is_bit_identical"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def added_test_files(base: str, commit: str) -> set[str]:
    out = git("diff", "--name-status", base, commit, "--", "tests/")
    return {line.split("\t")[1] for line in out.splitlines() if line.startswith("A\t")}


def added_functions(base: str, commit: str, path: str) -> set[str]:
    pat = re.compile(r"^\s*def (test_\w+)", re.M)
    before = set(pat.findall(git("show", f"{base}:{path}")))
    after = set(pat.findall(git("show", f"{commit}:{path}")))
    return after - before


def additions_only(base: str, commit: str) -> bool:
    for line in git("diff", "--numstat", base, commit, "--", "tests/").splitlines():
        if line.split("\t")[1] != "0":
            return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--junit", required=True)
    ap.add_argument("--base", default="main")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    commit = git("rev-parse", args.commit).strip()

    root = ET.parse(args.junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}

    new_files = added_test_files(args.base, commit)
    modified = {line.split("\t")[1] for line in
                git("diff", "--name-status", args.base, commit, "--", "tests/").splitlines()
                if line.startswith("M\t") and line.endswith(".py")}
    new_funcs = {p: added_functions(args.base, commit, p) for p in modified}

    new_tests: dict[str, str] = {}
    new_skips: list[dict] = []
    a3 = None
    for case in root.iter("testcase"):
        path = case.get("classname", "").split(".")
        # classname "tests.test_x" or "tests.test_x.TestClass"
        mod = next((i for i, p in enumerate(path) if p.startswith("test_")), None)
        if mod is None:
            continue
        file = "/".join(path[:mod + 1]) + ".py"
        name = case.get("name", "")
        func = name.split("[")[0]
        outcome = ("FAIL" if case.find("failure") is not None or case.find("error") is not None
                   else "SKIP" if case.find("skipped") is not None else "PASS")
        if func == A3_TEST:
            a3 = outcome == "PASS"
        if file in new_files or func in new_funcs.get(file, set()):
            tid = f"{file}::{name}"
            new_tests[tid] = outcome
            if outcome == "SKIP":
                sk = case.find("skipped")
                new_skips.append({"test": tid, "reason": (sk.get("message") or "").strip()})

    subprocess.check_call(["git", "fetch", "--quiet", "origin", args.branch])
    pushed = subprocess.call(["git", "merge-base", "--is-ancestor", commit,
                              f"origin/{args.branch}"]) == 0
    evidence = {
        "pytest": {"passed": tot["tests"] - tot["failures"] - tot["errors"] - tot["skipped"],
                   "failed": tot["failures"], "errors": tot["errors"], "skipped": tot["skipped"]},
        "tests_diff_additions_only": additions_only(args.base, commit),
        "tests_diff_base": git("rev-parse", args.base).strip(),
        "new_tests": dict(sorted(new_tests.items())),
        "new_skips": new_skips,
        "smoke_resume_bit_identical": a3,
        "solver_commit": commit,
        "solver_commit_pushed_ancestor": pushed,
        "junit": pathlib.Path(args.junit).name,
    }
    pathlib.Path(args.out).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "new_tests"}, indent=2))
    print(f"new tests: {len(new_tests)} ({sum(v == 'PASS' for v in new_tests.values())} PASS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
