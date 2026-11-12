"""Write (or check) the committed standalone sky page, ``docs/sky.html``.

    python -m tools.sky_page            # regenerate docs/sky.html
    python -m tools.sky_page --check    # exit 1 if it no longer matches

The page is ``tropoi.representation.visual.sky.standalone_page()``: the sky
template with the Earth preset and every preset's data built in, so it opens
by double-clicking, with no Python. It is generated, never edited by hand.

``--check`` compares the page text exactly (line endings aside) and the
embedded numbers to a relative 1e-12, so a last-bit difference in NumPy's
cross sections between platforms is not reported as drift.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "docs" / "sky.html"
DATA = re.compile(r'(<script id="sky-data" type="application/json">)(.*?)'
                  r'(</script>)', re.S)
RTOL = 1e-12


def _split(html: str) -> tuple[str, object]:
    html = html.replace("\r\n", "\n")
    m = DATA.search(html)
    if m is None:
        raise ValueError("no sky-data block in the page")
    raw = m.group(2).replace("\\u003c", "<").replace("\\u003e", ">")
    return DATA.sub(r"\1\3", html, count=1), json.loads(raw)


def _numbers_differ(a, b, path="data") -> list[str]:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None \
            or isinstance(a, str) or isinstance(b, str):
        return [] if a == b else [path]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return [] if math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0) else [path]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [path]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in _numbers_differ(x, y, f"{path}[{i}]")]
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path} keys"]
        return [d for k in a for d in _numbers_differ(a[k], b[k],
                                                      f"{path}.{k}")]
    return [path]


def differences(committed: str, expected: str) -> list[str]:
    """Where ``committed`` departs from ``expected`` (empty: they match)."""
    page_a, data_a = _split(committed)
    page_b, data_b = _split(expected)
    out = [] if page_a == page_b else ["page text outside the data block"]
    return out + _numbers_differ(data_a, data_b)


def expected_page() -> str:
    from tropoi.representation.visual.sky import standalone_page
    return standalone_page().replace("\r\n", "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true",
                   help="compare instead of writing; exit 1 on a difference")
    args = p.parse_args(argv)
    expected = expected_page()
    if args.check:
        if not PAGE.exists():
            print(f"{PAGE} is missing; run python -m tools.sky_page")
            return 1
        diff = differences(PAGE.read_text(encoding="utf-8"), expected)
        if diff:
            print(f"{PAGE.relative_to(REPO_ROOT)} is stale "
                  f"({', '.join(diff[:5])}{' …' if len(diff) > 5 else ''}); "
                  "run python -m tools.sky_page")
            return 1
        print(f"{PAGE.relative_to(REPO_ROOT)} is current")
        return 0
    PAGE.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {PAGE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
