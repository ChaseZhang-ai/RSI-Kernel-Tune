"""Parse GPU job stdout (correctness + perf) emitted by the frozen FA test suite.

The kernel test scripts print:
  correctness: "  PASS|FAIL|ERROR  <case>" per case, then "all N cases PASS" or "N/M FAILED".
  perf:        a final `=== JSON ===` line with json.dumps({"correctness":{...,"pass"},
               "performance":{"bwd_median_ms","bwd_mfu_pct",...}}).
"""

from __future__ import annotations

import json
import re

_CASE_RE = re.compile(r"^\s+(PASS|FAIL|ERROR)\s+(.+)$")
_ALL_PASS_RE = re.compile(r"all (\d+) cases PASS")
_FAILED_RE = re.compile(r"(\d+)/(\d+) FAILED")


def parse_correctness(stdout: str) -> dict:
    cases = [{"status": m.group(1), "name": m.group(2).strip()}
             for m in (_CASE_RE.match(ln) for ln in stdout.splitlines()) if m]
    all_pass = _ALL_PASS_RE.search(stdout)
    failed = _FAILED_RE.search(stdout)
    if failed:
        # "N/M FAILED" is a count: trust the parsed number. "0/M FAILED" used to
        # read passed=False with fails=0 — a verdict the line's own counter
        # contradicts (zero counted failures is not a failure).
        fails = int(failed.group(1))
        passed = fails == 0
    elif all_pass:
        fails = 0
        passed = True
    else:
        # fall back to per-case tally when no summary line is present
        fails = sum(1 for c in cases if c["status"] in ("FAIL", "ERROR"))
        passed = bool(cases) and fails == 0
    return {"passed": passed, "fails": fails, "cases": cases}


def parse_perf(stdout: str) -> dict:
    """Extract the combined `=== JSON ===` block (single-line json.dumps)."""
    empty = {"passed": None, "bwd_median_ms": None, "bwd_mfu_pct": None,
             "dq": None, "dk": None, "dv": None}
    marker = stdout.rfind("=== JSON ===")
    region = stdout[marker:] if marker != -1 else stdout
    obj = _first_json_with_key(region, "performance") or _first_json_with_key(stdout, "performance")
    if obj is None:
        return empty
    corr = obj.get("correctness", {})
    perf = obj.get("performance", {})
    return {
        "passed": corr.get("pass"),
        "bwd_median_ms": perf.get("bwd_median_ms"),
        "bwd_mfu_pct": perf.get("bwd_mfu_pct"),
        "dq": corr.get("dq"),
        "dk": corr.get("dk"),
        "dv": corr.get("dv"),
    }


def _first_json_with_key(text: str, key: str) -> dict | None:
    """Scan for a top-level JSON object containing `key` (handles one-line dumps).

    Each ``{`` is decoded from with json.JSONDecoder.raw_decode rather than a
    hand-rolled brace count: a character counter cannot tell a closing brace
    inside a string literal from a structural one, so a JSON value containing
    ``"}"`` used to cut every candidate region short and return None for input
    that was otherwise clean."""
    dec = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None
