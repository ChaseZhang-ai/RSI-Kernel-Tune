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
        # the count decides, not the marker's existence: "0/M FAILED" is a pass
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
        # say why: an unparseable block after the marker and a missing marker are
        # different diagnoses, and an all-None result without a note reads as
        # "the bench said nothing" when it may have said something unreadable
        return {**empty,
                "note": ("the === JSON === block does not parse into an object "
                         "carrying 'performance'" if marker != -1 else
                         "no === JSON === block in the output")}
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
    """Scan for a JSON object containing `key` (handles one-line dumps).

    The scan goes through JSONDecoder.raw_decode, which knows where a JSON string
    ends: the old hand-rolled brace counter read a ``}`` inside a string value as
    a closing brace, lost the depth, truncated every candidate, and quietly
    returned None for the whole output."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None
