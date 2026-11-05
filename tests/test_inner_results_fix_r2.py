"""Regression tests for the audit corps' inner results fix (r2, BUG-SEED-28).

Covers: a "0/N FAILED" summary no longer reading as a failure, and the perf
JSON scan surviving braces inside string values.
"""

from __future__ import annotations

from rsi_kernel_tune.inner.results import parse_correctness, parse_perf


def test_parse_correctness_zero_failed_summary_is_not_a_fail():
    r = parse_correctness("  PASS  a\n  PASS  b\n0/2 FAILED\n")
    assert r["passed"] is True and r["fails"] == 0
    # a real N/M FAILED still fails, whatever the per-case lines say
    r = parse_correctness("  PASS  a\n1/2 FAILED\n")
    assert r["passed"] is False and r["fails"] == 1


def test_json_scan_survives_braces_inside_strings():
    text = 'noise {"performance": {"bwd_median_ms": 1.0, "label": "a}b{c"}} {"x": 1}'
    assert parse_perf(text)["bwd_median_ms"] == 1.0
