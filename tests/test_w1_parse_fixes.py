"""Regression tests for the W1 parse/inner fix line (audit-corps FA5a, dedup rebuild).

Only the W1-unique faces live here; the corps' own tests on this branch cover
the shared ones (SEED-22 / SEED-25 / SEED-28 / SEED-43 / SEED-50). The ID in a
docstring names the finding; one test per audit item.
"""

import json

from peregrine_kernel.loop.parse import parse_eval, parse_probe_verdict


# --- SEED-47 [P2]: the numeric FAIL's tensor names come from the bench JSON ------

def test_numeric_fail_detail_names_non_attention_tensors():
    """SEED-47: the per-tensor summary used a hardcoded ("dq", "dk", "dv") — the
    vocabulary of ONE operator. A numeric FAIL of any other operator degraded to
    "no failure detail in the eval JSON"; the names now come from the JSON."""
    log = ('=== JSON ===\n{"correctness": {"pass": false, '
           '"out_main": {"max_diff": 0.4, "tolerance": 0.01}, '
           '"w_grad": {"error": "shape mismatch: got (64, 32) want (32, 64)"}}, '
           '"performance": null}')
    ev = parse_eval(log)
    assert "out_main: max_diff=0.4 tol=0.01" in ev.error
    assert "w_grad: shape mismatch" in ev.error
    assert "no failure detail" not in ev.error


def test_numeric_fail_keeps_the_attention_tensor_names():
    """SEED-47, no regression: the attention triple still reads as before."""
    log = ('=== JSON ===\n{"correctness": {"pass": false, '
           '"dq": {"max_diff": 3.1, "tolerance": 0.9}, '
           '"dk": {"error": "bad dk: shape=() dtype=None"}}, '
           '"performance": null}')
    ev = parse_eval(log)
    assert "dq: max_diff=3.1" in ev.error
    assert "dk: bad dk" in ev.error
