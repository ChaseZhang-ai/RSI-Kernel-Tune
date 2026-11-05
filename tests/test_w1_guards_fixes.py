"""Regression tests for the W1 audit fix P3-8 (goal-ref window/key hygiene):

* the accepted reference declares each reading under its own window's key — a
  device-window session never writes a device reading into the e2e arch_ref_ms
  key -- loop/goal_ref.py.

(The wave's SEED-08 / SEED-09 faces are covered by the corps' own
 test_corps_fix_guards.py / test_corps_port_guards.py on this branch; only the
 W1-unique P3-8 cases live here.)

Ported from the R2 audit's untracked test batch, rewritten for the
rsi_kernel_tune tree. CPU only.
"""

from __future__ import annotations

import pytest

from rsi_kernel_tune.loop import goal_ref as gr  # noqa: E402


# -- goal-ref: the declared reference lands each reading under its own window's key ----

def _accepted_state(gate_window: str, *, reading_ms: float, device_ms: float | None,
                    e2e_ms: float | None) -> dict:
    return {"op": "demo", "status": "accepted", "gate_window": gate_window,
            "accepted": {"candidate": 1, "sha256": "abc", "reference": ["ref.py"],
                         "reading_ms": reading_ms, "device_ms": device_ms,
                         "e2e_ms": e2e_ms, "runs": 1, "jobs": ["j1"], "bar_met": False}}


def test_a_device_window_reading_never_declares_the_e2e_key():
    out, arch = gr.apply({}, _accepted_state("device", reading_ms=1.5,
                                             device_ms=1.5, e2e_ms=None))
    lc = out["loop"]
    assert arch["declare"]
    assert "arch_ref_ms" not in lc          # not the e2e key: a device reading was it
    assert lc["arch_ref_device_ms"] == pytest.approx(1.5)


def test_a_true_e2e_reading_still_declares_the_e2e_key_under_a_device_window():
    out, _ = gr.apply({}, _accepted_state("device", reading_ms=1.5,
                                          device_ms=1.5, e2e_ms=2.0))
    lc = out["loop"]
    assert lc["arch_ref_ms"] == pytest.approx(2.0)
    assert lc["arch_ref_device_ms"] == pytest.approx(1.5)


def test_an_e2e_window_reading_still_declares_the_e2e_key():
    out, _ = gr.apply({}, _accepted_state("e2e", reading_ms=2.5,
                                          device_ms=1.5, e2e_ms=2.5))
    lc = out["loop"]
    assert lc["arch_ref_ms"] == pytest.approx(2.5)
    assert lc["arch_ref_device_ms"] == pytest.approx(1.5)
    # the gate_reading fallback: an e2e window whose record lacks e2e_ms read device
    out2, _ = gr.apply({}, _accepted_state("e2e", reading_ms=1.5,
                                           device_ms=1.5, e2e_ms=None))
    assert out2["loop"]["arch_ref_ms"] == pytest.approx(1.5)
