"""Regression tests for the audit-corps fix wave 4 (driver.py probe face).

Covers: BUG-SEED-35 (the meta verdict probe degrades on a backend error
like every sibling probe path already does — one GPU flake must not end
the whole run through the probe channel).
"""

from __future__ import annotations

from peregrine_kernel.loop.driver import (
    BackendError,
    _run_meta_verdict_probe,
    run_meta_step,
)

from test_loop_driver import FakeBackend, _ctx, _patching


class _GpuFlakeBackend(FakeBackend):
    """submit_gpu raises BackendError (a GPU flake, queue 503): the shape every
    sibling probe path already degrades through."""

    def submit_gpu(self, branch: str, mode: str) -> str:
        self.calls.append(("gpu", mode))
        raise BackendError("GPU flake: queue 503")


def test_verdict_probe_backend_error_degrades_to_no_verdict(tmp_path):
    """submit_gpu raising mid-probe must not escape run_meta_step: one flaky
    job used to end the whole run and waste the remaining budget. The round
    degrades to a no-verdict reading, the teacher hears the reason, the run
    carries on to its diagnosis."""
    from peregrine_kernel.meta_trajectory import load_meta_records
    diagnosis = "cause_class: R1\ndeficient_locus: D1.1"
    patch = _patching(tmp_path, "\n### D0.9 loosened\nx\n",
                      "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n")
    be = _GpuFlakeBackend(
        ["PROBE_REQUEST: notes/refsched_ab.py", diagnosis, patch], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "structural"
    assert "the probe job never ran: GPU flake: queue 503" in be.prompts[1]
    assert "PROBE PASS/FAIL" in be.prompts[1]          # the standing protocol note
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].probe_rounds == 1
    assert recs[0].probe_readings == []


def test_run_meta_verdict_probe_returns_a_no_verdict_reading(tmp_path):
    """Unit shape of the degrade: (no-verdict pv, reason log, no readings)."""
    be = _GpuFlakeBackend([], [])
    pv, log, readings = _run_meta_verdict_probe(
        _ctx(tmp_path, be), be, "fs-demo-metaprobe", "notes/p.py", 1)
    assert pv.found is False and readings == []
    assert "the probe job never ran: GPU flake: queue 503" in log


def test_probe_manifest_backend_error_falls_back_and_degrades(tmp_path):
    """The manifest chain (commit/push/batch job) raising BackendError takes
    the same wholesale-failure fallback as a marker-less log, and the
    per-script path it falls back to carries the same net per script: the run
    carries on instead of dying on the batch job's flake."""
    from peregrine_kernel.meta_trajectory import load_meta_records
    manifest = tmp_path / "batch.txt"
    scripts = ["s0.py", "s1.py"]
    manifest.write_text("\n".join(scripts) + "\n", encoding="utf-8")
    for s in scripts:
        (tmp_path / s).write_text("print('probe')\n", encoding="utf-8")
    diagnosis = "cause_class: R1\ndeficient_locus: D1.1"
    patch = _patching(tmp_path, "\n### D0.9 loosened\nx\n",
                      "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n")
    be = _GpuFlakeBackend(["PROBE-MANIFEST: batch.txt\n", diagnosis, patch], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "structural"
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].probe_rounds == 1 and recs[0].probe_readings == []
    # both scripts' degrade reasons ride the one re-prompt
    assert "the probe job never ran: GPU flake: queue 503" in be.prompts[1]
    assert be.prompts[1].count("the probe job never ran") == 2
