"""Regression test for the R3A audit wave: recon launch-facts tie-break.

BUG-CH-C-02 (P3): run_recon_step compared cluster products with ``>=`` while
its own comment rules that the clustered main kernel's reading stands — so a
later helper round with an equal CTA count silently overwrote the recorded
launch facts, and the finish gate then quoted the helper's shape.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from test_loop_driver import FakeBackend, _ctx  # noqa: E402

from rsi_kernel_tune.loop import driver

_LOG_CLUSTERED = ("LAUNCH-OK kernel=main_kernel grid=(16,1,1) block=(128,1,1) ctas=16 "
                  "cluster=(2,8,1) cluster_max_active=16\n")
_LOG_HELPER = ("LAUNCH-OK kernel=setup_helper grid=(16,1,1) block=(32,1,1) ctas=16 "
               "cluster=(16,1,1) cluster_max_active=16\n")


def test_a_tied_helper_round_does_not_overwrite_the_clustered_reading(
        tmp_path, monkeypatch):
    """A later round whose product ties the recorded clustered reading must not
    demote it: product ties keep the first (main kernel's) facts."""
    recon_dir = tmp_path / "recon"
    (recon_dir / "notes").mkdir(parents=True)
    for name in ("p1.py", "p2.py"):
        (recon_dir / "notes" / name).write_text("print('probe')\n", encoding="utf-8")
    be = FakeBackend(["RECON_REQUEST: notes/p1.py", "RECON_REQUEST: notes/p2.py",
                      "Reconnaissance complete. All measurements delivered."], [])
    ctx = _ctx(tmp_path, be)
    # recon jobs run on the dedicated reconnaissance tree's backend
    ctx.recon_backend = FakeBackend([], [_LOG_CLUSTERED, _LOG_HELPER])
    ctx.recon_dir = recon_dir
    ctx.max_recon_rounds = 4
    captured = {}
    monkeypatch.setattr(driver, "_seed_sass_tools", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_load_recon_state", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_recon_prompt", lambda *a, **k: "recon prompt")
    monkeypatch.setattr(driver, "_recon_artifacts_written", lambda *a, **k: True)
    monkeypatch.setattr(driver, "_ensure_ref_pricing", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_missing_recon_evidence", lambda *a, **k: [])
    monkeypatch.setattr(driver, "_launch_contradictions_open", lambda *a, **k: [])
    monkeypatch.setattr(driver, "_save_ref_stall_ledger", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_save_recon_state", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_require_recon_artifacts",
                        lambda *a, **k: captured.update(k.get("launch") or {}))
    result = driver.run_recon_step(ctx)
    assert result.finished
    assert captured.get("cluster") == (2, 8, 1)   # the clustered reading stands
    assert captured.get("round") == 0             # from the first (main) round
