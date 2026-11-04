"""Regression tests for the audit-corps fix wave 3 (driver.py banking face).

Covers: BUG-SEED-32 — role exposure tables keep their account of record: the
ref side's full-vs-full overwrite gets the dossier's size guard, and the
student side stops letting a head≠best pricing overwrite the best tree's
roles.
"""

from __future__ import annotations

from peregrine_kernel.inner.trajectory import (
    IterationRecord,
    append_record,
    load_records,
)

from test_loop_driver import (
    FakeBackend,
    _ORCH_PRICING_BLOCK,
    _ORCH_PRICING_BLOCK_HEAD,
    _ROLE_CORE_REF,
    _ROLE_CORE_STU,
    _STALL_LOG_OK,
    _ctx,
    _patching,
    _stall_ctx,
)


# ---------------------------------------------------------------- BUG-SEED-32

# The head tree's role core: every reading distinct from the best tree's, so a
# overwrite cannot hide ("tensor_active": 55.5 vs 83.0, 9000 vs 20000 samples).
_ROLE_CORE_HEAD = ('ROLE-JSON: {"duration_ns": 999111, "resolved_pct": 96.0, '
                   '"roles": {"copy-producer": {"exposed_ns": 4242, "exposed_pct": 4.2, '
                   '"regions": 1, "share_pct": 25.0, "stalls": {"long_sb": 80.0}}}, '
                   '"tensor_active": 55.5, "tensor_source": "pm-plateau"}')


def _role_block(core, *, samples=20000, tensor="83.0"):
    """Parametrised variant of test_loop_driver._role_block: the stall-sample
    weight and the tensor reading are what the guard tests tell apart."""
    return ("=== ROLE EXPOSURE BEGIN ===\n"
            f"[join] 900 sampled PCs onto 12 regions; 97.2% of {samples} stall samples resolved\n"
            f"[tensor] active={tensor}% source=pm-plateau duration_ns=874880 exposed_frac=0.170\n"
            "ROLE label=mma-consumer regions=3 samples=16000 share=80.0 exposed_pct=13.6 "
            "exposed_ns=118984 stalls=wait:51\n"
            + core + "\n"
            "=== ROLE EXPOSURE END ===\n"
            f"ROLE-OK roles=2 samples={samples} resolved=97.2 tensor={tensor}\n")


def _recon_ctx(tmp_path):
    """A ctx whose evidence lands in a recon tree (the ref-side channel)."""
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    return ctx, recon_dir


def test_ref_role_table_full_overwrite_needs_a_bigger_collection(tmp_path):
    """Full-vs-full used to be unconditional: a later probe log carrying a
    bystander kernel's smaller collection silently replaced the banked FULL
    table (the dossier channel guarded exactly this since the gdn_prefill_bwd
    incident). The heavier-sampled account of record stays banked."""
    from peregrine_kernel.loop.driver import _save_ref_role_exposure
    ctx, recon_dir = _recon_ctx(tmp_path)
    dst = recon_dir / "ref_role_exposure.txt"
    _save_ref_role_exposure(
        ctx, "noise\n" + _role_block(_ROLE_CORE_REF, samples=30000, tensor="88.0"))
    assert dst.read_text().startswith("[join]")      # the big account is banked
    _save_ref_role_exposure(
        ctx, "noise\n" + _role_block(_ROLE_CORE_STU, samples=12000, tensor="83.0"))
    banked = dst.read_text()
    assert "30000 stall samples" in banked and '"tensor_active": 88.0' in banked, \
        "the banked heavier-sampled table must not be replaced"
    assert "12000 stall samples" not in banked


def test_ref_role_table_a_bigger_later_collection_still_supersedes(tmp_path):
    """The guard is not a first-writer-wins latch: a later FULL collection with
    more stall samples (a retry, a heavier pricing) still supersedes."""
    from peregrine_kernel.loop.driver import _save_ref_role_exposure
    ctx, recon_dir = _recon_ctx(tmp_path)
    dst = recon_dir / "ref_role_exposure.txt"
    _save_ref_role_exposure(
        ctx, "noise\n" + _role_block(_ROLE_CORE_STU, samples=12000, tensor="83.0"))
    _save_ref_role_exposure(
        ctx, "noise\n" + _role_block(_ROLE_CORE_REF, samples=30000, tensor="88.0"))
    banked = dst.read_text()
    assert "30000 stall samples" in banked and '"tensor_active": 88.0' in banked


def _two_tree_ctx(tmp_path, best_log, head_log):
    """A segment whose best KEEP ("shabest") is not the tree under judgment
    (the fake's HEAD is "sha0"): two pricing jobs per teacher entry."""
    class _RefPushing(FakeBackend):
        def push_ref(self, ref, branch):
            self.calls.append(("push_ref", ref, branch))

    be = _RefPushing([], [best_log, head_log])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, device_median_ms=2.0, sha="shabest"))
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="M0-gate", change="rebuild", correctness=True,
        verdict="KEEP", bwd_median_ms=2.4, device_median_ms=2.4, sha="sha0"))
    return be, ctx, recon_dir


def test_student_roles_file_keeps_the_best_tree_account(tmp_path):
    """head != best: the best tree's role ledger is the account of record (the
    diagnose's role rows read the ledger-source tree), but the HEAD tree's
    pricing wrote the single-name file last and its account ended up banked as
    the run's student roles. The HEAD pricing must not overwrite that file."""
    from peregrine_kernel.loop.driver import _price_student_trees
    best_log = (_STALL_LOG_OK + _ORCH_PRICING_BLOCK
                + _role_block(_ROLE_CORE_STU, samples=20000, tensor="83.0"))
    head_log = (_STALL_LOG_OK + _ORCH_PRICING_BLOCK_HEAD
                + _role_block(_ROLE_CORE_HEAD, samples=9000, tensor="55.5"))
    be, ctx, recon_dir = _two_tree_ctx(tmp_path, best_log, head_log)
    priced = _price_student_trees(ctx, load_records(ctx.status_path))
    assert priced.head_is_best is False
    # the HEAD pricing job really ran and its dossier really banked — the
    # roles assertion below is not green merely because the second job failed
    assert "2600 instructions" in (recon_dir / "student_orch_dossier.txt").read_text()
    assert "2487 instructions" in (recon_dir / "student_orch_dossier_best.txt").read_text()
    banked = (recon_dir / "student_role_exposure.txt").read_text()
    assert "20000 stall samples" in banked and '"tensor_active": 83.0' in banked, \
        "the banked student roles must stay the best tree's account"
    assert "9000 stall samples" not in banked and '"tensor_active": 55.5' not in banked


def test_student_roles_banked_when_head_is_the_only_tree(tmp_path):
    """Control: with no best KEEP the single pricing prices HEAD and its role
    ledger is still banked (the guard must not latch the file shut)."""
    from peregrine_kernel.loop.driver import _price_student_trees
    be = FakeBackend([], [_STALL_LOG_OK + _ORCH_PRICING_BLOCK
                          + _role_block(_ROLE_CORE_STU, samples=20000)])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    priced = _price_student_trees(ctx, load_records(ctx.status_path))
    assert priced.best_sha is None and priced.head_is_best is False
    banked = (recon_dir / "student_role_exposure.txt").read_text()
    assert "20000 stall samples" in banked and '"tensor_active": 83.0' in banked
