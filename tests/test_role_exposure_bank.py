"""The role-exposure evidence banks carry identity (BUG-SEED-32).

Two overwrites used to happen silently:
* a recon probe pricing an aux kernel banked its FULL role table over the
  reference's own account (the dossier grew a banked-header guard for exactly
  this accident; the role ledger had none);
* the second (HEAD-tree) student pricing pass wrote its roles over the
  uniquely-named student role ledger, while the teacher view consumes the
  primary (best-tree) pass's roles — banked evidence describing a different
  tree than the verdicts it sits beside.
"""

from __future__ import annotations

from peregrine_kernel.loop import driver

from test_loop_driver import (  # noqa: F401 -- shared fixtures/helpers
    _ROLE_CORE_REF, _STALL_LOG_OK, FakeBackend, _ctx, _role_block, _stall_ctx,
)


def _log_for(core: str, kernel: str | None) -> str:
    """A pricing log whose role block carries (or omits) a kernel identity."""
    log = ""
    if kernel is not None:
        log += f"[stall_price] dominant launch index=0 name={kernel}\n"
    return log + _role_block(core)


def _save(ctx, log):
    driver._save_ref_role_exposure(ctx, log)
    ev = driver._evidence_dir(ctx)
    return (ev / driver._REF_ROLE_EXPOSURE_NAME).read_text()


def _ref_ctx(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    return ctx


def test_a_bystander_kernel_full_table_does_not_replace_the_banked_one(tmp_path,
                                                                       capsys):
    ctx = _ref_ctx(tmp_path)
    first = _save(ctx, _log_for(_ROLE_CORE_REF, "main_kernel"))
    assert "# banked-roles: kernel=main_kernel verdict=OK" in first
    second = _save(ctx, _log_for(_ROLE_CORE_REF, "aux_prepare_kernel"))
    assert second == first, "a different kernel's FULL table must not overwrite"
    assert "kept banked kernel=main_kernel" in capsys.readouterr().out


def test_the_same_kernel_refreshes_the_banked_table(tmp_path):
    ctx = _ref_ctx(tmp_path)
    fresher = _ROLE_CORE_REF.replace('"duration_ns": 843424',
                                     '"duration_ns": 900000')
    first = _save(ctx, _log_for(_ROLE_CORE_REF, "main_kernel"))
    second = _save(ctx, _log_for(fresher, "main_kernel"))
    assert second != first
    assert '"duration_ns": 900000' in second
    assert "# banked-roles: kernel=main_kernel verdict=OK" in second


def test_identity_less_logs_keep_the_legacy_overwrite_behavior(tmp_path):
    # the recon fixtures without dominant-launch/STALL-OK lines carry no
    # kernel identity: the old full-refresh behavior must stand (the guard
    # only fires when BOTH identities are known and differ)
    ctx = _ref_ctx(tmp_path)
    _save(ctx, _role_block(_ROLE_CORE_REF))
    second = _save(ctx, _role_block(_ROLE_CORE_REF))
    assert second.startswith("[join]")          # readers' first-line pin intact
    assert "# banked-roles: kernel=? verdict=OK" in second


def test_partial_still_never_downgrades_a_banked_full(tmp_path):
    ctx = _ref_ctx(tmp_path)
    first = _save(ctx, _log_for(_ROLE_CORE_REF, "main_kernel"))
    second = _save(ctx, _role_block_partial_stub())
    assert second == first
    assert driver._ref_overlap_banked(ctx)


def _role_block_partial_stub():
    # a minimal PARTIAL block: a role table whose verdict line lacks the
    # tensor term; parse_core still finds no tensor_active
    return ("ROLE EXPOSURE BEGIN\n"
            "ROLE-JSON: {\"duration_ns\": 100, \"resolved_pct\": 90.0}\n"
            "ROLE-PARTIAL roles=1 samples=5 resolved=90.0\n"
            "ROLE EXPOSURE END\n")


def test_head_tree_pricing_pass_leaves_the_student_role_ledger_alone(tmp_path):
    # the HEAD-tree pass comes home with a parsable role table, yet must not
    # write it: the teacher view consumes the primary pass's roles
    be = FakeBackend([], [_STALL_LOG_OK + _role_block(_ROLE_CORE_REF)])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    driver._run_stall_pricing(ctx, bank_as=(driver._STUDENT_ORCH_DOSSIER_NAME,),
                              label="head", persist_roles=False)
    assert not (tmp_path / "evidence" /
                driver._STUDENT_ROLE_EXPOSURE_NAME).exists()
