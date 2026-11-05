"""Corps-fix wave 8 regression tests (driver.py / parse.py misc batch).

Each test pins one audit finding from handoff/triage-wk6/triage-T5.md
(BUG-SEED-32/33/34/35/49, BUG-T5C-01) and triage-T1.md (BUG-T1C-01).
"""
import json
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_kernel_tune.inner.trajectory import IterationRecord, append_record
from rsi_kernel_tune.loop.driver import _save_ref_struct_table
from rsi_kernel_tune.loop.parse import (
    RESOLVABLE_DIFF_CAP,
    RESOLVABLE_DIFF_FLOOR,
    resolvable_diff,
)

from test_loop_driver import FakeBackend, _ctx
from test_wp1_measurement import _ratio_rec

REPO = Path(__file__).resolve().parents[1]


# --- BUG-T5C-01: _save_ref_struct_table's OSError log lost the destination ---


def test_struct_table_write_failure_log_names_the_destination(tmp_path, capsys):
    """Observed: the OSError log read `could not persist ref_struct_table.txt to
    {e}` — the exception text sat in the destination slot and the path it tried
    to write was absent, so a write failure said what went wrong but never where.
    The sibling bank logs all read `could not persist <name> to <dst>: <e>`."""
    from rsi_kernel_tune.loop.driver import _REF_STRUCT_TABLE_NAME

    recon = tmp_path / "recon"
    recon.mkdir()
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = recon / "report.md"
    # the destination is a directory: write_text fails with an OSError whose
    # message does not contain the path — only the log can carry it
    (recon / _REF_STRUCT_TABLE_NAME).mkdir()
    log = ("=== STRUCT TABLE BEGIN ===\nphase table body\n"
           "=== STRUCT TABLE END ===\n")
    _save_ref_struct_table(ctx, log)
    out = capsys.readouterr().out
    dst = recon / _REF_STRUCT_TABLE_NAME
    assert f"could not persist {_REF_STRUCT_TABLE_NAME} to {dst}:" in out, out
    assert "Is a directory" in out, "the error text stays in its own slot"


def test_struct_table_bank_still_writes_on_a_normal_path(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.recon_path.parent.mkdir()
    log = ("=== STRUCT TABLE BEGIN ===\nphase table body\n"
           "=== STRUCT TABLE END ===\n")
    _save_ref_struct_table(ctx, log)
    assert (tmp_path / "recon" / "ref_struct_table.txt").read_text(
        encoding="utf-8") == "phase table body"


# --- BUG-SEED-49: _quarantine_leaky_kb collapsed same-basename entries --------


def _kb_ctx(tmp_path):
    """A ctx whose kb repo root is tmp_path/repo (the quarantine's parent)."""
    from rsi_kernel_tune.loop import lang_mode

    ctx = _ctx(tmp_path, FakeBackend([], []))
    repo = tmp_path / "repo"
    kb = repo / "hw_kb" / "cutedsl"
    (kb / "sm100").mkdir(parents=True)
    (kb / "hopper").mkdir(parents=True)
    ctx.kb_dir = kb
    assert lang_mode.kb_repo_root(kb) == repo
    return ctx, kb


def test_quarantine_keeps_same_basename_entries_from_different_archs(tmp_path):
    """Observed: dst = qdir / src.name dropped the <arch>/ prefix, so two
    subtrees' same-named leaky entries renamed onto one file — rename(2)
    replaces silently, the earlier entry vanished from both its tree and the
    quarantine, and the reviewing human saw one leak, not two."""
    from rsi_kernel_tune.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    (kb / "sm100" / "leak.md").write_text("sm100 leak body\n", encoding="utf-8")
    (kb / "hopper" / "leak.md").write_text("hopper leak body\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/leak.md:1: banned narrative",
                               "hopper/leak.md:1: banned narrative"])
    qdir = tmp_path / "repo" / "kb_quarantine"
    landed = sorted(p.name for p in qdir.iterdir())
    assert len(landed) == 2, landed
    assert not (kb / "sm100" / "leak.md").exists()
    assert not (kb / "hopper" / "leak.md").exists()
    bodies = {p.name: p.read_text(encoding="utf-8")
              for p in qdir.iterdir()}
    assert "sm100 leak body\n" in bodies.values()
    assert "hopper leak body\n" in bodies.values()


def test_quarantine_does_not_replace_a_previous_run_s_entry(tmp_path):
    """A second quarantine (a later run) with the same entry name must not
    replace the file the first run left for the human — it flattens its own
    relative path beside it."""
    from rsi_kernel_tune.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    qdir = tmp_path / "repo" / "kb_quarantine"
    qdir.mkdir(parents=True)
    (qdir / "leak.md").write_text("earlier run's leak\n", encoding="utf-8")
    (kb / "sm100" / "leak.md").write_text("this run's leak\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/leak.md:1: banned narrative"])
    assert (qdir / "leak.md").read_text(encoding="utf-8") == "earlier run's leak\n"
    assert (qdir / "sm100__leak.md").read_text(encoding="utf-8") == "this run's leak\n"


def test_quarantine_unique_names_stay_flat(tmp_path):
    """The quarantine name always carries the entry's relative path, so a unique
    basename still lands readable -- and two same-named entries from different
    subtrees can never rename onto one destination (the incident's silent
    replace). Adapted at the merge: the merged implementation flattens
    unconditionally (``<rel with / -> __>``), not only on collision."""
    from rsi_kernel_tune.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    (kb / "sm100" / "oddity.md").write_text("body\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/oddity.md:1: banned narrative"])
    assert (tmp_path / "repo" / "kb_quarantine" / "sm100__oddity.md").is_file()


# --- BUG-SEED-33: the seeders read/write without a fixed encoding -------------

_SEED_C_LOCALE_SNIPPET = """
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {src!r})
from rsi_kernel_tune.loop.driver import _seed_pricing_toolkits, _seed_sass_tools


class _Be:
    def commit(self, msg):
        pass


ws = Path({ws!r})
_seed_pricing_toolkits(ws, _Be())          # first seeding: writes the modules
_seed_pricing_toolkits(ws, _Be())          # second: reads them back (no-op)
recon = ws / "recon"
ctx = SimpleNamespace(recon_dir=recon, oracle_mode="blackbox")
_seed_sass_tools(ctx, _Be())               # the recon-side seeder, same defect
print("SEEDED")
"""


def test_seeding_survives_a_non_utf8_locale(tmp_path):
    """Observed: the seeding functions read/wrote every vendored module with the
    locale-preferred codec. All 13 PRICING_SEEDED_MODULES (and the recon set)
    carry non-ASCII UTF-8 bytes, so under LC_ALL=C the ASCII codec raised
    UnicodeDecodeError on the first vendor read — the loop died at its first
    step, before any GPU job was bought. Reads AND writes pin
    encoding="utf-8" now (the write side mattered on the read-back: a file the
    same call had just written could not be re-read under the same locale)."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.update(LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0",
               PYTHONPATH=str(REPO / "src"))
    env.pop("PYTHONIOENCODING", None)
    ws = tmp_path / "ws"
    ws.mkdir()
    code = _SEED_C_LOCALE_SNIPPET.format(src=str(REPO / "src"), ws=str(ws))
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert "SEEDED" in r.stdout
    assert (ws / "notes" / "orch_tools.py").is_file()
    assert (ws / "recon" / "notes" / "stall_tools.py").is_file()


# --- BUG-SEED-34: the scaffold and the pricing paths judged a malformed -------
# --- ``part k of n`` declaration with opposite semantics -----------------------


def _pkg_tail_row(change, verdict="KEEP"):
    return SimpleNamespace(verdict=verdict, change=change)


@pytest.mark.parametrize("decl", ["part 0 of 3: scaffolding the tile loop",
                                  "part 1 of 99: grand rebuild, uncut"])
def test_malformed_package_declaration_is_rejected_on_both_paths(decl):
    """Observed: _declared_construction accepted ``part 0 of 3`` (k < n, no
    lower bound) and ``part 1 of 99`` (no cap) and scaffolded the tree — never
    evaluated — while _open_package_tail rejected the SAME delivery as
    malformed and priced over it: the row was construction-protected AND
    REVERT-eligible at once. Both paths now share _pkg_step_ok's bar."""
    from rsi_kernel_tune.loop.driver import (
        _declared_construction,
        _open_package_tail,
    )

    assert _declared_construction(decl) is None
    assert _open_package_tail([_pkg_tail_row(decl)]) is None


def test_wellformed_package_declaration_still_works_on_both_paths():
    """The unification must not tighten away the real contract: a valid
    ``part 1 of 3`` opening still scaffolds and still opens the tail."""
    from rsi_kernel_tune.loop.driver import (
        _declared_construction,
        _open_package_tail,
        _PKG_MAX_STEPS,
    )

    decl = "part 1 of 3: smem access atoms, write side"
    assert _declared_construction(decl) == "part 1 of 3"
    assert _open_package_tail([_pkg_tail_row(decl)]) == (1, 3)
    assert _declared_construction(f"part 2 of {_PKG_MAX_STEPS} — second step") \
        == f"part 2 of {_PKG_MAX_STEPS}"
    assert _open_package_tail(
        [_pkg_tail_row(f"part 2 of {_PKG_MAX_STEPS}")]) == (2, _PKG_MAX_STEPS)


def test_malformed_declaration_does_not_spend_the_package_budget():
    """An ordinary (malformed) KEEP is not a package row: it must not count
    toward the n-rows-of-grace budget any more than it extends the plea."""
    from rsi_kernel_tune.loop.driver import _open_package_tail

    segment = [_pkg_tail_row("part 0 of 3 — malformed opener"),
               _pkg_tail_row("part 1 of 2 — real package step")]
    assert _open_package_tail(segment) == (1, 2)


# --- BUG-SEED-35: a verdict probe whose transport failed ended the whole run --


from rsi_kernel_tune.loop.driver import BackendError


class _DyingTransportBackend(FakeBackend):
    """A backend whose transport raises BackendError (a node drain mid-run, a
    lost push) — the GPU-jitter shape the triage reproduced."""

    def __init__(self, die_on):
        super().__init__([], [])
        self._die_on = die_on

    def commit(self, message):
        if "commit" in self._die_on:
            raise BackendError("transient: node drain mid-run (GPU jitter)")
        return super().commit(message)

    def push_branch(self, branch):
        if "push" in self._die_on:
            raise BackendError("transient: push lost (GPU jitter)")
        return super().push_branch(branch)

    def submit_gpu(self, branch, mode):
        if "submit" in self._die_on:
            raise BackendError("transient: node drain mid-run (GPU jitter)")
        return super().submit_gpu(branch, mode)


@pytest.mark.parametrize("die_on", ["submit", "push", "commit"])
def test_verdict_probe_transport_failure_is_a_no_verdict_not_a_crash(
        tmp_path, die_on):
    """Observed: _run_meta_verdict_probe ran commit/push/submit bare — one
    BackendError escaped the diagnose step and terminated the whole run, while
    the sibling paths degrade locally (_run_ablation_probe catches the same
    sequence into a "job failed (...)" reason, _run_stall_pricing diagnoses
    without a ledger). The probe is now this round's NO VERDICT and the run
    survives."""
    from rsi_kernel_tune.loop.driver import (
        _probe_label,
        _run_meta_verdict_probe,
    )

    ctx = _ctx(tmp_path, FakeBackend([], []))
    be = _DyingTransportBackend(die_on)
    pv, probe_log, readings = _run_meta_verdict_probe(
        ctx, be, "fs-demo-metaprobe", "notes/probe_x.py", 1)
    assert probe_log == "" and readings == []
    assert pv.found is False and pv.passed is False
    assert _probe_label(pv) == "NO VERDICT"
    assert "job failed" in pv.reason and "GPU jitter" in pv.reason


def test_verdict_probe_transport_feedback_does_not_preach_protocol(tmp_path):
    """The failed-transport round must not reach the teacher as the probe's
    protocol problem: the no-verdict-line sermon (pre-register a criterion)
    would send it editing a script that never executed."""
    from rsi_kernel_tune.loop.driver import (
        _verdict_probe_feedback,
        _run_meta_verdict_probe,
    )

    ctx = _ctx(tmp_path, FakeBackend([], []))
    be = _DyingTransportBackend("submit")
    pv, probe_log, _readings = _run_meta_verdict_probe(
        ctx, be, "fs-demo-metaprobe", "notes/probe_x.py", 1)
    feedback = _verdict_probe_feedback(
        "notes/probe_x.py", pv, probe_log, 1, 3)
    assert "the probe job never ran" in feedback
    assert "GPU jitter" in feedback
    assert "pre-register" not in feedback


def test_verdict_probe_no_verdict_line_feedback_keeps_the_sermon(tmp_path):
    """A probe that RAN but printed no verdict line keeps the original
    protocol feedback — the transport wording is only for jobs that never
    started."""
    from rsi_kernel_tune.loop.driver import _verdict_probe_feedback
    from rsi_kernel_tune.loop.parse import parse_probe_verdict

    pv = parse_probe_verdict("some output with no verdict line", audit=False)
    feedback = _verdict_probe_feedback("notes/probe_x.py", pv, "some output", 1, 3)
    assert "pre-register" in feedback
    assert "the probe job never ran" not in feedback


# --- BUG-SEED-32: the role exposure tables had no overwrite guard -------------


def _ref_role_block(samples, tensor=88.0):
    """A FULL ref role block whose ROLE row carries ``samples`` — the account
    size the overwrite guard reads off the banked evidence."""
    core = ('ROLE-JSON: {"duration_ns": 843424, "resolved_pct": 99.5, '
            '"roles": {"mma-consumer": {"exposed_ns": 70847, "exposed_pct": 8.4, '
            '"regions": 3, "share_pct": 70.0, "stalls": {"dispatch": 47.0}}}, '
            f'"tensor_active": {tensor}, "tensor_source": "pm-plateau"}}')
    return ("=== ROLE EXPOSURE BEGIN ===\n"
            f"[join] 900 sampled PCs onto 12 regions; 99.5% of {samples} stall samples resolved\n"
            f"[tensor] active={tensor}% source=pm-plateau duration_ns=843424 exposed_frac=0.120\n"
            f"ROLE label=mma-consumer regions=3 samples={samples} share=70.0 "
            f"exposed_pct=8.4 exposed_ns=70847 stalls=dispatch:47\n"
            + core + "\n"
            "=== ROLE EXPOSURE END ===\n"
            f"ROLE-OK roles=1 samples={samples} resolved=99.5 tensor={tensor}\n")


def test_ref_role_table_keeps_the_bigger_banked_full_account(tmp_path):
    """Observed: the ref side only guarded partial-not-downgrading-FULL; a FULL
    block overwrote the banked account unconditionally — the dossier's own
    incident (a bystander kernel's blocks replacing the main kernel's banked
    dossier) applied to the role table verbatim. A later FULL table built from
    strictly fewer stall samples is now refused; an equal-or-larger account
    still supersedes."""
    from rsi_kernel_tune.loop.driver import _save_ref_role_exposure

    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    _save_ref_role_exposure(ctx, _ref_role_block(20000))
    banked = tmp_path / "evidence" / "ref_role_exposure.txt"
    assert "samples=20000" in banked.read_text(encoding="utf-8")
    # a later probe log's FULL table with fewer samples must not replace it
    _save_ref_role_exposure(ctx, _ref_role_block(4000))
    assert "samples=20000" in banked.read_text(encoding="utf-8")
    assert "samples=4000" not in banked.read_text(encoding="utf-8")
    # an equal-or-larger account still supersedes (same rule as the dossier's)
    _save_ref_role_exposure(ctx, _ref_role_block(30000))
    assert "samples=30000" in banked.read_text(encoding="utf-8")


def test_student_role_tables_split_head_and_best(tmp_path):
    """Observed: one fixed file name for both priced trees — head != best priced
    best first (writing its account), then HEAD, whose table overwrote the best
    tree's in student_role_exposure.txt while the diagnose prompt still rendered
    the best tree's table from the same step: evidence file and prompt table
    contradicted each other, and a PARTIAL HEAD table replaced a FULL best one.
    The names now follow the dossiers' head/best split."""
    from rsi_kernel_tune.loop.driver import (
        _STUDENT_BEST_ORCH_DOSSIER_NAME,
        _STUDENT_ORCH_DOSSIER_NAME,
        _STUDENT_ROLE_EXPOSURE_BEST_NAME,
        _STUDENT_ROLE_EXPOSURE_NAME,
        _run_stall_pricing,
    )
    from test_loop_driver import _ROLE_CORE_STU, _STALL_LOG_OK, _role_block, \
        _role_block_partial

    be = FakeBackend([], [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU),
                          _STALL_LOG_OK + _role_block_partial()])
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.evidence_dir = tmp_path
    # the exact sequence _price_student_trees runs when head != best
    _, _, _, best_roles = _run_stall_pricing(
        ctx, ref="sha-best",
        bank_as=(_STUDENT_BEST_ORCH_DOSSIER_NAME,), label="best")
    _, _, _, head_roles = _run_stall_pricing(
        ctx, ref="sha-head",
        bank_as=(_STUDENT_ORCH_DOSSIER_NAME,), label="head")
    assert best_roles is not None and head_roles is not None
    assert best_roles != head_roles
    assert (tmp_path / _STUDENT_ROLE_EXPOSURE_BEST_NAME).read_text(
        encoding="utf-8") == best_roles
    assert (tmp_path / _STUDENT_ROLE_EXPOSURE_NAME).read_text(
        encoding="utf-8") == head_roles


def test_student_role_tables_coincident_head_and_best_bank_both_names(tmp_path):
    """One job prices both trees when they coincide: both bank names carry the
    same table (student_role_exposure.txt keeps its historical content)."""
    from rsi_kernel_tune.loop.driver import (
        _STUDENT_BEST_ORCH_DOSSIER_NAME,
        _STUDENT_ORCH_DOSSIER_NAME,
        _STUDENT_ROLE_EXPOSURE_BEST_NAME,
        _STUDENT_ROLE_EXPOSURE_NAME,
        _run_stall_pricing,
    )
    from test_loop_driver import _ROLE_CORE_STU, _STALL_LOG_OK, _role_block

    be = FakeBackend([], [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.evidence_dir = tmp_path
    _, _, _, roles = _run_stall_pricing(
        ctx, ref="sha-same",
        bank_as=(_STUDENT_BEST_ORCH_DOSSIER_NAME, _STUDENT_ORCH_DOSSIER_NAME),
        label="best")
    assert roles is not None
    assert (tmp_path / _STUDENT_ROLE_EXPOSURE_NAME).read_text(
        encoding="utf-8") == roles
    assert (tmp_path / _STUDENT_ROLE_EXPOSURE_BEST_NAME).read_text(
        encoding="utf-8") == roles


# --- BUG-T1C-01: reused readings entered the resolvable_diff sample pool ------


def _reuse_ratio_rec(it, ms, ratio, origin):
    """A reuse-unchanged / declared-gate-close row: replace(base, ...) kept the
    base row's vs_ref and stamped reading_of (driver._archive_reused_reading)."""
    return replace(IterationRecord(iter=it, decision_id="D", change="c",
                                   correctness=True, verdict="KEEP",
                                   device_median_ms=ms, vs_ref_device=ratio),
                   reading_of=origin)


def test_resolvable_diff_excludes_reused_readings_from_the_pool():
    """Observed (triage's ratio-only direction): 4 reuse duplicates + 2 honest
    rows read rd at the FLOOR — the duplicated values occupied the
    leave-one-out center, their residuals collapsed to 0, and sigma_between
    with it: a resolution the run never earned (KEEP_PATCH margins and plateau
    margins ride the same band). A reused reading is the same measurement
    twice; the pool now excludes it like the driver's own pools do, and the
    same history honestly reports data-poor (the cap)."""
    dup = _ratio_rec(0, 1.0, 1.200)
    reused = [_reuse_ratio_rec(10 + i, 1.0, 1.200, origin=0) for i in range(4)]
    honest = [_ratio_rec(20, 1.0, 1.203), _ratio_rec(21, 1.0, 1.197)]
    assert resolvable_diff(reused + honest, "device") == RESOLVABLE_DIFF_CAP


def test_resolvable_diff_honest_tight_pool_still_resolves():
    """The exclusion is of reuse rows only: an honest tight series still
    resolves to the floor."""
    recs = [_ratio_rec(i, 1.0, 1.20 + (i % 3) * 0.001) for i in range(6)]
    assert resolvable_diff(recs, "device") == RESOLVABLE_DIFF_FLOOR


def test_resolvable_diff_honest_spread_pool_still_lands_between():
    """And an honest spread series still lands strictly inside [floor, cap] —
    the filter did not silently empty the pool."""
    recs = [_ratio_rec(i, 1.0, r) for i, r in
            enumerate([1.200, 1.202, 1.198, 1.201, 1.199, 1.2016])]
    rd = resolvable_diff(recs, "device")
    assert RESOLVABLE_DIFF_FLOOR < rd < RESOLVABLE_DIFF_CAP


def test_resolvable_diff_reused_rows_add_no_resolution():
    """A mixed history reads exactly like its honest subset when the reuse rows
    would have fit the window: duplicating one honest row as a reused reading
    must not move the band."""
    honest = [_ratio_rec(i, 1.0, r) for i, r in
              enumerate([1.200, 1.203, 1.197, 1.201, 1.199, 1.202])]
    base = resolvable_diff(honest[:5], "device")
    dup = _reuse_ratio_rec(99, 1.0, honest[4].vs_ref_device, origin=honest[4].iter)
    # 5 honest rows + 1 reused copy of the last honest row: same 5 real
    # measurements either way
    assert resolvable_diff(honest[:5] + [dup], "device") == base
