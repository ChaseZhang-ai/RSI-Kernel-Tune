"""Seeding and evidence-bank integrity: encoding, quarantine, probe degradation.

The driver's side channels (vendored probe toolkits, the KB quarantine, the
verdict-probe jobs, the banked pricing evidence, the package declarations) each
had one integrity hole:

- seeding read/wrote the vendored files with the locale codec, so a non-UTF-8
  node died on the first non-ASCII byte (BUG-SEED-33);
- the KB quarantine flattened ``<arch>/`` to the file name and POSIX-renamed
  over whatever was there, so two archs' same-named entries destroyed each
  other's evidence (BUG-SEED-49);
- a verdict probe's commit/push/submit ran bare, so one flaked GPU job
  clean-stopped the whole run_two_layer loop (BUG-SEED-35);
- the ref role-exposure table had no FULL-vs-FULL identity guard, and the
  student roles ledger had one name for two trees, so a head pricing job
  overwrote the best tree's just-banked table (BUG-SEED-32);
- ``part 0 of n`` was a malformed declaration the pricing side refused but the
  scaffold side accepted, exempting a no-op row from eval (BUG-SEED-34);
- the struct-table bank log put the exception text in the destination slot
  (BUG-T5C-01).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from rsi_kernel_tune.inner.trajectory import IterationRecord, append_record, load_records
from rsi_kernel_tune.loop.driver import (
    BackendError,
    _declared_construction,
    _open_package_tail,
    _price_student_trees,
    _quarantine_leaky_kb,
    _read_ref_struct_table,
    _run_meta_verdict_probe,
    _run_probe_manifest_batch,
    _save_ref_role_exposure,
    _save_ref_struct_table,
)
from rsi_kernel_tune.loop.driver import run_inner_loop
from test_loop_driver import (
    EVAL,
    NCU,
    _PART,
    _STALL_LOG_OK,
    FakeBackend,
    _ctx,
    _stall_ctx,
)

REPO = Path(__file__).resolve().parents[1]


# --- BUG-SEED-33: seeding pins encoding="utf-8" -------------------------------

_CHILD = textwrap.dedent("""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from rsi_kernel_tune.loop import driver

    class RBE:
        def commit(self, msg):
            pass

    root = Path(tempfile.mkdtemp(prefix="seed-locale-"))
    done = []

    def check(name, fn):
        fn()
        done.append(name)

    # the vendored skills modules carry non-ASCII text (arrows, multiplication
    # signs); every read/write here is locale-decoded without the fix
    check("_seed_sass_tools", lambda: driver._seed_sass_tools(
        SimpleNamespace(recon_dir=root / "recon", oracle_mode="clearbox"), RBE()))
    check("_seed_pricing_toolkits", lambda: driver._seed_pricing_toolkits(
        root / "price", RBE()))
    check("_seed_sweep_toolkits", lambda: driver._seed_sweep_toolkits(
        SimpleNamespace(workspace=root / "sweep"), RBE()))
    check("_mc_seed_toolkit", lambda: driver._mc_seed_toolkit(
        SimpleNamespace(workspace=root / "mc"), RBE()))

    # the settled path: an already-banked prescreen copy is compared against
    # the (non-ASCII) source before the skip is decided
    (root / "mc2" / "notes").mkdir(parents=True)
    (root / "mc2" / "notes" / "mc_prescreen.py").write_text(
        driver._MC_PRESCREEN_SRC, encoding="utf-8")
    check("_mc_seed_toolkit (settled compare)", lambda: driver._mc_seed_toolkit(
        SimpleNamespace(workspace=root / "mc2"), RBE()))

    # the struct table banks probe output verbatim, so it may be non-ASCII too
    (root / "st").mkdir()
    st = SimpleNamespace(recon_path=root / "st" / "report.md")
    check("_save_ref_struct_table", lambda: driver._save_ref_struct_table(
        st, "=== STRUCT TABLE BEGIN ===\\nwarps=8 ~ 64x64, UE-8? no: \\u2248 tile\\n"
        "=== STRUCT TABLE END ==="))
    table = driver._read_ref_struct_table(st)
    assert table is not None and "\\u2248" in table, "banked table lost its text"

    assert list((root / "recon" / "notes").glob("*.py")), "no vendor file landed"
    print("OK", len(done))
""")


def test_seeding_survives_a_non_utf8_locale(tmp_path):
    """Under LC_ALL=C every vendored skills file (they all carry non-ASCII
    text) made the seeding raise UnicodeDecodeError on read or
    UnicodeEncodeError on write; the struct table lost the same way. Each
    read/write now pins encoding=utf-8, so a non-UTF-8 node seeds and reads
    back byte-true."""
    script = tmp_path / "seed_under_c_locale.py"
    script.write_text(_CHILD, encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LC_", "LANG", "PYTHONUTF8", "PYTHONCOERCECLOCALE"))}
    env.update(LC_ALL="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0",
               PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, str(script)], env=env, cwd=str(REPO),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"seeding failed under LC_ALL=C:\n{r.stdout}{r.stderr}"
    assert r.stdout.startswith("OK ")


# --- BUG-SEED-49: the quarantine keeps <arch>/ and never renames over ----------

def test_quarantine_keeps_each_archs_path_and_never_renames_over(tmp_path):
    """sm90/overview.md and sm100/overview.md are two different leaky entries;
    flattened to qdir/<src.name> the second rename silently destroyed the
    first's quarantined evidence. The quarantine keeps the finding's
    ``<arch>/<entry>.md`` shape, and an occupied destination is renamed aside,
    not overwritten."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    kb = tmp_path / "hw_kb" / "cutedsl"
    for arch in ("sm90", "sm100"):
        (kb / arch).mkdir(parents=True)
        (kb / arch / "overview.md").write_text(f"leak narrative {arch}\n", encoding="utf-8")
    ctx.kb_dir = kb
    ctx.kb_arch = "sm100"

    _quarantine_leaky_kb(ctx, ["sm90/overview.md:1: banned narrative",
                               "sm100/overview.md:2: banned narrative"])
    qdir = tmp_path / "kb_quarantine"
    # Adapted at the merge to this tree's quarantine naming (BUG-SEED-49): the
    # destination flattens the entry's relative path (``sm90/overview.md`` ->
    # ``sm90__overview.md``), so two archs' same-named entries never rename onto
    # one file; a residual collision (a root-level lookalike) breaks the tie by
    # number.
    assert (qdir / "sm90__overview.md").read_text(encoding="utf-8") == "leak narrative sm90\n"
    assert (qdir / "sm100__overview.md").read_text(encoding="utf-8") == "leak narrative sm100\n"
    assert not (kb / "sm90" / "overview.md").exists()
    assert not (kb / "sm100" / "overview.md").exists()

    # a re-quarantine of the same relative path lands beside the first one
    (kb / "sm100" / "overview.md").write_text("leak again\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/overview.md:1: banned narrative"])
    assert (qdir / "sm100__overview.md").read_text(encoding="utf-8") == "leak narrative sm100\n"
    assert (qdir / "sm100__overview__1.md").read_text(encoding="utf-8") == "leak again\n"


# --- BUG-SEED-35: a flaked verdict probe degrades, the run continues -----------

class _FlakedBackend:
    """Every side-effect fails after its retries: one GPU flake, no job."""

    def commit(self, msg):
        raise BackendError("git push failed after retries")

    def push_branch(self, branch):
        raise BackendError("git push failed after retries")

    def submit_gpu(self, branch, mode):
        raise BackendError("node lost the job: no runner came back")


def test_a_flaked_verdict_probe_degrades_instead_of_ending_the_run(tmp_path, capsys):
    """The ablation probe and the stall pricing degrade locally on a flaked job
    (empty reading + one log line); the verdict probe's commit/push/submit let
    the BackendError climb into run_two_layer and clean-stop the whole run.
    Both verdict-probe paths now degrade the same way, and the caller's
    per-script fallback consumes the degraded verdict without dying."""
    from rsi_kernel_tune.loop.driver import (
        _probe_label,
        _probe_log_suffix,
        _verdict_probe_feedback,
    )

    ctx = _ctx(tmp_path, FakeBackend([], []))
    be = _FlakedBackend()
    pv, log, readings = _run_meta_verdict_probe(ctx, be, "fs-demo-metaprobe",
                                                "notes/p.py", 1)
    assert pv.passed is False and pv.found is False
    assert pv.reason.startswith("job failed (git push failed after retries)")
    assert log == "" and readings == []
    # this tree announces the degrade through the verdict reason (the teacher
    # prompt), not a stdout log line

    # the WP19 batch path degrades the same way: None is its "never really ran"
    # answer, which the caller reads as "fall back to the per-script path"
    assert _run_probe_manifest_batch(ctx, be, "fs-demo-metaprobe",
                                     ["notes/a.py", "notes/b.py"]) is None
    assert "meta: probe manifest batch job failed" in capsys.readouterr().out

    # the fallback loop (per-script probes over the same flaked backend) runs
    # to the end of the manifest: every round degrades, nothing re-raises
    for mp in ("notes/a.py", "notes/b.py"):
        pv_mp, log_mp, readings_mp = _run_meta_verdict_probe(
            ctx, be, "fs-demo-metaprobe", mp, 2)
        assert _probe_label(pv_mp) == "NO VERDICT" and readings_mp == []
        # the round's loop-log line names the failure (the caller appends this)
        assert "job failed" in _probe_log_suffix(pv_mp)
        feedback = _verdict_probe_feedback(mp, pv_mp, log_mp, 2, 4)
        assert "NO VERDICT" in feedback


# --- BUG-SEED-32: role-exposure banking guards ---------------------------------

def _role_block_text(role, samples, tensor):
    """A full ROLE-OK block whose identity (role label / sample count / tensor
    reading) is parametrizable, so two blocks stand for two different kernels."""
    core = ('ROLE-JSON: {"duration_ns": 874880, "resolved_pct": 97.2, "roles": {'
            f'"{role}": {{"exposed_ns": 118984, "exposed_pct": 13.6, "regions": 3, '
            f'"share_pct": 80.0, "stalls": {{"wait": 51.0}}}}}}, '
            f'"tensor_active": {tensor}, "tensor_source": "pm-plateau"}}')
    return ("=== ROLE EXPOSURE BEGIN ===\n"
            f"[join] 900 sampled PCs onto 12 regions; 97.2% of {samples} stall samples "
            "resolved\n"
            f"[tensor] active={tensor}% source=pm-plateau duration_ns=874880\n"
            f"ROLE label={role} regions=3 samples={samples} share=80.0 stalls=wait:51\n"
            + core + "\n"
            "=== ROLE EXPOSURE END ===\n"
            f"ROLE-OK roles=1 samples={samples} resolved=97.2 tensor={tensor}\n")


def test_a_smaller_full_role_table_does_not_replace_the_banked_one(tmp_path, capsys):
    """The dossier banking guard, in roles form: a later log carrying only a
    bystander kernel's FULL table (fewer stall samples) must not silently
    replace the main kernel's banked account; an equal-or-larger full table
    still supersedes."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    main = _role_block_text("mma-consumer", 20000, 83.0)
    bystander = _role_block_text("copy-producer", 800, 90.0)

    _save_ref_role_exposure(ctx, "noise\n" + main + "tail\n")
    _save_ref_role_exposure(ctx, "noise\n" + bystander + "tail\n")
    banked = (tmp_path / "evidence" / "ref_role_exposure.txt").read_text(encoding="utf-8")
    assert "mma-consumer" in banked and "20000" in banked
    assert "copy-producer" not in banked
    assert "kept the banked account (20000 stall samples) over this log's smaller one (800)" \
        in capsys.readouterr().out

    # a larger full table supersedes, as before
    _save_ref_role_exposure(ctx, _role_block_text("mma-consumer", 26000, 84.0))
    banked = (tmp_path / "evidence" / "ref_role_exposure.txt").read_text(encoding="utf-8")
    assert "26000" in banked


def _two_tree_ctx(tmp_path, best_log, head_log):
    """A segment whose best KEEP (shabest) is not the tree under judgment
    (sha0): two pricing jobs, the shape of every entry after the first."""
    class _RefPushing(FakeBackend):
        def push_ref(self, ref, branch):
            self.calls.append(("push_ref", ref, branch))

    be = _RefPushing([], [best_log, head_log])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, device_median_ms=2.0, sha="shabest"))
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="M0-gate", change="rebuild", correctness=True,
        verdict="KEEP", bwd_median_ms=2.4, device_median_ms=2.4, sha="sha0"))
    return ctx


def test_head_and_best_role_tables_bank_under_their_own_names(tmp_path):
    """The dossiers' best/HEAD name split, carried over to the roles ledger:
    with head != best the head pricing job must not overwrite the one file the
    best tree's job just banked (both wrote student_role_exposure.txt before,
    and the head job ran second)."""
    best_log = _STALL_LOG_OK + _role_block_text("mma-consumer", 20000, 83.0)
    head_log = _STALL_LOG_OK + _role_block_text("copy-producer", 4000, 61.0)
    ctx = _two_tree_ctx(tmp_path, best_log, head_log)

    priced = _price_student_trees(ctx, load_records(ctx.status_path))
    ev = tmp_path / "evidence"
    # Adapted at the merge to this tree's banking contract: the best tree's
    # account rides student_role_exposure_best.txt; the HEAD-tree pricing pass
    # reads roles only to discard them (persist_roles=False) -- the banked
    # evidence must not describe a different tree than the verdicts beside it.
    best_banked = (ev / "student_role_exposure_best.txt").read_text(encoding="utf-8")
    assert "mma-consumer" in best_banked and "83.0" in best_banked
    assert not (ev / "student_role_exposure.txt").exists()
    # the diagnose view's ledger and the banked best file are the same reading
    assert priced.roles == best_banked


# --- BUG-SEED-34: `part 0 of n` is malformed on both sides ----------------------

def test_part_zero_declaration_is_malformed_on_both_sides():
    """``part 0 of n`` built nothing: the pricing side (_open_package_tail)
    refused it, but the scaffold side exempted the row from eval — one
    declaration, two opposite readings. Both now read it as malformed, and a
    well-formed step still scaffolds/prices as before."""
    assert _declared_construction("part 0 of 3 of build step — D0.1: skeleton") is None
    assert _declared_construction("part 1 of 3 of build step — D0.1: skeleton") \
        == "part 1 of 3"
    assert _declared_construction("part 3 of 3 — done") is None      # finished
    seg = [IterationRecord(iter=0, decision_id="D0.1",
                           change="part 0 of 3 of build step — D0.1: skeleton",
                           correctness=True, verdict="KEEP", device_median_ms=2.0,
                           sha="sha0")]
    assert _open_package_tail(seg) is None          # pricing side: unchanged


def test_a_part_zero_row_is_evaluated_like_any_other_row(tmp_path):
    """The harm was the free pass: a 'part 0 of 3' delivery used to be committed
    NEUTRAL (scaffold) without a GPU eval. It now prices like any other row —
    the full eval runs and its verdict lands."""
    be = FakeBackend([_PART("D0.12", 0, 3)], [EVAL(0.0, ok=False), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    assert run_inner_loop(ctx, start_iter=0, initial_green=None) == "max_iters"
    recs = load_records(ctx.status_path)
    assert [r.verdict for r in recs] == ["REVERT"]     # evaluated, not scaffolded
    assert ("gpu", "all") in be.calls                  # a full eval was bought


# --- BUG-T5C-01: the struct-table bank log names dst, then the error -----------

def test_struct_table_log_names_the_destination_then_the_error(tmp_path, capsys):
    """The OSError line put the exception text in the destination slot
    ('could not persist ref_struct_table.txt to [Errno 2] ...'). Every sibling
    bank log reads 'to <dst>: <error>'; match them."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = tmp_path / "notadir" / "report.md"   # parent missing: write fails
    _save_ref_struct_table(
        ctx, "=== STRUCT TABLE BEGIN ===\nwarps=8\n=== STRUCT TABLE END ===")
    dst = tmp_path / "notadir" / "ref_struct_table.txt"
    assert f"could not persist ref_struct_table.txt to {dst}: [Errno" \
        in capsys.readouterr().out


def test_struct_table_round_trips_non_ascii_text(tmp_path):
    """Same encoding family as the seeding: the table is probe output and may
    carry non-ASCII; bank it and read it back byte-true."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.recon_path.parent.mkdir(parents=True)
    _save_ref_struct_table(
        ctx, "=== STRUCT TABLE BEGIN ===\nwarps=8 ≈ 64x64 tile\n=== STRUCT TABLE END ===")
    assert "≈" in _read_ref_struct_table(ctx)
