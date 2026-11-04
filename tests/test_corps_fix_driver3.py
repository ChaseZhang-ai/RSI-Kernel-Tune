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

from peregrine_kernel.inner.trajectory import IterationRecord, append_record
from peregrine_kernel.loop.driver import _save_ref_struct_table
from peregrine_kernel.loop.parse import (
    RESOLVABLE_DIFF_CAP,
    RESOLVABLE_DIFF_FLOOR,
    resolvable_diff,
)

from test_loop_driver import FakeBackend, _ctx

REPO = Path(__file__).resolve().parents[1]


# --- BUG-T5C-01: _save_ref_struct_table's OSError log lost the destination ---


def test_struct_table_write_failure_log_names_the_destination(tmp_path, capsys):
    """Observed: the OSError log read `could not persist ref_struct_table.txt to
    {e}` — the exception text sat in the destination slot and the path it tried
    to write was absent, so a write failure said what went wrong but never where.
    The sibling bank logs all read `could not persist <name> to <dst>: <e>`."""
    from peregrine_kernel.loop.driver import _REF_STRUCT_TABLE_NAME

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
    from peregrine_kernel.loop import lang_mode

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
    from peregrine_kernel.loop.driver import _quarantine_leaky_kb

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
    from peregrine_kernel.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    qdir = tmp_path / "repo" / "kb_quarantine"
    qdir.mkdir(parents=True)
    (qdir / "leak.md").write_text("earlier run's leak\n", encoding="utf-8")
    (kb / "sm100" / "leak.md").write_text("this run's leak\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/leak.md:1: banned narrative"])
    assert (qdir / "leak.md").read_text(encoding="utf-8") == "earlier run's leak\n"
    assert (qdir / "sm100__leak.md").read_text(encoding="utf-8") == "this run's leak\n"


def test_quarantine_unique_names_stay_flat(tmp_path):
    """The incident fix must not rename the common case: a unique basename with
    no collision keeps its plain quarantine name (the existing consumers and
    the human-facing layout are unchanged)."""
    from peregrine_kernel.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    (kb / "sm100" / "oddity.md").write_text("body\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/oddity.md:1: banned narrative"])
    assert (tmp_path / "repo" / "kb_quarantine" / "oddity.md").is_file()


# --- BUG-SEED-33: the seeders read/write without a fixed encoding -------------

_SEED_C_LOCALE_SNIPPET = """
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {src!r})
from peregrine_kernel.loop.driver import _seed_pricing_toolkits, _seed_sass_tools


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
    from peregrine_kernel.loop.driver import (
        _declared_construction,
        _open_package_tail,
    )

    assert _declared_construction(decl) is None
    assert _open_package_tail([_pkg_tail_row(decl)]) is None


def test_wellformed_package_declaration_still_works_on_both_paths():
    """The unification must not tighten away the real contract: a valid
    ``part 1 of 3`` opening still scaffolds and still opens the tail."""
    from peregrine_kernel.loop.driver import (
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
    from peregrine_kernel.loop.driver import _open_package_tail

    segment = [_pkg_tail_row("part 0 of 3 — malformed opener"),
               _pkg_tail_row("part 1 of 2 — real package step")]
    assert _open_package_tail(segment) == (1, 2)
