"""Ports of the audit-r2 archive's guard / eval regression tests.

Source: the audit-r2-2026-11-05 archive's untracked test directory (legacy-snapshot
naming), file test_guard_fixes.py, ported to the latest tree per the correction
order of 2026-11-05.

Subset rationale (full inventory in the porting report of CHANGES-corps-fix.md):
the archive's other eleven tests are covered by this session's wave
(test_corps_fix_guards.py covers the root-shadow rule, the multi-name import
scan, load_entry's sys.path position; test_corps_fix_misc.py covers the
host-path note; test_corps_fix_benchkit.py covers the precision length
mismatch) or their premises were removed with the WP24 de-boarding (the
external benchmark-board judge, the board task cards, card_harness listings of
the board surface) or no longer hold on the current timing.py aggregate (the
workloads_performance degrade-to-None behavior: the aggregates are rounded
directly today). The four tests below add coverage nothing else pins:

* the git pathspec wiring that carries the root-module glob into the delivery
  and hygiene checks;
* the card/tree harness listings naming a root shadow module (card_harness_
  files is only ever asserted None elsewhere);
* the import-scan spelling variants the session wave does not cover (submodule
  imports, as-bindings, two statements on one line, parenthesized from-imports,
  and the quiet non-literal import_module target);
* _geomean's usable-reading contract (the happy aggregates are covered in
  test_multi_workload_perf.py; the degradation edge is asserted nowhere) --
  rewritten, because the archive's contract changed: a missing reading is
  skipped today, only a non-positive one degrades the aggregate.

Note the archive's wrong-source import bug (it imported BASIS_CUPTI from the
timing_audit module): BASIS_CUPTI lives in ``rsi_kernel_tune.baseline`` -- not
needed here because the host-path-note tests it served are covered by
test_corps_fix_misc.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from rsi_kernel_tune.loop import goal_ref as gr
from rsi_kernel_tune.loop import harness_guard

from test_template_timing import FakeGpu, load_timing


# --- harness: the git pathspecs carry the root-module glob ---------------------

def test_git_pathspecs_carry_the_root_module_glob():
    """The root-shadow rule is enforced by is_harness_path, but the git-level
    delivery and hygiene checks (git diff / ls-files in loop/cli.py) must ask
    for the root glob too -- only the depth is narrowed later, by
    is_harness_path. Without the glob in HARNESS_PATHSPECS a root
    sitecustomize.py would be invisible to every git query before the depth
    filter ever runs."""
    assert harness_guard.ROOT_PY_PATHSPEC == "*.py"
    assert "*.py" in harness_guard.HARNESS_PATHSPECS
    assert tuple(p.rstrip("/") for p in harness_guard.HARNESS_PREFIXES) \
        == harness_guard.HARNESS_PATHSPECS[:-1]
    # the exclusion argument is optional: with no kernel file named, a root
    # shadow module still reads as harness (the archive's "no kernel exclusion"
    # default survives; the old "every root *.py is harness" rule it served was
    # replaced by the shadow-name rule, so kernel.py itself is not harness)
    assert harness_guard.is_harness_path("sitecustomize.py")
    assert not harness_guard.is_harness_path("kernel.py")


# --- harness: the card and tree listings carry root modules --------------------


def _git(card: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(card), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _card_repo(tmp_path: Path) -> Path:
    card = tmp_path / "card"
    (card / "eval").mkdir(parents=True)
    (card / "kernel.py").write_text("def run(): pass\n")
    (card / "eval/bench.py").write_text("print('judge')\n")
    (card / "gpu_entry.sh").write_text("#!/bin/bash\n")
    (card / "sitecustomize.py").write_text("# the card's own root helper\n")
    (card / "deep").mkdir()
    (card / "deep/lib.py").write_text("x = 1\n")          # not root: not harness
    _git(card, "init", "-q", "-b", "main")
    _git(card, "add", "-A")
    _git(card, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "card")
    return card


def test_tree_and_card_listings_carry_root_modules(tmp_path):
    """tree_harness_files (the delivery/hygiene diff source) and card_harness_
    files (the overlay source) must both surface a root shadow module the card
    itself ships, alongside the classic eval/ + gpu_entry.sh harness -- with or
    without a kernel file named (the root shadow is not the kernel file, so no
    exclusion touches it)."""
    card = _card_repo(tmp_path)
    head = _git(card, "rev-parse", "HEAD")
    want = {"eval/bench.py", "gpu_entry.sh", "sitecustomize.py"}
    assert set(harness_guard.tree_harness_files(card, head, "kernel.py")) == want
    assert set(harness_guard.card_harness_files(card, "kernel.py")) == want
    assert "sitecustomize.py" in harness_guard.tree_harness_files(card, head)
    assert "sitecustomize.py" in harness_guard.card_harness_files(card)


# --- goal-ref: spelling variants the scan cannot be escaped with ---------------

def test_the_import_scan_cannot_be_escaped_by_spelling_variants():
    """Ported (subset) from the archive's test_every_import_name_is_scanned:
    the comma list, the parenthesized single import and the dynamic
    import_module/__import__ forms are pinned by test_corps_fix_guards
    (BUG-SEED-08); the variants here are the ones nothing else covers."""
    cases = [
        "import torch\nimport flashinfer.rope\n",        # a submodule still names the lib
        "import flashinfer as fi, torch\n",              # an as-binding on the way
        "import a; import flashinfer\n",                 # two statements on one line
        "from flashinfer.rope import (\n    apply_rope,\n    Rope,\n)\n",
    ]
    for i, src in enumerate(cases):
        found = gr.library_findings(src, "kernel.py")
        assert any("flashinfer" in f for f in found), f"case {i} escaped: {found}"
    # a variable target cannot be read: the heuristic names what it can see only
    assert gr.library_findings('importlib.import_module(flashinfer_name)\n',
                               "kernel.py") == []


# --- timing: an aggregate never averages over a subset -------------------------

def test_geomean_skips_missing_readings_and_degrades_on_a_non_positive_one():
    """Rewritten from the archive's test_geomean_degrades_to_none_on_a_missing_
    or_non_positive_reading. The archive's contract degraded the aggregate to
    None on ANY unusable reading, a missing one included; the current contract
    splits the cases: a missing (None) reading is skipped (the geomean runs
    over the usable readings), while a non-positive reading still poisons the
    WHOLE aggregate to None -- never a geomean over the surviving subset. The
    poisoning half is what the workload verdicts read, and it is asserted
    nowhere else."""
    timing = load_timing(FakeGpu(device_ms=1.0))
    assert timing._geomean([1.0, 2.0]) == pytest.approx(math.sqrt(2.0))
    # a missing reading is skipped, not fatal: the usable subset aggregates
    assert timing._geomean([1.0, None, 2.0]) == pytest.approx(math.sqrt(2.0))
    # a non-positive reading poisons: no subset average may stand in
    assert timing._geomean([1.0, 0.0, 2.0]) is None
    assert timing._geomean([1.0, -2.0, 2.0]) is None
    assert timing._geomean([]) is None
