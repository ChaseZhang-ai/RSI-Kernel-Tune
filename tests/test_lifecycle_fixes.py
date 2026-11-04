"""Regression tests ported from the upstream fix wave (lifecycle batch).

Only the mechanisms that exist in this lineage are pinned; each names the fix it
carries:

* a run the loop itself stopped at its target AND with a clean confirmation on
  file completes (exit 0, done, target_met) -- the positive control of the
  confirmed-at-target contract;
* milestone section presence is reported when a required heading is missing, and
  section order is reported when two required headings are swapped.

Dropped from the source file, with reasons (see the BUG-PORT ledger where the
old test exposed a live bug here):

* stop=target with an alarmed best / with no confirmation keeping the run at 2:
  drive() here reads ``stop == target`` alone as DONE (BUG-PORT: the
  confirmed-at-target gate only tightens ``met``, it never gates the stop);
* task names validated at the creation entry points: cmd_init accepts any name,
  including ones that escape tasks/ (BUG-PORT);
* a malformed board card being a TaskError: the retired board task loader
  does not exist in this package (the card judge is benchkit's task protocol);
* git failures around committing/reporting being loud: taskcard._commit treats a
  failed ``git status`` as "nothing to do" (BUG-PORT) and kb_changes returns []
  silently; the run-log tee files are never closed;
* the status command surviving a torn status.jsonl line: status_text json.loads every
  nonempty line and crashes on a torn one (BUG-PORT);
* the baseline CLI degrading narrowly and reporting bad archives: _submit_cluster
  deliberately degrades to "" on any exception (its docstring owns that), and
  --show / --from-json surface a corrupt archive or bad JSON as an uncaught
  traceback instead of "[baseline] error:" + rc 4 (BUG-PORT);
* a "### 4." sub-heading standing in for the real "## 4.": the required-section
  presence check here is still substring-based (``text.find``), while the §3
  body extraction below it is line-start anchored (BUG-PORT) -- so only the
  missing-section and out-of-order halves of that fix port.
"""

from __future__ import annotations

import pytest

from peregrine_kernel import launch
from peregrine_kernel.milestone import validate_milestone

from test_launch import (  # noqa: F401 -- the repo fixture and the end-to-end stand-ins
    _confirmed, _loop_ends, _start, _task, _Teacher, repo,
)
from test_milestone import VALID


# -- fix: `peregrine run` completes only on a confirmed-at-target best (positive
#    control; the unconfirmed halves are a BUG-PORT entry) -------------------------


def test_stop_at_target_with_a_confirmed_best_still_completes(repo, monkeypatch):
    from peregrine_kernel.loop import cli as loop_cli
    from peregrine_kernel.runs import list_runs
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop="target",
                                                     confirm=_confirmed(1.3)))
    rc = _start(_task(repo))
    rec = list_runs()[0].record()
    assert rc == 0
    assert rec["status"] == "done" and rec["target_met"] is True


# -- fix: milestone section presence / order (the line-start halves are a
#    BUG-PORT entry; these two halves hold) ----------------------------------------


def test_a_missing_section_4_is_still_reported():
    text = VALID.replace("## 4. Known dead ends", "## 4b. Dead ends live elsewhere")
    assert any("missing required section: ## 4." in x for x in validate_milestone(text))


def test_sections_out_of_order_are_still_reported():
    sec5 = "## 5. Global correctness / performance gate\natol=5, rtol=0.05; MFU ≥95% target.\n\n"
    sec6 = "## 6. Rules\nOne variable per round; anti-plagiarism clause; INFEASIBLE clause.\n\n"
    swapped = VALID.replace(sec5 + sec6, sec6 + sec5)
    assert swapped != VALID
    assert any("out of order" in x for x in validate_milestone(swapped))
