"""Ports of the audit-r2 archive's monitor / resume regression tests.

Source: audit-r2-2026-11-05/forgekernel-untracked-tests/test_monitor_resume_
fixes.py, ported to the latest tree per the correction order of 2026-11-05.

Subset rationale (full inventory in the porting report of CHANGES-corps-fix.md):
the archive's other twenty tests are covered by this session's wave
(test_corps_fix_lifecycle.py covers the hook-never-ran classification
SEED-14, the marker/JSON rebuild SEED-15, companion pairs SEED-16, hook dump
traces SEED-41, per-process call merges SEED-42; test_corps_fix_parse.py
covers the diff signature SEED-25, the results parser SEED-28 and the torn
trajectory line SEED-43; test_wp15_resolution_confirmation.py covers the
digest's clock header) or their premises were replaced (load_records now drops
any unparseable line instead of raising on a middle one; the digest's clock
telemetry moved to the WP15 clock header; parse_perf no longer carries a
"note"). The two tests below add coverage nothing else pins:

* the final-JSON-killed-mid-dump input shape for parse_log (the session's
  SEED-15 test covers a marker with NO JSON line; a marker followed by a
  TRUNCATED JSON line is a distinct shape a regex/parse fix could separate);
* the companion-closure fixpoint over a three-decision chain (the session's
  SEED-16 tests pin two-decision pairs; a chain must close to all members
  from every changed set).
"""

from __future__ import annotations

import json

from peregrine_kernel.monitor import MARKER, PROFILE_MARKER, classify, parse_log
from peregrine_kernel.resume import Decision, compute_reopen


# --- a killed final JSON still yields the finished profiles --------------------


def _p_rec(profile, card_pass=True):
    """A per-profile record in the shape the runtime runner writes."""
    return {"profile": profile, "card_pass": card_pass, "rc": 0, "seconds": 1.0,
            "cases": [],
            "hook": {"processes": 1, "calls": [], "perturbed": 0,
                     "reseeded": 0, "errors": []}}


def test_parse_log_rebuilds_when_the_final_json_was_killed_mid_dump():
    """Ported from the archive (test_parse_log_rebuilds_partial_when_final_
    json_is_truncated): the job died while DUMPING the final result -- the
    marker printed, a partial JSON line followed. The per-profile blocks still
    on the log are recovered, the verdict flip survives classification, and
    the recovery is named ("cut short after N profiles")."""
    blocks = [_p_rec("base"), _p_rec("repeat", card_pass=False)]
    text = "".join(f"{PROFILE_MARKER}\n{json.dumps(b)}\nnoise\n" for b in blocks)
    text += MARKER + "\n" + '{"profiles":[{"profile":"base"'   # killed mid-dump
    res = parse_log(text)
    assert res is not None and res["complete"] is False
    assert [r["profile"] for r in res["profiles"]] == ["base", "repeat"]
    v = classify(res)
    assert v["status"] == "alarm"          # the verdict flip survives the kill
    assert any("cut short after 2" in f["what"] for f in v["findings"])


# --- a companion chain closes to the fixpoint ----------------------------------

def test_reopen_follows_a_companion_chain_to_the_fixpoint():
    """Ported from the archive (test_reopen_follows_a_companion_chain_to_the_
    fixpoint): companions are symmetric edges, so a chain A (x) B (x) C must
    close to all three members no matter which end is changed -- one ordered
    pass over the trajectory is not enough, the closure runs to the fixpoint."""
    traj = [Decision("A", companions={"B"}), Decision("B", companions={"C"}),
            Decision("C")]
    for changed in ("A", "B", "C"):
        assert compute_reopen(traj, {changed}).reopen == ["A", "B", "C"]
