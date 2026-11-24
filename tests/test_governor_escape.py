"""The governor's architecture-escape rule (task 14 section 4.5).

Three conditions together: >= 3 student iterations since the M0 gate passed, the best
reading still > 2x the target, and neither of the last 2 iterations a KEEP that improved
the best beyond the 1.5 % noise floor. Then the next diagnose / patch turn is prefixed
with the fixed [ESCAPE RULE] text, once per meta round.
"""

from __future__ import annotations

import json
from pathlib import Path

from forgekernel.loop import governor as gov

M0 = "M0-gate"
DIAGNOSE = "# Meta-Agent — Diagnosis step (operator-agnostic prompt)\n\nbody\n"
PATCH = "# Meta-Agent — Patch step (operator-agnostic prompt)\n\nbody\n"
M0_AUDIT = "# Meta-Agent — Mandatory M0 finish audit (operator-agnostic prompt)\n\nbody\n"
STUDENT = "# Student — One iteration (operator-agnostic prompt)\n\nbody\n"

ESCAPE_PREFIX = "[ESCAPE RULE] "


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Inner:
    def __init__(self):
        self.prompts: list[str] = []

    def run_agent(self, prompt: str, *, role: str | None = None) -> str:
        self.prompts.append(prompt)
        return "ok"


def row(i: int, ms: float, *, did: str = "D1.1", verdict: str = "KEEP",
        correctness=True, id_key: str = "iter") -> dict:
    return {id_key: i, "decision_id": did, "change": "c", "verdict": verdict,
            "correctness": correctness, "device_median_ms": ms, "bwd_median_ms": ms}


def status(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "student" / "status.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def fire_state(rows, target=10.0, **kw):
    return gov.escape_state(rows, target_ms=target, **kw)


# a run that satisfies all three conditions: M0 at 100 ms, one big KEEP to 70 ms, then
# two KEEPs of +0.7 % and +0.4 % — below the 1.5 % floor
FIRING = [row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5), row(3, 69.2)]


def test_all_three_conditions_fire_with_the_actual_numbers():
    state = fire_state(FIRING)
    assert state.fire
    assert state.n == 3
    assert state.best_ms == 69.2 and state.target_ms == 10.0
    assert state.x == 6.92


def test_condition_one_needs_the_m0_gate_and_three_iterations():
    no_gate = [row(0, 100.0), row(1, 70.0), row(2, 69.5), row(3, 69.2)]
    assert not fire_state(no_gate).fire
    assert "M0-gate" in fire_state(no_gate).reason
    # an M0-gate row that did not KEEP does not count as the foundation
    reverted_gate = [row(0, 100.0, did=M0, verdict="REVERT", correctness=False)] + FIRING[1:]
    assert not fire_state(reverted_gate).fire
    # two iterations since the gate is not three
    early = [row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5)]
    state = fire_state(early)
    assert not state.fire and state.n == 2 and "<" in state.reason


def test_condition_two_needs_more_than_twice_the_target():
    # the big win lands before the recent window, so condition three holds throughout
    at_2x = [row(0, 100.0, did=M0), row(1, 20.1), row(2, 20.05), row(3, 20.0)]
    state = fire_state(at_2x)
    assert not state.fire and "within 2x" in state.reason
    above_2x = [row(0, 100.0, did=M0), row(1, 20.1), row(2, 20.05), row(3, 20.1)]
    assert fire_state(above_2x).fire


def test_condition_three_a_recent_keep_beyond_the_noise_floor_prevents_it():
    improved = [row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5), row(3, 60.0)]
    state = fire_state(improved)
    assert not state.fire and "noise floor" in state.reason
    # 2.1 % is well past the 1.5 % floor; 1.4 % is under it
    assert not fire_state([row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5),
                           row(3, 68.5)]).fire
    assert fire_state([row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5),
                       row(3, 69.0)]).fire


def test_a_reverted_recent_row_is_not_an_improvement():
    # a lower reading on a REVERT row (relaxed-revert keeps regressions) must not be
    # mistaken for progress — progress is a KEEP
    rows = [row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5),
            row(3, 60.0, verdict="REVERT", correctness=False)]
    assert fire_state(rows).fire


def test_too_few_evaluated_iterations_since_m0_do_not_fire():
    # three rows since M0, but only one of them carries a device reading
    rows = [row(0, 100.0, did=M0), row(1, 70.0),
            row(2, 69.5, verdict="NEUTRAL", correctness=None),
            row(3, 69.2, verdict="NEUTRAL", correctness=None)]
    rows[2]["device_median_ms"] = None
    rows[3]["device_median_ms"] = None
    state = fire_state(rows)
    assert not state.fire and "evaluated" in state.reason and state.n == 3


def test_no_target_or_no_readings_never_fires():
    assert not gov.escape_state(FIRING, target_ms=None).fire
    assert "no target" in gov.escape_state(FIRING, target_ms=None).reason
    blank = [row(0, 100.0, did=M0), row(1, 70.0, verdict="NEUTRAL", correctness=None),
             row(2, 69.5, verdict="NEUTRAL", correctness=None),
             row(3, 69.2, verdict="NEUTRAL", correctness=None)]
    for r in blank[1:]:
        r["device_median_ms"] = None
        r["bwd_median_ms"] = None
    state = gov.escape_state(blank, target_ms=10.0)
    assert not state.fire and ("no reading" in state.reason
                               or "evaluated" in state.reason)


def test_the_gate_window_selects_the_reading():
    rows = [row(0, 100.0, did=M0)]
    rows[0]["device_median_ms"] = None                  # e2e only
    rows += [row(1, 70.0), row(2, 69.5), row(3, 69.2)]
    for r in rows[1:]:
        r["device_median_ms"] = None
    assert not gov.escape_state(rows, target_ms=10.0, gate_window="device").fire
    assert gov.escape_state(rows, target_ms=10.0, gate_window="e2e").fire


# -- the prompt prefix ----------------------------------------------------------------------

def make(tmp_path: Path, rows: list[dict], *, target=10.0):
    status_path = status(tmp_path, rows)
    policy = gov.GovernorPolicy(hours=100.0, min_student_iters=0, target_ms=target,
                                status_path=status_path)
    ledger = status_path.parent / gov.LEDGER_NAME
    inner = Inner()
    return gov.GovernedBackend(inner, policy, ledger, clock=Clock()), inner, ledger


def test_the_escape_text_prefixes_diagnose_and_patch_once_per_meta_round(tmp_path):
    be, inner, ledger = make(tmp_path, FIRING)
    be.run_agent(DIAGNOSE)
    assert inner.prompts[0].startswith(ESCAPE_PREFIX)
    assert "After 3 student iterations the best reading is still 6.92 times the target" \
        in inner.prompts[0]
    assert "body" in inner.prompts[0]
    be.run_agent(PATCH)                       # same meta round: no second escape
    assert not inner.prompts[1].startswith(ESCAPE_PREFIX)
    be.run_agent(DIAGNOSE)
    assert not inner.prompts[2].startswith(ESCAPE_PREFIX)
    be.run_agent(STUDENT)                     # the meta round ends
    be.run_agent(DIAGNOSE)                    # a new meta round may fire again
    assert inner.prompts[4].startswith(ESCAPE_PREFIX)
    assert not inner.prompts[3].startswith(ESCAPE_PREFIX)
    events = [e["event"] for e in gov.read_ledger(ledger).events]
    assert events.count("escape_fired") == 2
    fired = [e for e in gov.read_ledger(ledger).events if e["event"] == "escape_fired"]
    assert fired[0]["iters"] == 3 and fired[0]["x"] == 6.92
    assert fired[0]["best_ms"] == 69.2 and fired[0]["target_ms"] == 10.0


def test_the_escape_text_goes_to_teacher_phases_only(tmp_path):
    be, inner, _ = make(tmp_path, FIRING)
    be.run_agent(M0_AUDIT)
    be.run_agent("# Recon — Blackbox reconnaissance of the reference (teacher)\n\nbody\n")
    be.run_agent("# Unknown teacher prompt\n")
    assert not any(p.startswith(ESCAPE_PREFIX) for p in inner.prompts)


def test_each_missing_condition_keeps_the_prompt_clean(tmp_path):
    cases = {
        "no gate": [row(0, 100.0), row(1, 70.0), row(2, 69.5), row(3, 69.2)],
        "too few iters": [row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5)],
        "close to target": [row(0, 100.0, did=M0), row(1, 30.0), row(2, 20.5),
                            row(3, 20.0)],
        "recent improvement": [row(0, 100.0, did=M0), row(1, 70.0), row(2, 69.5),
                               row(3, 60.0)],
    }
    for name, rows in cases.items():
        be, inner, _ = make(tmp_path / name, rows)
        be.run_agent(DIAGNOSE)
        assert inner.prompts[0] == DIAGNOSE, name


def test_a_run_without_a_target_never_escapes(tmp_path):
    be, inner, _ = make(tmp_path, FIRING, target=None)
    be.run_agent(DIAGNOSE)
    assert inner.prompts[0] == DIAGNOSE


def test_escape_and_soft_brake_can_share_a_turn(tmp_path):
    status_path = status(tmp_path, FIRING)
    policy = gov.GovernorPolicy(hours=0.001, min_student_iters=10, target_ms=10.0,
                                status_path=status_path)   # share = 1.8 s
    ledger = status_path.parent / gov.LEDGER_NAME
    ledger.write_text(json.dumps({"event": "turn", "role": "teacher", "phase": "diagnose",
                                  "elapsed_s": 2.0}) + "\n", encoding="utf-8")
    inner = Inner()
    be = gov.GovernedBackend(inner, policy, ledger, clock=Clock())
    be.run_agent(DIAGNOSE)
    assert inner.prompts[0].startswith(ESCAPE_PREFIX)
    assert gov.SOFT_BRAKE_TEXT in inner.prompts[0]


def test_a_malformed_status_line_does_not_break_the_turn(tmp_path):
    status_path = tmp_path / "student" / "status.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("not json\n" + json.dumps(FIRING[0]) + "\n"
                           + json.dumps(FIRING[1]) + "\n" + json.dumps(FIRING[2]) + "\n"
                           + json.dumps(FIRING[3]) + "\n", encoding="utf-8")
    policy = gov.GovernorPolicy(hours=100.0, min_student_iters=0, target_ms=10.0,
                                status_path=status_path)
    inner = Inner()
    be = gov.GovernedBackend(inner, policy, status_path.parent / gov.LEDGER_NAME,
                             clock=Clock())
    be.run_agent(DIAGNOSE)
    assert inner.prompts[0].startswith(ESCAPE_PREFIX)
