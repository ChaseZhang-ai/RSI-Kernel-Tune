"""The throughput governor (loop/governor.py): phases, the two brakes, the ledger, and
the policy. The escape rule has its own file (test_governor_escape.py)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from rsi_kernel_tune.loop import governor as gov
from rsi_kernel_tune.loop.driver import AGENT_ROLE_STUDENT, AGENT_ROLE_TEACHER, AgentError

REPO = Path(__file__).resolve().parents[1]

STUDENT_PROMPT = "# Student — One iteration (operator-agnostic prompt)\n\nbody\n"
DIAGNOSE_PROMPT = "# Meta-Agent — Diagnosis step (operator-agnostic prompt)\n\nbody\n"
PATCH_PROMPT = "# Meta-Agent — Patch step (operator-agnostic prompt)\n\nbody\n"
RECON_PROMPT = "# Recon — Blackbox reconnaissance of the reference (teacher)\n\nbody\n"


class Clock:
    """A controllable monotonic clock."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class Inner:
    """A DriverBackend whose agent turn records the prompt and costs clock time."""

    def __init__(self, clock: Clock | None = None, cost: float = 0.0, reply: str = "ok"):
        self.clock = clock
        self.cost = cost
        self.reply = reply
        self.prompts: list[str] = []
        self.roles: list[str | None] = []
        self.kernel_file = "kernel.py"

    def run_agent(self, prompt: str, *, role: str | None = None) -> str:
        self.prompts.append(prompt)
        self.roles.append(role)
        if self.clock is not None:
            self.clock.advance(self.cost)
        return self.reply

    def submit_gpu(self, branch: str, mode: str) -> str:
        return f"job {branch} {mode}"

    def marker(self) -> str:
        return "inner-marker"


class RolelessInner:
    def __init__(self):
        self.prompts: list[str] = []

    def run_agent(self, prompt: str) -> str:      # no role= kwarg, like old backends
        self.prompts.append(prompt)
        return "ok"


def write_status(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(i: int, ms: float | None = 10.0, *, did: str = "D1.1", verdict: str = "KEEP",
        correctness=True) -> dict:
    return {"iter": i, "decision_id": did, "change": "c", "verdict": verdict,
            "correctness": correctness, "device_median_ms": ms, "bwd_median_ms": ms}


def make(tmp_path: Path, *, hours=1.0, teacher_share=0.5, min_student_iters=5,
         max_teacher_turns_per_phase=12, iters=0, target_ms=None) -> tuple:
    status = tmp_path / "student" / "status.jsonl"
    status.parent.mkdir(parents=True, exist_ok=True)
    write_status(status, [row(i) for i in range(iters)])
    policy = gov.GovernorPolicy(hours=hours, teacher_share=teacher_share,
                                min_student_iters=min_student_iters,
                                max_teacher_turns_per_phase=max_teacher_turns_per_phase,
                                target_ms=target_ms, status_path=status)
    ledger = tmp_path / "student" / gov.LEDGER_NAME
    return policy, ledger, status


# -- phase identification -------------------------------------------------------------------

_META_PHASES = {
    "recon_prompt.md": "recon",
    "distill_prompt.md": "distill",
    "distill_blackbox_prompt.md": "distill",
    "diagnose_prompt.md": "diagnose",
    "patch_prompt.md": "patch",
    "m0_audit_prompt.md": "m0_audit",
    "kb_harvest_prompt.md": "kb",
    "kb_recipe_prompt.md": "kb",
    "student_prompt.md": "student",
    # not phases of the two-layer loop (KLIB one-turn prompts / appended fragments; the
    # cost-model prompt is `forge roofline --estimate`'s one teacher turn, outside a run)
    "seed_adapt_prompt.md": "other",
    "baseline_rewrite_prompt.md": "other",
    "cost_model_prompt.md": "other",
    "verdict_probe_brief.md": "other",
    "verdict_probe_brief_audit.md": "other",
}


def test_every_meta_prompt_first_line_maps_to_its_phase():
    """Every prompt family in meta/ (language variants included) is recognised."""
    seen = set()
    for path in sorted((REPO / "meta").rglob("*.md")):
        expected = _META_PHASES.get(path.name)
        assert expected is not None, f"meta/{path.relative_to(REPO / 'meta')} has no phase"
        first = gov.first_line(path.read_text(encoding="utf-8"))
        assert gov.detect_phase(first) == expected, (path, first)
        seen.add(path.name)
    assert seen == set(_META_PHASES)


def test_the_judge_prompt_is_recognised():
    first = ('You are writing the correctness judge for the operator task "rmsnorm".\n\n'
             "The task's description:")
    assert gov.detect_phase(first) == "judge"


def test_leading_blank_lines_and_unknown_titles():
    assert gov.detect_phase("\n\n" + STUDENT_PROMPT) == "student"
    assert gov.detect_phase("# Something else entirely\n") == "other"
    assert gov.detect_phase("") == "other"
    assert gov.detect_phase("   \n\n") == "other"


def test_resolve_role_prefers_the_explicit_role():
    assert gov.resolve_role("teacher", "student") == "teacher"
    assert gov.resolve_role("student", "diagnose") == "student"
    assert gov.resolve_role(None, "student") == "student"
    assert gov.resolve_role(None, "diagnose") == "teacher"
    assert gov.resolve_role(None, "other") == "student"   # never stop what we cannot name
    assert gov.resolve_role("nonsense", "patch") == "teacher"


# -- transparency ---------------------------------------------------------------------------

def test_below_every_limit_the_prompt_is_byte_identical(tmp_path):
    policy, ledger, _ = make(tmp_path, hours=10.0)
    clock = Clock()
    inner = Inner(clock, cost=0.0)
    be = gov.GovernedBackend(inner, policy, ledger, clock=clock)
    out = be.run_agent(DIAGNOSE_PROMPT)
    assert out == "ok"
    assert inner.prompts == [DIAGNOSE_PROMPT]
    out = be.run_agent(STUDENT_PROMPT, role=AGENT_ROLE_STUDENT)
    assert inner.prompts[-1] == STUDENT_PROMPT


def test_attributes_and_methods_pass_through_and_writes_reach_the_inner(tmp_path):
    policy, ledger, _ = make(tmp_path)
    inner = Inner()
    be = gov.GovernedBackend(inner, policy, ledger)
    assert be.kernel_file == "kernel.py"
    assert be.marker() == "inner-marker"
    assert be.submit_gpu("b", "all") == "job b all"
    be.kernel_file = "other.py"                 # post-construction assignment
    assert inner.kernel_file == "other.py"
    assert be.kernel_file == "other.py"


def test_a_roleless_inner_backend_is_called_with_the_prompt_only(tmp_path):
    policy, ledger, _ = make(tmp_path)
    inner = RolelessInner()
    be = gov.GovernedBackend(inner, policy, ledger)
    assert be.run_agent(STUDENT_PROMPT) == "ok"
    assert inner.prompts == [STUDENT_PROMPT]


def test_govern_with_no_policy_returns_the_inner_backend(tmp_path):
    inner = Inner()
    assert gov.govern(inner, None, tmp_path / "l.jsonl") is inner
    policy, ledger, _ = make(tmp_path)
    wrapped = gov.govern(inner, policy, ledger)
    assert isinstance(wrapped, gov.GovernedBackend)
    assert gov.govern(wrapped, policy, ledger) is wrapped     # idempotent


# -- the teacher-share rule -----------------------------------------------------------------

def test_teacher_share_soft_brakes_at_the_share_and_stops_at_1_25x(tmp_path):
    # hours 0.1 x share 0.5 = 180 s; hard = 225 s; each turn costs 100 s
    policy, ledger, _ = make(tmp_path, hours=0.1, iters=0)
    clock = Clock()
    inner = Inner(clock, cost=100.0)
    be = gov.GovernedBackend(inner, policy, ledger, clock=clock)

    be.run_agent(DIAGNOSE_PROMPT)              # used 0 -> 100
    be.run_agent(DIAGNOSE_PROMPT)              # used 100 -> 200, still < 180? no: 100 < 180
    assert not inner.prompts[0].startswith(gov.SOFT_BRAKE_TEXT)
    assert not inner.prompts[1].startswith(gov.SOFT_BRAKE_TEXT)

    be.run_agent(DIAGNOSE_PROMPT)              # used 200 >= 180 -> soft brake
    assert inner.prompts[2].startswith(gov.SOFT_BRAKE_TEXT)
    assert "body" in inner.prompts[2]          # the turn still ran

    with pytest.raises(gov.GovernorStop) as ei:   # used 300 >= 225 -> hard stop
        be.run_agent(DIAGNOSE_PROMPT)
    assert isinstance(ei.value, AgentError)
    assert "teacher wall clock" in str(ei.value)
    assert "teacher_share" in ei.value.rules
    # the refused turn never reached the agent
    assert len(inner.prompts) == 3

    summary = gov.read_ledger(ledger)
    assert [e["event"] for e in summary.events].count("soft_brake") == 1
    assert [e["event"] for e in summary.events].count("hard_stop") == 1
    assert [e["event"] for e in summary.events].count("over_share") == 1
    assert summary.by_role[AGENT_ROLE_TEACHER]["turns"] == 3
    assert summary.by_role[AGENT_ROLE_TEACHER]["seconds"] == pytest.approx(300.0)


def test_the_share_rule_lifts_once_the_student_has_its_iterations(tmp_path):
    policy, ledger, status = make(tmp_path, hours=0.1, min_student_iters=5, iters=0)
    clock = Clock()
    inner = Inner(clock, cost=200.0)
    be = gov.GovernedBackend(inner, policy, ledger, clock=clock)
    be.run_agent(DIAGNOSE_PROMPT)              # 200 s: over the soft share, no stop
    write_status(status, [row(i) for i in range(5)])
    be.run_agent(DIAGNOSE_PROMPT)              # 400 s, but 5 iterations: ungoverned
    assert not inner.prompts[1].startswith(gov.SOFT_BRAKE_TEXT)
    summary = gov.read_ledger(ledger)
    assert summary.policy["min_student_iters"] == 5


def test_student_turns_are_never_braked_or_stopped(tmp_path):
    policy, ledger, status = make(tmp_path, hours=0.1, iters=0)
    # a ledger already far past the hard threshold — the teacher would be refused here
    ledger.write_text(json.dumps({"event": "policy", **policy.as_dict()}) + "\n"
                      + json.dumps({"event": "turn", "role": "teacher",
                                    "phase": "diagnose", "elapsed_s": 100000.0}) + "\n",
                      encoding="utf-8")
    inner = Inner(Clock(), cost=100000.0)
    be = gov.GovernedBackend(inner, policy, ledger)
    for _ in range(20):
        be.run_agent(STUDENT_PROMPT)
    assert all(p == STUDENT_PROMPT for p in inner.prompts)
    assert len(inner.prompts) == 20


def test_an_explicit_student_role_is_never_stopped_even_on_a_teacher_prompt(tmp_path):
    policy, ledger, _ = make(tmp_path, hours=0.1, iters=0)
    ledger.write_text(json.dumps({"event": "turn", "role": "teacher", "phase": "diagnose",
                                  "elapsed_s": 100000.0}) + "\n", encoding="utf-8")
    inner = Inner()
    be = gov.GovernedBackend(inner, policy, ledger)
    assert be.run_agent(DIAGNOSE_PROMPT, role=AGENT_ROLE_STUDENT) == "ok"


# -- the consecutive-teacher-turns rule -----------------------------------------------------

def test_a_hard_stop_from_the_share_also_records_over_share(tmp_path):
    # the whole teacher budget was spent before the first governed turn (a resume, or
    # a ledger written by the reconnaissance backend): the first teacher turn is refused
    # and the report's teacher_over_share flag must still be true
    policy, ledger, _ = make(tmp_path, hours=0.1, iters=0)
    ledger.write_text(json.dumps({"event": "turn", "role": "teacher", "phase": "recon",
                                  "elapsed_s": 300.0}) + "\n", encoding="utf-8")
    inner = Inner(Clock())
    be = gov.GovernedBackend(inner, policy, ledger, clock=Clock())
    with pytest.raises(gov.GovernorStop):
        be.run_agent(DIAGNOSE_PROMPT)
    events = [e["event"] for e in gov.read_ledger(ledger).events]
    assert events == ["hard_stop", "over_share"] or events == ["over_share", "hard_stop"]
    assert gov.throughput_flags(gov.read_ledger(ledger), student_iters=0)[
        "budget_starved"] is True


def test_consecutive_turns_soft_brake_at_the_cap_and_stop_four_later(tmp_path):
    policy, ledger, _ = make(tmp_path, hours=100.0, max_teacher_turns_per_phase=3,
                             min_student_iters=0)
    clock = Clock()
    inner = Inner(clock)
    be = gov.GovernedBackend(inner, policy, ledger, clock=clock)
    for _ in range(2):
        be.run_agent(DIAGNOSE_PROMPT)
    assert not any(p.startswith(gov.SOFT_BRAKE_TEXT) for p in inner.prompts)
    for _ in range(4):                          # turns 3, 4, 5, 6: soft brakes
        be.run_agent(DIAGNOSE_PROMPT)
    assert all(p.startswith(gov.SOFT_BRAKE_TEXT) for p in inner.prompts[2:])
    with pytest.raises(gov.GovernorStop) as ei:  # turn 7 = 3 + 4
        be.run_agent(DIAGNOSE_PROMPT)
    assert "consecutive_phase_turns" in ei.value.rules
    assert "teacher turn(s) in one phase" in str(ei.value)
    assert len(inner.prompts) == 6


def test_a_student_turn_resets_the_consecutive_counter(tmp_path):
    policy, ledger, _ = make(tmp_path, hours=100.0, max_teacher_turns_per_phase=3,
                             min_student_iters=0)
    inner = Inner(Clock())
    be = gov.GovernedBackend(inner, policy, ledger, clock=Clock())
    for _ in range(2):
        be.run_agent(DIAGNOSE_PROMPT)
    be.run_agent(STUDENT_PROMPT)
    for _ in range(2):
        be.run_agent(DIAGNOSE_PROMPT)           # would be turns 4 and 5 without the reset
    assert not any(p.startswith(gov.SOFT_BRAKE_TEXT) for p in inner.prompts)
    assert inner.roles == ["teacher", "teacher", "student", "teacher", "teacher"]


def test_a_different_phase_starts_its_own_count(tmp_path):
    policy, ledger, _ = make(tmp_path, hours=100.0, max_teacher_turns_per_phase=3,
                             min_student_iters=0)
    inner = Inner(Clock())
    be = gov.GovernedBackend(inner, policy, ledger, clock=Clock())
    be.run_agent(DIAGNOSE_PROMPT)
    be.run_agent(DIAGNOSE_PROMPT)
    be.run_agent(PATCH_PROMPT)                  # count 1 of "patch"
    be.run_agent(DIAGNOSE_PROMPT)               # count 1 of "diagnose" again
    assert not any(p.startswith(gov.SOFT_BRAKE_TEXT) for p in inner.prompts)


def test_both_brakes_can_apply_to_one_turn(tmp_path):
    # share already exceeded and the phase cap reached at the same time
    policy, ledger, _ = make(tmp_path, hours=0.01, max_teacher_turns_per_phase=1,
                             min_student_iters=5, iters=0)
    ledger.write_text(json.dumps({"event": "turn", "role": "teacher", "phase": "diagnose",
                                  "elapsed_s": 20.0}) + "\n", encoding="utf-8")
    inner = Inner(Clock())
    be = gov.GovernedBackend(inner, policy, ledger, clock=Clock())
    be.run_agent(DIAGNOSE_PROMPT)
    text = inner.prompts[0]
    assert text.startswith(gov.SOFT_BRAKE_TEXT)
    events = [e["event"] for e in gov.read_ledger(ledger).events]
    assert events.count("soft_brake") == 2
    assert {e["rule"] for e in gov.read_ledger(ledger).events
            if e["event"] == "soft_brake"} == {"teacher_share", "consecutive_phase_turns"}


# -- the ledger -----------------------------------------------------------------------------

def test_the_ledger_has_one_policy_line_and_one_turn_line_per_turn(tmp_path):
    policy, ledger, _ = make(tmp_path)
    be = gov.GovernedBackend(Inner(Clock()), policy, ledger, clock=Clock())
    be.run_agent(DIAGNOSE_PROMPT, role=AGENT_ROLE_TEACHER)
    be.run_agent(STUDENT_PROMPT)
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert rows[0]["event"] == "policy"
    assert rows[0]["teacher_share"] == 0.5 and rows[0]["min_student_iters"] == 5
    turns = [r for r in rows if r["event"] == "turn"]
    assert [(t["role"], t["phase"]) for t in turns] == [
        ("teacher", "diagnose"), ("student", "student")]
    assert turns[0]["student_iters"] == 0 and turns[1]["student_iters"] == 0
    assert turns[0]["error"] == ""


def test_a_failing_turn_is_recorded_and_the_error_is_re_raised(tmp_path):
    policy, ledger, _ = make(tmp_path)

    class Boom(Inner):
        def run_agent(self, prompt, *, role=None):
            raise AgentError("agent command failed")

    be = gov.GovernedBackend(Boom(), policy, ledger, clock=Clock())
    with pytest.raises(AgentError, match="agent command failed"):
        be.run_agent(DIAGNOSE_PROMPT)
    turn = [r for r in gov.read_ledger(ledger).turns][0]
    assert "AgentError: agent command failed" in turn["error"]


def test_an_unwritable_ledger_warns_once_and_keeps_working(tmp_path, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    ledger = blocker / gov.LEDGER_NAME             # parent is a file: mkdir/open both fail
    policy = gov.GovernorPolicy(hours=0.1, status_path=tmp_path / "status.jsonl")
    clock = Clock()
    inner = Inner(clock, cost=100.0)
    be = gov.GovernedBackend(inner, policy, ledger, clock=clock)
    be.run_agent(DIAGNOSE_PROMPT)
    be.run_agent(DIAGNOSE_PROMPT)
    be.run_agent(DIAGNOSE_PROMPT)                  # soft brake still applied (memory)
    assert inner.prompts[-1].startswith(gov.SOFT_BRAKE_TEXT)
    err = capsys.readouterr().err
    assert err.count("[governor] warning") == 1


# -- the policy -----------------------------------------------------------------------------

def test_policy_defaults(tmp_path):
    p = gov.policy_from_env(hours=2.0, status_path=tmp_path / "s.jsonl", environ={})
    assert (p.teacher_share, p.min_student_iters, p.max_teacher_turns_per_phase) == \
        (0.5, 5, 12)
    assert p.share_s == pytest.approx(3600.0) and p.hard_share_s == pytest.approx(4500.0)
    assert p.hard_turns == 16


def test_policy_reads_the_environment_over_the_caller(tmp_path):
    env = {gov.POLICY_ENV: json.dumps({"hours": 0.5, "teacher_share": 0.25,
                                       "min_student_iters": 2,
                                       "max_teacher_turns_per_phase": 4})}
    p = gov.policy_from_env(hours=99.0, target_ms=0.5, gate_window="e2e",
                            status_path=tmp_path / "s.jsonl", environ=env)
    assert p.hours == 0.5 and p.teacher_share == 0.25
    assert p.min_student_iters == 2 and p.max_teacher_turns_per_phase == 4
    assert p.target_ms == 0.5 and p.gate_window == "e2e"


@pytest.mark.parametrize("payload", [
    "not json",
    "[]",
    json.dumps({"hours": 1, "max_tokens": 1000}),     # no token key exists, ever
    json.dumps({"teacher_share": 0.5}),               # no hours and no caller default
    json.dumps({"hours": 0}),
    json.dumps({"hours": 1, "teacher_share": 0}),
    json.dumps({"hours": 1, "teacher_share": 1.5}),
    json.dumps({"hours": 1, "min_student_iters": -1}),
    json.dumps({"hours": 1, "min_student_iters": 1.5}),
    json.dumps({"hours": 1, "min_student_iters": True}),
    json.dumps({"hours": 1, "max_teacher_turns_per_phase": 0}),
])
def test_policy_rejects_bad_payloads(payload):
    with pytest.raises(ValueError):
        gov.policy_from_env(status_path="/tmp/x", environ={gov.POLICY_ENV: payload})


def test_policy_off_wins_over_a_bad_payload(tmp_path):
    env = {gov.DISABLE_ENV: "off", gov.POLICY_ENV: "not json"}
    assert gov.policy_from_env(hours=1.0, environ=env) is None


@pytest.mark.parametrize("value,off", [("off", True), ("OFF", True), ("0", True),
                                        ("false", True), ("no", True), ("disabled", True),
                                        ("on", False), ("", False), ("1", False)])
def test_governor_disabled_spellings(value, off):
    assert gov.governor_disabled({gov.DISABLE_ENV: value}) is off


def test_a_bad_gate_window_or_target_is_refused(tmp_path):
    with pytest.raises(ValueError, match="gate_window"):
        gov.policy_from_env(hours=1.0, gate_window="wall", environ={})
    with pytest.raises(ValueError, match="target_ms"):
        gov.policy_from_env(hours=1.0, target_ms=0, environ={})


# -- status / report / usage ----------------------------------------------------------------

def test_student_iterations_counts_every_row():
    assert gov.student_iterations(None) == 0
    assert gov.student_iterations("/nonexistent/status.jsonl") == 0


def test_flags_truth_table():
    def summary(events, min_iters=5):
        return gov.LedgerSummary(policy={"min_student_iters": min_iters},
                                 events=[{"event": e} for e in events])

    assert gov.throughput_flags(summary([]), student_iters=0) == {
        "teacher_over_share": False, "budget_starved": False, "escape_fired": False}
    # over share but the student got its iterations: not starved
    assert gov.throughput_flags(summary(["over_share"]), student_iters=5) == {
        "teacher_over_share": True, "budget_starved": False, "escape_fired": False}
    # short of iterations but the teacher never went over: not starved
    assert gov.throughput_flags(summary([]), student_iters=1) == {
        "teacher_over_share": False, "budget_starved": False, "escape_fired": False}
    # both: starved
    assert gov.throughput_flags(summary(["over_share"]), student_iters=1) == {
        "teacher_over_share": True, "budget_starved": True, "escape_fired": False}
    assert gov.throughput_flags(summary(["escape_fired"]), student_iters=9) == {
        "teacher_over_share": False, "budget_starved": False, "escape_fired": True}


def test_status_lines_report_turns_and_the_active_rules(tmp_path):
    policy, ledger, status = make(tmp_path, hours=0.01, iters=0)
    ledger.write_text(
        json.dumps({"event": "policy", **policy.as_dict()}) + "\n"
        + json.dumps({"event": "turn", "role": "teacher", "phase": "diagnose",
                      "elapsed_s": 3.5}) + "\n"
        + json.dumps({"event": "turn", "role": "student", "phase": "student",
                      "elapsed_s": 2.0}) + "\n"
        + json.dumps({"event": "over_share"}) + "\n"
        + json.dumps({"event": "escape_fired"}) + "\n", encoding="utf-8")
    write_status(status, [row(0)])
    lines = gov.status_lines(ledger, status)
    assert len(lines) == 2
    assert "teacher 1 / 3.5 s" in lines[0] and "student 1 / 2.0 s" in lines[0]
    assert "teacher_over_share" in lines[1] and "budget_starved" in lines[1]
    assert "escape_fired" in lines[1] and "soft brakes 0, hard stops 0" in lines[1]


def test_status_lines_are_empty_without_a_ledger(tmp_path):
    assert gov.status_lines(tmp_path / "none.jsonl", tmp_path / "status.jsonl") == []


def test_report_section_without_a_ledger(tmp_path):
    lines = gov.report_lines(tmp_path / "none.jsonl", tmp_path / "status.jsonl")
    assert lines[0] == "## Throughput"
    assert any("no ledger" in line for line in lines)


def test_report_section_with_turns_flags_and_usage(tmp_path):
    policy, ledger, status = make(tmp_path, hours=0.01, iters=0)
    ledger.write_text(
        json.dumps({"event": "policy", **policy.as_dict()}) + "\n"
        + json.dumps({"event": "turn", "role": "teacher", "phase": "recon",
                      "elapsed_s": 10.0}) + "\n"
        + json.dumps({"event": "turn", "role": "student", "phase": "student",
                      "elapsed_s": 5.0}) + "\n"
        + json.dumps({"event": "over_share"}) + "\n", encoding="utf-8")
    write_status(status, [row(0), row(1)])
    usage = tmp_path / gov.USAGE_NAME
    usage.write_text(json.dumps({"role": "teacher", "phase": "recon",
                                 "total_tokens": 1200, "duration_s": 10.0}) + "\n"
                     + json.dumps({"role": "student", "phase": "student",
                                   "prompt_tokens": 100, "completion_tokens": 40,
                                   "seconds": 5.0}) + "\n", encoding="utf-8")
    text = "\n".join(gov.report_lines(ledger, status, usage_path=usage))
    assert "student iterations: 2" in text
    assert "teacher_over_share=yes" in text
    assert "budget_starved=yes" in text            # 2 < 5 iterations and over share
    assert "escape_fired=no" in text
    assert "turns by role: teacher 1 / 10.0 s; student 1 / 5.0 s" in text
    assert "turns by phase: recon 1 / 10.0 s; student 1 / 5.0 s" in text
    assert "1200 tokens" in text and "140 tokens" in text

    # the usage section disappears with the file, everything else stays
    text = "\n".join(gov.report_lines(ledger, status))
    assert "usage" not in text and "student iterations: 2" in text


def test_report_section_tolerates_garbage_lines(tmp_path):
    policy, ledger, status = make(tmp_path)
    ledger.write_text("not json\n" + json.dumps({"event": "turn", "role": "teacher",
                                                 "elapsed_s": "much"}) + "\n",
                      encoding="utf-8")
    lines = gov.report_lines(ledger, status)
    assert lines[0] == "## Throughput" and any("teacher 1" in x for x in lines)
