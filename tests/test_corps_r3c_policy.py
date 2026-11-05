"""Regression pins for the BUG-CORPS-C-08 and BUG-CORPS-C-09 fixes (audit wave R3C).

C-08: a student with the network on (student_online / no_cluster without offline) has
no sandbox network wall, so the cluster denial is only as deep as its deny rules —
``bash -c 'ssh …'`` must be denied too, not just a leading ``ssh``.

C-09: the goal-ref spool lives inside the agent's working directory; the side the
harness writes (responses, logs, context.json) must be in the session's deny_write, and
the rendered settings must hold it read-only at the file level.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from rsi_kernel_tune.loop import agent_policy as ap
from rsi_kernel_tune.loop import goal_ref as gr


def _policy(tmp_path: Path) -> ap.AgentPolicy:
    return ap.AgentPolicy(
        oracle_mode="source",
        repo_root=tmp_path / "repo",
        ref_source_paths=(),
        ref_source_urls=("github.com/Org/RefLib",),
        teacher_trees=(tmp_path / "repo" / "operators" / "op" / "goal_ref",),
        dsl_source_dirs=(),
        student_online=True,
    )


# -- BUG-CORPS-C-08 ------------------------------------------------------------------------

def test_a_wrapped_cluster_command_is_denied_for_an_online_student(tmp_path):
    """`bash -c 'ssh …'` walks past every prefix rule (the rule stops at `bash`) and
    the online student's sandbox has no network wall: the wrapper payload needs its
    own deny rules."""
    ws = tmp_path / "ws"
    deny = _policy(tmp_path).settings("student", ws)["permissions"]["deny"]
    for rule in ("Bash(bash *-c *ssh*)", "Bash(bash *-c *scp*)",
                 "Bash(sh *-c *rsync*)", "Bash(zsh *-c *ssh*)",
                 "Bash(bash *-c *python* -m rsi_kernel_tune.submit*)"):
        assert rule in deny, rule


def test_an_offline_student_carries_the_wrapper_rules_too(tmp_path):
    pol = dataclasses.replace(_policy(tmp_path), student_online=False)
    ws = tmp_path / "ws"
    deny = pol.settings("student", ws)["permissions"]["deny"]
    assert "Bash(bash *-c *ssh*)" in deny
    # and the offline tools stay removed as before
    assert "WebFetch" in deny and "Bash(curl:*)" in deny


def test_a_teacher_without_the_cluster_denial_has_no_wrapper_rules(tmp_path):
    """The teacher keeps the network on purpose (source mode); nothing new for it."""
    pol = dataclasses.replace(_policy(tmp_path), student_online=False)
    ws = tmp_path / "ws"
    s = pol.settings("teacher", ws)
    assert "Bash(bash *-c *ssh*)" not in s["permissions"]["deny"]
    assert not any("-c *" in r for r in s["permissions"]["deny"])


# -- BUG-CORPS-C-09 (rendering side) -------------------------------------------------------

def test_the_spool_side_the_harness_writes_is_read_only_to_the_session(tmp_path):
    """responses/ + logs/ + context.json are the session policy's deny_write;
    requests/ stays writable — the agent-side eval command creates requests there."""
    op_dir = tmp_path / "repo" / "operators" / "op"
    op_dir.mkdir(parents=True)
    for d in ("state.json", "ledger.jsonl"):
        (op_dir / d).touch()
    gr.Paths.of(op_dir).spool.joinpath("requests").mkdir(parents=True)
    gr.Paths.of(op_dir).spool.joinpath("responses").mkdir(parents=True)
    gr.Paths.of(op_dir).spool.joinpath("logs").mkdir(parents=True)
    (gr.Paths.of(op_dir).spool / "context.json").touch()
    ro = gr.session_read_only_paths(op_dir)
    spool = gr.Paths.of(op_dir).spool
    assert {spool / "responses", spool / "logs", spool / "context.json"} <= set(ro)
    assert spool / "requests" not in ro

    pol = dataclasses.replace(_policy(tmp_path), repo_root=tmp_path / "repo",
                              deny_write=tuple(ro))
    ws = gr.Paths.of(op_dir).ws          # created above with the spool dirs
    s = pol.settings("student", ws)
    deny = s["permissions"]["deny"]
    fs = s["sandbox"]["filesystem"]
    for p in (spool / "responses", spool / "logs", spool / "context.json"):
        assert f"Edit(/{p})" in deny and f"Edit(/{p}/**)" in deny
        # both spellings in the sandbox too: the file-level guarantee must not depend
        # on how the runtime reads a directory entry
        assert str(p) in fs["denyWrite"] and f"{p}/**" in fs["denyWrite"]
    # the agent's own working directory is not denied (the kernel is written there)
    assert f"Edit(/{ws})" not in deny


def test_the_ledger_and_state_stay_denied_but_readable(tmp_path):
    """The pre-existing deny_write contract is unchanged: read-refused for the file
    tools never happens (the check reads the ledger from the shell)."""
    op_dir = tmp_path / "repo" / "operators" / "op"
    op_dir.mkdir(parents=True)
    ro = gr.session_read_only_paths(op_dir)
    pol = dataclasses.replace(_policy(tmp_path), repo_root=tmp_path / "repo",
                              deny_write=tuple(ro))
    s = pol.settings("student", gr.Paths.of(op_dir).ws)
    state, ledger = gr.Paths.of(op_dir).state, gr.Paths.of(op_dir).ledger
    assert f"Edit(/{state})" in s["permissions"]["deny"]
    assert f"Read(/{ledger})" not in s["permissions"]["deny"]
    doc = pol.policy_document("student")
    assert "no_cluster" in doc and doc["no_cluster"] is True
    assert json.dumps(doc)  # the policy the hook reads still serializes
