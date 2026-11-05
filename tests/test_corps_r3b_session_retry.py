"""BUG-CORPS-C-03 regression: a retry after a failed session RESUME starts a
fresh session instead of re-entering the same dead one.

With resume_session on, a scope's session id lives in a file under the
workspace that outlives the process that minted it. When a turn resumed a stale
id, every remaining attempt re-entered the same dead session (the wrapper's
``--resume`` fails hard on an unknown id) and the run died as agent_failed after
the full retry budget — three wasted turns and an uncollected phase, on a file
one unlink away from fixed. Now a hard failure (anything outside the API-outage
budget) on a resumed session forgets the id; the next attempt CREATES. A failure
on a freshly created session still keeps it: the retry resumes it with the
error tail, which is the WP19 feature working as designed.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from peregrine_kernel.loop import cli

DEAD = "dead-id"
FRESH = re.compile(r"scope-student-[0-9a-f]+")
SID_FILE = Path(".pk-sessions") / "scope.student"


def _backend(tmp_path: Path, script) -> cli.RealBackend:
    be = cli.RealBackend("op", tmp_path, "true", "n", "e",
                         agent_retries=3, agent_retry_sleep=0.0)
    be.resume_session = True
    be.agent_session_scope = "scope"
    be._agent_turn = script          # the fake turn
    (tmp_path / SID_FILE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / SID_FILE).write_text(DEAD, encoding="utf-8")
    return be


def _script(seen):
    """A turn that fails hard on the dead id (unknown-session error) and
    succeeds otherwise, recording (id, mode, prompt-tail) per attempt."""
    def turn(cmd, prompt, role):
        m = re.search(r"PK_AGENT_SESSION_ID=(\S+) PK_AGENT_SESSION_MODE=(\S+)", cmd)
        seen.append((m.group(1), m.group(2),
                     "## Previous attempt failed" in prompt))
        if m.group(1) == DEAD:
            return 1, "", "No conversation found with session ID " + DEAD
        return 0, "ok out\n", ""
    return turn


def test_a_stale_session_id_is_forgotten_and_the_retry_creates(tmp_path):
    seen = []
    be = _backend(tmp_path, _script(seen))
    out = be.run_agent("do the turn")
    assert out == "ok out\n"
    # attempt 1 resumed the dead id and failed; attempt 2 created a fresh one
    assert seen[0][:2] == (DEAD, "resume")
    assert FRESH.fullmatch(seen[1][0]) and seen[1][1] == "create"
    # the failure tail still rides the retry prompt
    assert seen[1][2] is True
    # the sid file now names the fresh session, not the dead one
    assert (tmp_path / SID_FILE).read_text(encoding="utf-8") == seen[1][0]


def test_a_created_session_survives_its_first_hard_failure(tmp_path):
    """The WP19 feature is intact: a turn that failed on a session it just
    CREATED leaves the id in place, so the retry RESUMES it with the tail."""
    seen = []

    def turn(cmd, prompt, role):
        m = re.search(r"PK_AGENT_SESSION_ID=(\S+) PK_AGENT_SESSION_MODE=(\S+)", cmd)
        seen.append((m.group(1), m.group(2),
                     "## Previous attempt failed" in prompt))
        if m.group(2) == "create":
            return 1, "", "half-written turn died"     # hard, non-transient
        return 0, "ok out\n", ""

    be = _backend(tmp_path, turn)
    (tmp_path / SID_FILE).write_text(DEAD, encoding="utf-8")
    # clear the dead id first so attempt 1 is a CREATE
    (tmp_path / SID_FILE).unlink()
    assert be.run_agent("do the turn") == "ok out\n"
    assert seen[0][1] == "create"
    assert seen[1] == (seen[0][0], "resume", True)     # resumed, tail attached
    assert (tmp_path / SID_FILE).read_text(encoding="utf-8") == seen[0][0]


def test_resume_is_off_no_reset_and_no_env(tmp_path):
    """resume_session off: no session env is built, nothing is written or
    dropped — the baseline behavior."""
    be = cli.RealBackend("op", tmp_path, "true", "n", "e",
                         agent_retries=1, agent_retry_sleep=0.0)
    be.agent_session_scope = "scope"
    be._agent_turn = lambda cmd, prompt, role: (0, "ok\n", "")
    assert be._session_env("student") == {}
    assert not (tmp_path / SID_FILE).exists()
