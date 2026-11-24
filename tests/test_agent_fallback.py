"""Agent turns that did not run on the model the task asked for.

Three layers, tested separately because they fail differently:

1. ``scripts/agent_trailer.py`` — the parsing seam. It turns a structured turn into the
   loop's text reply, the usage record, and the refusal / mismatch verdict. Its input
   shapes were taken from a real ``claude -p --output-format json --verbose`` turn (an
   array of system/assistant/result records) and from the Codex CLI's documented event
   stream.
2. the launcher scripts — ``FK_AGENT_USAGE_FILE`` turns the structured turn on and the
   ledger gets its line; ``FK_AGENT_ON_FALLBACK=fail`` ends a downgraded turn with the
   marker.
3. ``RealBackend.run_agent`` — sees the marker, does not retry, and raises ``AgentError``
   with the reason.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAILER = ROOT / "scripts" / "agent_trailer.py"
CLAUDE_SH = ROOT / "scripts" / "agent-claude.sh"
CODEX_SH = ROOT / "scripts" / "agent-codex.sh"


def _trailer():
    spec = importlib.util.spec_from_file_location("_fk_agent_trailer", TRAILER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A turn as `claude -p --output-format json --verbose` prints it (2.1.316): one JSON array
# of a system/init record, the assistant message and the terminal result record.
CLAUDE_TURN = [
    {"type": "system", "subtype": "init", "session_id": "sess-1",
     "model": "claude-opus-5-5", "claude_code_version": "2.1.316"},
    {"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-5-5-20260101",
                                      "content": [{"type": "text", "text": "the reply"}]},
     "session_id": "sess-1"},
    {"type": "result", "subtype": "success", "is_error": False, "duration_ms": 12000,
     "duration_api_ms": 11000, "num_turns": 1, "result": "the reply", "session_id": "sess-1",
     "total_cost_usd": 0.25,
     "usage": {"input_tokens": 100, "cache_creation_input_tokens": 200,
               "cache_read_input_tokens": 300, "output_tokens": 40},
     "modelUsage": {"claude-opus-5-5-20260101": {"inputTokens": 100, "outputTokens": 40}}},
]

# The shape Codex documents for `codex exec --json`.
CODEX_TURN = "\n".join([
    json.dumps({"type": "thread.started", "thread_id": "th-1"}),
    json.dumps({"type": "turn.started"}),
    json.dumps({"type": "item.completed",
                "item": {"id": "i1", "type": "agent_message", "text": "codex says hi"}}),
    json.dumps({"type": "turn.completed",
                "usage": {"input_tokens": 1000, "cached_input_tokens": 800,
                          "output_tokens": 50, "reasoning_output_tokens": 0}}),
])


def _run(tmp_path, argv, *, raw: str | Path | None = None, capsys=None):
    module = _trailer()
    raw_path = None
    if raw is not None:
        raw_path = Path(raw) if isinstance(raw, Path) else tmp_path / "raw.out"
        if not isinstance(raw, Path):
            raw_path.write_text(raw, encoding="utf-8")
        argv = [*argv, "--raw", str(raw_path)]
    return module.main(argv), capsys.readouterr()


def _transcript(tmp_path, session: str, records: list[dict]) -> Path:
    d = tmp_path / "cfg" / "projects" / "-tmp-work"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _refusal_record() -> dict:
    return {"type": "system", "subtype": "model_refusal_fallback", "trigger": "refusal",
            "scope": "session", "originalModel": "claude-opus-5-5",
            "fallbackModel": "claude-sonnet-4-5", "apiRefusalCategory": "cyber",
            "apiRefusalExplanation": "…"}


# -- the parsing seam ------------------------------------------------------------------


def test_a_claude_turn_gives_the_reply_and_a_full_record(tmp_path, capsys):
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "claude-opus-5-5",
                              "--role", "teacher", "--usage-file", str(tmp_path / "l.jsonl")],
                   raw=json.dumps(CLAUDE_TURN), capsys=capsys)
    assert rc == 0
    assert out.out == "the reply\n"                       # stdout contract: plain text
    usage = [line for line in out.err.splitlines() if line.startswith("FK_USAGE ")]
    assert len(usage) == 1
    record = json.loads(usage[0][len("FK_USAGE "):])
    assert record["role"] == "teacher" and record["backend"] == "claude"
    assert record["model_used"] == "claude-opus-5-5-20260101"
    assert record["session_id"] == "sess-1"
    assert (record["input_tokens"], record["output_tokens"]) == (100, 40)
    assert record["cache_read_input_tokens"] == 300
    assert record["cache_creation_input_tokens"] == 200
    assert record["total_tokens"] == 640                  # 100+40+300+200
    assert record["cost_usd"] == 0.25 and record["duration_s"] == 12.0
    assert record["refusal_check"] == "unknown"           # no transcript was reachable
    ledger = (tmp_path / "l.jsonl").read_text().splitlines()
    assert len(ledger) == 1 and json.loads(ledger[0])["session_id"] == "sess-1"


def test_a_result_object_without_usage_still_gives_the_reply(tmp_path, capsys):
    raw = json.dumps({"type": "result", "result": "ok", "session_id": "s"})
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "m"], raw=raw, capsys=capsys)
    assert rc == 0 and out.out == "ok\n"
    record = json.loads([l for l in out.err.splitlines()
                         if l.startswith("FK_USAGE ")][0][len("FK_USAGE "):])
    assert record["input_tokens"] is None and record["total_tokens"] is None
    assert record["cost_usd"] is None and record["model_used"] is None


def test_plain_text_is_passed_through_and_the_status_is_kept(tmp_path, capsys):
    rc, out = _run(tmp_path, ["claude", "--rc", "124", "--model", "m"],
                   raw="the raw text\n", capsys=capsys)
    assert rc == 124 and out.out == "the raw text\n"
    assert "FK_USAGE" not in out.err


def test_a_claude_error_result_is_not_reported_as_a_delivered_turn(tmp_path, capsys):
    raw = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                      "result": "Not logged in · Please run /login", "session_id": "s",
                      "usage": {"input_tokens": 0, "output_tokens": 0}})
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "m"], raw=raw, capsys=capsys)
    assert rc == 1                                        # the CLI said it failed
    assert out.out == "Not logged in · Please run /login\n"
    record = json.loads([l for l in out.err.splitlines()
                         if l.startswith("FK_USAGE ")][0][len("FK_USAGE "):])
    assert record["rc"] == 1


def test_a_codex_stream_gives_the_last_agent_message_and_the_turn_usage(tmp_path, capsys):
    rc, out = _run(tmp_path, ["codex", "--rc", "0", "--model", "gpt-5.5-codex",
                              "--role", "teacher"], raw=CODEX_TURN, capsys=capsys)
    assert rc == 0 and out.out == "codex says hi\n"
    record = json.loads([l for l in out.err.splitlines()
                         if l.startswith("FK_USAGE ")][0][len("FK_USAGE "):])
    assert record["backend"] == "codex" and record["session_id"] == "th-1"
    assert record["cache_read_input_tokens"] == 800       # cached_input_tokens
    assert record["total_tokens"] == 1050                 # input + output (cache is inside input)


def test_a_codex_stream_without_a_reply_fails_loudly(tmp_path, capsys):
    raw = "\n".join([json.dumps({"type": "thread.started", "thread_id": "t"}),
                     json.dumps({"type": "turn.completed", "usage": {}})])
    rc, out = _run(tmp_path, ["codex", "--rc", "0"], raw=raw, capsys=capsys)
    assert rc == 3 and "no reply text" in out.err and "FK_USAGE" not in out.err


def test_a_failed_codex_turn_is_a_failure_and_says_why(tmp_path, capsys):
    raw = "\n".join([json.dumps({"type": "turn.failed",
                                 "error": {"message": "stream disconnected"}})])
    rc, out = _run(tmp_path, ["codex", "--rc", "0"], raw=raw, capsys=capsys)
    assert rc == 1 and "stream disconnected" in out.err


def test_an_older_codex_item_shape_is_tolerated(tmp_path, capsys):
    raw = "\n".join([json.dumps({"id": "0", "msg": {"type": "agent_message",
                                                    "message": "older shape"}}),
                     json.dumps({"type": "result", "result": "older shape",
                                 "usage": {"inputTokens": 5, "outputTokens": 2}})])
    rc, out = _run(tmp_path, ["codex", "--rc", "0"], raw=raw, capsys=capsys)
    assert rc == 0 and out.out == "older shape\n"


def test_usage_from_a_json_array_and_from_lines_agree(tmp_path, capsys):
    one = json.dumps(CLAUDE_TURN[-1])
    many = "\n".join(json.dumps(r) for r in CLAUDE_TURN)
    _, a = _run(tmp_path, ["claude", "--rc", "0"], raw=one, capsys=capsys)
    _, b = _run(tmp_path, ["claude", "--rc", "0"], raw=many, capsys=capsys)
    assert a.out == b.out == "the reply\n"


# -- the refusal / mismatch verdict -----------------------------------------------------


def test_a_refusal_record_ends_the_turn_in_fail_mode(tmp_path, capsys):
    _transcript(tmp_path, "sess-1", [{"type": "user", "message": {"role": "user"}},
                                     _refusal_record()])
    ledger = tmp_path / "l.jsonl"
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "claude-opus-5-5",
                              "--config-dir", str(tmp_path / "cfg"), "--usage-file", str(ledger)],
                   raw=json.dumps(CLAUDE_TURN), capsys=capsys)
    assert rc == 78                                        # FK_AGENT_ON_FALLBACK defaults to fail
    assert "FK_AGENT_FALLBACK" in out.err
    alert = [l for l in out.err.splitlines() if l.startswith("FK_ALERT ")][0]
    kind = json.loads(alert[len("FK_ALERT "):])
    assert kind["kind"] == "refusal_fallback"
    assert kind["original_model"] == "claude-opus-5-5"
    assert kind["fallback_model"] == "claude-sonnet-4-5"
    assert kind["category"] == "cyber"
    record = json.loads(ledger.read_text().splitlines()[0])
    assert record["refusal_check"] == "refusal" and record["warning"] == "refusal_fallback"


def test_warn_mode_delivers_the_reply_and_records_the_warning(tmp_path, capsys):
    _transcript(tmp_path, "sess-1", [_refusal_record()])
    ledger = tmp_path / "l.jsonl"
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "claude-opus-5-5",
                              "--config-dir", str(tmp_path / "cfg"), "--on-fallback", "warn",
                              "--usage-file", str(ledger)],
                   raw=json.dumps(CLAUDE_TURN), capsys=capsys)
    assert rc == 0 and out.out == "the reply\n"
    assert "FK_ALERT" in out.err and "FK_AGENT_FALLBACK" not in out.err
    assert json.loads(ledger.read_text().splitlines()[0])["warning"] == "refusal_fallback"


def test_a_model_mismatch_is_its_own_alert(tmp_path, capsys):
    # no fallback record, but the turn ran on another model than the one asked for
    _transcript(tmp_path, "sess-1", [{"type": "assistant",
                                      "message": {"model": "claude-sonnet-4-5"}}])
    ledger = tmp_path / "l.jsonl"
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "claude-opus-5-5",
                              "--config-dir", str(tmp_path / "cfg"), "--usage-file", str(ledger)],
                   raw=json.dumps(CLAUDE_TURN), capsys=capsys)
    # the modelUsage of the result object names the requested model, so the transcript's
    # last assistant record is not what decides: no mismatch here
    assert rc == 0 and "FK_ALERT" not in out.err
    raw = json.dumps({"type": "result", "result": "hi", "session_id": "sess-1",
                      "modelUsage": {"claude-sonnet-4-5": {"outputTokens": 40}}})
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "claude-opus-5-5",
                              "--config-dir", str(tmp_path / "cfg")], raw=raw, capsys=capsys)
    assert rc == 78
    alert = json.loads([l for l in out.err.splitlines()
                        if l.startswith("FK_ALERT ")][0][len("FK_ALERT "):])
    assert alert["kind"] == "model_mismatch"
    assert (alert["model_requested"], alert["model_used"]) == ("claude-opus-5-5",
                                                              "claude-sonnet-4-5")


def test_a_missing_transcript_is_unknown_and_blocks_nothing(tmp_path, capsys):
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "claude-opus-5-5",
                              "--config-dir", str(tmp_path / "nowhere")],
                   raw=json.dumps(CLAUDE_TURN), capsys=capsys)
    assert rc == 0 and "FK_ALERT" not in out.err
    record = json.loads([l for l in out.err.splitlines()
                         if l.startswith("FK_USAGE ")][0][len("FK_USAGE "):])
    assert record["refusal_check"] == "unknown"


def test_a_clean_transcript_is_recorded_as_clean(tmp_path, capsys):
    _transcript(tmp_path, "sess-1", [{"type": "assistant",
                                      "message": {"model": "claude-opus-5-5-20260101"}}])
    rc, out = _run(tmp_path, ["claude", "--rc", "0", "--model", "claude-opus-5-5",
                              "--config-dir", str(tmp_path / "cfg")],
                   raw=json.dumps(CLAUDE_TURN), capsys=capsys)
    assert rc == 0
    record = json.loads([l for l in out.err.splitlines()
                         if l.startswith("FK_USAGE ")][0][len("FK_USAGE "):])
    assert record["refusal_check"] == "clean"


def test_an_alias_and_a_dated_id_are_the_same_model():
    t = _trailer()
    assert t.same_model("claude-opus-5-5", "claude-opus-5-5-20260101")
    assert t.same_model("claude-opus-5-5[1m]", "claude-opus-5-5-20260101")
    assert t.same_model("opus", "claude-opus-5-5-20260101")
    assert t.same_model(None, "claude-opus-5-5")
    assert not t.same_model("claude-opus-5-5", "claude-sonnet-4-5")
    assert not t.same_model("claude-opus-5-5", "claude-opus-5-4-20251201")
    # nothing comparable: no cry of wolf (a false mismatch ends a run in fail mode)
    assert t.same_model("mystery-model", "claude-opus-5-5")


# -- the launcher scripts ---------------------------------------------------------------


def _fake_claude(tmp_path, payload: str) -> Path:
    fake = tmp_path / "claude"
    fake.write_text("#!/usr/bin/env bash\ncat >/dev/null\nprintf '%s' \"$*\" > \"$ARGV_FILE\"\n"
                    f"cat <<'JSON'\n{payload}\nJSON\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_the_claude_launcher_writes_the_ledger_and_keeps_stdout_text(tmp_path):
    fake = _fake_claude(tmp_path, json.dumps(CLAUDE_TURN))
    ledger = tmp_path / "agent_usage.jsonl"
    env = {**os.environ, "FK_CLAUDE_BIN": str(fake), "FK_AGENT_USAGE_FILE": str(ledger),
           "FK_AGENT_ROLE": "teacher", "FK_AGENT_PY": sys.executable,
           "ARGV_FILE": str(tmp_path / "argv")}
    r = subprocess.run(["bash", str(CLAUDE_SH), "claude-opus-5-5", "high", "600"],
                       input="prompt", capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "the reply\n"
    argv = (tmp_path / "argv").read_text()
    assert "--output-format json" in argv and "--verbose" in argv
    assert "--permission-mode" not in argv and "--dangerously-skip-permissions" in argv
    record = json.loads(ledger.read_text().splitlines()[0])
    assert record["role"] == "teacher" and record["session_id"] == "sess-1"


def test_without_a_ledger_the_claude_launcher_is_a_plain_text_turn(tmp_path):
    fake = tmp_path / "claude"
    fake.write_text('#!/usr/bin/env bash\ncat >/dev/null\necho "P $*"\n')
    fake.chmod(0o755)
    env = {**os.environ, "FK_CLAUDE_BIN": str(fake), "FK_AGENT_PY": sys.executable}
    env.pop("FK_AGENT_USAGE_FILE", None)
    r = subprocess.run(["bash", str(CLAUDE_SH), "m"], input="prompt", capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0
    assert r.stdout.strip() == "P -p --model m --dangerously-skip-permissions"
    assert "FK_USAGE" not in r.stderr


def test_the_codex_launcher_writes_the_ledger(tmp_path):
    fake = tmp_path / "codex"
    fake.write_text("#!/usr/bin/env bash\ncat >/dev/null\ncat <<'JSON'\n" + CODEX_TURN
                    + "\nJSON\n", encoding="utf-8")
    fake.chmod(0o755)
    ledger = tmp_path / "agent_usage.jsonl"
    r = subprocess.run(["bash", str(CODEX_SH), "gpt-5.5-codex", "high", "60"], input="prompt",
                       capture_output=True, text=True,
                       env={**os.environ, "FK_CODEX_BIN": str(fake),
                            "FK_AGENT_USAGE_FILE": str(ledger), "FK_AGENT_PY": sys.executable})
    assert r.returncode == 0, r.stderr
    assert r.stdout == "codex says hi\n"
    assert json.loads(ledger.read_text().splitlines()[0])["backend"] == "codex"


def test_the_codex_launcher_does_not_retry_a_downgraded_turn(tmp_path):
    refusal = _transcript(tmp_path, "th-1", [_refusal_record()])
    fake = tmp_path / "codex"
    fake.write_text("#!/usr/bin/env bash\ncat >/dev/null\ncat <<'JSON'\n" + CODEX_TURN
                    + "\nJSON\n", encoding="utf-8")
    fake.chmod(0o755)
    r = subprocess.run(["bash", str(CODEX_SH), "m", "high", "60"], input="prompt",
                       capture_output=True, text=True,
                       env={**os.environ, "FK_CODEX_BIN": str(fake), "FK_AGENT_PY": sys.executable,
                            "FK_AGENT_ON_FALLBACK": "fail",
                            "CODEX_HOME": str(refusal.parent.parent.parent)})
    assert r.returncode == 78 and "FK_AGENT_FALLBACK" in r.stderr


# -- RealBackend.run_agent -------------------------------------------------------------


def _backend(monkeypatch, **kw):
    from forgekernel.loop.cli import RealBackend
    monkeypatch.setattr("forgekernel.loop.cli.time.sleep", lambda *_: None)
    b = RealBackend("op", Path("/ws"), "fake-agent", "n", "e", **kw)
    return b


def test_run_agent_does_not_retry_a_downgraded_turn(monkeypatch):
    from forgekernel.loop.cli import AgentError
    b = _backend(monkeypatch, agent_retries=3, agent_retry_sleep=0.0)
    calls = {"n": 0}

    def turn(cmd, prompt, role):
        calls["n"] += 1
        return 78, "the reply\n", ('FK_USAGE {"role": "teacher"}\nFK_ALERT {"kind": '
                                   '"refusal_fallback"}\nFK_AGENT_FALLBACK {"kind": '
                                   '"refusal_fallback", "why": "downgraded"}\n')

    monkeypatch.setattr(b, "_agent_turn", turn)
    with pytest.raises(AgentError) as e:
        b.run_agent("go", role="teacher")
    assert calls["n"] == 1                    # a retry would refuse the same context again
    assert "fallback model" in str(e.value) and "not retrying" in str(e.value)
    assert "refusal_fallback" in str(e.value)


def test_run_agent_returns_the_reply_when_only_a_warning_was_written(monkeypatch):
    b = _backend(monkeypatch, agent_retries=1, agent_retry_sleep=0.0)

    def turn(cmd, prompt, role):
        return 0, "delivered\n", ('FK_USAGE {"role": "teacher"}\n'
                                  'FK_ALERT {"kind": "model_mismatch"}\n')

    monkeypatch.setattr(b, "_agent_turn", turn)
    assert b.run_agent("go", role="teacher") == "delivered\n"


def test_a_marker_in_stdout_is_enough(monkeypatch):
    from forgekernel.loop.cli import AgentError
    b = _backend(monkeypatch, agent_retries=3, agent_retry_sleep=0.0)
    calls = {"n": 0}

    def turn(cmd, prompt, role):
        calls["n"] += 1
        return 0, "FK_AGENT_FALLBACK refusal_fallback\n", ""

    monkeypatch.setattr(b, "_agent_turn", turn)
    with pytest.raises(AgentError):
        b.run_agent("go")
    assert calls["n"] == 1
