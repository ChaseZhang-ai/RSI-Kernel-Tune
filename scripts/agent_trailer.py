#!/usr/bin/env python3
"""agent_trailer.py — turn an agent CLI's structured output into the loop's contract.

    python3 agent_trailer.py <claude|codex> --rc N [--raw FILE] [--model M] [--role R]
        [--usage-file F] [--on-fallback fail|warn] [--config-dir D]... [--transcript F]

    stdin:  the raw stdout of the turn (when --raw is not given)
    stdout: the reply text, exactly what a caller of scripts/agent-<name>.sh expects
    stderr: `FK_USAGE {...}` (one JSON line: this turn's usage record), and when the turn
            was quietly downgraded `FK_ALERT {...}` plus, in fail mode, a line carrying
            the marker `FK_AGENT_FALLBACK`
    exit:   0 for a delivered turn; the child's own status when the child failed; 3 when
            the output was structured but held no reply; 78 when a downgrade happened and
            FK_AGENT_ON_FALLBACK=fail

Why this exists: the loop's contract is text in / text out (loop/cli.py
RealBackend.run_agent), but the usage ledger and the refusal check need what the CLIs
print in their structured modes (`claude -p --output-format json`, `codex exec --json`).
This is that seam, and only this: the loop never parses an agent's JSON.

The two CLIs are not parsed the same way, deliberately:

- **claude**: one JSON object (`{"type":"result","result":"…","usage":{…},
  "session_id":…}`); `result` is the reply. If the text is not JSON at all, the turn ran
  in plain-text mode (an older CLI, a fake in a test): the raw text is passed through and
  the child's status is kept — a parse failure must never fail a turn that worked.
- **codex**: a JSONL event stream (`thread.started`, `item.completed`, `turn.completed`,
  `turn.failed`); the reply is the last completed `agent_message` item, the usage comes
  from `turn.completed`. This stream is a newer surface whose shape may change between
  versions, so several historical shapes are tolerated; when no reply can be found the
  turn fails (exit 3) instead of handing raw JSON to the loop as if it were a reply.

Refusal check (docs/safeguard-refusals.md): Claude Code answers a refused request by
switching the whole session to a fallback model and writing a `system` +
`model_refusal_fallback` record into the session transcript. A turn that was downgraded
is not the turn anyone asked for, and its context will be refused again — so the default
(`FK_AGENT_ON_FALLBACK=fail`) is to end it, and RealBackend.run_agent does not retry.
The transcript is looked up by `session_id` under the CLI's configuration directory
(`$CLAUDE_CONFIG_DIR` or `~/.claude`; `$CODEX_HOME` or `~/.codex` for codex rollouts).
When no transcript is found the check is "unknown": no error, no block, and the record
says so.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REFUSAL_SUBTYPE = "model_refusal_fallback"
USAGE_MARKER = "FK_USAGE"
ALERT_MARKER = "FK_ALERT"
FALLBACK_MARKER = "FK_AGENT_FALLBACK"
EXIT_NO_REPLY = 3
EXIT_DOWNGRADED = 78

# The record's token counters, in the order they are summed and rendered. A missing value
# is null, never 0: "the CLI did not say" and "no tokens" are different facts.
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens", "total_tokens")
_SYNONYMS = {
    "input_tokens": ("input_tokens", "inputTokens", "input"),
    "output_tokens": ("output_tokens", "outputTokens", "output"),
    "cache_read_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens",
                                "cacheRead", "cached_input_tokens", "cachedInputTokens"),
    "cache_creation_input_tokens": ("cache_creation_input_tokens",
                                    "cacheCreationInputTokens", "cacheWrite"),
    "total_tokens": ("total_tokens", "totalTokens"),
}
# A family word, for telling "the same model under another version string" from "a
# different model" when neither name is a prefix of the other.
_FAMILY = re.compile(r"opus|sonnet|haiku|fable|nano|gpt|o\d", re.I)


class Turn:
    """What one structured turn said: the reply, its usage, and who produced it."""

    def __init__(self, reply: str | None, *, usage: dict | None = None,
                 cost_usd: float | None = None, duration_s: float | None = None,
                 session_id: str | None = None, model_used: str | None = None,
                 is_error: bool = False, error: str = "") -> None:
        self.reply = reply
        self.usage = usage if usage is not None else {k: None for k in USAGE_KEYS}
        self.cost_usd = cost_usd
        self.duration_s = duration_s
        self.session_id = session_id
        self.model_used = model_used
        self.is_error = is_error
        self.error = error


# -- counters -----------------------------------------------------------------------


def _number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return None
    return None


def _counter(raw: dict, *names: str) -> int | None:
    for name in names:
        value = _number(raw.get(name))
        if value is not None:
            return int(value)
    return None


def normalize_usage(raw: object, *, cache_within_input: bool = False) -> dict:
    """The five counters from a CLI's usage block, under one set of names.

    ``cache_within_input``: the CLI's ``input_tokens`` already counts the cached tokens
    (codex), so the total must not add the cache again."""
    if not isinstance(raw, dict):
        return {k: None for k in USAGE_KEYS}
    out = {key: _counter(raw, *_SYNONYMS[key]) for key in USAGE_KEYS}
    if out["total_tokens"] is None:
        if cache_within_input:
            parts = [out["input_tokens"], out["output_tokens"]]
        else:
            parts = [out[k] for k in USAGE_KEYS[:-1]]
        parts = [p for p in parts if p is not None]
        if parts:
            out["total_tokens"] = sum(parts)
    return out


# -- the CLIs' output ----------------------------------------------------------------


def _json_objects(text: str) -> list[dict]:
    """Every JSON object ``text`` holds: a whole object, a whole array of them
    (`claude -p --output-format json --verbose` prints one array: system, assistant,
    result), or the objects that are one per line (a JSONL event stream)."""
    text = text.strip()
    if not text:
        return []
    try:
        whole = json.loads(text)
    except ValueError:
        whole = None
    if isinstance(whole, dict):
        return [whole]
    if isinstance(whole, list):
        return [x for x in whole if isinstance(x, dict)]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _ms(value: object) -> float | None:
    n = _number(value)
    return None if n is None else float(n) / 1000.0


def _dominant_model(model_usage: object) -> str | None:
    """The model that produced most of the turn: the result object's ``modelUsage`` maps
    each model to its token counts, so an auxiliary model's few tokens lose."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    best, best_out = None, None
    for name, block in model_usage.items():
        if not isinstance(name, str) or not name:
            continue
        out = None
        if isinstance(block, dict):
            out = _counter(block, "outputTokens", "output_tokens", "output")
        if best is None or (out is not None and (best_out is None or out > best_out)):
            best, best_out = name, out
    return best


def parse_claude(text: str) -> Turn | None:
    """The reply of a ``claude -p --output-format json`` turn; None when the output is not
    this CLI's structured output at all (a plain-text turn)."""
    objects = _json_objects(text)
    if not objects:
        return None
    result = None
    for obj in objects:
        if obj.get("type") == "result" or ("result" in obj and obj.get("type") is None):
            result = obj
    if result is None:
        return Turn(None, error="the JSON output has no result record")
    reply = result.get("result")
    if not isinstance(reply, str):
        reply = None
    revision = result.get("usage")
    if revision is None and isinstance(result.get("message"), dict):
        revision = result["message"].get("usage")
    return Turn(reply, usage=normalize_usage(revision),
                cost_usd=float(result["total_cost_usd"])
                if _number(result.get("total_cost_usd")) is not None else None,
                duration_s=_ms(result.get("duration_ms")) or _ms(result.get("duration_api_ms")),
                session_id=result.get("session_id")
                if isinstance(result.get("session_id"), str) else None,
                model_used=_dominant_model(result.get("modelUsage")),
                is_error=bool(result.get("is_error")),
                error=str(result.get("subtype") or "") if result.get("is_error") else "")


def parse_codex(text: str) -> Turn | None:
    """The reply of a ``codex exec --json`` turn; None when there is no JSON in it."""
    events = _json_objects(text)
    if not events:
        return None
    reply: str | None = None
    usage: dict = {}
    session_id: str | None = None
    error = ""
    failed = False
    for event in events:
        kind = event.get("type")
        if kind == "thread.started" and isinstance(event.get("thread_id"), str):
            session_id = event["thread_id"]
        elif kind == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message" \
                    and isinstance(item.get("text"), str):
                reply = item["text"]
        elif kind == "turn.completed":
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
        elif kind == "turn.failed":
            failed = True
            detail = event.get("error")
            if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                error = detail["message"]
            elif isinstance(detail, str):
                error = detail
        elif kind == "error":
            failed = True
            error = error or "the event stream reported an error"
        elif kind == "result" and isinstance(event.get("result"), str):
            # An older event shape: one terminal result object.
            reply = event["result"]
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
        else:
            # The pre-`item.*` shape: {"id": …, "msg": {"type": "agent_message", …}}.
            msg = event.get("msg")
            if isinstance(msg, dict) and msg.get("type") == "agent_message" \
                    and isinstance(msg.get("message"), str):
                reply = msg["message"]
    return Turn(reply, usage=normalize_usage(usage, cache_within_input=True),
                session_id=session_id, is_error=failed, error=error)


PARSERS = {"claude": parse_claude, "codex": parse_codex}


# -- the session transcript ------------------------------------------------------------


def default_config_dirs(agent: str) -> list[Path]:
    if agent == "codex":
        home = os.environ.get("CODEX_HOME") or "~/.codex"
    else:
        home = os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"
    return [Path(home).expanduser()]


def find_transcripts(config_dirs: list[Path], session_id: str) -> list[Path]:
    """The session's transcript files: a name containing ``session_id`` under the CLI's
    ``projects/`` (claude) or ``sessions/`` (codex) tree."""
    found: list[Path] = []
    for directory in config_dirs:
        if not directory.is_dir():
            continue
        for sub in ("projects", "sessions"):
            base = directory / sub
            if not base.is_dir():
                continue
            for path in base.rglob(f"*{session_id}*.jsonl"):
                if path.is_file() and path not in found:
                    found.append(path)
    return found


def inspect_transcripts(paths: list[Path]) -> dict:
    """{"refusal": the last model_refusal_fallback record or None, "models": [...]}."""
    refusal = None
    models: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("type") == "system" and record.get("subtype") == REFUSAL_SUBTYPE:
                refusal = record
            message = record.get("message")
            if isinstance(message, dict):
                model = message.get("model")
                if isinstance(model, str) and model and not model.startswith("<"):
                    models.append(model)
    return {"refusal": refusal, "models": models}


def _norm(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(model).lower())


def _bare(model: str) -> str:
    """The model without a CLI-side alias suffix such as ``[1m]``."""
    return str(model).split("[", 1)[0]


def same_model(requested: str | None, used: str | None) -> bool:
    """Whether ``used`` is the model that was asked for (an alias, a dated id, or the same
    model). Deliberately conservative: a name it cannot relate is *not* reported — a false
    mismatch ends a run under FK_AGENT_ON_FALLBACK=fail."""
    if not requested or not used:
        return True
    req, use = _norm(_bare(requested)), _norm(_bare(used))
    if not req or not use:
        return True
    if req == use or use.startswith(req) or req.startswith(use):
        return True
    if not any(ch.isdigit() for ch in req) and req in use:
        return True                     # a bare alias ("opus") resolved to a full id
    req_family, use_family = _FAMILY.search(req), _FAMILY.search(use)
    if req_family and use_family:
        return req_family.group(0) == use_family.group(0) and not any(
            ch.isdigit() for ch in req)
    return True                         # nothing comparable: do not cry wolf


# -- the record ------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_record(*, role: str, agent: str, requested_model: str | None, turn: Turn,
                 rc: int, refusal_check: str, warning: str | None,
                 session_id: str | None) -> dict:
    record = {"ts": _now(), "role": role, "backend": agent,
              "model_requested": requested_model, "model_used": turn.model_used,
              "session_id": session_id or turn.session_id, "rc": rc}
    record.update(turn.usage)
    record["cost_usd"] = turn.cost_usd
    record["duration_s"] = turn.duration_s
    record["refusal_check"] = refusal_check
    if warning:
        record["warning"] = warning
    return record


def refusal_check(agent: str, turn: Turn, *, config_dirs: list[Path] | None = None,
                  transcripts: list[Path] | None = None,
                  requested_model: str | None = None) -> dict:
    """{"alerts": [...], "refusal_check": "clean|refusal|mismatch|unknown",
    "warning": first kind or None, "model_used": ...} for one finished turn."""
    paths = list(transcripts or [])
    if not paths:
        session_id = turn.session_id
        if session_id:
            paths = find_transcripts(
                config_dirs if config_dirs is not None else default_config_dirs(agent),
                session_id)
    inspected = inspect_transcripts(paths) if paths else {"refusal": None, "models": []}
    model_used = turn.model_used or (inspected["models"][-1] if inspected["models"] else None)
    alerts: list[dict] = []
    state = "unknown"
    if inspected["refusal"] is not None:
        record = inspected["refusal"]
        alerts.append({"kind": "refusal_fallback",
                       "session_id": turn.session_id,
                       "original_model": record.get("originalModel"),
                       "fallback_model": record.get("fallbackModel"),
                       "category": record.get("apiRefusalCategory"),
                       "scope": record.get("scope"),
                       "detail": "the request context was refused and the whole session "
                                 "was moved to a fallback model; a retry refuses the same "
                                 "context"})
        state = "refusal"
    elif model_used and not same_model(requested_model, model_used):
        alerts.append({"kind": "model_mismatch", "session_id": turn.session_id,
                       "model_requested": requested_model, "model_used": model_used,
                       "detail": "the turn did not run on the model the task asked for"})
        state = "mismatch"
    elif paths:
        state = "clean"
    return {"alerts": alerts, "refusal_check": state,
            "warning": alerts[0]["kind"] if alerts else None, "model_used": model_used}


# -- main ------------------------------------------------------------------------------


def _say(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")
    sys.stderr.flush()


def _emit_reply(reply: str) -> None:
    sys.stdout.write(reply)
    if not reply.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _append(path: str, record: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as e:      # a ledger that cannot be written is not a failed turn
        _say(f"agent_trailer.py: cannot append to {path}: {e}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent_trailer.py",
                                description="One agent turn's structured output -> the "
                                            "loop's reply + usage record + refusal check.")
    p.add_argument("agent", choices=sorted(PARSERS), help="the backend that produced it")
    p.add_argument("--rc", type=int, required=True, help="the CLI's own exit status")
    p.add_argument("--raw", default=None,
                   help="the CLI's raw stdout; omitted: read stdin")
    p.add_argument("--model", default=None, help="the model the task asked for")
    p.add_argument("--role", default=None,
                   help="the turn's role (default: $FK_AGENT_ROLE, else 'unknown')")
    p.add_argument("--usage-file", default=None,
                   help="append the record here as JSONL (default: $FK_AGENT_USAGE_FILE)")
    p.add_argument("--on-fallback", choices=("fail", "warn"), default=None,
                   help="what a downgraded turn does (default: $FK_AGENT_ON_FALLBACK, "
                        "else fail)")
    p.add_argument("--config-dir", action="append", default=None,
                   help="where the CLI keeps its session transcripts (repeatable)")
    p.add_argument("--transcript", action="append", default=None,
                   help="a session transcript file to check (repeatable)")
    args = p.parse_args(argv)

    rc = args.rc
    try:
        raw = Path(args.raw).read_text(encoding="utf-8", errors="replace") if args.raw \
            else sys.stdin.read()
    except OSError as e:
        _say(f"agent_trailer.py: cannot read {args.raw}: {e}")
        return rc or EXIT_NO_REPLY
    role = args.role or os.environ.get("FK_AGENT_ROLE") or "unknown"
    usage_file = args.usage_file if args.usage_file is not None \
        else os.environ.get("FK_AGENT_USAGE_FILE") or ""
    on_fallback = args.on_fallback or os.environ.get("FK_AGENT_ON_FALLBACK") or "fail"
    config_dirs = [Path(d).expanduser() for d in args.config_dir] if args.config_dir else None
    transcripts = [Path(t).expanduser() for t in args.transcript] if args.transcript else None

    turn = PARSERS[args.agent](raw)
    if turn is None:
        # Not the CLI's structured output: plain text is already the reply.
        sys.stdout.write(raw)
        if raw and not raw.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        if args.agent != "claude":
            _say(f"agent_trailer.py: the {args.agent} output is not a JSON event stream; "
                 f"no reply could be extracted")
            return rc or EXIT_NO_REPLY
        if rc == 0:
            _say(f"agent_trailer.py: the {args.agent} output is not JSON (a plain-text "
                 f"turn); no usage record")
        return rc
    if turn.is_error and rc == 0:
        rc = 1               # the CLI said the turn failed: do not report it as delivered
    if turn.reply is None:
        sys.stdout.write(raw)
        if raw and not raw.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        _say(f"agent_trailer.py: no reply text in the {args.agent} output"
             + (f": {turn.error}" if turn.error else ""))
        return rc or EXIT_NO_REPLY

    _emit_reply(turn.reply)
    check = refusal_check(args.agent, turn, config_dirs=config_dirs,
                          transcripts=transcripts, requested_model=args.model)
    record = build_record(role=role, agent=args.agent, requested_model=args.model,
                          turn=turn, rc=rc, refusal_check=check["refusal_check"],
                          warning=check["warning"], session_id=turn.session_id)
    if usage_file:
        _append(usage_file, record)
    _say(f"{USAGE_MARKER} " + json.dumps(record, ensure_ascii=False, sort_keys=True))
    for alert in check["alerts"]:
        _say(f"{ALERT_MARKER} " + json.dumps(alert, ensure_ascii=False, sort_keys=True))
    if check["alerts"] and on_fallback == "fail":
        _say(f"{FALLBACK_MARKER} " + json.dumps(
            {"kind": check["warning"], "role": role,
             "why": "a downgraded turn is not the turn the task asked for, and its context "
                    "will be refused again (docs/safeguard-refusals.md)",
             "alerts": check["alerts"]}, ensure_ascii=False, sort_keys=True))
        return EXIT_DOWNGRADED
    return rc


if __name__ == "__main__":
    sys.exit(main())
