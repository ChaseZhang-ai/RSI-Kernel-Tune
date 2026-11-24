#!/usr/bin/env bash
# agent-claude.sh — one agent turn on Claude Code, the way `forge run` runs every turn.
#
#   bash agent-claude.sh <model> [effort] [timeout_s]
#     prompt on stdin, reply on stdout.
#
# A turn is `claude -p` under a wall-clock timeout: `claude -p` has none of its own, and a
# stream that opens and then never delivers its last frame would hang the run with it.
# A caller that needs a /goal session (loop/goal_ref.py, when a task has no reference)
# sets FK_AGENT_DRIVER=goal; the turn is then an interactive claude driven by /goal
# (scripts/claude-goal.sh), with the same model and arguments.
#
# Credentials come from the environment (local.env): ANTHROPIC_API_KEY or
# CLAUDE_CODE_OAUTH_TOKEN, or the CLI's own login (provider = "native"). Another provider
# (global_config/backends.toml) sets ANTHROPIC_BASE_URL and FK_PROVIDER_KEY_ENV, the name of
# the variable that holds its key.
#
# Two modes, one contract (prompt in, reply out, the reply on stdout either way):
#
#   FK_AGENT_USAGE_FILE set     `--output-format json --verbose`. scripts/agent_trailer.py
#                               takes `.result` out of the JSON and writes it to stdout,
#                               and writes the turn's usage record (`FK_USAGE {...}`, and
#                               `FK_AGENT_USAGE_FILE` gets it as one JSONL line) plus the
#                               refusal / model-mismatch check to stderr. A run always sets
#                               it (launch._entered); a downgraded turn then ends here with
#                               FK_AGENT_ON_FALLBACK=fail (the default) and exit 78.
#   FK_AGENT_USAGE_FILE empty   plain `-p` text output, exec'd through as before: no
#                               record, no refusal check (a caller outside a run).
#
# FK_AGENT_PERMISSION_MODE=default|acceptEdits|plan (or any other mode `claude --help`
# lists) replaces `--dangerously-skip-permissions` with `--permission-mode <value>`. A
# non-bypass mode needs a human at the prompt: it is for interactive debugging only.
set -u
MODEL=${1:?model, e.g. claude-opus-5-5}
EFFORT=${2:-}
TIMEOUT=${3:-7200}
if [ -n "${FK_PROVIDER_KEY_ENV:-}" ]; then
    KEY="${!FK_PROVIDER_KEY_ENV:-}"
    if [ -z "$KEY" ]; then
        echo "agent-claude.sh: $FK_PROVIDER_KEY_ENV is not set (the provider's key)" >&2
        exit 2
    fi
    # claude reads the token from AUTH_TOKEN but refuses to start without API_KEY also set.
    export ANTHROPIC_API_KEY="$KEY" ANTHROPIC_AUTH_TOKEN="$KEY"
fi
# The agent policy is delivered in the working directory, so nothing here changes it
# (forge run names this script by its absolute path).
HERE="$(dirname "${BASH_SOURCE[0]}")"
PY="${FK_AGENT_PY:-python3}"
CLAUDE="${FK_CLAUDE_BIN:-claude}"

if [ -n "${FK_AGENT_PERMISSION_MODE:-}" ]; then
    ARGS=(--permission-mode "$FK_AGENT_PERMISSION_MODE")
else
    ARGS=(--dangerously-skip-permissions)
fi
if [ -n "$EFFORT" ]; then
    ARGS+=(--effort "$EFFORT")
fi

if [ "${FK_AGENT_DRIVER:-}" = "goal" ]; then
    exec bash "$HERE/claude-goal.sh" "$MODEL" "$MODEL" "${ARGS[@]}"
fi

if [ -z "${FK_AGENT_USAGE_FILE:-}" ]; then
    exec "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "$CLAUDE" -p --model "$MODEL" "${ARGS[@]}"
fi

# The structured turn: capture the CLI's JSON, then hand it to the trailer helper, which
# owns the stdout contract, the usage record and the refusal check.
RAW="$(mktemp "${TMPDIR:-/tmp}/fk-claude-turn.XXXXXX")" || exit 1
trap 'rm -f "$RAW"' EXIT
"$PY" "$HERE/timeout-run.py" "$TIMEOUT" "$CLAUDE" -p --output-format json --verbose \
    --model "$MODEL" "${ARGS[@]}" >"$RAW"
RC=$?
"$PY" "$HERE/agent_trailer.py" claude --rc "$RC" --raw "$RAW" --model "$MODEL" \
    --role "${FK_AGENT_ROLE:-unknown}" --usage-file "$FK_AGENT_USAGE_FILE" \
    --on-fallback "${FK_AGENT_ON_FALLBACK:-fail}"
exit $?
