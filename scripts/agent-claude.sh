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

ARGS=(--dangerously-skip-permissions)
if [ -n "$EFFORT" ]; then
    ARGS+=(--effort "$EFFORT")
fi

if [ "${FK_AGENT_DRIVER:-}" = "goal" ]; then
    exec bash "$HERE/claude-goal.sh" "$MODEL" "$MODEL" "${ARGS[@]}"
fi
# WP19 (task 6): session resume within one scope (a meta entry, an iteration).
# FK_AGENT_SESSION_MODE=create passes --session-id (first turn of the scope);
# resume passes --resume (the scope's later turns, and a failed turn's retry).
# Default off: neither flag is passed and the turn is byte-identical to the
# baseline. The per-turn policy rewrite and the skills mirror are the loop's
# business (agent_policy.install runs before every prompt) — resuming a
# session does not weaken any of the four offline layers, because the project
# settings the session resumes INTO are rewritten every turn.
SID="${FK_AGENT_SESSION_ID:-}"
MODE="${FK_AGENT_SESSION_MODE:-}"
if [ -n "$SID" ] && [ "$MODE" = "create" ]; then
    exec "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "${FK_CLAUDE_BIN:-claude}" -p \
        --model "$MODEL" --session-id "$SID" "${ARGS[@]}"
elif [ -n "$SID" ] && [ "$MODE" = "resume" ]; then
    exec "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "${FK_CLAUDE_BIN:-claude}" -p \
        --model "$MODEL" --resume "$SID" "${ARGS[@]}"
fi
exec "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "${FK_CLAUDE_BIN:-claude}" -p --model "$MODEL" "${ARGS[@]}"
