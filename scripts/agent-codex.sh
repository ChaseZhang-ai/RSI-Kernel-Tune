#!/usr/bin/env bash
# agent-codex.sh — one agent turn on Codex, the way `forge run` runs a teacher turn.
#
#   bash agent-codex.sh <model> [effort] [timeout_s]
#     prompt on stdin, reply on stdout (the contract of agent-claude.sh).
#
#   FK_AGENT_ECHO=1 prints the argv it would run, one argument per line, and exits 0
#   without running anything (how a test or a human checks the invocation).
#
# The turn is `codex exec --json` under a wall-clock timeout, in a fresh session, with the
# prompt read from stdin (`codex exec -`). `--json` makes stdout a JSONL event stream;
# scripts/agent_trailer.py takes the last completed agent_message out of it and writes the
# reply to stdout, with the usage record (`FK_USAGE {...}`) and the refusal check on
# stderr, exactly as for the Claude backend. Credentials: the CLI's own login, or
# OPENAI_API_KEY / CODEX_API_KEY in the environment; a provider table (wire = "openai")
# sets OPENAI_BASE_URL and FK_PROVIDER_KEY_ENV, the name of the variable holding its key.
#
# --dangerously-bypass-approvals-and-sandbox is the default for the same reason the Claude
# turn skips permissions: the harness's defences are the agent policy, the read-only deny
# lists and the workspace, and no human is present to answer an approval prompt. The
# `none` policy_support of this backend is why it may only serve the teacher, and only on
# a task without a sealed reference (forgekernel.task._read_agent).
#
# --skip-git-repo-check: a teacher turn runs in the operator tree and in the judge's check
# directory as well as in the workspace clone, and only the last is a git repository;
# codex otherwise refuses to start outside one.
#
# NOT EXERCISED HERE: codex is not installed on this machine. The argv follows Atrex's
# codex.py (Apache-2.0, see NOTICE) and the CLI's own clap definitions
# (codex-rs/exec/src/cli.rs: --json, --color, -o/--output-last-message, PROMPT `-` = stdin)
# plus docs/exec.md's event sample; `codex exec --help` on the machine that has it is the
# last word. See docs/agent-backends.md.
set -u
MODEL=${1:?model, e.g. gpt-5.5-codex}
EFFORT=${2:-}
TIMEOUT=${3:-7200}
if [ "${FK_AGENT_DRIVER:-}" = "goal" ]; then
    echo "agent-codex.sh: the codex backend cannot serve a /goal session (the session is a Claude Code feature, scripts/claude-goal.sh)" >&2
    exit 2
fi
if [ -n "${FK_PROVIDER_KEY_ENV:-}" ]; then
    KEY="${!FK_PROVIDER_KEY_ENV:-}"
    if [ -z "$KEY" ]; then
        echo "agent-codex.sh: $FK_PROVIDER_KEY_ENV is not set (the provider's key)" >&2
        exit 2
    fi
    export OPENAI_API_KEY="$KEY" CODEX_API_KEY="$KEY"
fi
HERE="$(dirname "${BASH_SOURCE[0]}")"
PY="${FK_AGENT_PY:-python3}"
CODEX="${FK_CODEX_BIN:-codex}"

ARGS=(exec --json --color never --dangerously-bypass-approvals-and-sandbox
      --skip-git-repo-check)
if [ -n "$EFFORT" ]; then
    ARGS+=(-c "model_reasoning_effort=\"$EFFORT\"")
fi
ARGS+=(-m "$MODEL" -)

if [ "${FK_AGENT_ECHO:-}" = "1" ]; then
    printf '%s\n' "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "$CODEX" "${ARGS[@]}"
    exit 0
fi
if ! command -v "$CODEX" >/dev/null 2>&1; then
    echo "agent-codex.sh: $CODEX is not installed (or not executable); install the Codex CLI or point FK_CODEX_BIN at it" >&2
    exit 127
fi

RAW="$(mktemp "${TMPDIR:-/tmp}/fk-codex-turn.XXXXXX")" || exit 1
trap 'rm -f "$RAW"' EXIT
"$PY" "$HERE/timeout-run.py" "$TIMEOUT" "$CODEX" "${ARGS[@]}" >"$RAW"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "agent-codex.sh: the turn failed (rc=$RC); run \`codex login status\` and \`codex exec \"reply ok\"\` to diagnose" >&2
fi
"$PY" "$HERE/agent_trailer.py" codex --rc "$RC" --raw "$RAW" --model "$MODEL" \
    --role "${FK_AGENT_ROLE:-unknown}" \
    --usage-file "${FK_AGENT_USAGE_FILE:-}" \
    --on-fallback "${FK_AGENT_ON_FALLBACK:-fail}"
exit $?
