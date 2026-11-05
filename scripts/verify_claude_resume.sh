#!/usr/bin/env bash
# verify_claude_resume.sh — WP19 (task 6) CLI capability verification.
#
# The 10-minute gate before any session-resume work is trusted: does THIS
# machine's `claude -p` support `--session-id` / `--resume`, and (with
# RKT_RESUME_VERIFY_LIVE=1) does a real two-turn session actually resume?
# The cheap checks run always and cost nothing; the live round-trip spends a
# few API calls and runs only when asked for.
#
# Output: one CAPABILITY line per check, exit 0 iff the flags exist.
set -u
BIN="${RKT_CLAUDE_BIN:-claude}"

echo "=== verify_claude_resume ==="
if ! command -v "$BIN" >/dev/null 2>&1; then
    echo "CAPABILITY cli-present: NO ($BIN not on PATH)"
    echo "VERDICT: FAIL (no CLI)"
    exit 1
fi
echo "CAPABILITY cli-present: YES"
"$BIN" --version 2>/dev/null | head -1 || true

HELP="$("$BIN" --help 2>&1 || true)"
for FLAG in --session-id --resume --print; do
    if printf '%s' "$HELP" | grep -q -- "$FLAG"; then
        echo "CAPABILITY flag$FLAG: YES"
    else
        echo "CAPABILITY flag$FLAG: NO"
        MISSING=1
    fi
done
if [ "${MISSING:-0}" = "1" ]; then
    echo "VERDICT: FAIL (flag missing — resume stays off, the loop runs the baseline)"
    exit 1
fi

if [ "${RKT_RESUME_VERIFY_LIVE:-0}" != "1" ]; then
    echo "VERDICT: FLAGS-OK (live round-trip skipped; set RKT_RESUME_VERIFY_LIVE=1 to spend it)"
    exit 0
fi

# --- the live round-trip: create a session, then resume it --------------------
SID="rkt-resume-verify-$(date +%s)"
T1=$(printf 'Reply with exactly the single word: PINE' | "$BIN" -p --model "${RKT_RESUME_MODEL:-claude-sonnet-4-5}" --session-id "$SID" 2>&1 | tail -1)
echo "CAPABILITY live-turn1: ${T1:-<empty>}"
T2=$(printf 'What single word did I ask you to reply with one turn ago? Reply with that word only.' \
    | "$BIN" -p --model "${RKT_RESUME_MODEL:-claude-sonnet-4-5}" --resume "$SID" 2>&1 | tail -1)
echo "CAPABILITY live-turn2-resumed: ${T2:-<empty>}"
if [ "$T2" = "PINE" ]; then
    echo "VERDICT: PASS (the resumed turn saw the prior turn's context)"
    exit 0
fi
echo "VERDICT: FAIL (the resumed turn lost the context — resume stays off)"
exit 1
