# Agent backends: which CLI runs a turn, what it costs, and when a turn is not the turn

Three separate concerns live behind one task key (`[agent] backend`):

| | what it decides | where |
|---|---|---|
| **the backend** | which agent CLI runs a turn, and what that CLI may be asked to do | `src/forgekernel/agentbackends/` |
| **the usage ledger** | what each turn cost (tokens, time, money), per role and per model | `scripts/agent_trailer.py` writes, `src/forgekernel/usage.py` reads |
| **refusal / downgrade detection** | whether the turn ran on the model the task asked for | `scripts/agent_trailer.py` + `RealBackend.run_agent` |

None of it changes the loop's gate, the card's clock, or the agent policy. A backend that
cannot be held to the policy is restricted instead (see §2).

## 1. The two backends that ship

| | `claude` | `codex` |
|---|---|---|
| launcher | `scripts/agent-claude.sh` | `scripts/agent-codex.sh` |
| wire | `anthropic` (`ANTHROPIC_BASE_URL`) | `openai` (`OPENAI_BASE_URL`) |
| roles | teacher **and** student | **teacher only** |
| policy enforcement | `hooks` — `.claude/settings.json` + `scripts/agent_policy_hook.py` | `none` |
| may serve a task with a sealed reference | yes | **no** (refused at task load) |
| /goal session (`FK_AGENT_DRIVER=goal`) | yes (`scripts/claude-goal.sh`) | **no** (refused with a reason) |
| exercised on this machine | yes | **no** — codex is not installed |

`global_config/models.toml` names the default; a task overrides the teacher or the student:

```toml
[task]
language = "triton"
arch     = "sm100"

[agent.teacher]
backend = "codex"
model   = "gpt-5.5-codex"
effort  = "high"            # -> -c model_reasoning_effort="high"
```

What the task file is checked for (`forgekernel/task.py:_read_agent`, all problems listed
at once):

1. `backend` is a registered backend;
2. the role is in that backend's `roles` (`codex` + `[agent.student]` is refused);
3. a backend with `policy_support == "none"` cannot serve a task with a **sealed
   reference** — blackbox mode (`[reference]` without `source`, or `blackbox = true`), or a
   leaderboard task whose `[leaderboard] ref` recon tree exists.

Why (3): the only thing that keeps an agent out of a sealed reference is the policy's read
deny lists, and those are enforced by Claude Code's settings and hook. A Codex turn would
have the same instruction in its prompt and nothing behind it. The same reason is why the
student role is Claude-only: offline turns, no ssh, no self-submitted jobs and no reads of a
sealed reference are all hook-enforced.

### The commands, byte for byte

`forgekernel.agents.agent_command(spec, role, env)` is what the loop runs. It is unchanged
apart from the `FK_AGENT_ROLE` prefix (the launcher scripts need the role for the ledger and
the refusal check):

```
env FK_AGENT_ROLE=teacher bash /abs/scripts/agent-claude.sh claude-opus-5-5 high 7200
env FK_AGENT_ROLE=teacher bash /abs/scripts/agent-codex.sh gpt-5.5-codex high 7200
```

With a provider table in front (the key itself never appears):

```
env FK_AGENT_ROLE=teacher OPENAI_BASE_URL=https://gw.example/openai FK_PROVIDER_KEY_ENV=OA_KEY \
    bash /abs/scripts/agent-codex.sh gpt-5.5-codex high 7200
```

Both scripts take the prompt on **stdin** and print the reply on **stdout**
(`bash <script> <model> [effort] [timeout_s]`).

What the Codex script runs (print it yourself with `FK_AGENT_ECHO=1`, which runs nothing):

```
python3 scripts/timeout-run.py 7200 codex exec --json --color never \
    --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
    -c model_reasoning_effort="high" -m gpt-5.5-codex -
```

- `-` is the prompt on stdin (`codex exec -`), so a 100 kB prompt never becomes an argv.
- `--skip-git-repo-check`: a teacher turn runs in the operator tree and in the judge's check
  directory as well as in the workspace clone, and only the last is a git repository.
- `--dangerously-bypass-approvals-and-sandbox` for the same reason the Claude turn skips
  permissions: no human is present to answer an approval prompt, and the harness's defences
  are the policy, the deny lists and the workspace.

## 2. Adding a backend

A backend is one module in `src/forgekernel/agentbackends/`: the module name is the backend
name, and it exports one instance `BACKEND` (nothing else registers it — `task.BACKENDS` and
`provider.grant` both read this directory).

```python
# src/forgekernel/agentbackends/mine.py
from .base import ScriptBackend

class MineBackend(ScriptBackend):
    name = "mine"                 # must equal the module name
    wire = "openai"               # "anthropic" | "openai": what its provider speaks
    script = "scripts/agent-mine.sh"   # the launcher (prompt on stdin, reply on stdout)
    roles = ("teacher",)          # what it may serve
    policy_support = "none"       # "hooks": the policy can refuse a read; "none": it cannot

BACKEND = MineBackend()
```

Then:

1. write `scripts/agent-mine.sh` with the contract above — and, when it has a structured
   mode, parse it in `scripts/agent_trailer.py` (a `parse_mine()` next to `parse_claude` /
   `parse_codex`, registered in `PARSERS`) so the turn produces a usage record;
2. add a table example to `global_config/backends.toml` if it has its own endpoint;
3. document it here, and add a test in `tests/test_agentbackends.py`;
4. if it takes code from another project, record that in `NOTICE`.

Discovery is a directory listing: nothing of a backend is imported until a task names it or
`provider.grant` asks for its wire.

How the registry screens a module: `agentbackends.names()` lists the modules, and
`load(name)` refuses one whose `BACKEND.name` disagrees with its file name (a name that can
only be read by running the module could route nothing when the module is what is broken).

## 3. The usage ledger

Every turn's numbers, from the CLI's own structured output. **Display only:** there is no
budget, no cap and no blocking anywhere in it (token spend is not a constraint here; it is
reported so a run can be read).

- **Where**: `<run dir>/agent_usage.jsonl`, one JSON object per line (`agent_usage.jsonl`
  next to `run.json`, `run.log` and `report.md`). `launch._entered` sets
  `FK_AGENT_USAGE_FILE` for the whole run, so judge, teacher and student turns share it.
- **Who writes it**: `scripts/agent_trailer.py`, which every launcher calls when
  `FK_AGENT_USAGE_FILE` is set. It also writes `FK_USAGE {...}` (the same record) on stderr.
- **Who reads it**: `forgekernel.usage.read_ledger` / `summarize` / `render_markdown`; the
  run report (`## Agent usage`: totals, per role, per model, downgraded turns) and
  `forge status` (`agent usage: 12 turn(s), 1,234,567 tokens, $3.40, 2h05m agent time`).
  Other tasks (the throughput report) read the same file through `forgekernel.usage`.

A record:

| field | meaning |
|---|---|
| `ts`, `role`, `backend` | when, which role (`teacher` / `student` / `judge`), which CLI |
| `model_requested`, `model_used` | what the task asked for, what the CLI says ran |
| `session_id` | the CLI's session id (the refusal check needs it) |
| `rc` | the turn's exit status |
| `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `total_tokens` | as reported; **null** when the CLI did not say, never 0 |
| `cost_usd`, `duration_s` | when the CLI reports them (Codex reports neither) |
| `refusal_check` | `clean` / `refusal` / `mismatch` / `unknown` (see §4) |
| `warning` | present on a downgraded turn: `refusal_fallback` or `model_mismatch` |

Notes:

- No ledger path, no structured turn: `FK_AGENT_USAGE_FILE` empty means `claude -p` runs in
  its plain-text mode (as before), with no record and no refusal check. That is what a caller
  outside a run (a bare `forge lib adapt`) gets.
- The `/goal` path (`FK_AGENT_DRIVER=goal`, `scripts/claude-goal.sh`) produces no usage
  record: it drives an interactive session through tmux and has no structured output. Its
  turns show up as `unknown_usage` only if some other turn wrote a record without one.
- A `claude` turn with `is_error: true` is recorded with `rc: 1` even when the CLI exited 0,
  so a failed turn is never counted as delivered.
- `total_tokens` for Codex is `input + output`: Codex's `input_tokens` already includes the
  cached tokens, unlike Claude's.

## 4. Refusal / downgrade detection

What it is: Claude Code answers a request its safeguards refuse by switching **the whole
session** to a fallback model and writing a `system` + `model_refusal_fallback` record into
the session transcript — no error, no exit code. A retry sends the same context, which is
refused again, and what the fallback model produces cannot be attributed to the model the
run is about. See `docs/safeguard-refusals.md` for the record and how to find the trigger.

What a turn does now (`scripts/agent_trailer.py`, after a structured turn):

1. the transcript is looked up by `session_id` under the CLI's configuration directory
   (`$CLAUDE_CONFIG_DIR` or `~/.claude`, `projects/**/*<session_id>*.jsonl`; `$CODEX_HOME` or
   `~/.codex` for Codex rollouts) and searched for `model_refusal_fallback`;
2. `model_requested` is compared with `model_used` (the result's dominant `modelUsage`
   model, else the transcript's last assistant model). An alias (`opus`), a dated id
   (`claude-opus-5-5-20260101`) and the `[1m]` suffix are the same model; a different family
   or a different version is not. A name it cannot relate is **not** reported — a false
   mismatch would end a run;
3. either hit writes `FK_ALERT {...}` on stderr and a `warning` in the record.

Then `FK_AGENT_ON_FALLBACK` decides:

| value | behaviour |
|---|---|
| `fail` (**default**) | the launcher exits 78 with the marker `FK_AGENT_FALLBACK` on stderr; `RealBackend.run_agent` raises `AgentError` **immediately, without retrying** |
| `warn` | the reply is delivered as usual; the alert and the record stay |

No transcript found → `refusal_check: "unknown"`: no error, no block, nothing assumed.

## 5. Permission mode (debugging only)

`claude` runs with `--dangerously-skip-permissions` by default (the harness's defences are
the agent policy hooks and the workspace isolation). `FK_AGENT_PERMISSION_MODE=<mode>`
replaces it with `--permission-mode <mode>`: the value is passed through unchanged, and the
accepted values are the ones `claude --help` lists — on 2.1.316 that is `acceptEdits`,
`auto`, `bypassPermissions`, `manual`, `dontAsk`, `plan` (there is **no** literal `default`;
a value the CLI does not know is refused by the CLI itself, with its own message). A
non-bypass mode waits for a human at the prompt: it is for interactive debugging, never for
a run.

## 6. Not verified on this machine

- **Codex, end to end.** Codex is not installed here. The argv follows Atrex's `codex.py`
  (Apache-2.0, `NOTICE`) and the CLI's own definitions
  (`codex-rs/exec/src/cli.rs`: `--json`, `--color`, `-o/--output-last-message`, `PROMPT -` =
  stdin) plus the documented event sample; the launcher's `FK_AGENT_ECHO=1` prints it, and
  the tests cover its shape through a fake `codex`. Whether `codex exec` on the installed
  version accepts `--dangerously-bypass-approvals-and-sandbox`, `--skip-git-repo-check` and
  `-c model_reasoning_effort="…"` in that order is the one thing a machine with codex must
  confirm (`codex exec --help`).
- **A real refusal.** No turn here has refused: the check is exercised against synthetic
  transcripts in the test suite, and the `model_refusal_fallback` record's shape comes from
  `docs/safeguard-refusals.md`.
- **A logged-in Claude turn.** `claude` on this machine is not logged in
  (`claude --version` 2.1.316; a one-prompt probe returned `Not logged in`, 0 tokens). What
  the probe did confirm: `--output-format json --verbose` prints **one JSON array** of
  `system` / `assistant` / `result` records; the `result` record carries `result`, `usage`,
  `modelUsage`, `session_id`, `total_cost_usd`, `duration_ms`, `is_error`; and the
  transcript lands at `<config>/projects/<mangled-cwd>/<session_id>.jsonl`, which is where
  the refusal check looks.
