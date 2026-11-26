# opkb: the operator-family lesson base (`forge lessons`)

A finished run leaves behind a trajectory of objective facts: which change was kept and what
it bought, which change was reverted and with what error, what stopped the run, what the
reconnaissance measured on the reference. Until now that trajectory died with the run. The
board's 235 problems come from 124 models and many are the same operator in another model
(six RMSNorms, nine GEMMs in one track), so the next problem of the same family paid again
for everything the previous one had already measured.

`op_kb/` keeps those facts: one JSON record per run, filed under the run's operator family,
and a rendered file the next run's teacher reads as `{LESSONS_PATH}`.

```bash
forge lessons harvest .artifacts/069_rms_norm_251124-101500-ab12   # one run -> one record
forge lessons render tasks/069_rms_norm --out lessons.md           # records -> teacher file
FK_LESSONS_PATH=$PWD/lessons.md forge new tasks/069_rms_norm       # (or let campaign do it)
forge lessons lint                                                # check + quarantine
forge lessons query tasks/069_rms_norm --k 5                      # what would be read
forge lessons stats                                               # counts per family
```

The lesson base is `--root`, else `FK_OPKB_ROOT`, else `<repository>/op_kb`. It is a project
asset and **is tracked by git**; the run directories it is extracted from (`.artifacts/`) are
not, so what is committed is the extracted records, never a run's tree.

## 1. Why mechanical extraction

`docs/design/milestone-distillation.md` §4.1bis: *"do not pre-summarise with an LLM -- the
summary's blind spots become the meta agent's blind spots"*. A record is therefore assembled
by a parser from fields that are already structured; free text (a change description, an
error message, a recon table cell) is copied **verbatim**, truncated at a fixed length, and
marked `verbatim: true`. Nothing in a record is paraphrased, scored, ranked or explained.

§7 of the same document says a per-operator milestone is single-use and what generalises is
the meta-milestone plus the knowledge base. The knowledge base (`hw_kb/`) only holds
*hardware mechanisms* certified by GPU probes, so "this family's routes, dead ends and
prices" had no home. That is what `op_kb/` is.

`tests/test_repo_hygiene.py` guards the opposite boundary: a tracked operator tree would let
a later run of the same operator read the previous run's conclusions. That rule is about
*trees*, and the user has since dropped the "distill from zero" requirement: an earlier run of
the same problem **is** allowed to be read. The rendered file marks such a block
`same problem`; `--exclude-self` restores the old isolation for an experiment that needs it.

## 2. The record

```
op_kb/<family>/<id>.json      one record per run; the file name is the record's id
op_kb/index.json              one summary row per record (rebuilt by harvest and lint)
op_kb/README.md               what the directory is (written by the first harvest)
op_kb/_quarantine/            records that failed lint, file name preserved (plus .reason.txt)
```

`family` is `rsi_kernel_tune.family.family_key` of the board definition name (else of the task
name) -- the same keys `families.toml` declares. `id` is `<problem>-<run id>`, so re-harvesting
the same run writes the same bytes.

### Fields and where each one comes from

| field | source (all under `<run>/`) |
|---|---|
| `schema` | the constant `1` this build reads |
| `id` | `run.json`'s `id` + the problem label |
| `created` | `run.json`'s `updated` (else `created`, else the file's mtime) |
| `family` | `problem.definition` (else `problem.name`) through `rsi_kernel_tune.family` |
| `problem.suite` | the card's `eval/leaderboard/problem.json` → `suite` |
| `problem.name` | `run.json` → `task` (the task/operator name) |
| `problem.definition` | the card's `problem.json` → `definition.name` (the board's name) |
| `problem.track` | the card's `sol_board.kernel_id` through `campaign.solscore.track_of` |
| `problem.arch`, `problem.language` | the task snapshot's `task/task.toml` → `[task]` |
| `problem.workloads` | the card's `group` length (1 for a one-workload card) |
| `reference.kind` | `[reference]` (`blackbox` → `sealed`, else `eager`), else the card's `reference_impl.kind` (`definition` → `eager`; `binary`/`sealed`/`baseline_solution` → `sealed`), else `generated` (a task that shipped no reference) or `none` |
| `outcome.window` | the card's `loop.toml` → `[loop] gate_window`, else `[target] timing`, else the loop's default `device` |
| `outcome.metric_kind`, `.value` | the board's SOL mean over the card's `T_b`/`T_SOL` and the confirmation's per-workload device medians (`campaign.solscore`, the one implementation), else the confirmation's `median_ratio` as `speedup` |
| `outcome.stop` | `loop_summary.json` → `stop`, else `run.json` → `stop` |
| `outcome.hours` | `run.json`'s `updated - created` |
| `outcome.student_iterations` | the number of `operators/<op>/student/status.jsonl` rows |
| `kept[]` | `status.jsonl` rows with verdict `KEEP`: `iter`, `change`, `metric_before` → `before_ms`, the gate-window reading → `after_ms` |
| `failed[]` | `status.jsonl` rows with verdict `REVERT`: `iter`, `change`, `error`, `kind` = `correctness` when the row's `correctness` is `False`, else `revert` |
| `meta[]` | `operators/<op>/meta_status.jsonl`: `meta_step` → `round`, the raw `outcome` |
| `certified_kb[]` | `loop_summary.json` → `kb_harvest` (the `hw_kb/...` recipes the end-of-run harvest recorded) |
| `usage` | `<run>/agent_usage.jsonl` (the backends task's ledger; optional) |
| `recon` | the reconnaissance tree's `report.md` + `candidates.md`: only the `*-OK` verdict lines and the candidate-table rows |
| `source.run`, `source.best_sha` | the run directory's name; `best_confirmation.json` → `sha` |

A field whose source is absent is **omitted**, never defaulted: "no reading" and "a reading
of zero" are different facts. `harvest` prints one note per artifact it could not read.

`outcome.value` with `metric_kind=sol` is the board's arithmetic (the card's frozen `T_b` /
`T_SOL` and the per-workload device medians of `best_confirmation.json`) applied to the
**run's own** readings: it is the run machine's estimate of the board score, not the board's
score -- a B300 reading is optimistic on FP4-bound and SFU-bound kernels, and no clock lock is
involved. The records are ordered and compared against each other, not against the board.
`speedup` is the confirmation's `median_ratio` as the loop recorded it (same-job when the
reference rode the jobs, median-vs-median otherwise).

### The reconnaissance block

The user allowed the reference's measured structure into the lesson base (the reference is
readable now), but only in a mechanical form. `record.py` reads exactly two things out of the
recon tree:

- **verdict lines** — `SASS-OK`, `SCHED-OK`, `LIVE-OK`, `STALL-OK`, `ORCH-OK`, `STRUCT-OK`,
  `LAUNCH-OK`, `ROLE-OK`, `PMTL-OK`, ... Each is printed by a recon tool only after its own
  self-checks pass, and each carries `key=value` fields (grid, block, cluster, instruction
  counts, peak live registers, MMA runs, ...). The line is kept verbatim; its fields are
  parsed.
- **candidate-table rows** — `decision | value | ... | [VERIFY: ...]` from `candidates.md`
  (tile shape, stage count, register split, cluster, epilogue ...). Duplicate
  (decision, value) pairs collapse.

The report's prose is **not** read; neither are `notes/` scripts, the milestone, SASS dumps
or the other run artifacts. A fixture decoy test pins that: a token written into the report's
prose, the milestone and a probe script never appears in a record.

## 3. Reading a problem: same family, self marked, ordered

`query` / `render` resolve the problem from a task directory (`tasks/<op>`, which yields the
task name, the board definition name and the suite from the frozen card), from a run
directory (`.artifacts/<op>_<id>`, read through its `run.json` and the card under
`operators/`), or from a bare problem name. A record matches the problem when
`problem.name` **or** `problem.definition` agrees, so a run of `tasks/sol-069_rms_norm` and
one of `tasks/069_rms_norm` are the same problem when both resolve to definition
`069_rms_norm`.

- Only records of the **same family** are returned. A different family's numbers describe a
  different kernel. No records of the family -> an empty list, and `render` writes a minimal
  file saying so (not an error).
- The problem's own earlier runs are **included by default** and marked with the line
  `same problem`; `--exclude-self` drops them.
- Order: `outcome.value` descending (a record with no metric value sorts last), then
  `created` descending, then `id` ascending -- a total order, so the same input renders to
  the same bytes.

## 4. The rendered file

Written for a teacher agent, English, in this order:

1. A title and exactly three explanation lines: what the file is (mechanically extracted
   facts), where it comes from (other operators of the family; `same problem` = this
   problem's own run), and how far the numbers carry (only between close shapes and dtypes;
   nothing here is a decision for this problem).
2. One block per record: the problem and its family, the `same problem` line, the problem
   line (name, definition, suite, track, arch, language, workload count), the reference kind,
   the outcome (metric, value, window, stop, student iterations, hours), the creation time,
   the source run and best sha, then `kept` (iteration, change, before/after reading),
   `failed` (iteration, kind, change, error text), `meta` (round, patch outcome),
   `certified_kb`, the recon entries, and the usage totals.

Every number is printed in the record's own JSON spelling, so a rendered digit is always a
substring of the record; the render adds no number of its own (`0` indices, counts of blocks
and the like are avoided on purpose). No evaluative sentence is added -- the file names
values, not judgments, because it is bound into the prompt as run-time data and must not
outrank the prompt.

Handing it to a run:

```bash
forge lessons render tasks/069_rms_norm --out lessons.md
# prints: hand it to a run with: FK_LESSONS_PATH=/abs/path/lessons.md
```

or in code: `rsi_kernel_tune.runcontext.env_for({"lessons": "<file>"})`. `campaign` and `evolve`
call `render` and set that variable themselves; nothing in this task changes their code.

## 5. Lint

`forge lessons lint` checks every record, moves a failing one to `_quarantine/<same file
name>` (plus `<stem>.reason.txt`), and rebuilds `index.json`. Nothing is deleted. The rules:

1. `schema` is the integer this build reads.
2. Required fields (`schema`, `id`, `created`, `family`, `problem`, `reference`, `outcome`,
   `source`) are present.
3. Types are what the schema says (numbers are not bools, ids are strings, `kept` is a list,
   ...). Enums are checked: `reference.kind`, `outcome.metric_kind`, `outcome.window`,
   `failed[].kind`, the `usage` keys.
4. Unknown keys are findings (a typo is not silently ignored), at the top level and inside
   `problem`, `reference`, `outcome`, the row objects, `usage`, `recon` and `source`.
5. `family` is a key `families.toml` declares.
6. Length caps: id 120, change 1000, error 1500, 200 kept/failed/meta rows, 100 KB entries,
   24 verdict lines, 200 candidate rows, 512 KiB per file.
7. Path leakage: absolute machine paths (`/home/...`, `C:\...`, `\\wsl...`),
   run-internal relative paths (`.artifacts/`, `operators/`), `..` traversal, and
   `local.env` / `submit.local.toml`. A sha-shaped token is fine -- `source.best_sha` is one;
   a path into this machine is not.
8. Secret shapes: AWS key ids, private-key blocks, `sk-...` / `gh*_...` / `hf_...` /
   `xox*-...` tokens, board-token assignments, password assignments.
9. Free text carries `verbatim: true` (`kept[].change`, `failed[].change` / `.error`, every
   recon entry). Entries whose text was truncated are still verbatim: the truncation marker
   is the only thing added.
10. `id` is a safe file-name stem and matches the file's name.
11. `created` parses as ISO 8601.
12. `certified_kb` entries are `hw_kb/...`-relative `.md` paths.

`harvest` lints the record it just extracted before filing it; one that fails is written
straight to `_quarantine` (with its reasons) and harvest exits 2. So the lesson base never
contains a record a reader would reject.

A root that does not exist is a finding, not a clean pass (`kb_lint`'s fail-closed rule): a
mispointed `--root` must not certify a tree nobody scanned. `lint` does not create it; the
first `harvest` does.

The path rule is deliberately strict: a real run's compile-error text often carries a
container path (`/tmp/...`, `/root/.triton/...`), and such a record is quarantined rather
than laundered -- the fact it carries is useful, so a human deletes the machine-specific
part and moves the file back. That is the price of never rewriting a verbatim field.

## 6. Deliberately not collected

- **The milestone** (`milestone.md`) and any other agent-written narrative. It is a lesson
  plan somebody wrote; a summary's blind spots become the teacher's blind spots.
- **The recon report's prose**, probe scripts (`notes/`), SASS dumps, job logs. Only the
  fixed-format verdict lines and candidate-table rows come across.
- **Anything not on disk.** A missing artifact leaves its fields out; no value is estimated,
  scaled, or reconstructed from a mean.
- **The usage ledger if its format is not recognized.** The backends task owns
  `agent_usage.jsonl`; this reader accepts the plausible per-turn shapes and ignores what it
  does not understand, so a ledger in another format leaves `usage` out instead of inventing
  numbers.

## 7. Soft dependencies and what this task does not do

- `family.py` / `families.toml` decide the family (a wrong rule is a bug in `families.toml`;
  report it, do not work around it here).
- `campaign.solscore` is the one SOL arithmetic (the metric is not re-derived here).
- `loop_summary.json`, `best_confirmation.json` and the trajectory files are the loop's; a
  run that stopped early simply has fewer fields.
- The usage ledger is the backends task's; a governance ledger (`governor_ledger.jsonl`,
  throughput) is not read yet.
- No agent loop, no GPU job, no board upload: extraction, selection, rendering and linting are
  pure file I/O, and the tests run on CPU against a synthetic run fixture.

## 8. Tests

- `tests/fixtures/opkb_run/` -- a synthesized run with the real layout and field names, plus
  deliberate decoys (prose-only recon text, a milestone, a probe note) that must not reach a
  record.
- `tests/test_opkb_record.py` -- extraction field by field against the fixture's literals,
  omitted fields for absent artifacts, verbatim/truncation, the recon block, refusals, the
  CLI's harvest.
- `tests/test_opkb_query.py` -- target resolution (task directory / bare name), family-only
  selection, self inclusion and `--exclude-self`, the order, `k`, the CLI.
- `tests/test_opkb_render.py` -- every rendered number comes from a record, byte stability,
  the `same problem` marker, the empty-family file, no evaluative sentences, the env line.
- `tests/test_opkb_lint.py` -- one positive and one negative case per rule, quarantine with
  the file name preserved, index rebuild and idempotence, harvest's refusal, `op_kb/` not
  being git-ignored.
