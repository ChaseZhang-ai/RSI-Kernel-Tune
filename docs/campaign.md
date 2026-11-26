# campaign: from the board's gap list to a track archive

`forge campaign` is the orchestration layer (task 10). It decides what to work on from
the saved board snapshot, picks a starting point for each problem, drives the ordinary
loop, collects every correct version a run produced, composes them into the **single-file
`submission.py`** a collection submission takes, and packs a track's problems into the
upload archive. It optimizes nothing itself and changes nothing inside the loop: every
piece of machinery it uses is either the base repository's own command
(`forge init` / `new` / `run`) or another task's deliverable, reached through one
replaceable registry (`rsi_kernel_tune/campaign/tools.py`).

```
forge campaign targets   # the gap list: leader / best known / gap per problem
forge campaign plan      # choose the pilot, write plan.json
forge campaign run       # execute the plan (resumable, --parallel problems)
forge campaign status    # one line per planned problem
forge campaign report    # report.md / report.csv / report.json
forge campaign export    # show / copy one problem's submission.py
forge campaign pack      # one track -> <track>-<UTC>.zip + manifest.json
forge campaign sync      # saved board pages -> ledger.jsonl + ratios.json
forge campaign tools     # what is installed, and what each slot resolved to
```

The command lives in `src/rsi_kernel_tune/ext/campaign.py`. Until the step-0 `ext/` discovery
hook is in the checkout it runs as `python -m rsi_kernel_tune.ext.campaign …`; once the hook
lands, `forge campaign …` is the same code.

## Targets and priority

The board publishes a per-kernel leaderboard (one SOL score per team per problem) and a
collection leaderboard (one score per team per track). `targets` reduces both to three
numbers per problem:

| number | definition |
|---|---|
| `leader` | the score of **the incumbent**: the team with the highest *weighted overall* score (sum of its per-problem scores / 235, a problem it never submitted counting 0). `--incumbent NAME` overrides the pick. On the 2026-11-24 snapshot this is `Amir M. Mir \| SF Tensor`, 0.830080 overall. |
| `best_known` | the highest score any non-reference entry holds on that problem, and who holds it |
| `gap` | `best_known − (ours if we have one, else leader)` |

Rows sort by gap descending; a tie goes to a problem with an *exact* starting point over
one with only a family neighbour. The per-track instrument panel reports, per track, our
mean / the best-known mean / the incumbent's mean (missing problems count 0) and the same
three weighted into one overall number — that is "how far from first".

**Why the incumbent is a single team and not "rank 1 in each collection":** the overall
standings weight the four tracks by problem count (94 / 82 / 33 / 26), so the team to beat
is the one who wins *that* sum; its per-track means are then the per-track bar. The
numbers this produces are asserted against the board's own printed scores in
`tests/test_campaign_targets.py` (0.818852 / 0.820129 / 0.905855 / 0.805878 for the
incumbent, 0.841240 / 0.857025 / 0.936968 / 0.819011 for the best known).

**Feasibility prior.** `cursor_problem_level_metrics.csv` (the public Cursor-235
solution's per-problem score under the v1.0 evaluator) is a prior only: `--by priority`
multiplies the gap by 1.2 when the public solution reached ≥ 0.7, 0.7 when it stayed
below 0.4, and 1.0 otherwise (unknown included), printing the factor's reason next to the
pick. It never changes what the gap *is*, only the order of `--pilot auto`.

## How the pilot is chosen

* `--pilot default` (what `plan` does with no selector): the documented three —
  `069_rms_norm` (L1), `001_fp8_attention_output_projection` (Quant) and
  `030_attention_output_projection_with_residual` (L1). They are gap-list leaders, simple
  shapes, and they cross two tracks, so one pilot exercises two dataset subsets.
* `--pilot auto`: the top `--count` (default 3) of the gap × feasibility order.
* `--pilot N`: the top N by raw gap.
* `--problems a,b` (or `--pilot a,b`): explicit; `track:name` when a name is ambiguous.

The selection, its reason and the incumbent are written into `plan.json`'s `warnings`, so
a plan always says why it contains what it contains.

## One problem's pipeline (`run`)

Each problem is a fixed stage sequence; every stage's completion is recorded in
`state/<op>.json`, so a second invocation resumes instead of repeating.

1. **board record** — the board's `kernels/<id>` payload (T_b / T_SOL per workload) from
   `board_records/` if cached, else fetched once from the board's public API and cached.
   `forge init --sol-board <file>` freezes it into the card, which is what makes the card's
   eval JSON carry a SOL score and lets every later run work offline. A record that does
   not line up (wrong workload count, missing `sol_ms`) is *not* used: the card is built
   with `--no-sol-board` and the readings then carry no S (never a fake one).
2. **task** — `forge init <op> --leaderboard sol-execbench --dataset … --definition
   <subset>/<problem> --workload 0-(n-1)` (all workloads: the board scores the mean over
   them) plus `--init-kernel` when a start is already known. The task is then validated
   with `rsi_kernel_tune.task.load_task`.
3. **starts** — see below. Each candidate is scored with `benchx` (`--repeat 2`) on this
   card and sorted by mean SOL score (geomean latency when the card has no board pair).
4. **target** — `[target] ms` = the best scored start's geomean reading / `--step`
   (default 1.10), written into `task.toml`; the best start is also seeded as the card's
   kernel file (through `rsi_kernel_tune.warmstart.validate` when it exists).
5. **optimize** — `forge evolve <task> --lineages 3 --generations 1 --parallel P` when the
   evolve task is installed (one lineage per available language route), else `forge new` +
   `forge run`. The target picture rides the child's environment whenever
   `target/B200.json` exists (`FK_TARGET_PROFILE_PATH`, via `rsi_kernel_tune.runcontext` when
   that module is present); roofline / lessons files join when the plan or the environment
   names them.
6. **versions** — every KEEP iteration of every run that passed correctness, plus the
   starts. The source comes from the run workspace's git (`git show <sha>:<kernel>`), the
   readings from `student/status.jsonl` (per-workload `device_median_ms`, scored against
   the card's own T_b / T_SOL), and `best_confirmation.json` adds the run's final
   confirmation when it names a sha the trajectory only measured through a `reading_of`
   row. A CUDA-mode run's versions carry `cuda-cpp`.
7. **export** — `compose` (below) writes `exports/<op>/submission.py`.
8. **submit** — off by default; `--submit` hands the file to the board task.

A run that stops short is not a failure here: stages 6–7 use whatever versions exist.

## Where starting points come from

Every source is optional; whatever is installed contributes:

| source | how | origin tag |
|---|---|---|
| exact reference-library hits | `forge lib find <problem>` entries whose `problem_keys` name this problem | `starter` |
| same-family neighbours | the top 1–2 family entries through `forge lib adapt <problem> --seed <id>` (supplement 1) | `adapt` |
| torch baselines | `forge lib baselines <problem> --dataset …` (reference wrapped as `run`, `torch.compile` variants, TF32 variants) | `baseline` |
| the definition's own reference | always: the eager reference wrapped as a DPS `run` | `starter` |

All of them are benched and sorted together; the reference stays as the always-correct
floor, so a problem always has something exportable even when nothing else is installed.

## Composition and the fallback

`exports/<op>/submission.py` is one file with a top-level `run(*args)`:

* the chosen version is pasted in with its top-level `run` renamed to `_fk_candidate_run`
  (an AST edit — decorators keep seeing their source, which Triton and CuTe DSL require);
* the definition's own reference is pasted after it with **every top-level binding
  prefixed** `_fk_ref_`, and its entry renamed `_fk_reference_run` — the prefix is what
  makes the two sides unable to shadow each other's helpers;
* `run` calls the candidate, and any exception falls back to the reference, writing the
  pre-allocated outputs — so an edge shape a specialised kernel mishandles costs that
  workload's score instead of the whole problem's.

Candidate choice: `forge shapes compose` first (per-workload winners, when the shapes task
is installed); otherwise the correct version with the best mean SOL score. Then, in score
order, each candidate is tried until one exports: a `.cu` candidate without seedlib's
`cuda_to_inline` is recorded as **`not exported: needs klib`** and the next candidate is
used — never silently dropped, never exported as something a collection submission cannot
take. A candidate with a `--allow-slow-cold-start`-gated cold start above 100 s is skipped
with its measured number in the rejection list; an unmeasured cold start is exported with
`cold_start: "unmeasured"`.

Licences: a candidate's `meta` (`license` / `source`) goes to `guard`'s
`license_check(dict) -> {"ok", "problems"}` before anything is written; a reported problem
blocks the export. `guard`'s `notices.banner([meta…])` becomes the file's header when it
exists; with guard absent the header carries the same provenance lines and says
`license check: unavailable`.

**A known risk, recorded in the manifest.** The pinned official evaluator
(`SOL-ExecBench/src/sol_execbench/driver/templates/eval_driver.py`) replaces
`torch.utils.cpp_extension.load` / `load_inline` with a raising stub for Python-language
solutions *before* importing the user module. An inline-CUDA `submission.py` (and hence
the `cuda_to_inline` route) therefore needs verification against the live v1.1 stack
before it is trusted; every such export carries
`risk: ["inline-cuda: …"]`, and the `.cu` itself remains usable for **single-kernel**
submissions (which accept `cuda_cpp`).

## Packing a track

`campaign pack --track <track> --dataset <tree>` walks the track's problems by kernel id
and gives every one a directory `<kernel_name>/submission.py`:

1. this campaign's own export, else
2. a `forge lib cover` tree (`--cover`), else
3. the dataset definition's reference — **only with `--allow-reference-fallback`**, else
4. the pack **fails and names the problems** (a missing kernel scores 0, so a track is
   never uploaded with a hole).

Then it self-checks (the board task's `validate_collection` too, when installed):
directory names equal the board's kernel names, exactly one `submission.py` per
directory, a top-level `run` in each, ≤ 10 MB per file, ≤ 50 MB archive, ≤ 100 MB
uncompressed, no symlinks, no path traversal — on the tree *and* on the finished zip. The
output is `<track>-<UTC stamp>.zip` plus `manifest-<track>-<stamp>.json` (per problem: the
source, the local score, fallback or not, cold-start state, licence, risk flags), and the
summary prints the upload rules: **at most 2 collection submissions per track per UTC day,
about 6 h to a result, suggested slots 00:30 and 12:30 UTC**. Nothing is uploaded here.

## The ledger and the ratio

`ledger.jsonl` is append-only: one line per benchmark (`event: "bench"` / `"starter"`), per
board reading (`"board"`), per export (`"compose"`), per archive (`"pack"`), per refusal
(`"error"`). `forge board import <campaign>` writes the same lines through
`sync.record_results(rows, ledger_path)` (rows: `problem`/`kernel_name`, `mean_s` or
`sol_score`, optional `track`, `sha`, `note`), and `campaign sync --from-saved <path>`
reads JSON saved from the board's pages instead (a per-kernel page — every team's row, or
`--username`'s — or a submission result). No network, no token.

`ratios.json` derives, for every problem with both readings, `ratio = board S / dev S`,
plus means by operator family and by reading tag (`dev-only`, `fp4-bound`, …) and an
overall mean. `sync.correction(family, doc)` returns the family's mean or the overall one,
and `sync.corrected_estimate(dev_s, family, doc)` is `dev_s × correction`; the report shows
that next to every local score with its basis (`board reading`, `dev x family ratio`, or
`uncorrected`). The ratio exists because the development machine is a B300 and the board
times a B200: FP4-dense kernels read faster here than they score there, SFU-heavy ones much
faster, and an SM-count-sensitive kernel can go either way. Until a problem has a pair, the
family mean is the best available correction — and it is always shown *as* a correction,
never replacing the measurement.

## Soft dependencies

Everything below is optional; `campaign tools` prints what resolved and what did not, and
each missing tool narrows what is exported rather than what is claimed.

| tool | module probed | used for | if absent |
|---|---|---|---|
| benchx | `rsi_kernel_tune.benchx.api` (`CardBench`), else `rsi_kernel_tune.ext.bench --json` | scoring every candidate | starts are unscored (the task keeps its default target); readings carry no S |
| evolve | `rsi_kernel_tune.ext.evolve` / `rsi_kernel_tune.evolve` | multi-lineage optimization | `forge new` + `forge run` |
| seedlib | `rsi_kernel_tune.ext.lib` | `find` / `baselines` / `adapt` / `cover` | only the reference + baselines are absent → the reference is the start; pack needs `--allow-reference-fallback` |
| seedlib | `rsi_kernel_tune.klib.normalize.cuda_to_inline` | a `.cu` version → a single-file submission | `not exported: needs klib` |
| shapes | `rsi_kernel_tune.ext.shapes` | per-workload winners | the best mean-S version |
| guard | `rsi_kernel_tune.integrity.license_check`, `.notices.banner`, `.export_check` | licence gate + header + export check | header says unavailable |
| board | `rsi_kernel_tune.ext.board` | `--submit`, `sync --board`, `collection_validate` | no submission, `--from-saved` import, campaign's own self-check |
| targethw | `rsi_kernel_tune/runcontext.py`, `target/B200.json` | context files into every run | the same four `FK_*` names are built here |
| opkb/family | `rsi_kernel_tune.family.family_key`, `kernel_library/index.json` | the start column and the family grouping | the name without its `NNN_` prefix |

`rsi_kernel_tune.campaign.solscore.score_kernel(reading, board, mean=True)` is the shared entry
the evolve and shapes task files name; it uses `rsi_kernel_tune.leaderboard.sol_score`, so no
task re-derives the board's formula.

## Working files

```
.artifacts/campaign/            (or --work-dir)
  plan.json                     what was selected and how each problem runs
  ledger.jsonl   ratios.json    the append-only record and its derived view
  state/<op>.json               per-problem progress (resumable)
  board_records/<id>.json       the board's kernels/<id> payloads, cached
  starts/<op>/                  the starts' sources as they were benched
  exports/<op>/submission.py    the composed file per problem
  out/<track>-<UTC>.zip         the archive, plus manifest-<track>-<UTC>.json
  run.md, report.md/.csv/.json  the campaign's own paperwork
  tools.log                     the child processes' stdout
```

## Tests

`tests/test_campaign_targets.py` (the real snapshot's numbers and the gap order),
`test_campaign_plan.py` (selection, argv, language routes), `test_campaign_run.py` (the
whole pipeline with every tool faked: starts, adapt, shapes, target, versions, the cold
start gate, the CUDA/klib route, licences, resume, `--parallel`, export/fallback by
importing the produced file), `test_campaign_pack.py` (the archive contract and each
self-check rule) and `test_campaign_sync.py` (the ledger, saved pages, the ratio
arithmetic). Shared fixtures live in `tests/campaign_helpers.py`: the real snapshot pages
under `tests/fixtures/campaign/snapshot/` and a two-problem miniature dataset next to them.

## Not covered

* No real GPU job runs in the tests (the pilot on the B300 is a report item), so the
  numbers the fake bench returns are hand-made; the *orchestration* around them is what is
  tested.
* `--submit` and `sync --board` need the board task and a token; neither was exercised
  against the live service.
* One route per task: the language mix (supplement 2) becomes parallel lineages only when
  `forge evolve` is installed; without it the plan lists the routes and one task is built.
* `shapes compose`'s own output layout is probed (its `--out`, then
  `tasks/<op>/shapes/submission.py`); a shapes task that uses a different path needs one
  line in `tools._resolve_shapes`.
