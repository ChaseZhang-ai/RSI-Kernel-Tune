<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CHANGES-WP25 — Rename & repositioning (Peregrine-Kernel)

Branch `feat/wp25-rename`, cut from origin/main @ `4a4c51a` (the WP23 three-generation
integration tip; independent final review PASS-WITH-NOTES, 2586 passed). Nature: pure
mechanical rename + documentation restructuring + full-repo Englishization. **Zero behavior
change** — the only deliverable is the name. Full-suite regression green before and after
with the same count (2586; +8 new tests in `tests/test_branding.py` -> 2594).

## 0. Readiness gate (recorded)

- Workspace reuse per `handoff/106`, section 3: `git -C wk4-wp25 fetch origin && git checkout -B
  feat/wp25-rename origin/main` — HEAD = `4a4c51a`, worktree clean.
- Baseline: `~/venvs/forgekernel/bin/python -m pytest tests/ -q` -> **2586 passed in
  325.01s** (matches the 136 review's 317.70s run bit-for-bit in count). The venv directory
  name stays `forgekernel` (local environment, not a repo asset).
- GitHub rename `kernel-ZXtune` -> `peregrine-kernel`: already executed by the user; the
  origin remote of `zxtune-integration` already points at the new name. Out of scope here;
  `push origin main` belongs to the release manager.

## 1. Task 1 (P0): namespace migration

### Mapping table and replacement counts (per category)

| Object | Old | New | Count |
|---|---|---|---|
| Python package directory | `src/forgekernel/` | `src/peregrine_kernel/` | `git mv` (1 dir; incl. `card_template`) |
| Monitor runtime hook file | `monitor/runtime/hook/fk_monitor_hook.py` | `pk_monitor_hook.py` | `git mv` (1 file) |
| Package/module references (imports, dotted paths, docstrings) | `forgekernel` | `peregrine_kernel` | ~1455 (mechanical pass 1442 across 212 files + 13 hw_kb path pointers) |
| Distribution name (pyproject `name`) | `forgekernel` | `peregrine-kernel` | 1 |
| Console script | `forge` | `peregrine` | 1 (pyproject) |
| Env-var prefix | `FK_*` | `PK_*` | 486 |
| Full-name env vars | `FORGEKERNEL_{OPERATORS,RUNS,LOCK}_DIR` | `PEREGRINE_*` | 22 |
| Uppercase prompt labels | `FORGEKERNEL TURN` / `FORGEKERNEL REFERENCE KERNEL` | `PEREGRINE ...` | ~10 (scripts/claude-goal.py, loop/goal_ref.py, tests/fixtures/stub-claude.py, tests/test_claude_goal.py — both sides of every pin replaced) |
| Monitor log markers | `=== FK MONITOR JSON/PROFILE ===` | `=== PK MONITOR ... ===` | 8 (src definition + docs) |
| Internal identifiers | `fk_*` (incl. `_fk_`-prefixed and `"fk_"` string keys) | `pk_*` | 117 |
| Hyphenated runtime tokens | `.fk-jobs`, `.fk-turn`, `.fk-sessions`, `.fk-goalref-tree`, `fk-inflight.json`, `fk-overlay-index-`, `fk-probe-batch`, `<!-- fk-ref-census`, `[fk-monitor]`, `fk-goal-turn-`, `fk-goal-run-`, `fk-resume-verify-` | `pk-*` forms | 42 |
| CLI command word | `` `forge ``, `forge init/new/run/status/harvest/export`, `"forge"`, `forge <op>`, `[forge]` log prefix, `./forge` | `peregrine ...` / `[peregrine]` | ~225 (98 backticked + 41 bare + 19 quoted + 3 exact + 16 `[forge]` + 4 f-string + docs/skills/hw_kb/meta command examples) |
| Agent git identities | `forge-judge` / `forge-loop` / `forge-monitor` / `forge-goalref` (+`@local`/`@localhost` emails), `.forge-loop/` crash dir, `forge-student-` branch prefix | `peregrine-*` forms | ~15 |
| CamelCase self-brand | `ForgeKernel` | `Peregrine-Kernel` | 16 (package/module docstrings, docs self-references, hw_kb mode docs, agent-policy hook messages, test docstrings) |
| Protected (NOT replaced) | `~/venvs/forgekernel` local venv paths | kept | 3 mentions (decree: the venv is not a repo asset) |

### Engineering interpretations (recorded, per the "no riding along" fence)

1. **NOTICE stays byte-identical** (red line). Its derivative-work mapping names pre-WP25
   paths (`src/forgekernel/...`); after the rename those strings are historical records of
   which files carry adapted code. `NOTICE` is whitelisted in `test_branding.py`. The
   file identity behind each mapping is preserved by the rename history (git mv).
2. **hw_kb**: knowledge content untouched (fence respected). Only 13 framework module-path
   pointers (`src/forgekernel/loop/kb_lint.py` etc. in README/TEMPLATE docs — references to
   our own files that would otherwise become stale) and 3 command-name mentions
   (`` `forge loop --kb-arch` ``) were updated; every KB entry byte-identical.
3. **`fk-cuda-cflags` (hyphenated) is NOT renamed**: it is a runtime source-text contract —
   `cardkit.py`/`benchkit.py` parse it from kernel source by regex, and hw_kb/TEMPLATE.md
   (frozen) instructs agents to write it. The task mapping table covers `FK_*`/`fk_*`
   underscore forms only. Renaming would require breaking the frozen hw_kb content or the
   kernels/ corpus. Recorded as the one deliberate `fk-` residual; `test_branding.py`
   documents why its pattern (underscore) does not match it.
4. **Discipline text byte-frozen**: `loop/driver.py` line "ForgeKernel's evaluation
   numbers." (the human-guided-experiment discipline sentence written into
   infeasible_stop.md) and the `milestone.py` comments that mirror frozen
   `meta_milestone.md` terminology ("the ForgeKernel projection", "ForgeKernel measures
   whether the decision graph carries the student") keep the CamelCase brand — red line:
   the English anti-fraud discipline original stays byte-identical. `meta/` prompts and
   `meta_milestone.md` untouched except the two command-name examples (`forge <op>
   baseline` -> `peregrine <op> baseline`), which the package brief explicitly allows.
5. **`FORGEKERNEL TURN` / `FORGEKERNEL REFERENCE KERNEL` prompt labels renamed to
   PEREGRINE**: verified against the goal_ref design that "reference kernel" here is this
   framework's self-generated reference (goal-only mode), i.e. a self-brand, not upstream
   attribution; every test pin replaced on both sides (`test_claude_goal.py`,
   `tests/fixtures/stub-claude.py`).
6. **`git_name`/`git_email` defaults** for baseline/loop/monitor/goalref archive commits:
   `forge`/`forge@local` etc. -> `peregrine`/`peregrine@local` (no test pins them).
7. **Stale `ls .pk-jobs` in scripts/smoke_wp19.sh**: the script's log discovery predates
   WP24's `.jobs/` dirname and already relied on its `find` fallback; the rename keeps it
   byte-equivalent (renamed to `.pk-jobs`, still caught by the same fallback). Real-GPU
   smoke, not run in this package.

### Smoke (recorded)

- Editable reinstall in the shared venv: `pip uninstall forgekernel` ->
  `pip install -e .` -> dist `peregrine-kernel 0.1.0` at this worktree.
- `python -c "import peregrine_kernel; from peregrine_kernel.loop import driver"` -> OK.
- `peregrine --version` -> `peregrine_kernel 0.1.0`; `peregrine --help` shows the
  `peregrine init/new/run/status/harvest/export` surface (task 4 has the full record).

## 2. Task 2 (P0): legal exemption surface + `tests/test_branding.py`

New `tests/test_branding.py` (8 tests, user copyright header, Apache-2.0) pins the
zero-residual gate permanently. Patterns are built into the test; a hit outside the
whitelist = red.

| Pattern | Whitelist (files; line-anchored where noted) | Rationale |
|---|---|---|
| CJK `[\u4e00-\u9fff]` | **none — absolute zero** | single-language repo (2026-11-04 paper-internationalization adjudication) |
| `\bforgekernel\b` / `src/forgekernel` | NOTICE (frozen legal file); `docs/design/v1-launch.md` (historical archive); `scripts/smoke_wp19.sh`, `tasks/wp21mini/run_smoke.sh` — only on `venvs/forgekernel` lines | NOTICE red line; archive naming strings are records; local venv name not a repo asset |
| `ForgeKernel` (CamelCase) | LICENSE; NOTICE; README fork declaration (`OpenBMB/ForgeKernel` lines); CHANGELOG (`OpenBMB/ForgeKernel` / `upstream ForgeKernel`); `docs/paper/README.md` (upstream arm of the experiment matrix); `kernels/README.md` (upstream release); `meta_milestone.md`, `meta/recon_prompt.md`, `meta/cuda/recon_prompt.md` (frozen prompts); `priors/attention/attn_fwd_gqa16_h100_gen1.md` (gen-0 provenance); `scripts/floor_backtest.json` (`ForgeKernel gen` comparison attribution); v1-launch; `loop/driver.py` — only the "ForgeKernel's evaluation numbers" discipline line; `milestone.py` — only the meta_milestone-terminology lines; `run_smoke.sh` — only the upstream-checkout path line | Apache-2.0 attribution + byte-frozen discipline text |
| `\bFK_[A-Z]` / `FK MONITOR` / `FORGEKERNEL` | `docs/design/v1-launch.md` only | archive |
| `\bfk_[a-z_]` | none | `fk-cuda-cflags` is hyphenated and intentionally not matched (see §1.3) |
| `\bZXtune\b` / `zxtune` (any case) | none | old repo name fully retired |
| command forms: backticked `forge`, `[forge]`, `` `forge` ``, `forge init/new/run/status/harvest/export/<op>/--version` | v1-launch only | "forge" is a normal English word (`forged`); only command shapes are matched |
| structural | `pyproject.toml` name = `peregrine-kernel`; `peregrine = "peregrine_kernel.__main__:main"`; `src/peregrine_kernel/__init__.py` exists; `src/forgekernel` does not exist | — |

The test excludes its own source (it must name the forbidden patterns to define them) and
gitignored dot-directories (`.git`, `.pytest_cache`, `.artifacts`, `.venv`, egg-info).

Copyright headers: new files carry the user copyright under Apache-2.0 ("The
Peregrine-Kernel Authors" SPDX headers; NOTICE/LICENSE/upstream headers untouched).

## 3. Task 3 (P0): README rewrite + docs restructuring

- `README.md` fully rewritten in English: brand `# Peregrine-Kernel` (subtitle: a pure AI
  kernel-optimization framework); the five-layer mechanism model; task supply and
  `--from-kernels` closure; zero-benchmark-association statement; fork declaration
  ("Peregrine-Kernel is a fork of OpenBMB/ForgeKernel (@5030503)") + upstream acknowledgment
  and Apache-2.0 compliance section; the three-generation improvement table (one sentence
  per package: gen1 WP1-6 / gen2 WP8-12 / gen3 WP14-21+24); Quick Start on the `peregrine`
  CLI; paper placeholder (paper: TBD -> `docs/paper/`); `kernels/` asset note preserved;
  pointer to `docs/README.md`.
- `docs/` restructured (moves/creates only, no content deleted):
  - new `docs/README.md` — one-line-per-document index (framework docs, design rationale,
    prior art, paper);
  - `docs/design/` keeps the existing archive (`v1-launch.md` with its translated archive
    note; `milestone-distillation.md`);
  - `docs/prior-art/` unchanged;
  - new `docs/paper/` + placeholder README (WP26 fills; paper-integrity red lines
    pre-stated);
  - new `docs/external-eval.md` placeholder (WP27 fills; boundary red lines pre-stated);
  - `docs/wp-changes/` removed (distilled into CHANGELOG, see task 3b).
- new `CHANGELOG.md`: English per-package digests organized by generation (gen 1 WP1-6,
  gen 2 WP8-12, gen 3 WP14-21+WP24), integration (WP23), and this package; key numbers,
  mechanism names and code anchors preserved. It is the sole in-tree history carrier —
  the Chinese originals were removed from the tree and remain recoverable from git
  history (noted in the CHANGELOG header).
- new `docs/backlog.md`: the single frozen/deferred/rejected ledger (the landing point of
  the 2026-11-04 freeze order): 17 frozen assessment directions (A1-A17), 12 deferred P2
  items (B1-B12), 4 closed/rejected items recorded against re-proposals (C1-C4, incl. the
  WP7 precision-lane principle rejection), plus the SemVer version policy (D: after this
  package merges, the release manager tags `v1.0.0`). Every entry carries source +
  re-estimation trigger (= the first real-run A/B numbers landing).
- optional CI (GitHub Actions): NOT enabled — P2 pending user approval, recorded in
  `docs/backlog.md` B11.

## 4. Task 3b (P0): full-repo Englishization

Baseline at package start (this branch's own grep, superseding the main @19e885b count of
39 files): **75 files** contained CJK. Disposition:

| Class | Files | Disposition |
|---|---|---|
| CHANGES history | 20 root `CHANGES-WP*.md` + 6 `docs/wp-changes/*` | distilled into English `CHANGELOG.md` (per-package digests keeping key numbers, mechanism names, code anchors); **removed from the tree** (`git rm`); originals permanently in git history; `CHANGES-WP25.md` (this file) written in English |
| Chinese READMEs | `kernels/README_zh.md` | deleted — verified NOT an Apache-2.0 obligation file (NOTICE's mapping does not list it); the upstream `kernels/README.md` language-switcher line reduced to `**English**` (navigation line only; its body stays byte-identical upstream text — documented deviation serving the zero-CJK acceptance, which has no whitelist) |
| kernels/ provenance Chinese | 10 `kernel.py` provenance-comment segments (~180 lines) | translated to English, provenance honesty preserved sentence-for-sentence (quotes of KB/milestone text rendered as faithful translations — the recurring-term concordance is in the Translation concordance section below; see rows T5/T10/T11); kernel code tokens untouched |
| src comment/string mixing | 9 files (driver, trajectory, parse, timing, cardkit, milestone, direct, analytic_floor + the two injected prompt strings) | translated sentence-for-sentence; the two served strings (driver LANDABLE hint, restart-target discipline line) and the milestone coverage-failure error message translated with their test pin (`test_wp4_search.py` "lever candidates: ring") synced |
| tests Chinese | 13 files incl. `test_dataflow_audit.py` (83 lines: docstring + assert messages) and the two audit JSON registries (values only; keys untouched) | translated; audit mechanics unchanged (306-test subset green before the full run) |
| docs Chinese | `energize-protocol.md` (full rewrite, semantics preserved), `prior-supply.md` heading, `v1-launch.md` archive note, `dry-run-wp16.md` | translated |
| scripts/skills/tasks Chinese | `energize_ab.py` (docstring + matrix strings), `self_evolve.py`, `backtest_floors.py`, `build_priors.py`, `sweep_run.py`, `tasks/wp21mini/*` | translated |
| misc root artifacts | `dry_run_wp16.py` | translated (WP16 acceptance artifact, kept) |

### Translation concordance (semantic red lines — original -> English)

Applied sentence-for-sentence. This file is part of the delivered tree, and the tree is
100% English with an absolute zero-CJK gate (no whitelist) — so the concordance below
records each original phrase by an anchor into git history instead of by quoting it: the
Chinese original of every row is recoverable bit-exactly via
`git show 930da47:<path>` (the last commit before the Englishization commit `0b5c61d`;
same commit for every path below). The side-by-side zh->en table was delivered to the
release manager in the handoff layer alongside this file.

| # | Original anchor (`git show 930da47:<path>`) | Translation (en) | Where |
|---|---|---|---|
| T1 | `src/peregrine_kernel/loop/driver.py` — the revival-discipline term and the restart-target injected line | revival always re-verifies; the line reads "- Revival always re-verifies (red line (3)-2): the revived node re-passes the full suite with its next gated delivery — watermark / clock-drift / fresh-recheck apply as to any delivery — before any of its readings re-enter best, plateau or pricing." (English tail byte-unchanged) | driver.py comment/docstring; dry-run-wp16.md artifact |
| T2 | `src/peregrine_kernel/loop/driver.py` — the reading-reuse rule (comment + the ref-price/carry log line); `scripts/self_evolve.py` comment | reading reuse is acceleration, not measurement-skipping; "red-line addendum (2)" | driver.py; self_evolve.py |
| T3 | `src/peregrine_kernel/inner/trajectory.py`, `loop/driver.py`, `tests/test_wp11_throughput.py` — the methodology term in "merged-job / eval_job_s / previous / fallback" compounds | basis (merged-job basis / eval_job_s basis / previous basis / fallback basis) | trajectory.py; driver.py; test_wp11_throughput.py |
| T4 | `src/peregrine_kernel/milestone.py` — the regime-coverage validation error | coverage failure = freeze invalid | milestone.py error message |
| T5 | kernels provenance comments (gdn_prefill_bwd, recurrent_kda_prefill, attn_bwd_mha, attn_fwd_gqa16, fp8_gemm, fp4_grouped_gemm, conv2d_dgrad, attn_bwd_gqa8, dsa_indexer_bwd, gdn_prefill_fwd/varlen) — the recurring KB/milestone terms | yield ladder; same-side discipline; section B / section ledger / section C / section style / covered section; pit N; station A; mechanism N / "mechanism B missing"; "sub-term" | kernels provenance |
| T6 | the same kernels files — the assessment-lineage tags on src/tests/scripts comments | line A / line B / line C (tags `A-G3`, `B-B9`, `C-T5` — from the three independent assessment reports 96/97/99-100) | src/tests/scripts lineage tags |
| T7 | `src/peregrine_kernel/loop/driver.py` — the REMOVE_CANDIDATES tail + `tests/test_wp4_search.py:314` | lever candidates: (the test pin updated on both sides) | driver.py + test pin |
| T8 | `src/peregrine_kernel/loop/driver.py` — the ARCH_PROBE freeze comment | the candidate architecture recon settled on | driver.py |
| T9 | `src/peregrine_kernel/milestone.py` — the machine-checked-vocabulary comment | WP14 vocabulary section | milestone.py |
| T10 | kernels provenance — the D0.12 section B packing/conversion quote | packing/precision conversion must terminate the upstream f32 accumulator's live range — after the conversion no f32 copy may survive | kernels provenance |
| T11 | kernels provenance (recurrent_kda_prefill) — the staging quote, the topology-freeze clause, the section 6 INFEASIBLE clause, the D2.5(a) alignment clause, the D0.10 loader row, the hw_kb escape-hatch and gotcha quotes | the stage count must drop to 4; go through D2.3's joint staging decision; if some ceiling can only be broken by changing the topology -> M0 was written wrong, the meta repairs it and the student outputs EXHAUSTED; metrics the reference measurably achieves under the same resource envelope do not constitute grounds for INFEASIBLE; 16 B alignment at the run origin; the loader is not the reader; the PTX escape hatch for vector reads must carry side-effect markers; choosing the wrong member is silent | kernels provenance |
| T12 | `docs/energize-protocol.md` (full document) + `scripts/energize_ab.py` — the protocol rules and red-line labels | one default at a time; evidence before the flip; every default must have a rollback switch; the local 3060 verifies the chain only and produces no performance conclusions (red line 11); SPRT only stops early, it never manufactures a KEEP | energize protocol; energize_ab.py |
| T13 | `src/peregrine_kernel/loop/driver.py` — the LANDABLE hint injected line (test unpinned; dry-run artifact updated) | the teacher can land these levers as one merged decision line with a single milestone patch; the student delivers once per rung (ordinary bundle delivery, judged by the standard eval gate) | driver.py LANDABLE line; dry-run-wp16.md |

Interpretation recorded: the package brief asked for the concordance to be appended to
CHANGES-WP25 with the original sentences; the same brief's red line ("the delivered tree
is 100% English") and its acceptance grep (zero CJK hits, no whitelist, permanent
regression) take precedence. The zh column therefore lives in the handoff layer; the
git-history anchors above make every original bit-exactly recoverable from the tree.

The kernels/ translations preserve every provenance fact: decision-graph references
(D0.x/D1.x/D2.x), KB entry names, measured numbers, and honesty qualifiers ("uncertified",
"forbidden", negative knowledge) are all carried into the English text.

### Acceptance (verified below, §6)

`grep -rlP '[\x{4e00}-\x{9fff}]' . --exclude-dir=.git` -> **zero hits**; the same check is
`tests/test_branding.py::test_tree_is_100_percent_english` (permanent regression, no
whitelist).

## 5. Task 4 (P0): regression & acceptance evidence

- Baseline (before any change, @4a4c51a): **2586 passed in 325.01s**.
- After task 1 (namespace migration, commit 930da47): **2586 passed in 518.32s** — count
  identical, zero behavior change.
- Final branch tip `9b86156` (tasks 2/3/3b included, +8 `test_branding.py` tests):
  **2594 passed in 316.15s** — all green.
- CLI smoke: `peregrine --help` / `--version` (§1); `peregrine init/new/run` fixture path
  (§6).
- Ordinary-task dry-run (fixtures, no GPU): §6.

## 6. Evidence pastes

Recorded below (grep outputs, pytest counts, CLI smoke, dry-run).

### Zero-residual greps

```
$ git grep -in 'forgekernel' -- . ':(exclude)NOTICE' ':(exclude)docs/design/v1-launch.md' ':(exclude)scripts/smoke_wp19.sh' ':(exclude)tasks/wp21mini/run_smoke.sh'
(exit 1 — no output)
$ git grep -i 'zxtune'
(exit 1 — no output)
$ git grep -nE '\bFK_[A-Z]|FORGEKERNEL' -- . ':(exclude)docs/design/v1-launch.md'
(exit 1 — no output)
$ git grep -nE '\bfk_[a-z_]'
(exit 1 — no output)
$ grep -rlP '[\x{4e00}-\x{9fff}]' . --exclude-dir=.git
(exit 1 — no output)
$ ~/venvs/forgekernel/bin/python -m pytest tests/test_branding.py -q
8 passed in 4.19s
$ ~/venvs/forgekernel/bin/python -m pytest tests/ -q            # final branch tip
2594 passed in 316.15s
```

### CLI smoke & ordinary-task dry-run (fixtures, no GPU)

```
$ peregrine --version
peregrine_kernel 0.1.0
$ peregrine help internal | head -3
Lower-level commands. They act on the operator trees a run generates under
.artifacts/<run>/operators/ -- point PEREGRINE_OPERATORS_DIR at that directory first.
`peregrine run` drives all of them; ...
$ peregrine init brandsmoke
[peregrine] wrote .../tasks/brandsmoke/: fill in description.md and shapes.json, adjust
task.toml, add reference/ if you have one, then `peregrine new tasks/brandsmoke`
$ peregrine new tasks/brandsmoke --dry-run | tail -6
  agents: teacher claude-opus-5-5 (high), student claude-opus-5-5
  budget: 4 h, 3 teacher round(s), 8 h to write the reference
  GPU: this machine (GPU 0)
  note: no [reference]: a /goal session writes one first (budget 8 h)
  note: shape small: 0 input group(s) given, the teacher adds 3
  note: shape large: 0 input group(s) given, the teacher adds 3
$ peregrine init gdn_prefill_bwd --from-kernels --arch h100 --axes '{"num": 64}'
[peregrine] wrote .../tasks/gdn_prefill_bwd/ from
kernels/linear_attention/gdn_prefill_bwd/h100/kernel.py: fill shapes.json's axes and
description.md's signature from the frozen contract, then `peregrine new tasks/gdn_prefill_bwd`
(scaffolds cleaned after the smoke; `peregrine run`/`peregrine export` fixture paths are
pinned by tests/test_deboard_regression.py 5-6, green in the full suite)
```

## 7. Scope-fence compliance & known residuals (record)

- Untouched: `measure_paired` and the window protocol, precision formulas/constants,
  driver loop semantics, gate defaults, `meta/` discipline wording (command names only),
  kb_lint logic, hw_kb content, kernels/ kernel code, trajectory schema, C1 contract
  signatures. `timing.py measure_paired` byte-identity is pinned by the existing test
  suite (WP19 golden + 136-review anchors), all green.
- Known residuals (all whitelisted in test_branding.py): NOTICE's historical paths;
  v1-launch archive naming strings; two venv-path mentions in smoke scripts; CamelCase
  attribution/provenance/frozen-discipline spots (§2 table); `fk-cuda-cflags` contract.
- The venv directory `~/venvs/forgekernel` is unchanged (not a repo asset); the shared
  venv now has `peregrine-kernel 0.1.0` installed editable from this worktree.
