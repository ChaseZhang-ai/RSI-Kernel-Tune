# B300 runbook

Everything below assumes the reproduction is bootstrapped (`scripts/bootstrap.sh`) and the local
checks pass (`scripts/local_validate.sh`). Read the two warnings first — they change the plan.

---

## Two facts before you spend hours

1. **The leaderboard is measured on B200 at a locked clock.** SOL-ExecBench's FlashInfer-Bench
   collection scores on B200 (eval stack v1.1) and locks SM 1500 / memory 3996 MHz. **B300 is for
   generation and iteration** (faster, more memory, Blackwell-family); it does **not** produce
   numbers comparable to the board's B200 `T_b`/`T_SOL`. Plan: **develop and iterate on B300,
   then re-measure the finalists on a B200 at the board clock** and submit those. Do not score B300
   timings against B200 targets — the score would be meaningless.
2. **`T_b`/`T_SOL` are per-machine.** The board measured them on its B200. The residual between
   machines stays in the score. Keep the clock lock on and use the card's own SOL-score path.

If you have a *B300* leaderboard instead, swap the board snapshot
(`scripts/fetch_board.py --out data/board_b300.json`) — everything else in VULCAN is board-agnostic.

---

## Step 0 — environment (GPU box)

```bash
# upstream needs Python >= 3.11; CUDA 13.x for Blackwell
vulcan/scripts/bootstrap.sh                      # clones pin b2c8ae8, pip install -e
pip install cupti-python                         # the board's stopwatch (or let gpu_entry install it)

# datasets
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
# SOL-ExecBench data: run scripts/download_solexecbench.py in the SOL-ExecBench repo
```

Check the arch string upstream expects for your GPU (`task.toml` `[task] arch`): the shipped
configs use `sm90` (Hopper) and `sm100` (Blackwell). Confirm the B300/Blackwell-Ultra value
against `vendor/ForgeKernel/src/forgekernel/loop/lang_mode.py` and the card template before the
first run; a wrong arch silently changes what compiles.

## Step 1 — refresh the board and produce the plan

```bash
cd vulcan && export PYTHONPATH=src
python3 scripts/fetch_board.py                   # refresh data/ from the live API
python3 -m vulcan stats                          # confirm FK / GB / best-known
python3 -m vulcan plan --hours 300 --target best --json-out plan.json
```

`plan.json` gives, per kernel: allocated hours, the current score, the **target score** (best-known
by default), and per-workload **`target_ms`**. Matching every kernel's best-known score yields
collection **0.819011 > 0.815321** (rank 1).

Validate B300 ≠ board: on B300, compare a kernel's B300 latency to its B200 `T_b` only as a rough
gate, never as the score.

## Step 2 — smoke-test the pipeline on one kernel (canary)

Pick one easy, high-`S` kernel first (e.g. `025_rmsnorm_h4096`) to de-risk the whole path — clock
lock, tolerance, anti-cheat, export.

```bash
cd vulcan/vendor/ForgeKernel
python -m forgekernel init rmsnorm --leaderboard sol-execbench \
    --dataset ~/SOL-ExecBench/data/benchmark \
    --definition <definition_id> --list-workloads          # find the id + workload index
python -m forgekernel init rmsnorm --leaderboard sol-execbench \
    --dataset ~/SOL-ExecBench/data/benchmark \
    --definition <definition_id> --workload <index>
# edit tasks/rmsnorm/task.toml: set [target] ms from plan.json, arch, agent model/effort, budget
python -m forgekernel new  tasks/rmsnorm
python -m forgekernel run  .artifacts/rmsnorm_<id>
python -m forgekernel export .artifacts/rmsnorm_<id> --out solution.json --name vulcan --author zixinzenith
```

Confirm the eval JSON has `sol_board.sol_score` and `device_median_ms`, and that `gpu_entry.sh`
reports the clock lock held. If the clock lock cannot be held, upstream refuses the task (exit 3);
fix permissions rather than bypassing (`LB_REQUIRE_CLOCK_LOCK=0` makes readings non-comparable).

## Step 3 — run the portfolio on the priority kernels

Front-load the low-`S` / Fast₁-straggler kernels in this order (from `LEADERBOARD_ANALYSIS.md`):

```
009_gemm_n5120_k2048      <-- flagship: largest score headroom AND 8 Fast1 misses
007_gemm_n4096_k4096      (3 Fast1 misses)
010_gemm_n6144_k4096      (1)
011_gemm_n28672_k4096     (1)
006_gemm_n2048_k4096      (largest pure score gap after 009)
008 004 005 001 003 022 021 024   <-- remaining mid kernels
```

For each kernel, spend the budget as a portfolio rather than one chain:

| round | hours/arm | arms (architecturally distinct M0s) |
|---|---|---|
| 1 | 1 h | cta-per-tile, persistent-cta, warp-specialized, split-k, register-resident, fused-epilogue |
| 2 | 3 h | keep the better half |
| 3 | 7 h | keep the better quarter |
| final | remainder | commit everything to the survivor |

Gate every acceptance with the VULCAN criteria: correctness **and** the precision contract
(`precision_guard`) **and** a paired-ratio CI above the noise floor (`stats.accept_gain`). Reject
"faster but lower internal precision" immediately — it fails the board and is not a real speedup.

Fast₁ repair for `009`: aim first to cross `T_b` on all 8 straggler workloads (small-shape GEMMs —
launch overhead and tail effects usually dominate), then chase score.

## Step 4 — re-measure on the board's device and submit

Re-run the finalists on a **B200 at the board clock**, export, and submit per kernel via the card's
`forge export`. A task's verdict then equals the board's verdict (same judge, inputs, stopwatch).

Targets to clear:

| metric | rank 1 now | our target |
|---|---|---|
| collection SOL Score | 0.815321 | **> 0.816 (min), 0.82+ (safe), 0.84 (stretch)** |
| Fast₁ | 654/660 | **≥ 654, aim 660/660 (only 4 kernels needed)** |
| solved | 26/26 | 26/26 (mandatory) |

## Step 5 — budget shape (suggested)

| phase | GPU-hours (suggested) | purpose |
|---|---|---|
| canary (1 kernel, B200) | 2–4 | de-risk the submission path |
| portfolio rounds 1–2, 4 priority GEMMs | 30–50 | pick architectures cheaply |
| final rounds, 8 mid kernels | 120–200 | the score actually comes from here |
| Fast₁ repair (`009`, `007`, `010`, `011`) | 30–60 | close 13 workloads |
| B200 confirmation + submission | 10–20 | locked-clock re-measurement |
| **total** | **~200–330** | |

`vulcan plan --hours N` re-computes the per-kernel split for any `N`.

---

## Checklist before each submission

- [ ] Correctness passes the card's own judge (all input groups).
- [ ] Precision contract satisfied (`precision_guard.evaluate_gate` = PASS); no accumulation
      downgrade hidden behind a passing elementwise check.
- [ ] Adversarial probes pass (correlated keys, long memory, near-cancellation, wide range,
      zero-output, −inf parity).
- [ ] Clock lock held; `device_window_basis` is `cupti_kernel_span`.
- [ ] Paired reading is stable (two mirrored readings within ~1.2×; watch `unstable_sets`).
- [ ] No reward-hack flags from `reward_hack_scan`.
- [ ] Score improved with the paired-ratio CI clearing the noise floor (else revert and record a
      negative entry).

## What to report back

Per kernel: chosen architecture, hours spent, final `device_median_ms`, `sol_board.sol_score`,
Fast₁ (fast/total), and the per-generation gain of the self-evolution chain. Paste those into the
issue/PR so the projection model in `allocator.py` can be re-fit (improvement item 12) and the
whole plan re-run with measured rates.
