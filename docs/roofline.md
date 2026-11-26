# Roofline: measured peaks, and how far a kernel is from them

RSI-Kernel-Tune's loop only knows two targets — a ratio against the reference, or an absolute
millisecond count (`task.py`). Neither says whether the kernel is *near what the hardware
allows*, so a teacher cannot tell "this structure has reached its limit" from "this is
still far off", and an orchestrator cannot rank workloads by how much room they have. The
board's `T_SOL` is one estimate of that limit, but it is an analytic one (SOLAR) computed
without knowing our code, and on some workloads it sits below what the hardware can
physically do.

This module adds the missing reading in three parts:

| part | command | what it produces |
|---|---|---|
| measure | `forge calibrate` | one JSON document of the machine's **measured** peaks |
| analyse | `forge roofline tasks/<op> --calibration … --cost-model …` | the Markdown context file `{ROOFLINE_PATH}`, per workload: the floor, the bottleneck class, the distance |
| advise | `skills/roofline/` | `classify` (a profile → class), `suggest` (signal → cause → action), `distance` (the same arithmetic, standalone) |

Two rules shape all of it. **A peak that was not measured is labelled as such** (the
target machine's come from `target/B200.json` and `peaks.toml`, with a source per value),
and **nothing here changes the loop's gate**: `EvalResult.device_median_ms` is still the
only number a KEEP/REVERT rests on.

---

## 1. `forge calibrate` — measure a machine

```bash
forge calibrate                                   # local.env decides where (ssh when FK_DELIVERY=ssh)
forge calibrate --remote --seconds 10             # force the ssh host
forge calibrate --local --only hbm,launch --out /tmp/p.json
```

Six probes, each a standalone script under `src/rsi_kernel_tune/roofline/probes/`:

| probe | measures | units |
|---|---|---|
| `hbm` | a large device-to-device copy (read + write) and a read-only reduction | GB/s |
| `l2` | the same copy with a working set that fits in L2 | GB/s |
| `tensor` | a large square product in bf16, fp16, fp8 (`torch._scaled_mm`; skipped where torch has no fp8 path) | TFLOPS |
| `elementwise` | a bandwidth-limited ternary operation (torch) and an arithmetic-limited multiply-add chain (Triton) | GB/s, Gop/s |
| `sfu` | chains of `exp`, `rsqrt`, `sigmoid` in Triton | Gop/s |
| `launch` | a minimal kernel's round-trip, its device-side duration, and back-to-back launches | µs |

Every probe **checks its arithmetic before it times anything** (a small copy must
reproduce its source, a small product must match a float32 computation, a short chain
must match the same recurrence on the host), and every entry reports the median, the
number of repeats and the coefficient of variation. A probe that cannot run (no Triton,
no fp8 path, an OOM) is an entry with `skipped: <reason>` — never a zero.

**How it reaches a GPU.** Through this project's existing delivery, not a second one. The
calibration builds a tiny job tree (the probe files, `run_all.py`, a `gpu_entry.sh` whose
`calibrate` task runs them), and `submit.direct` ships it: the per-GPU lock (including
`FK_JOB_LOCK_CMD` on a shared workstation), the job timeout, the log, and — for the
development machine — the ssh delivery `local.env` names. `forge target probe` reaches its
machine the same way, so a failed delivery is reported the way every run reports one.
`run_all.py` runs each probe as its own subprocess: a probe that dies costs that probe and
not the calibration.

The job's entry runs `python3` on the machine that receives it (what the card's own entry
scripts assume). When that is not the interpreter holding torch — a workstation run from a
venv, a host with several Pythons — set `FK_CALIBRATE_PYTHON` to the one to use: `--local`
sets it to the harness's own interpreter automatically, and for an ssh host it can live in
`local.env` (it rides the job's environment).

**The document** (`calibration/<gpu>-<date>.json`, not committed):

```json
{"schema": "rsi_kernel_tune.calibration/1", "gpu": "NVIDIA B300", "arch": "sm_103",
 "sm_count": 160, "driver": "...", "cuda": "13.1", "torch": "2.9.0",
 "clocks": {"sm_mhz": 1500, "mem_mhz": 3996, "app_sm_mhz": 1500, "max_sm_mhz": 1965,
            "locked": true},
 "ts": "2026-11-24T10:00:00+00:00", "delivery": "ssh dev", "seconds": 10.0,
 "probes": {"hbm": {"entries": {"copy": {"value": 6300.0, "unit": "GB/s",
                                        "repeats": 7, "cv": 0.008, "method": "..."}}},
            "...": {}},
 "errors": {}}
```

`clocks.locked` is read from `nvidia-smi`'s application/current/maximum columns, never
assumed. An unpinned clock does not invalidate the reading, but it makes every peak a
reading at "whatever the machine ran at", and both the document and the summary say so.
The board locks SM 1500 / memory 3996 MHz; a calibration taken elsewhere is comparable
only within that caveat.

## 2. `forge roofline` — the distance from the floor

```bash
forge roofline tasks/<op> --calibration calibration/nvidia-b300-2026-11-24.json \
    --cost-model costs.json --run .artifacts/<op>_<id> --out roofline.md
forge roofline tasks/<op> --estimate --calibration cal.json          # one teacher turn
forge roofline tasks/<op> --cost-model costs.json --json
```

Inputs: the card's `problem.json` (every workload's axes and the board's T_b / T_SOL), the
**target** peaks (official: `target/B200.json` plus `peaks.toml`), optionally the
**development** peaks (a calibration), and a cost model. It writes English Markdown:

1. a header saying this is a first-order model, which calibration and which official
   figures produced it, the efficiency factors, the launch floor, and the score goal;
2. a peaks table (target / development / ratio) with the source of every target value;
3. one row per workload: axes, T_SOL, floor, class, T_b, T_SOL/floor, T_b/floor, the
   current reading, current/floor, the headroom grade, and the confidence flags;
4. a summary: class counts, the T_SOL/floor distribution, **the workloads whose T_SOL is
   below the floor** (named explicitly — the board thinks a time is possible that this
   model says is not), the time that scores `--sol-goal` and whether the floor allows it,
   and where the headroom sits;
5. caveats (an unmeasured launch floor, unmeasured target peaks, a cost model with extra
   entries, …).

Without `--out` the Markdown goes to stdout and the one-line hint ("hand it to a run …
`runcontext.env_for({"roofline": …})` or `--context roofline=…`") goes to stderr, so a
redirect keeps the file clean. The command sets no environment variable and changes no
task.

### The computation (pure, hand-checkable)

```
bytes_total = bytes_read + bytes_written
t_mem       = bytes_total / (bandwidth  x eff_hbm)         [ms]
t_tensor    = flops       / (peak(peak_name) x eff_tensor) [ms]
t_floor     = max(t_mem, t_tensor, t_launch)
```

- `peak_name` is the precision the cost model names (`bf16`, `fp16`, `fp8`, `fp4`, `fp32`,
  `sfu`, or `none` when the call does no arithmetic). A peak the peak set lacks is an
  **error naming it** — never a default.
- **Class**: `latency` when the launch floor is strictly the largest term; otherwise
  `balanced` when the two throughput terms are within 10% of the larger; otherwise
  `memory` or `compute` by which is larger.
- **Headroom**: `near-floor` (a reading within 1.10x of the floor), `mid` (up to 2x),
  `far` (over 2x).
- **Board comparison**: `T_SOL/floor` below 1 means the board's floor is unreachable under
  this model; the file says so. `T_B/floor` says how much the board's baseline could gain
  if it reached the floor.
- **Goal**: `time_for_score(goal, T_b, T_SOL)` (the repository's own inversion of the
  board's score, not a second implementation) is compared with the floor.

The arithmetic lives in `skills/roofline/roofline_tools.py` and `forge roofline` imports
it, so the skill's `distance` and the command's table cannot disagree (a test compares
them value by value).

### Two peak sets, and the confidence flags

`target/B200.json` gives the graded GPU's bandwidth, FP8/FP4 and SFU figures with its own
sources; `peaks.toml` adds the precisions it does not carry (BF16/FP16, derived from the
FP8 figure and labelled `derived:`), H100's datasheet numbers, and nothing where no
public figure exists (L2 bandwidth, launch latency). Each workload is flagged when a
*development-machine* reading of it can mislead:

| flag | when | why |
|---|---|---|
| `fp4-bound` | the floor is set by dense FP4 arithmetic | B300 runs dense NVFP4 1.5x B200 |
| `sfu-bound` | the floor is set by special functions | B300's SFU rate is ~2x B200's |
| `sm-count-sensitive` | a compute- or launch-bound floor, and the two SM counts differ | B300 has more SMs; bandwidth is 8 TB/s on both, so a bandwidth-bound floor does not move |

These labels are derived from the cost model and the peak sets, never guessed from code.

### Cost models

`--cost-model FILE` — JSON, validated hard (a missing field, a negative or non-finite
number, a name the card does not have: all refusals):

```json
{"schema": "rsi_kernel_tune.cost-model/1",
 "workloads": [{"name": "wl0", "flops": 2100000000, "bytes_read": 84000000,
                "bytes_written": 11000000, "peak": "bf16",
                "basis": {"flops": "…", "bytes_read": "…", "bytes_written": "…"}}]}
```

`--estimate` — one teacher turn with `meta/cost_model_prompt.md`. The prompt is
operator-agnostic (invariant L1): it defines what to count and which pipe the arithmetic
runs on; the definition and the workload axes enter it as data. The answer must be one
JSON object; a validation failure is fed back and a second turn asked; a second failure is
an error — a guessed number is never used.

### Efficiency factors

`--eff-hbm` (default 0.85) and `--eff-tensor` (default 0.80) are the share of a vendor
peak a well-written kernel reaches. 0.85 is the middle of the 80–95% band AutoKernel's
`calibrate.py` reports for real kernels at a manual peak; 0.80 for tensor work is lower on
purpose (a vendor peak assumes a perfect tile with no operand traffic). Both are CLI
flags because they are assumptions, and the file prints them so a reader can judge the
result's sensitivity. Their limits: they are not measured, they do not model occupancy,
tile shapes or the L2, and they are the same for every workload — a kernel at 0.6 of the
peak is not "unreachable", it is far from the floor, which is exactly what the distance
column says.

## 3. `skills/roofline/` — what the agents use

| subcommand | input | output |
|---|---|---|
| `classify` | `--metrics "name=value,…"`, `--csv <ncu --import … --csv>`, or `--json` | `CLASS memory compute_sol=41.2 memory_sol=93.4 …` + `EVIDENCE …` |
| `suggest` | a signal id, or nothing | `SUGGEST signal=… cause=… action=…` rows, hardware-level only |
| `distance` | a calibration file + FLOPs/bytes | the floor, class and distance of one workload |

`classify` follows KernelAgent's thresholds (both SOL percentages under 60 → neither
resource saturated) and adds the ncu-report-skill's latency distinction (DRAM under 10%
with long-scoreboard stalls dominating → latency-bound, not bandwidth-bound). It reports
which metric name supplied each number, so an old spelling or the DRAM-only memory metric
is visible in the verdict.

`suggest` is the playbook distilled from ncu-report-skill's diagnosis manual: sixteen
rows of *signal → likely cause → candidate action*, all about resources, scheduling and
instruction patterns, none about what the kernel computes. It is **first-order advice**:
it never measures, and it never replaces a profile or a calibration.

## Why `hw_kb/` does not take any of this

Entries in `hw_kb/` are certified by a probe on the architecture they describe, with a
recipe that can be re-run (`docs/kb-harvest.md`). A roofline is a *model*: counted work
divided by vendor or measured peaks, corrected by two global efficiency factors, with
thresholds for a class name. Putting that into the knowledge base would make a heuristic
look like a certified measurement — and the one thing `hw_kb/` must never contain is an
entry that reads as measured but is not. The roofline lives in run context files, which
are data for a run; `hw_kb/` stays what a probe proved.

## Not covered / known limits

- The **target machine's launch latency is unknown**: B200 is the board's, and this
  campaign does not calibrate it. The target floor therefore has `t_launch = 0` and the
  file says a launch-bound workload's floor is a lower bound only.
- `l2` and `elementwise` are measured and archived but not yet used in the floor: no cost
  model says "this working set fits in L2", and a Torch-level elementwise reading is a
  memory reading, not a peak. They are there for comparison and for a future model.
- The `alu` and `sfu` probes need Triton; without it those entries are `skipped`, and a
  cost model that names `sfu` then fails by name (which is the intended behaviour).
- Whether a teacher actually uses `{ROOFLINE_PATH}` is not verified here — that belongs to
  the pilot (see the task report's unverified items).
