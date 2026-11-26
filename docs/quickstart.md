# Quickstart: clone, install, run

This is the shortest path from a fresh clone to a working `forge`. It assumes nothing about
your machine except Python and git; the GPU parts are marked. The full command map (including
the SOL-ExecBench campaign loop) is in the README's "Board commands" section; each command's
own `--help` is the reference.

## 0. What you need

| | |
|---|---|
| Control machine (runs `forge`) | Linux or macOS, Python ≥ 3.11, git |
| GPU machine (same machine, or an ssh host) | an NVIDIA Hopper (`sm90`) or Blackwell (`sm100`/`sm103`) GPU, a CUDA build of PyTorch there; `nvcc` for the target compile checks; Nsight Compute (`ncu`) recommended |
| A model | the **Codex CLI** (logged in, or `OPENAI_API_KEY` / `CODEX_API_KEY` in `local.env`); the default teacher is Codex. Keep Claude for the student unless a task explicitly overrides it. |
| Only for board submissions | `SOL_EXECBENCH_TOKEN` in the environment (never in a file), and the SOL-ExecBench dataset (§4) |

## 1. Clone and install

```bash
git clone https://github.com/ChaseZhang-ai/RSI-Kernel-Tune.git
cd RSI-Kernel-Tune
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # torch and safetensors come with it
pip install -e ".[dev]"          # optional: pytest, to run the test suite
pip install nvidia-cutlass-dsl   # optional: the CuTe DSL API the agent reads
cp local.env.example local.env   # say where the GPU is (local or an ssh host)
forge --help                     # built-ins + the 11 extension commands
```

If the GPU machine needs a specific CUDA wheel, install it first, then `pip install -e .`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

`local.env` keys: `FK_DELIVERY=local|ssh`, `FK_GPU`, `REMOTE_HOST` / `REMOTE_WORKDIR` (ssh),
`FK_JOB_LOCK_CMD` (optional, local), `FK_TARGET_SMS` / `FK_CUDA_ARCH` (optional overrides).
The example file explains each one; credentials go here too, never into the repository.

## 2. Verify without a GPU, a model or a token

Everything in this block is CPU-only and needs no network:

```bash
forge --version
forge target show                  # the graded-hardware profile the agents read
forge lib find 069_rms_norm        # the reference library (134 entries, in-tree)
forge campaign targets             # gap list, from the in-tree board snapshot
forge campaign plan --pilot default --dry-run
forge lessons --help               # and: roofline / shapes / calibrate / bench / board / ...
python -m pytest -q                # the full suite (a few minutes; 1 test skips without _reference/)
```

## 3. Run the loop on a GPU (one task)

```bash
forge target probe                 # measure this machine -> target/dev.json
forge new tasks/rmsnorm --dry-run  # the example task; prints the plan
forge new tasks/rmsnorm            # -> .artifacts/rmsnorm_<id>
tmux new -s forge 'forge run .artifacts/rmsnorm_<id>'   # a run takes hours
forge status .artifacts/rmsnorm_<id>                    # progress; also `forge run ...` to resume
```

The board cards want locked clocks (`SM 1500 MHz / memory 3996 MHz` on a B200); on an unlocked
machine either lock it or expect the reading to carry a `clock_note`. `forge target check`
tells you what differs from the graded machine.

Other vehicles, same card and judge:

```bash
forge bench <task> --candidate kernel.py           # one candidate through the loop's own job
forge evolve tasks/<op> --lineages 3 --generations 2   # multiple lineages of the loop
forge calibrate && forge roofline tasks/<op> --calibration calibration/<gpu>-<date>.json
forge shapes plan tasks/<op> && forge shapes run <plan.json> && forge shapes compose tasks/<op>
```

## 4. The SOL-ExecBench campaign path

```bash
# dataset: the official downloader (needs `pip install datasets huggingface_hub`)
git clone https://github.com/NVIDIA/SOL-ExecBench
python SOL-ExecBench/scripts/download_solexecbench.py   # -> SOL-ExecBench/data/benchmark/<subset>/<problem>

forge campaign plan --pilot default --dataset SOL-ExecBench/data/benchmark
forge campaign run --parallel 2                          # init -> bench -> evolve/shapes -> export
forge campaign export 069_rms_norm
forge target compile-check <the exported submission.py>
forge campaign pack --track l1                           # needs the whole track covered
```

The board snapshot ships in-tree (`reports/board-snapshot-2026-11-24/`), so `targets` and
`plan` work without any extra download. Submission needs a token, only from the environment:

```bash
export SOL_EXECBENCH_TOKEN=...
forge board submit 069_rms_norm <submission.py>          # first five a day are instant
forge board submit-collection l1 dist/l1-<UTC>.zip       # twice per UTC day, ~6 h
forge board import .artifacts/campaign                   # ledger + board/dev ratios
forge campaign report
```

## 5. Where to read next

`README.md` (the command map) and, in `docs/`:

| topic | file |
|---|---|
| the campaign layer | `campaign.md` |
| the board client, rate limits, token | `board.md` |
| evaluating one candidate | `benchx.md` |
| the export gates (static review, licences, multi-seed, cold start) | `integrity.md` |
| target hardware, B200 vs the development GPU | `target-hardware.md` |
| Triton as a language | `triton-mode.md` |
| starting from an existing kernel | `warmstart.md`, `kernel-library.md` |
| per-shape specialisation | `shapes.md` |
| lessons between runs | `opkb.md` |
| roofline / calibration | `roofline.md` |
| the teacher's budget governor | `throughput.md` |
| agent backends and the usage ledger | `agent-backends.md` |

## Status: what is verified, and what is not

- **Verified on a fresh clone, CPU only:** the full test suite (3342 passed, 0 failed, 1
  skipped), every command's `--help`, and the §2 smokes above. No secret, no GPU, no model,
  no token is needed for any of it.
- **Not verified yet:** the framework has not been run end-to-end on a real GPU with a real
  model, and nothing has been submitted to the board. Expect the first real run to surface
  small things (that is the honest state of the project, not a hidden condition).
- Two things to know before trusting a number: the pinned evaluator replaces
  `torch.utils.cpp_extension.load/load_inline` for Python submissions, so the inline-CUDA
  route must be confirmed on the live v1.1 stack before betting on it (`docs/campaign.md`
  writes the risk into the manifest); and the Triton knowledge-base entries are
  `[PENDING-CERT]` — the mechanisms are trustworthy, the numbers are not.
