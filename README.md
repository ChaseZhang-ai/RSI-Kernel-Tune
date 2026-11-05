# RSI-Kernel-Tune

**A self-evolving GPU kernel-optimization framework**: give it an operator (a description, a
reference implementation, a correctness tolerance, and inputs) and it iteratively produces
faster kernels — an architecture curriculum distilled from the reference by a teacher agent,
implemented by a student agent under anti-cheat guardrails, judged by honest paired
measurement, and promoted into the next reference so optimization keeps going past the best
known implementation. Findings certified by hardware probes accumulate in a knowledge base
shared across operators.

**English** | License: [Apache-2.0](LICENSE) | Python >= 3.11

**RSI-Kernel-Tune** is a fork of [OpenBMB/ForgeKernel](https://github.com/OpenBMB/ForgeKernel)
(baseline commit `5030503`); the upstream acknowledgment and license obligations are preserved
below and in [NOTICE](NOTICE).

## Highlights

- 🔁 **Self-evolution, one step past every reference.** Every accepted kernel becomes the
  next run's reference, so each generation must beat the one before it; the chain stops only
  when the hardware stops giving (a gain below the measured noise floor).
- 🗺️ **Architecture-first curriculum.** The decisions that are most expensive to change —
  kernel split, tile shapes, warp roles, persistence — are settled first in an M0 milestone
  the teacher distills from black-box behavior, and only frozen after candidate skeletons
  are measured on real hardware.
- 📏 **Honest measurement.** Mirror-ordered paired timing windows, a precision gate that
  compares fp64 error ratios against the reference's own error, fresh-input rechecks, a
  replay probe that catches memoization cheats, and a clock the task declares and the
  harness verifies: machine faults never count as verdicts.
- 🛡️ **Red-team tested anti-cheat.** A cheat-fixture suite (cache replay, seed probing,
  stream injection, timer tampering) runs in CI and must be caught by the defenses.
- 📚 **Knowledge that outlives a run.** Only hardware-probe-certified mechanisms enter the
  per-architecture knowledge base; falsified levers are remembered too, so the same dead
  end is never paid for twice.
- 🔍 **Priced search.** Every lever carries a measured price, sweeps eliminate losers
  between rounds, runner-up architectures wait in a pool for stagnation restarts, and
  sub-noise levers compound in a pool of their own.
- 🏭 **Two contracts, one framework.** `STRICT` (default) measures the kernel's honest
  compute for capability benchmarks; the opt-in `MEMO` lane lets a production workload
  with repeating inputs keep a content-keyed cache and measures serving latency — the
  lanes are never mixed in one comparison.
- 🔌 **Zero network egress on the evaluation path.** Tasks come from you (or are scaffolded
  from the released kernel corpus); nothing is uploaded anywhere.

## Quick start

```bash
rsi-kernel-tune init myop                       # tasks/myop/ scaffold: fill description.md / shapes.json / reference/
rsi-kernel-tune new tasks/myop --dry-run        # plan preview (judge, calibration, loop config)
rsi-kernel-tune new tasks/myop                  # create the run directory
rsi-kernel-tune run .artifacts/myop_<id>        # judge -> card -> calibrate -> distill -> student iterations -> report
rsi-kernel-tune export .artifacts/myop_<id>     # export the confirmed best kernel
```

- Tests run without installing: `python3 -m pytest tests/ -q` (`pyproject.toml` sets
  `pythonpath=["src"]`).
- Scaffold a task from a published kernel's frozen contract (the published kernel is sealed
  as a black-box reference): `rsi-kernel-tune init myop --from-kernels --axes '{"seqs": 64}'`.
- GPU jobs (evaluation, NCU, reference acquisition) submit through the backend configured in
  `submit.toml` (`local.env`); see [docs/](docs/README.md).
- The measurement clock is declarable (`task.toml [protocol]`): a declared lock is verified
  by gpu_entry, and a lock that cannot be taken refuses to time.

## kernels/

`kernels/` releases the published-kernel corpus (26 kernels, upstream artifact release with
honest provenance records). It is prior/teaching material, and via
`rsi-kernel-tune init --from-kernels` it is also the framework's own task corpus.

## Documentation

- [docs/README.md](docs/README.md) — one-line-per-document index
- [CHANGELOG.md](CHANGELOG.md) — release history
- [docs/backlog.md](docs/backlog.md) — frozen/deferred proposals

## Acknowledgment and license

- This repository is a fork of [OpenBMB/ForgeKernel](https://github.com/OpenBMB/ForgeKernel)
  (baseline commit `5030503`). Upstream is licensed under the **Apache License 2.0**; this
  repository follows the same license — [LICENSE](LICENSE) and the derivative-work mapping in
  [NOTICE](NOTICE) are preserved.
- `kernels/` is the upstream artifact release; its provenance records are kept as-is.

## Paper

Paper: TBD. The experiment bundle (result tables, matrices, reproduction notes) lands under
[docs/paper/](docs/paper/).
