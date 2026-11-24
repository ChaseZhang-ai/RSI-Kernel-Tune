# RSI-Kernel-Tune

A research repository on **recursive self-improving GPU kernel generation**: whether an agent
that writes kernels can be made to improve its own process, and what has to be true — about
verification, about search, about the feedback the agent reads — for that improvement to be
real rather than apparent.

Two studies live here. They ask different questions, share the metric and the harness
conventions, and are meant to be read together.

| folder | study |
|---|---|
| [`vulcan/`](vulcan/README.md) | Reproduces the **ForgeKernel** architecture and studies score-aware orchestration: how a fixed budget should be divided across operators and architecture hypotheses, and when a gain is real. |
| [`argus/`](argus/README.md) | Reproduces the **CudaForge** agent loop and studies the machinery around it: what a generator must verify, what its search must respect, and what its judge should be told. |

The two reproduce different base systems on purpose. ForgeKernel separates architectural
planning from coding and promotes accepted kernels into the next reference; CudaForge is a
compact Coder/Judge loop that reads profiler counters. Studying both lets a claim about
"agentic kernel optimization" be checked against two different loops instead of one, and the
two axes they investigate — budget allocation and evaluation integrity — are complementary
rather than competing.

## Shared conventions

- **One metric, stated once.** Both studies report against the same scoring function
  (`S = 1/(1 + (T − T_SOL)/(T_b − T_SOL))`, geometric mean per operator, arithmetic mean per
  collection). `vulcan/docs/LEADERBOARD_ANALYSIS.md` recovers it and verifies it against the
  published numbers; `argus/docs/metric.md` states it and documents where the anchors come from.
- **CPU-first validation.** Every claim about the *reasoning* stack has a GPU-free test. Claims
  about hardware are marked as pending and carry the protocol that would measure them.
- **Pinned upstreams, vendored and gitignored.** Each study records the upstream commit it
  builds against (`vulcan/vendor/UPSTREAM.lock`, `argus/scripts/bootstrap.sh`), so a result can
  be reproduced against the same code.
- **No unmeasured numbers.** A gain is reported only from a paired measurement; a projected or
  synthetic number is labelled as such wherever it appears.

## Where to start

- Read a study's README for its questions, findings and limitations:
  [`vulcan/README.md`](vulcan/README.md) · [`argus/README.md`](argus/README.md).
- Run either study's local validation without a GPU:
  `bash vulcan/scripts/local_validate.sh` · `bash argus/scripts/local_validate.sh`.
- The planned GPU protocols are written down before any hardware is requested:
  [`vulcan/docs/B300_RUNBOOK.md`](vulcan/docs/B300_RUNBOOK.md) ·
  [`argus/docs/experiments.md`](argus/docs/experiments.md).

## License

MIT; see [`LICENSE`](LICENSE). Vendored upstreams keep their own licenses and are not
redistributed here.
