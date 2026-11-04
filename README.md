# Peregrine-Kernel

**A pure AI kernel-optimization framework**: given an operator (a description, a reference
implementation, a correctness tolerance, and inputs), the system iteratively produces faster
kernels through five interlocking mechanism layers —

1. **Teacher-student decision-graph distillation** — the teacher distills an architecture
   decision graph (milestone) from the reference's black-box behavior (running, timing,
   profiling); the student implements it line by line;
2. **Honest measurement** — paired window timing (`measure_paired` mirror order), a
   precision gate (STRICT everywhere by default), fresh-input rechecks (fresh_recheck), a
   declarable and verified measurement clock (`task.toml [protocol]`), and clock-lock refusal
   safety (a lock that cannot be taken refuses to time; machine faults never count as
   verdicts);
3. **Certified knowledge compounding** — a cross-run knowledge base (KB) with supply and
   lifecycle management, a falsified-lever ledger, and deterministic KB-MENU injection;
4. **Priced search governance** — lever pricing, budget and stopping governance, and the
   structural-ceiling channel (analytic floors, structural escape);
5. **Structural ceiling channel** — NCU-evidence-driven bottleneck location aligned with the
   hardware limit.

Task supply is user-side (`tasks/<op>/` ordinary tasks: teacher judging + user-supplied
inputs); `peregrine init --from-kernels <op>` scaffolds a task from a published kernel's
frozen contract in `kernels/` and seals the published kernel as a black-box reference via
the existing sealref mechanism — the framework brings its own task corpus, so
self-evolution always has something to beat, and beating a published kernel is a real win.

**Zero association with any external leaderboard or benchmark organization**: the evaluation
path has zero network egress (the agent CLI's LLM backend and the teacher's survey channel
remain; neither is benchmark-related). The upstream-inherited leaderboard machinery (the
dual-suite supply line, the board API fetch, the board evaluation template, leaderboard-pinned
clocks, leaderboard docs) was removed in one cut by WP24 — see `CHANGELOG.md`.

Peregrine-Kernel is a fork of OpenBMB/ForgeKernel (@5030503).

## Upstream acknowledgment and compliance

- This repository is a fork of [OpenBMB/ForgeKernel](https://github.com/OpenBMB/ForgeKernel),
  baseline commit `5030503`.
- Upstream code is licensed under the **Apache License 2.0**; this repository follows the
  same license — [LICENSE](LICENSE) is retained, and [NOTICE](NOTICE) is the WP24-reduced
  derivative-work mapping (per-file review recorded in the WP24 history, git-only).

## What this fork adds (gen 1 WP1–WP6 / gen 2 WP8–WP12 / gen 3 WP14–WP21 + WP24)

| Generation | Package | One-line summary |
|---|---|---|
| 1 | WP1 measurement statistics | Judging resolution becomes the run's own measured quantity (`resolvable_diff`, C1); raw readings + spread retained; confirmation band; clock-drift guard; gpu_state telemetry |
| 1 | WP2 telemetry expansion | NCU metrics 18 -> 25 (SOL/L2/waves/coalescing); arch-adaptive filter; build-resource `resusage`; roofline header in the digest |
| 1 | WP3 knowledge compounding | Cross-run falsified-lever ledger; probe/recipe archive; KB certification queue (`hw_kb/CANDIDATES.md`) |
| 1 | WP4 search mechanisms | LANDABLE combo landing; SWEEP_REQUEST teacher sweeps; slow-decision ledger + REMOVE_CANDIDATES |
| 1 | WP5 budget & stopping | Format-retry budget accounting; meta budget x remaining wall clock; headroom stopping + extension; reopen uncap; `ncu_on_revert` |
| 1 | WP6 M0 + structural | ARCH_PROBE pre-freeze topology measurement; R2-structural skeletons must carry measured prices |
| 2 | WP8 prior supply | Teacher-side prior channels (survey / prior_art / carry), default-off; `priors/` library (25 operator graphs); `skills/sota-survey/` |
| 2 | WP9 measurement anchoring | NCU regime-aligned flags everywhere; analytic floor (`analytic_floor.py`); fresh_recheck (non-reproducing best hard-REVERTs as suspicious); reanchor memo |
| 2 | WP10 structural search | STAGNATION escape trigger (`stagnation_k`/`stagnation_override`); MC-DELIVERY single-job fan-out prescreen; compile-flags decision axis |
| 2 | WP11 process speedup | eval+NCU merged single job (zero-behavior golden pin); content-addressed build cache (default off); wall-clock composition telemetry; SPRT early stop (default off) |
| 2 | WP12 knowledge injection | Deterministic KB-MENU synthesizer (`kbmenu.py`); KB entry mortality + dead-end propagation; probe-recipe library |
| 3 | WP14 shape regimes | Per-shape ledger (C8); `per_shape:`/`regime:` decision vocabulary (pure predicates, full-coverage families); per-regime NCU evidence; MC-DISPATCH |
| 3 | WP15 resolution & confirmation | resolvable_diff v2; routine confirmations budgeted in the trajectory; SPRT on by backtest; Stouffer advisory; log-domain best confidence |
| 3 | WP16 search & population | Shrink-grid pattern search; noise-gated SWEEP halving; MC near-tie defense; sub-noise compounding pool; runner-up pool + restart targets + graveyard; floor_progress |
| 3 | WP17 screening & M0 | Single screening primitive (`loop/screening.py`, C9); ARCH_PROBE fidelity tiers; (family, arch) ledger; in-process variant switching |
| 3 | WP18 knowledge closure | `kb_used:` per-entry ledger with KEEP/REVERT feedback; batch certification + regime fields; kernel-artifact prior collector; PRICED_NULL stamping |
| 3 | WP19 process speedup 3 | Build-cache key narrowing (C11); probe batch manifest; idle-GPU background lane (C15); context bounding (C13); single-re-prompt bounce (C12); session resume |
| 3 | WP20 red team & defenses | `tests/redteam/` cheat fixtures; stream/timer/process sentinels (accounting only); judge-edge recheck; gate-degraded / mirror-divergence flags |
| 3 | WP21 energize & audit | Dataflow audit CI (every field has a reader); mini smoke task; the energize protocol (one default at a time, evidence first) |
| 3 | WP24 deboarding | Leaderboard machinery removed in one cut (template -> `benchkit.py`, `eval/problem/`, NOTICE reduced); `--from-kernels` supply closure with sealed black-box references |

Per-package detail: [CHANGELOG.md](CHANGELOG.md) (English digests; the original Chinese
documents live in git history).

## Quick start

- **Python >= 3.11**;
- tests run without installing: `python3 -m pytest tests/ -q` (`pyproject.toml` sets
  `pythonpath=["src"]`);
- a minimal loop for one task:

```
peregrine init myop                     # tasks/myop/ scaffold: fill description.md / shapes.json / reference/
peregrine new tasks/myop --dry-run      # plan preview (judge, calibration, loop config)
peregrine new tasks/myop                # create the run directory
peregrine run .artifacts/myop_<id>      # judge -> card -> calibrate -> distill -> student iterations -> report
peregrine export .artifacts/myop_<id>   # export the confirmed best kernel
```

- `peregrine init myop --from-kernels --axes '{"seqs": 64}'`: scaffold a task from a
  published kernel's frozen contract in `kernels/` (the published kernel is sealed as a
  black-box reference via sealref);
- GPU jobs (evaluation, NCU, reference acquisition) submit through the backend configured in
  `submit.toml` (`local.env`), see `docs/design/`;
- the measurement clock is declarable (`task.toml [protocol]`): a declared lock is verified
  by gpu_entry, and a lock that cannot be taken refuses to time.

## Paper

Paper: TBD. The paper engineering bundle (result tables, experiment matrix, reproduction
notes) lands under `docs/paper/` (WP26).

## kernels/

`kernels/` releases the published-kernel corpus (26 kernels; upstream artifact release;
provenance records kept honest). It is prior/teaching material: the provenance records state
where each kernel came from, and honest provenance is not leaderboard association. The
directory's own README (inherited from the upstream release, attribution preserved) describes
the layout.

## Documentation index

See [docs/README.md](docs/README.md) for the one-line-per-document index.
