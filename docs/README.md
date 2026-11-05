# Documentation index

One line per document. Historical per-package change documents were distilled into
[../CHANGELOG.md](../CHANGELOG.md); their Chinese originals live in git history (pre-WP25).

## Framework documents

| Path | Content |
|---|---|
| [blackbox-mode.md](blackbox-mode.md) | Sealed black-box references: what gets sealed, who can read what, and the known failure modes of the guard |
| [correctness-monitor.md](correctness-monitor.md) | The advisory correctness monitor riding into GPU jobs (`.rkt_monitor/` tree, stressed-input checks, output watch) |
| [energize-protocol.md](energize-protocol.md) | The energize-and-harvest protocol: the acceptance A/B matrix, one-default-at-a-time flipping, JustGoal comparisons |
| [backlog.md](backlog.md) | The single ledger of frozen / deferred / rejected items (do not re-propose) |
| [external-eval.md](external-eval.md) | The external evaluation export lane: suite submission formats and protocols, the packager (`scripts/eval-export/`), the leased-card self-evaluation discipline, and the lane's boundary rules |
| [industrial-lane.md](industrial-lane.md) | The two-lane contract: the STRICT measurement lane (default, memoization forbidden) vs the opt-in MEMO industrial lane (content-addressed caching legal, honest fallback mandatory), its honesty boundaries and the M3 wiring specification |
| [kb-harvest.md](kb-harvest.md) | How certified recipes get into the knowledge base at run end (written for people, not agents) |
| [kb-provenance/](kb-provenance/README.md) | Provenance discipline for hw_kb entries and the mechanical lint |
| [milestones-template.md](milestones-template.md) | The milestones meta-template — how to write a kernel-optimization milestone |
| [prior-supply.md](prior-supply.md) | The prior supply discipline (survey / prior-art / carry channels) |
| [safeguard-refusals.md](safeguard-refusals.md) | What a model-safeguard refusal looks like in the transcript and why it is dangerous |
| [wp20r-known-gaps.md](wp20r-known-gaps.md) | The red-team matrix's open rows: the recorded not-caught cheat forms (G1 closed by WP29's replay probe) and their regression-visible fixtures |

## Design rationale

| Path | Content |
|---|---|
| [design/v1-launch.md](design/v1-launch.md) | Historical design archive (V1 upstream design; the leaderboard sections are obsolete post-WP24 — see the header note) |
| [design/milestone-distillation.md](design/milestone-distillation.md) | The original design rationale for milestone distillation and its invariants |

## Prior art

| Path | Content |
|---|---|
| [prior-art/README.md](prior-art/README.md) | The offline prior-art library: what is here and how it is anchored |
| [prior-art/template-op.md](prior-art/template-op.md) | The per-operator prior-art template |

## Paper (WP26 fills this)

| Path | Content |
|---|---|
| [paper/README.md](paper/README.md) | Placeholder for the paper engineering bundle (tables, experiment matrix, reproduction notes) |

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
