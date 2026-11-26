"""campaign: the orchestration layer over one board problem, one track, the whole board.

`forge campaign` decides what to work on (the board snapshot's per-problem gap to the
best any team has reached), picks a starting point for each problem (an existing kernel,
a torch baseline, or nothing), drives the ordinary loop (`forge init` / `forge new` /
`forge run`, or `forge evolve` when that task has landed), collects every correct
version the run produced, composes them into the **single-file `submission.py`** a
collection submission takes, and packs a track's problems into the upload archive.

It optimizes nothing itself and changes nothing inside the loop. Everything it needs
from another task is a *soft* dependency, resolved through `rsi_kernel_tune.campaign.tools`
(`benchx` to score a candidate, `seedlib` to find / adapt / cover kernels, `shapes` to
compose per-workload winners, `guard` for the licence banner, `evolve` for the
multi-lineage loop, `board` to submit); a missing tool is recorded as "unavailable",
never guessed at.

The score math all of it shares lives in `rsi_kernel_tune.campaign.solscore` (the board's
`S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))`, per workload then averaged), which
`forge evolve` and `forge shapes` also import when they are present.

Layout of a campaign's working files (`--work-dir`, default
``.artifacts/campaign/<name>/``)::

    plan.json            what was selected and how each problem will be run
    ledger.jsonl         append-only: one line per benchmark / board reading / error
    ratios.json          board S / dev S per problem, aggregated by family and by tag
    state/<op>.json      per-problem progress (resumable)
    board_records/       the board's kernels/<id> payloads, cached for offline reuse
    exports/<op>/submission.py   what `campaign export` produced for that problem
    out/<track>-<UTC>.zip + manifest.json   what `campaign pack` produced
"""

from __future__ import annotations

__all__ = ["solscore", "targets", "plan", "runner", "compose", "pack", "sync", "report",
           "tools"]
