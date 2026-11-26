"""shapes: one kernel per shape, and one entry point that dispatches between them.

A board problem carries a *set* of workloads (``forge init --workload 0,3,5-9`` freezes a
group card), and one kernel serving every size in that set is how a run ends up at 1.037x
on the large shape and 0.289x on the small one -- the same problem, two different optima
(issue #8). This package turns the set into a *cluster* per size class, runs one ordinary
task per cluster, cross-evaluates every cluster's best kernel (plus the run's other
correct versions, ``--also`` solutions and the definition's own eager reference) on the
problem's **own card**, and composes the per-workload winners into **one**
``submission.py`` whose ``run(*args)`` routes on the input shapes.

Why per-workload dispatch is the optimal decomposition, not a trick: a workload's score
depends only on its own latency (the board's ``S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))``
is per workload, and a problem's score is their arithmetic mean), so every workload
running its own fastest correct implementation maximises the score. The dispatch cost is
not on the board's stopwatch: the card times a CUPTI *device* window around the kernel
launches, and the Python routing happens before the first launch.

Modules (each one file, no GPU in the pure ones)::

    cluster.py   the card's workloads and the deterministic clustering (pure)
    plan.py      plan.json: clusters, targets, the `forge init` argv of each
    run.py       plan -> `forge init`/`new`/`run`/`export` per cluster (resumable)
    evalx.py     candidates and the benchx adapter: one candidate x every workload
    matrix.py    the candidate x workload readings, and the routing table (pure)
    compose.py   the dispatcher `submission.py`, its validator, and the report

Products land in ``tasks/<op>/shapes/`` (a working directory, kept out of git through
``.git/info/exclude`` -- never ``.gitignore``)::

    plan.json            the clusters and how each one is run (see docs/shapes.md)
    runs.json            per-cluster progress: planned -> running -> solved | failed
    matrix.json          every reading of every candidate on every workload
    shapes_compose.json  the routing table: who won each workload, its S, mean S
    submission.py        the deliverable: one file, top-level run(*args)
    solution.json        the same source as an official single-problem solution
    report.md            the per-workload table a human reads

``forge shapes plan | run | compose`` are the commands (``rsi_kernel_tune.ext.shapes``); the
structures above are documented in ``docs/shapes.md`` because campaign calls
``forge shapes compose`` as a soft dependency and reads the same files.
"""

from __future__ import annotations

__all__ = ["cluster", "compose", "evalx", "matrix", "plan", "run"]

PRODUCT_DIR = "shapes"
"""``tasks/<op>/shapes/``: where plan/run/compose write (a working directory)."""

SCHEMA = 1
"""The schema number each JSON product carries."""


class ShapesError(ValueError):
    """A shapes command cannot do what it was asked: said in full, not half-done."""
