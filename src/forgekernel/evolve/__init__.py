"""``forge evolve``: N lineages on one problem, with elite migration between them.

The population is a directory under ``.artifacts/evolve/``; a generation is one
ordinary ``forge run`` per active lineage.  See ``docs/evolve.md``.

    population  state machine, resume, per-generation bookkeeping
    lineage     task directories (ordinary source reference / board sealed-ref)
    migration   elites, stalls, migration, refuge, retirement (pure functions)
    metric      one generation's reading from the run's own evidence
    archive     variants/, per-version history, archive.md
"""

from __future__ import annotations

from forgekernel.evolve.migration import Decision, Plan, View, plan  # noqa: F401
from forgekernel.evolve.population import Config, Population  # noqa: F401

__all__ = ["Config", "Population", "View", "Decision", "Plan", "plan"]
