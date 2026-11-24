"""benchx: the one entry that evaluates a candidate kernel on a board problem.

Only the interface is here (forgekernel.benchx.api): the ``Bench`` protocol, the data
classes a reading comes in, and ``FakeBench`` for tests. The real ``Bench`` is task
benchx's; until it lands, nothing else evaluates a candidate.
"""

from forgekernel.benchx.api import (
    COLD_START_BUDGET_S,
    CONF_DEV_ONLY,
    CONF_FP4_BOUND,
    CONF_SFU_BOUND,
    CONF_SM_COUNT_SENSITIVE,
    MODE_DEV,
    MODE_TARGET,
    MODES,
    Bench,
    BenchCall,
    BenchResult,
    FakeBench,
    WorkloadReading,
    cold_start_over_budget,
)

__all__ = [
    "Bench",
    "BenchCall",
    "BenchResult",
    "COLD_START_BUDGET_S",
    "CONF_DEV_ONLY",
    "CONF_FP4_BOUND",
    "CONF_SFU_BOUND",
    "CONF_SM_COUNT_SENSITIVE",
    "FakeBench",
    "MODE_DEV",
    "MODE_TARGET",
    "MODES",
    "WorkloadReading",
    "cold_start_over_budget",
]
