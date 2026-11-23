"""VULCAN -- Value-of-information Utilization for Leaderboard-Conditioned Architecture search.

VULCAN is an orchestration layer that sits *on top of* a kernel-generation system
(upstream: OpenBMB/ForgeKernel) and turns "make every kernel a bit faster" into
"maximize the benchmark's actual published objective under a fixed GPU budget".

The name is a label for this branch of the repository so it can be told apart from
the other contributor's work.  Nothing in ``vulcan`` imports ``forgekernel``: the
package is pure standard-library Python and is therefore validated on any CPU.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "objective",
    "stats",
    "allocator",
    "portfolio",
    "kb_graph",
    "cluster",
    "precision_guard",
    "roofline",
    "calibration",
    "kb_hw_adapter",
    "curriculum",
    "supervisor",
]
