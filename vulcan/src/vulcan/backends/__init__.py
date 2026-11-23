"""Backends for VULCAN.

* ``base``  -- the ``CampaignBackend`` protocol and ``EvalResult``.
* ``mock``  -- deterministic synthetic GPU (CPU validation, A/B experiment).
* ``forge`` -- adapter to the upstream OpenBMB/ForgeKernel ``forge`` CLI (GPU).
"""

from __future__ import annotations

from .base import CampaignBackend, EvalResult

__all__ = ["CampaignBackend", "EvalResult"]
