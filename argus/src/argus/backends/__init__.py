"""Backends: the mock (CPU validation), the local numpy runner, and the CudaForge adapter."""

from __future__ import annotations

from .base import BackendError, CampaignBackend
from .mock import MockBackend, MockConfig

__all__ = ["CampaignBackend", "BackendError", "MockBackend", "MockConfig"]
