"""ARGUS -- Adversarial Reward-hack Guard with Unified Search.

A verifier-first, attribution-driven agentic GPU-kernel optimizer.  ARGUS reproduces
the *CudaForge* agent architecture (OptimAI-Lab/CudaForge, MIT) as its base and adds
three capabilities that CudaForge lacks:

* ``verifier`` / ``probes`` / ``fingerprint`` -- board-faithful, reward-hack-resistant
  verification (CudaForge accepts a single ``torch.allclose`` on one shape).
* ``attribution`` / ``roofline`` -- causal, counter-grounded performance attribution and
  model-based pre-screening (CudaForge dumps raw counters into a prompt).
* ``search`` -- a population search with regression-proof lineage and mechanism-level
  mutation (CudaForge is a single greedy chain that derails on any regression).

Everything in this package is pure Python (+ numpy where numerics are involved) so the
whole reasoning stack can be validated on a CPU; only the real kernel evaluations need a
GPU.  See ``README.md`` and ``ARCHITECTURE.md``.
"""

from __future__ import annotations

__all__ = [
    "objective",
    "roofline",
    "attribution",
    "verifier",
    "probes",
    "fingerprint",
    "search",
    "memory",
    "schedule",
    "cudaforge",
]

__version__ = "0.1.0"
