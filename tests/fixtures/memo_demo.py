# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed memoization demo fixture for the MEMO industrial lane.

This asset pre-builds the delivery shape of the MEMO lane (M1 design,
``docs/industrial-lane.md`` — the path is referenced here before that document
lands) so it can be unit-tested and reused as an integration fixture once the
lane is wired in M3. It is fully self-contained: no driver, benchkit, or
task-card plumbing is touched or required.

The two-lane contract, in one paragraph:

- **MEMO lane** (opt-in, industrial production scenario): repeated inputs are
  the workload (same weights re-inferred, fixed dataset re-processed), so a
  content-addressed cache in front of the kernel is a legitimate system
  optimization. What is timed is *service latency over a repeated-input
  stream*, and this module is the reference shape of that deliverable.
- **STRICT lane** (default, academic measurement scenario): the exact same
  wrapper shape is **cheating** — it memorizes answers instead of computing,
  and the WP29 replay probe exists precisely to capture that pattern (a run
  whose "kernel" never recomputes a repeated input). Under STRICT this module
  must never be wired into a measured path; the two lanes' numbers are never
  compared in one column. See ``docs/industrial-lane.md`` for the full
  contract difference.

Design (per the M2 task pack):

- ``MemoKernel`` wraps an injectable fallback kernel with the kernel delivery
  signature ``__call__(*inputs) -> output``.
- The cache key is content-addressed: per input buffer, a deterministic hash
  of shape + dtype + a **streamed, single-pass, O(data)** sha256 over the raw
  bytes; per-buffer digests are combined into one key. Two tensors with equal
  values at different memory addresses therefore share a key, and any value
  change produces a different key.
- On a cache hit the wrapped kernel is **not** called at all — the cached
  output is returned (a clone of it, so callers cannot poison the cache by
  mutating the returned tensor). On a miss the fallback kernel computes
  honestly and its output is cached. Correctness on *any* input is the iron
  rule: a hit returns the previously computed (correct) output, a miss
  computes honestly.

Known limitations (stated, not hidden):

- **Single-threaded by contract.** The cache dict has no locking; concurrent
  ``__call__`` from multiple threads is undefined (duplicate misses are
  possible and harmless for correctness, but stats may race). The MEMO lane's
  measured regime is a sequential repeated-input stream, so this surface is
  deliberately minimal.
- Inputs must be ``torch.Tensor``. Non-tensor arguments are rejected up front
  rather than silently weakening the key.
- Outputs are cached on CPU as detached clones; device-to-host traffic is
  part of the miss cost, not the hit cost (CPU-only in this repo's tests).
"""

from __future__ import annotations

import hashlib

import torch

# Chunk size for the streaming byte hash: hash the buffer through fixed-size
# windows so peak memory stays O(chunk) regardless of tensor size. The whole
# buffer is still visited exactly once (O(data), single pass).
_HASH_CHUNK_NUMEL = 1 << 20  # 1 Mi worth of uint8 elements per update


def buffer_digest(tensor: torch.Tensor) -> str:
    """Return the deterministic content digest of one tensor buffer.

    The digest binds shape and dtype (so identical byte patterns under
    different types do not collide) and then streams the buffer's raw bytes
    through sha256 in a single pass. The tensor's storage identity, device
    placement, and memory address play no role: equal values yield equal
    digests.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"MemoKernel keys torch.Tensor inputs only, got {type(tensor).__name__}"
        )
    detached = tensor.detach().cpu().contiguous()
    flat = detached.flatten().view(torch.uint8)
    h = hashlib.sha256()
    h.update(f"{tuple(detached.shape)}:{detached.dtype}".encode("utf-8"))
    for start in range(0, flat.numel(), _HASH_CHUNK_NUMEL):
        chunk = flat[start : start + _HASH_CHUNK_NUMEL]
        h.update(chunk.numpy().tobytes())
    return h.hexdigest()


def content_key(*inputs: torch.Tensor) -> str:
    """Combine per-buffer digests into one cache key for an input tuple.

    Order-sensitive: permuting the input tuple changes the key, which matches
    the kernel delivery contract (inputs are positional arguments).
    """
    parts = [buffer_digest(t) for t in inputs]
    h = hashlib.sha256()
    h.update(f"memo-demo-v1:{len(parts)}".encode("utf-8"))
    for p in parts:
        h.update(p.encode("ascii"))
    return h.hexdigest()


class MemoKernel:
    """Content-addressed memoization wrapper around an injectable kernel.

    The wrapped kernel has the delivery signature ``__call__(*inputs) ->
    output``. On the first sight of an input tuple the fallback kernel runs
    honestly and its output is cached; on every later call with
    content-equal inputs the cached output is served and the fallback kernel
    is not invoked (observable through the wrapper's ``compute_calls``
    counter, which mirrors the injected kernel's own invocations).

    A run under the MEMO lane may deliver this shape; the same shape inside a
    STRICT-lane measurement is answer memorization and is exactly what the
    WP29 replay probe is built to catch. Contract boundary:
    ``docs/industrial-lane.md``.
    """

    def __init__(self, kernel, max_entries: int | None = None):
        """Wrap ``kernel``; ``max_entries`` bounds the cache (FIFO eviction,
        ``None`` = unbounded)."""
        if not callable(kernel):
            raise TypeError("kernel must be callable: __call__(*inputs) -> output")
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive or None")
        self._kernel = kernel
        self._max_entries = max_entries
        self._cache: dict[str, torch.Tensor] = {}
        self.compute_calls = 0  # honest fallback invocations
        self.hits = 0
        self.misses = 0

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Serve ``output`` for ``inputs``: cached on hit, honestly computed
        on miss. The returned tensor is a fresh copy either way."""
        key = content_key(*inputs)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached.clone()
        self.misses += 1
        self.compute_calls += 1
        output = self._kernel(*inputs)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"wrapped kernel must return a torch.Tensor, "
                f"got {type(output).__name__}"
            )
        stored = output.detach().cpu().clone()
        if self._max_entries is not None and len(self._cache) >= self._max_entries:
            oldest = next(iter(self._cache))  # dict preserves insertion order
            del self._cache[oldest]
        self._cache[key] = stored
        return stored.clone()

    @property
    def cached_keys(self) -> list[str]:
        """Cache keys currently resident (insertion order, oldest first)."""
        return list(self._cache)

    def clear(self) -> None:
        """Drop all cached entries and reset the hit/miss/compute counters."""
        self._cache.clear()
        self.compute_calls = 0
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Snapshot of the wrapper's observable counters and cache size."""
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "compute_calls": self.compute_calls,
        }
