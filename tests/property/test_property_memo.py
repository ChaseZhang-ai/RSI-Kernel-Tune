# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests (hypothesis) for the MEMO-lane MemoKernel fixture.

Hunt 2 (task 178): the content-addressed cache is the MEMO lane's deliverable,
so its identity invariants are load-bearing (M3 / docs/industrial-lane.md):

* P15 equal values at different memory addresses share one cache key, and a
  key seen twice is served from the cache (the fallback kernel runs exactly
  once) with equal outputs;
* P16 any content change changes the key: a value flip, a dtype change, a
  shape change, and a permutation of the input tuple each produce a
  different key;
* P17 FIFO eviction: with ``max_entries=k`` the cache never holds more than k
  entries, the evicted entries are always the OLDEST resident ones, and an
  evicted key recomputes on its next sight;
* P18 the served output is a clone: a caller poisoning its copy never poisons
  the cache, and the hit/miss/compute counters stay consistent.

Runs on CPU (torch required); no GPU is needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "memo_demo.py"
_SPEC = importlib.util.spec_from_file_location(
    "rsi-kernel-tune_tests_property_memo_demo", _FIXTURE)
assert _SPEC is not None and _SPEC.loader is not None
memo_demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = memo_demo
_SPEC.loader.exec_module(memo_demo)

content_key = memo_demo.content_key
buffer_digest = memo_demo.buffer_digest
MemoKernel = memo_demo.MemoKernel

DTYPES = st.sampled_from([torch.float32, torch.float64, torch.int64, torch.int32])
SHAPES = st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=3)
SEEDS = st.integers(min_value=0, max_value=2**31 - 1)


def _tensor(shape, dtype, seed):
    """A deterministic tensor of the requested shape/dtype (its own storage)."""
    g = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g, dtype=dtype)
    return torch.randint(-100, 100, shape, generator=g, dtype=dtype)


def _identity_kernel(*inputs):
    """The honest fallback: return the first input (MemoKernel clones it)."""
    return inputs[0]


@settings(deadline=None, max_examples=200)
@given(shape=SHAPES, dtype=DTYPES, seed=SEEDS)
def test_property_equal_values_at_different_addresses_share_one_key(shape, dtype, seed):
    """P15: the key is content-addressed -- shape/dtype/value equal tensors at
    different storages (a clone, and an independently generated equal tensor)
    produce one key; the wrapper computes once and serves every later sight
    from the cache with an equal output."""
    t = _tensor(shape, dtype, seed)
    cloned = t.clone()
    independent = _tensor(shape, dtype, seed)  # second generator, same values
    assert not t.data_ptr() == cloned.data_ptr()

    assert buffer_digest(t) == buffer_digest(cloned)
    assert content_key(t) == content_key(cloned)
    assert content_key(t) == content_key(independent)

    memo = MemoKernel(_identity_kernel)
    out1 = memo(t)
    out2 = memo(cloned)
    out3 = memo(independent)
    assert memo.compute_calls == 1
    assert memo.misses == 1
    assert memo.hits == 2
    assert torch.equal(out1, out2)
    assert torch.equal(out2, out3)
    assert memo.stats() == {"entries": 1, "hits": 2, "misses": 1, "compute_calls": 1}


@settings(deadline=None, max_examples=200)
@given(shape=SHAPES, dtype=DTYPES, seed=SEEDS,
       mode=st.sampled_from(["value", "dtype", "shape", "order"]))
def test_property_any_content_change_changes_the_key(shape, dtype, seed, mode):
    """P16: value, dtype, shape and input-tuple order each move the key. Two
    tensors that differ in any one of them must not collide (shape+dtype are
    bound into the digest; the tuple order is bound into the combined key)."""
    t = _tensor(shape, dtype, seed)
    other = _tensor(shape, dtype, (seed * 7 + 13) % 2**31)

    if mode == "value":
        perturbed = t.clone()
        flat = perturbed.flatten()
        if dtype.is_floating_point:
            flat[0] = flat[0] + 1.0
        else:
            flat[0] = flat[0] + 1
        perturbed = flat.view(perturbed.shape)
        assume(not torch.equal(t, perturbed))
        assert content_key(t) != content_key(perturbed)
    elif mode == "dtype":
        promoted = torch.float64 if dtype != torch.float64 else torch.float32
        recast = t.to(promoted)
        assert content_key(t) != content_key(recast)
    elif mode == "shape":
        # flatten a multi-dim row to 1-D, or grow a 1-D row to 2-D: same bytes,
        # different shape -> the digest's shape binding must move the key
        reshaped = t.reshape(1, -1) if t.dim() == 1 else t.reshape(-1)
        assert reshaped.shape != t.shape
        assert torch.equal(reshaped.flatten(), t.flatten())
        assert content_key(t) != content_key(reshaped)
    else:  # order
        differing = other.clone()
        flat = differing.flatten()
        if dtype.is_floating_point:
            flat[0] = flat[0] + 1.0
        else:
            flat[0] = flat[0] + 1
        differing = flat.view(differing.shape)
        assume(not torch.equal(t, differing))
        assert content_key(t, differing) != content_key(differing, t)


@settings(deadline=None, max_examples=200)
@given(max_entries=st.integers(min_value=1, max_value=6),
       extra=st.integers(min_value=1, max_value=6))
def test_property_fifo_eviction_evicts_the_oldest(max_entries, extra):
    """P17: FIFO eviction -- the cache never exceeds max_entries, the entries
    evicted are exactly the oldest residents, an evicted key recomputes (a
    miss) on its next sight, and a re-inserted key becomes the youngest."""
    n = max_entries + extra
    inputs = [torch.full([1], float(i)) for i in range(n)]
    memo = MemoKernel(_identity_kernel, max_entries=max_entries)

    for i, x in enumerate(inputs):
        memo(x)
        assert len(memo.cached_keys) <= max_entries
        if i >= max_entries:
            assert len(memo.cached_keys) == max_entries
    assert memo.compute_calls == n  # every distinct input computed exactly once

    keys_after = memo.cached_keys
    assert len(keys_after) == max_entries
    # the residents are exactly the last max_entries inserted (FIFO, not LRU --
    # nothing was touched since insertion); they all still hit
    residents = inputs[extra:]
    assert keys_after == [content_key(x) for x in residents]
    for x in residents:
        before = memo.compute_calls
        memo(x)
        assert memo.compute_calls == before
    # the evicted prefix recomputes on its next sight (oldest evicted first),
    # and each re-insertion evicts the oldest resident in turn
    evicted = inputs[:extra]
    for x in evicted:
        before = memo.compute_calls
        memo(x)
        assert memo.compute_calls == before + 1
    assert content_key(evicted[-1]) == memo.cached_keys[-1]


@settings(deadline=None, max_examples=200)
@given(shape=SHAPES, seed=SEEDS)
def test_property_a_poisoned_return_never_poisons_the_cache(shape, seed):
    """P18: the served output is a fresh copy on hit and miss alike -- a caller
    mutating its copy leaves the cache intact, and the counters stay
    consistent (hits + misses == calls, misses == compute_calls for an honest
    kernel)."""
    t = _tensor(shape, torch.float32, seed)
    memo = MemoKernel(lambda *xs: xs[0] * 2.0 + 1.0)

    out1 = memo(t)
    reference = out1.clone()
    out1.fill_(-999.0)  # the caller poisons ITS copy

    out2 = memo(t.clone())  # content-equal input at another address -> hit
    assert torch.equal(out2, reference)
    out3 = memo(t)
    assert torch.equal(out3, reference)
    assert memo.compute_calls == 1
    assert memo.hits == 2 and memo.misses == 1
    assert memo.stats() == {"entries": 1, "hits": 2, "misses": 1, "compute_calls": 1}
