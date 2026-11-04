# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MEMO-lane memoization demo fixture (``tests/fixtures/memo_demo.py``).

These tests pin the delivery shape of the MEMO industrial lane before it is
wired into the driver (that wiring is M3 scope). The contract under test:

- **Hit correctness**: a content-equal second call returns the cached output
  and does not recompute anything (the injected kernel's call counter is the
  assertion point).
- **Fallback correctness**: a novel input is computed honestly by the
  injected kernel and the result matches the unwrapped kernel's output.
- **Latency signature**: with a deliberately slow toy kernel, the first call
  is expensive (>= 50 ms) and a served hit is cheap (< 5 ms) — a coarse,
  wall-clock assertion on CPU, never a nanosecond-grade one.
- **Key stability**: equal values at different memory addresses share one
  key; different values, dtypes, shapes, or input order produce different
  keys.

Single-threaded semantics only: the fixture's cache is unlocked by contract
(see the module docstring there); no concurrency test exists because
concurrent use is out of the MEMO lane's measured regime. No test here is
skipped or expected to fail; everything runs CPU-only on the repo's CPU
torch.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "memo_demo.py"
_SPEC = importlib.util.spec_from_file_location("peregrine_tests_memo_demo", _FIXTURE)
assert _SPEC is not None and _SPEC.loader is not None
memo_demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = memo_demo
_SPEC.loader.exec_module(memo_demo)


class CountingKernel:
    """Toy kernel with the delivery signature ``__call__(*inputs) -> output``
    that counts its own invocations (the honest-recompute assertion point).
    Optionally sleeps, to model a genuinely expensive kernel."""

    def __init__(self, sleep_s: float = 0.0):
        self.sleep_s = sleep_s
        self.calls = 0
        self.seen_inputs: list[tuple] = []

    def __call__(self, *inputs):
        self.calls += 1
        self.seen_inputs.append(inputs)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return self._compute(*inputs)

    def _compute(self, *inputs):
        if len(inputs) == 1:
            (a,) = inputs
            return a @ a.T + 1.0
        a, b = inputs
        return a @ b + 1.0


def make_memo(sleep_s: float = 0.0):
    kernel = CountingKernel(sleep_s=sleep_s)
    return memo_demo.MemoKernel(kernel), kernel


def sample(seed: int, shape: tuple[int, ...] = (32, 32)) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen)


# --- Hit correctness ----------------------------------------------------------


def test_hit_serves_cached_output_without_recompute():
    memo, kernel = make_memo()
    x = sample(0)
    first = memo(x)
    second = memo(x)
    assert torch.allclose(first, second)
    assert kernel.calls == 1, "a content-equal second call must not recompute"
    assert memo.compute_calls == 1
    assert memo.hits == 1
    assert memo.misses == 1


def test_repeated_hits_never_recompute():
    memo, kernel = make_memo()
    x = sample(1)
    reference = memo(x)
    for _ in range(5):
        assert torch.allclose(memo(x), reference)
    assert kernel.calls == 1
    assert memo.stats() == {"entries": 1, "hits": 5, "misses": 1, "compute_calls": 1}


def test_hit_output_is_independent_of_cache_and_caller():
    memo, _ = make_memo()
    x = sample(2)
    first = memo(x)
    first.fill_(0.0)  # a caller mutating its result must not poison the cache
    second = memo(x)
    assert torch.allclose(second, memo(x))
    assert not torch.allclose(second, torch.zeros_like(second))


# --- Fallback correctness -------------------------------------------------------


def test_miss_falls_back_honestly_on_novel_input():
    memo, kernel = make_memo()
    x0 = sample(3)
    memo(x0)  # warm the cache with a different value
    x_new = sample(4)
    expected = kernel._compute(x_new)  # the unwrapped kernel's honest output
    got = memo(x_new)
    assert torch.allclose(got, expected)
    assert kernel.calls == 2, "a novel input must be computed by the fallback"
    assert memo.misses == 2 and memo.compute_calls == 2


def test_fallback_receives_inputs_verbatim():
    memo, kernel = make_memo()
    a, b = sample(5), sample(6, (32, 48))
    out = memo(a, b)
    assert torch.allclose(out, kernel._compute(a, b))
    assert kernel.seen_inputs == [(a, b)], "inputs pass through unmodified"


def test_value_change_is_a_miss_even_after_many_hits():
    memo, kernel = make_memo()
    x = sample(7)
    for _ in range(3):
        memo(x)
    y = x.clone()
    y[0, 0] += 1.0  # one element differs -> different content -> honest compute
    memo(y)
    assert kernel.calls == 2
    assert memo.hits == 2 and memo.misses == 2


# --- Latency signature (coarse, CPU-measurable) ---------------------------------


def test_latency_signature_hit_is_cheap_and_miss_is_expensive():
    memo, kernel = make_memo(sleep_s=0.06)  # toy kernel: >= 50 ms per compute
    x = sample(8)
    t0 = time.perf_counter()
    memo(x)
    first_s = time.perf_counter() - t0
    assert first_s >= 0.05, "first call runs the (deliberately slow) kernel"
    hit_samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        memo(x)
        hit_samples.append(time.perf_counter() - t0)
    best_hit_s = min(hit_samples)  # min-of-N is robust to scheduler noise
    assert best_hit_s < 0.005, "a served hit must not run any kernel compute"
    assert kernel.calls == 1
    assert best_hit_s < first_s / 10


# --- Key stability and collision resistance -------------------------------------


def test_equal_values_at_different_addresses_share_one_key():
    x = sample(9)
    y = x.clone()  # fresh storage, identical content
    assert x.data_ptr() != y.data_ptr()  # genuinely different memory
    assert memo_demo.buffer_digest(x) == memo_demo.buffer_digest(y)
    assert memo_demo.content_key(x) == memo_demo.content_key(y)
    memo, kernel = make_memo()
    memo(x)
    memo(y)
    assert kernel.calls == 1, "a clone of a seen input must be a cache hit"


def test_different_values_produce_different_keys():
    a = sample(10)
    b = sample(11)
    assert memo_demo.buffer_digest(a) != memo_demo.buffer_digest(b)
    memo, kernel = make_memo()
    memo(a)
    memo(b)
    assert kernel.calls == 2


def test_key_binds_shape_and_dtype_not_just_raw_bytes():
    buf = torch.arange(4, dtype=torch.float32)
    as_int = buf.view(torch.int32)  # identical bytes, different dtype
    assert memo_demo.buffer_digest(buf) != memo_demo.buffer_digest(as_int)
    vec = torch.randn(8, generator=torch.Generator().manual_seed(12))
    mat = vec.clone().reshape(2, 4)  # identical bytes and dtype, different shape
    assert memo_demo.buffer_digest(vec) != memo_demo.buffer_digest(mat)


def test_input_order_changes_the_key():
    a, b = sample(13), sample(14)  # both square so either order computes
    assert memo_demo.content_key(a, b) != memo_demo.content_key(b, a)
    memo, kernel = make_memo()
    memo(a, b)
    memo(b, a)
    assert kernel.calls == 2, "positional inputs are keyed jointly and in order"


def test_changing_only_one_input_of_several_is_a_miss():
    memo, kernel = make_memo()
    a, b = sample(15), sample(16, (32, 48))
    memo(a, b)
    b2 = b.clone()
    b2[0, 0] += 1.0
    memo(a, b2)
    assert kernel.calls == 2


def test_non_tensor_input_is_rejected_rather_than_weakly_keyed():
    memo, _ = make_memo()
    x = sample(17)
    try:
        memo(x, [1, 2, 3])
    except TypeError:
        pass
    else:
        raise AssertionError("non-tensor input must fail loudly, not hash silently")


# --- Cache management surface ---------------------------------------------------


def test_bounded_cache_evicts_oldest_entry_and_recomputes_honestly():
    memo, kernel = make_memo()
    bounded = memo_demo.MemoKernel(kernel, max_entries=1)
    x, y = sample(18), sample(19)
    bounded(x)
    bounded(y)  # evicts x
    assert bounded.cached_keys == [memo_demo.content_key(y)]
    bounded(x)  # x is unseen again -> honest recompute (a hit would be wrong here)
    assert kernel.calls == 3
    assert bounded.misses == 3 and bounded.hits == 0


def test_clear_resets_cache_and_counters():
    memo, kernel = make_memo()
    x = sample(20)
    memo(x)
    memo(x)
    memo.clear()
    assert memo.stats() == {"entries": 0, "hits": 0, "misses": 0, "compute_calls": 0}
    memo(x)
    assert kernel.calls == 2, "after clear() the input must be recomputed"


def test_wrapper_rejects_non_callable_and_bad_bounds():
    for bad in (None, 42):
        try:
            memo_demo.MemoKernel(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"MemoKernel({bad!r}) must raise TypeError")
    kernel = CountingKernel()
    try:
        memo_demo.MemoKernel(kernel, max_entries=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_entries=0 must raise ValueError")
