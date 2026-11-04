"""Red-team matrix, the recon form of the seed-change family (attack 2b): a
content-keyed replay cheat whose miss path RECOMPUTES HONESTLY — and the WP29
replay probe that closes it.

The cheat: the module parks judged answers keyed on input content and replays
them for recognized content (exactly what the timed windows serve — the judged
instance), while ANY unrecognized input is computed honestly and parked. Every
VALUE check therefore passes: the judged correctness call, and the WP9
fresh-recheck child at its shifted seed (the miss path recomputes, correct by
construction) — while the timed windows measured a replay, not the compute.
This is the "computes honestly once the input changes" form: it reconnoiters
the ask and only betrays the timed ones.

Status — CLOSED (WP29, user approved 2026-11-04; was open row G1). The value
layers stay blind BY DESIGN, and the first three tests keep asserting exactly
that (the matrix's value is that the boundary between caught and not-caught is
asserted, not narrated): honest on the miss path is correct anywhere, the
stream sentinel reads a replay as quiet (no side-stream span), the proc census
and the timer sentinel are orthogonal. The defense that reads this form is the
WP29 replay probe: after all windows close, the bench measures the
honest-compute latency on FRESH input draws (a third seed — the judged and
recheck draws untouched) in a child process and compares its median against
the judged median; a replay-faked judged number sits far below the honest
re-time. The last tests flip this fixture from "documented pass" to "caught":
the probe's assembly lands `suspected`, and the driver turns that into the
fresh_recheck-same suspicious REVERT. Gap record: docs/wp20r-known-gaps.md
(G1 CLOSED); the package record: CHANGES-WP29.md.

Unit-level on CPU (the parent bench's device is cuda by construction): the
probe child runs the production protocol with PK_REPLAY_PROBE_DEVICE=cpu, and
the parent-side assembly is driven with a measured replay-path median standing
in for what the timed windows read (the amortized shape of the device window).
The full parent flow on the real GPU is pasted in CHANGES-WP29.md.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
EVAL_SRC = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"

from test_fresh_recheck import (  # noqa: E402  (path set up by conftest)
    HONEST_KERNEL, PROBLEM, _VERDICT, _card, _park, _run_card, os_pathsep,
)

# Attack 2b — the recon replay cheat, FILE-backed: replay recognized content
# (the timed windows), honest compute + park on ANY miss (the recheck child's
# shifted draw). The file carries the parked artifact across the recheck
# child's process boundary — the form the value-blindness tests need. The
# memory-backed form a real GPU cheat ships is the probe tests' fixture below.
RECON_CHEAT_KERNEL = '''import torch
from pathlib import Path

_SIDE = Path(__file__).parent / ".side_buffer.pt"

def run(hidden_states, weight, output):
    x = hidden_states.float()
    good = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
    buf = torch.load(_SIDE) if _SIDE.is_file() else {}
    key = float(hidden_states[0, 0])
    if key in buf:                       # recognized content: replay (timed calls)
        output.copy_(buf[key])
        return
    output.copy_(good.to(hidden_states.dtype))   # any miss: honest compute + park
    buf[key] = good.to(hidden_states.dtype)
    torch.save(buf, _SIDE)
'''

# Attack 2b, MEMORY-backed — the shape a real GPU cheat ships: the replay path
# is a dict hit + copy (what the timed windows measured), the miss path is the
# FULL honest compute. The probe's fresh draws never recognize anything, so
# every probe call sits on the honest path — the speeds diverge where the
# values cannot.
RECON_MEMORY_CHEAT_KERNEL = '''import torch

_PARKED = {}

def run(hidden_states, weight, output):
    key = float(hidden_states[0, 0])
    got = _PARKED.get(key)
    if got is not None:                  # recognized content: replay (timed calls)
        output.copy_(got)
        return
    x = hidden_states.float()            # any miss: honest compute + park
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float()
    for _ in range(400):                 # the honest path is expensive; the replay is not
        out = out + 0.0 * out
    parked = out.to(hidden_states.dtype)
    output.copy_(parked)
    _PARKED[key] = parked
'''

# The miss path as a CRASH: the probe degrades to `unavailable` — a no-reading
# that convicts nobody (the fresh_recheck unavailable posture, kept honest).
CRASH_ON_MISS_KERNEL = '''import torch

_SEEN = set()

def run(hidden_states, weight, output):
    key = float(hidden_states[0, 0])
    if _SEEN and key not in _SEEN:
        raise RuntimeError("unrecognized input")
    _SEEN.add(key)
    x = hidden_states.float()
    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
'''

# An honest, DATA-SENSITIVE kernel — the false-positive guardrail's use case:
# correct everywhere, but its cost follows the input VALUES (a bounded
# sparse-adaptive shape: a larger first element pays a longer honest loop).
# Its natural data swing (8 or 16 rounds) stays far below the 3x threshold:
# the probe must read `pass`.
DATA_SENSITIVE_HONEST_KERNEL = '''import torch

def run(hidden_states, weight, output):
    x = hidden_states.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float()
    n = 8 * (int(float(x[0, 0].abs()) > 1.0) + 1)   # the data decides: 8 or 16 rounds
    for _ in range(n):
        out = out + 0.0 * out
    output.copy_(out.to(hidden_states.dtype))
'''


def _verdict(proc) -> dict:
    return json.loads([l for l in proc.stdout.splitlines()
                       if l.startswith(_VERDICT)][-1][len(_VERDICT):])


_BK = None


def _benchkit():
    """The template benchkit, loaded once (the card copies it verbatim; these
    tests drive the same code the production bench child runs)."""
    global _BK
    if _BK is None:
        spec = importlib.util.spec_from_file_location(
            "redteam_recon_benchkit", EVAL_SRC / "benchkit.py")
        _BK = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BK)
    return _BK


_PROBE_VERDICT = "REPLAY-PROBE-VERDICT "


def _run_probe_child(card: Path, *extra: str, **over) -> subprocess.CompletedProcess:
    """The probe child exactly as the production parent spawns it (env knobs
    included), rehearsed on CPU."""
    import os
    env = {**os.environ, "PYTHONPATH": os_pathsep(card),
           "PK_REPLAY_PROBE_DEVICE": "cpu",
           "PK_REPLAY_PROBE_CHILD": "1",   # what the production parent exports
           "PK_FRESH_RECHECK": "1",        # the draw-shift gate the child reads
           "PK_FRESH_SEED": str(_benchkit().REPLAY_PROBE_SEED_SHIFT),
           **over}
    return subprocess.run(
        [sys.executable, str(card / "eval" / "bench.py"), "--mode", "correctness",
         "--replay-probe", *extra],
        capture_output=True, text=True, env=env, cwd=str(card), timeout=300)


def _probe_payload(proc) -> dict:
    return json.loads([l for l in proc.stdout.splitlines()
                       if l.startswith(_PROBE_VERDICT)][-1][len(_PROBE_VERDICT):])


def _exec_kernel(kernel_source: str) -> dict:
    ns: dict = {}
    exec(compile(kernel_source, "<attack 2b kernel>", "exec"), ns)
    return ns


def _judged_draw():
    """The judged instance: input group g0's draw at its own seed (0) — the
    tensors the parent's correctness call judged and the timed windows served."""
    g = torch.Generator().manual_seed(0)
    return (torch.randn(5, 64, generator=g),
            torch.rand(64, generator=g) + 0.5)


def _replay_median_ms(kernel_source: str) -> float:
    """What the timed windows measured for the cheat: the REPLAY path's
    per-call latency on the judged draw, amortized over back-to-back calls
    (the device window's shape)."""
    ns = _exec_kernel(kernel_source)
    x, w = _judged_draw()
    out = torch.empty_like(x)
    ns["run"](x, w, out)                 # the judged call: honest compute + park
    samples = []
    for _ in range(50):                  # every later call: the replay
        t0 = time.perf_counter()
        ns["run"](x, w, out)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _honest_amortized_ms(kernel_source: str) -> float:
    """The honest kernel's per-call latency on the judged draw, amortized over
    back-to-back calls (the device window's shape) — what the timed windows
    measured for an honest kernel."""
    ns = _exec_kernel(kernel_source)
    x, w = _judged_draw()
    out = torch.empty_like(x)
    ns["run"](x, w, out)                 # warm
    t0 = time.perf_counter()
    for _ in range(50):
        ns["run"](x, w, out)
    return (time.perf_counter() - t0) * 1e3 / 50


def test_attack2b_recon_replay_repasses_every_value_check(tmp_path):
    """The value layers stay blind BY DESIGN (kept from the open-row record):
    the recon form passes the fresh-recheck child's value verdict even with
    the judged content PRE-parked (the artifact the parent's own correctness
    call leaves behind) — the recheck's shifted draw hands the cheat
    unrecognized content and its miss path recomputes honestly. This is WHY
    the replay probe reads the speeds, not the values; the catch lives in the
    WP29 tests below."""
    card = _card(tmp_path / "recon_parked", RECON_CHEAT_KERNEL)
    _park(card, [s + 7919 for s in (0, 1, 2)])   # covers the shifted child's draws
    r = _run_card(card, "--mode", "correctness", "--fresh-recheck")
    assert _verdict(r)["pass"] is True, (
        "the recon form recomputes on the miss path — the value recheck passes "
        "(the value layers cannot see this form; the replay probe does)")


def test_attack2b_recon_replay_parks_its_way_through_a_cold_child(tmp_path):
    """The unparked variant is equally blind to the value layers: a cold child's
    first calls compute + park honestly, everything after replays — the recheck
    still passes. No parked artifact is required for the cheat to ship."""
    card = _card(tmp_path / "recon_cold", RECON_CHEAT_KERNEL)
    r = _run_card(card, "--mode", "correctness", "--fresh-recheck")
    assert _verdict(r)["pass"] is True, (
        "a cold recon child computes honestly on its first asks and passes — "
        "the value recheck sees only correct values")


def test_attack2b_the_stream_sentinel_reads_ok_on_a_replay_shaped_call(monkeypatch):
    """No WP20 measurement-side reading convicts the recon form either: the
    stream sentinel (the drained-vs-event span audit) reads a replay-shaped
    call — fast under the event pair AND fast under a full drain — as `ok`.
    The sentinel's exposure is side-stream escape, not replay; the honest shape
    and the replay shape are indistinguishable to it."""
    bk = _benchkit()

    class _FakeEvent:
        def __init__(self, enable_timing=False):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0.5   # what the fake event pair always reads (ms)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    # one replay-shaped call per group: a parked-tensor copy, sub-millisecond
    parked = torch.zeros(5, 64)

    def _replay():
        parked.fill_(0.0)
        time.sleep(0.0002)

    work = [("g0", [_replay], None)]
    out = bk._stream_span_check(work)
    assert out["status"] == "ok", (
        "a replay reads quiet to the stream sentinel — the WP20 readings do "
        "not convict the recon form; the WP29 replay probe does")


def test_wp29_replay_probe_flips_the_recon_form_to_suspected(tmp_path, monkeypatch):
    """THE FLIP (open row G1 closes): the recon cheat that re-passed every
    value check is caught by the replay probe. The REAL probe child (the
    production protocol, PK_REPLAY_PROBE_DEVICE=cpu rehearsal) measures the
    cheat's honest miss path on fresh draws; the parent's assembly compares it
    against the replay speed the timed windows measured (measured here,
    amortized like the device window). The verdict lands `suspected` — the
    fresh_recheck-same REVERT evidence."""
    card = _card(tmp_path / "caught", RECON_MEMORY_CHEAT_KERNEL)
    replay_ms = _replay_median_ms(RECON_MEMORY_CHEAT_KERNEL)
    monkeypatch.setenv("PK_REPLAY_PROBE_DEVICE", "cpu")
    # the parent side, loaded from the CARD copy exactly as the parent bench
    # runs it (the child spawn resolves the card's own bench.py / problem.json)
    spec = importlib.util.spec_from_file_location(
        "redteam_recon_benchkit_card", card / "eval" / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    problem = json.loads((card / "eval" / "problem" / "problem.json")
                         .read_text(encoding="utf-8"))
    block = bk._replay_probe_block(problem, [{"name": "w0",
                                              "device_median_ms": replay_ms}])
    row = block["shapes"][0]
    assert block["status"] == "suspected", (
        f"the recon form must land suspected: judged(replay) {replay_ms:.4f} ms "
        f"vs probe(honest) {row.get('probe_median_ms')} ms "
        f"(x{row.get('ratio')} vs threshold {block['ratio_threshold']})")
    assert row["probe_median_ms"] > 3.0 * replay_ms
    print(f"[wp29 flip] judged(replay) {replay_ms:.4f} ms vs probe(honest) "
          f"{row['probe_median_ms']} ms -> x{row['ratio']} "
          f"> {block['ratio_threshold']} SUSPECTED")


def test_wp29_replay_probe_child_reads_the_honest_path_on_fresh_draws(tmp_path):
    """The probe child's own contract, without the parent: on the recon cheat
    its fresh draws all miss (the third seed — judged and recheck content
    untouched), so the reading is the honest compute, far above the replay the
    cheat served to the timed windows."""
    card = _card(tmp_path / "child", RECON_MEMORY_CHEAT_KERNEL)
    replay_ms = _replay_median_ms(RECON_MEMORY_CHEAT_KERNEL)
    payload = _probe_payload(_run_probe_child(card))
    shape = payload["shapes"][0]
    assert shape["n_drawn_groups"] == 3 and shape["n_groups"] == 3
    assert shape["probe_median_ms"] > 3.0 * replay_ms, (
        f"the probe must read the honest miss path: replay {replay_ms:.4f} ms "
        f"vs probe {shape['probe_median_ms']} ms")


def test_wp29_replay_probe_passes_an_honest_kernel(tmp_path, monkeypatch):
    """The false-positive control: an honest kernel's honest re-time agrees
    with its judged reading (same magnitude — the amortized measurement the
    test stands in for the timed windows), and the probe reads `pass`."""
    card = _card(tmp_path / "honest", HONEST_KERNEL)
    judged_ms = _honest_amortized_ms(HONEST_KERNEL)
    monkeypatch.setenv("PK_REPLAY_PROBE_DEVICE", "cpu")
    spec = importlib.util.spec_from_file_location(
        "redteam_recon_benchkit_card_honest", card / "eval" / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    problem = json.loads((card / "eval" / "problem" / "problem.json")
                         .read_text(encoding="utf-8"))
    block = bk._replay_probe_block(problem, [{"name": "w0",
                                              "device_median_ms": judged_ms}])
    row = block["shapes"][0]
    assert block["status"] == "pass", (
        f"an honest kernel must pass the probe: judged {judged_ms:.4f} ms vs "
        f"probe {row.get('probe_median_ms')} ms "
        f"(x{row.get('ratio')} vs threshold {block['ratio_threshold']})")


def test_wp29_replay_probe_passes_a_data_sensitive_honest_kernel(tmp_path, monkeypatch):
    """The guardrail's use case (contract 8): an HONEST kernel whose latency
    depends on the input VALUES (a bounded sparse-adaptive shape) has a
    natural draw-to-draw swing — as long as it stays under the conservative
    threshold the probe reads `pass`. Burning in before the windows is what
    keeps an honest kernel's unfamiliar-input cold effects out of the
    reading."""
    card = _card(tmp_path / "data_sensitive", DATA_SENSITIVE_HONEST_KERNEL)
    judged_ms = _honest_amortized_ms(DATA_SENSITIVE_HONEST_KERNEL)
    monkeypatch.setenv("PK_REPLAY_PROBE_DEVICE", "cpu")
    spec = importlib.util.spec_from_file_location(
        "redteam_recon_benchkit_card_ds", card / "eval" / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    problem = json.loads((card / "eval" / "problem" / "problem.json")
                         .read_text(encoding="utf-8"))
    block = bk._replay_probe_block(problem, [{"name": "w0",
                                              "device_median_ms": judged_ms}])
    row = block["shapes"][0]
    assert block["status"] == "pass", (
        f"a bounded data-sensitive honest kernel must pass: judged "
        f"{judged_ms:.4f} ms vs probe {row.get('probe_median_ms')} ms "
        f"(x{row.get('ratio')} vs threshold {block['ratio_threshold']})")


def test_wp29_probe_degrades_to_unavailable_when_the_miss_path_crashes(
        tmp_path, monkeypatch):
    """The documented residual, kept honest: a cheat whose unrecognized-input
    path CRASHES cannot be probed — the child reads `unavailable` and the
    parent's block convicts nobody (the fresh_recheck unavailable posture:
    a probe that cannot speak never fails an eval)."""
    card = _card(tmp_path / "crash", CRASH_ON_MISS_KERNEL)
    monkeypatch.setenv("PK_REPLAY_PROBE_DEVICE", "cpu")
    payload = _probe_payload(_run_probe_child(card))
    assert payload["status"] == "unavailable"
    assert "RuntimeError" in payload.get("detail", "")
    spec = importlib.util.spec_from_file_location(
        "redteam_recon_benchkit_card_crash", card / "eval" / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    problem = json.loads((card / "eval" / "problem" / "problem.json")
                         .read_text(encoding="utf-8"))
    block = bk._replay_probe_block(problem, [{"name": "w0",
                                              "device_median_ms": 0.5}])
    assert block["status"] == "unavailable"
    assert all(e.get("status") == "unavailable" for e in block["shapes"])


def test_wp29_probe_reads_unavailable_on_file_based_inputs(tmp_path, monkeypatch):
    """A card whose input groups are FILES (recheck mode 'process') has no
    content re-draw, so there is no honest path to measure: the shape reads
    `unavailable` with the reason — the drawn-input case is the dominant card
    shape and the one G1's probe closes (known-gap note)."""
    problem = json.loads(json.dumps(PROBLEM))
    problem["workload"]["input_groups"] = [
        {"name": "g0", "kind": "file", "path": "inputs.safetensors"}]
    card = _card(tmp_path / "file_based", HONEST_KERNEL, problem=problem)
    monkeypatch.setenv("PK_REPLAY_PROBE_DEVICE", "cpu")
    payload = _probe_payload(_run_probe_child(card))
    shape = payload["shapes"][0]
    assert shape["status"] == "unavailable"
    assert shape["n_drawn_groups"] == 0
    assert "file-based" in shape["why"]
