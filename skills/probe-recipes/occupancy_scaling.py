"""PROBE RECIPE: occupancy-scaling curve (grid-scaling) — is the current
bottleneck latency, throughput, or the wave tail?

WP12 preset measurement recipe (skills/probe-recipes/). Operator-agnostic:
this file is copied into `notes/<name>.py` (unchanged or with the CONFIG and
`launch()` hook below adapted to the tree's kernel) and run through the
verdict-probe channel — `PROBE_REQUEST: notes/<name>.py`, which the driver
executes as `cmd:python3 notes/<name>.py` in the student tree, cwd = repo
root. It produces READINGS only: no gate, no verdict about the operator —
the interpretation lines name the curve shape, the teacher judges.

The question it answers: launch the SAME kernel with the grid dimension
multiplied by s in {0.25, 0.5, 1.0} (the loop body unchanged — only fewer
blocks are issued; for a persistent kernel scale the num-SM-equivalent knob
instead) and time each. The curve shape separates the three first-split
bottleneck classes that the SOL header alone cannot:

    flat    t(s) ~ constant across scales   -> latency-bound (SMs idle, the
                                               critical path is inside one
                                               block/warp; grid changes move
                                               nothing)
    linear  t(s) ~ s (quarter grid = ~4x    -> throughput-bound (SMs are the
            slower)                            saturated resource; reducing
                                               grid costs proportionally)
    stepped t(s) drops in waves (a jump     -> wave quantization (the tail
            between adjacent scales)           wave dominates; tile height /
                                               grid rounding is the lever)

Adaptation (the only operator-specific spot): fill `launch(scale)` to launch
the tree's kernel with its grid dimension(s) multiplied by `scale`. The
default implementation below launches a built-in streaming kernel so the
template runs end-to-end on a GPU node without edits (a self-check that the
harness measures); it is NOT the operator's kernel and its curve says nothing
about the operator.

Protocol (the verdict-probe rules, unchanged): all scales measured
interleaved in ONE job (round r times scale s), CUDA events, median per
scale; readings are ratios to s=1.0, so node drift cancels. Output lines:

    RECIPE occupancy-scaling scale=<s> grid=<blocks> ms=<median> ratio=<t/t1>
    READ occupancy-scaling shape=<flat|linear|stepped|mixed> verdict=<...> detail=<...>
    PROBE PASS max_diff=n/a (occupancy_curve_shape=<shape>)  # occupancy curve shape

`PROBE PASS max_diff=n/a` is this channel's "the instrument ran and produced
its reading" line (a non-numerical contract, per the probe brief) — it is a
reading the diagnosis may cite (`via probe:<path>#<n>`), never a quality
judgment.
"""
import argparse
import statistics
import sys

# --- CONFIG (edit when adapting to a tree's kernel) ---------------------------
SCALES = (0.25, 0.5, 1.0)     # grid multipliers, ascending; 1.0 = the base launch
ROUNDS = 15                   # interleaved rounds (each round times every scale once)
WARMUP = 5                    # untimed launches of the base grid before round 0
WORK_MIB = 64                 # streaming-kernel payload for the default launch

# --- curve reading (pure; no torch, unit-tested) -------------------------------
FLAT_BAND = 0.15              # |ratio - 1| within this at every sub-scale = flat
LINEAR_BAND = 0.25            # |ratio - s| within this at every sub-scale = linear
STEP_JUMP = 1.30              # adjacent-scale ratio jump beyond this = a wave step


def read_curve(points):
    """[(scale, median_ms), ...] ascending in scale -> (shape, verdict, detail).

    Deterministic classification: all sub-scale ratios within FLAT_BAND of 1
    -> flat; all within LINEAR_BAND of their scale -> linear; any adjacent
    pair jumping more than STEP_JUMP (time drops when the grid crosses a wave
    boundary) -> stepped; else mixed."""
    pts = sorted(points, key=lambda p: p[0])
    base = next((ms for s, ms in pts if abs(s - 1.0) < 1e-9), None)
    if base is None or base <= 0 or len(pts) < 2:
        return "mixed", "no base reading (s=1.0 missing or zero) — no shape", "n/a"
    sub = [(s, ms / base) for s, ms in pts if abs(s - 1.0) >= 1e-9]
    if not sub:
        return "mixed", "no sub-scale points", "n/a"
    if all(abs(r - 1.0) <= FLAT_BAND for _, r in sub):
        return ("flat", "latency-bound: shrinking the grid does not slow the kernel "
                "(SMs idle; the critical path is inside the block)", 
                " ".join(f"s={s:g}:r={r:.3f}" for s, r in sub))
    if all(abs(r - s) <= LINEAR_BAND for s, r in sub):
        return ("linear", "throughput-bound: time scales with the grid (SMs are the "
                "saturated resource; grid is not the lever to shrink)",
                " ".join(f"s={s:g}:r={r:.3f}" for s, r in sub))
    ordered = sorted(sub)
    for (s0, r0), (s1, r1) in zip(ordered, ordered[1:]):
        if r0 / max(r1, 1e-9) >= STEP_JUMP:
            return ("stepped", "wave quantization: a wave boundary sits between these "
                    "scales (tail wave dominates; tile height / grid rounding is the lever)",
                    f"s={s0:g}:r={r0:.3f} -> s={s1:g}:r={r1:.3f}")
    return ("mixed", "no clean shape: re-read with the SOL header and the stall ledger "
            "before attributing", " ".join(f"s={s:g}:r={r:.3f}" for s, r in sub))


def report_lines(pts, shape, verdict, detail):
    """The block this recipe prints, byte-identical for identical readings."""
    base = next((ms for s, ms in pts if abs(s - 1.0) < 1e-9), None)
    out = []
    for s, ms in sorted(pts):
        ratio = "n/a" if not base else f"{ms / base:.3f}"
        out.append(f"RECIPE occupancy-scaling scale={s:g} ms={ms:.4f} ratio={ratio}")
    out.append(f"READ occupancy-scaling shape={shape} verdict={verdict} detail=[{detail}]")
    out.append(f"PROBE PASS max_diff=n/a (occupancy_curve_shape={shape})  "
               f"# occupancy curve shape")
    return out


# --- GPU harness (lazy torch; runs only under the job) --------------------------

def launch(scale):
    """Launch the kernel at grid x scale; returns (runner, description) where
    runner blocks until the kernel is done. DEFAULT = a size-scaled streaming
    self-check so the template runs end-to-end on a GPU node (it exercises the
    harness, NOT the operator's kernel — its curve says nothing about the
    operator). When adapting: launch the tree's kernel here with its grid
    dimension(s) multiplied by `scale` — the loop body unchanged; a persistent
    kernel scales its num-SM-equivalent knob instead."""
    import torch
    n = max(1024, int((WORK_MIB * 1024 * 1024 // 4) * scale))
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.ones_like(a)

    def run():
        torch.add(a, b, out=a)                         # streaming, grid follows n
        torch.cuda.synchronize()
    return run, f"elems={n} (built-in size-scaled streaming self-check)"


def measure(launcher, rounds, warmup):
    """Interleaved rounds x scales, CUDA events, per-scale median ms."""
    import torch
    runners = {}
    descs = {}
    for s in SCALES:
        runners[s], descs[s] = launcher(s)
    for _ in range(max(1, warmup)):
        runners[SCALES[-1]]()
    samples = {s: [] for s in SCALES}
    for _ in range(max(1, rounds)):
        for s in SCALES:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[s]()
            stop.record()
            torch.cuda.synchronize()
            samples[s].append(start.elapsed_time(stop))
    return [(s, statistics.median(samples[s])) for s in SCALES], \
        "; ".join(f"s={s:g}: {descs[s]}" for s in SCALES)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Occupancy-scaling curve probe recipe (grid-scaling: flat="
                    "latency-bound, linear=throughput-bound, stepped=wave quantization). "
                    "Readings only; adapt launch() to the tree's kernel before relying "
                    "on the default hook.")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    args = ap.parse_args(argv)
    pts, desc = measure(launch, args.rounds, args.warmup)
    shape, verdict, detail = read_curve(pts)
    for line in report_lines(pts, shape, verdict, detail):
        print(line)
    print(f"RECIPE occupancy-scaling launchers: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
