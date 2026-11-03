"""PROBE RECIPE: phase ablation — decompose the kernel into its memory face
and its compute face.

WP12 preset measurement recipe (skills/probe-recipes/). Operator-agnostic:
copied into `notes/<name>.py` and run through the verdict-probe channel
(`PROBE_REQUEST: notes/<name>.py` -> `cmd:python3 notes/<name>.py`, cwd =
repo root). Readings only — no gate, no operator verdict; the READ lines
name the split, the teacher judges.

The question it answers: build two degraded versions of the tree's kernel —
one that only READS the operands and throws them away (no math), one that
only COMPUTES from register-resident data (no loads) — and time them beside
the full kernel in the same job. The split attributes the kernel time to its
faces:

    read_ms ~ full_ms, compute_ms << full  -> memory-face bound (the loads are
                                              the kernel; tiling/supply levers)
    compute_ms ~ full_ms, read_ms << full  -> compute-face bound (the math is
                                              the kernel; issue/scheduling levers)
    read + compute << full                 -> the faces overlap today (the sum
                                              names the floor the overlap hides;
                                              check the analytic floor next)

When both faces are individually far below the full time, neither alone is
the bottleneck — the residual lives in the overlap (ping-pong, dependency
shape), which is a timeline question, not an ablation one.

This recipe stands alone: it does not require (but cites) the analytic-floor
solver.

Adaptation (the only operator-specific spots): fill `phase_read()` and
`phase_compute()` with the two degraded versions of the tree's kernel. The
defaults below launch built-in streaming / FMA self-checks so the template
runs end-to-end on a GPU node; their split says nothing about the operator.

Protocol: three variants interleaved in ONE job, CUDA events, medians;
readings are shares of the full time, so node drift cancels. Output lines:

    RECIPE phase-ablation variant=<full|read|compute> ms=<median>
    RECIPE phase-ablation read_share=<pct>% compute_share=<pct>% sum=<pct>%
    READ phase-ablation face=<memory|compute|overlap> verdict=<...> detail=<...>
    PROBE PASS max_diff=n/a (phase_split=read=<pct>%/compute=<pct>%)  # phase split

`PROBE PASS max_diff=n/a` = "the instrument ran and produced its reading" (a
non-numerical contract, per the probe brief) — citable, never a quality
judgment.
"""
import argparse
import statistics
import sys

# --- CONFIG (edit when adapting to a tree's kernel) ---------------------------
ROUNDS = 15                   # interleaved rounds (each round times every variant once)
WARMUP = 5                    # untimed full-variant launches before round 0
WORK_MIB = 64                 # payload size for the built-in self-check variants

# --- split reading (pure; no torch, unit-tested) --------------------------------
FACE_BAND_PCT = 75.0          # a face whose share reaches this is "the kernel"
OVERLAP_SUM_PCT = 85.0        # read+compute below this = the faces overlap


def split_phases(full_ms, read_ms, compute_ms):
    """The three medians -> (face, verdict, detail). Deterministic."""
    if read_ms is None or compute_ms is None or not full_ms or full_ms <= 0:
        return "unknown", "no base reading (full ms missing or zero)", "n/a"
    read_share = read_ms / full_ms * 100.0
    compute_share = compute_ms / full_ms * 100.0
    detail = f"read={read_share:.1f}% compute={compute_share:.1f}% sum={read_share + compute_share:.1f}%"
    if read_share >= FACE_BAND_PCT >= compute_share:
        return ("memory", "memory-face bound: the loads are the kernel — tiling / supply "
                "(TMA, cp.async, L2 policy) levers, not math ones", detail)
    if compute_share >= FACE_BAND_PCT >= read_share:
        return ("compute", "compute-face bound: the math is the kernel — issue / scheduling "
                "(unroll, accumulator shape, pipe mix) levers, not supply ones", detail)
    if read_share + compute_share <= OVERLAP_SUM_PCT:
        return ("overlap", "both faces individually cheap: the time lives in the overlap "
                "(dependency shape / anti-phase) — a timeline question, not an ablation one",
                detail)
    return ("mixed", "both faces carry real time: split further (per operand / per segment) "
            "before attributing", detail)


def report_lines(medians, face, verdict, detail):
    """medians = {"full": ms, "read": ms, "compute": ms} -> the printed block."""
    full = medians.get("full")
    read, compute = medians.get("read"), medians.get("compute")
    if not full or read is None or compute is None:
        return [f"RECIPE phase-ablation variant=n/a ms=n/a",
                f"READ phase-ablation face={face} verdict={verdict} detail=[{detail}]"]
    out = [f"RECIPE phase-ablation variant=full ms={full:.4f}",
           f"RECIPE phase-ablation variant=read ms={read:.4f}",
           f"RECIPE phase-ablation variant=compute ms={compute:.4f}",
           f"RECIPE phase-ablation read_share={read / full * 100.0:.1f}% "
           f"compute_share={compute / full * 100.0:.1f}% "
           f"sum={(read + compute) / full * 100.0:.1f}%",
           f"READ phase-ablation face={face} verdict={verdict} detail=[{detail}]",
           f"PROBE PASS max_diff=n/a (phase_split=read={read / full * 100.0:.0f}%/"
           f"compute={compute / full * 100.0:.0f}%)  # phase split"]
    return out


# --- GPU harness (lazy torch; runs only under the job) --------------------------

def variants():
    """The three timed variants: (name, runner blocking until done, desc).
    DEFAULTS = built-in streaming (read face) / FMA (compute face) self-checks
    sharing the full variant's payload. When adapting: return the tree's own
    full kernel plus its read-only and compute-only degradations."""
    import torch
    n = WORK_MIB * 1024 * 1024 // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.ones_like(a)

    def full():
        torch.add(a, b, out=a)
        torch.copysign(a, b, out=a)                    # a second touch: keeps both faces busy
        torch.cuda.synchronize()

    def read():
        torch.copysign(a, b, out=a)                    # moves bytes, no arithmetic of note
        torch.cuda.synchronize()

    def compute():
        torch.mul(a, 1.0001, out=a)                    # register-resident math, no new bytes
        torch.cuda.synchronize()

    return [("full", full, "built-in self-check"), ("read", read, "built-in self-check"),
            ("compute", compute, "built-in self-check")]


def measure(variants_fn, rounds, warmup):
    """Interleaved rounds x variants, CUDA events, per-variant median ms."""
    import torch
    vs = variants_fn()
    for _ in range(max(1, warmup)):
        vs[0][1]()
    samples = {name: [] for name, _, _ in vs}
    for _ in range(max(1, rounds)):
        for name, run, _d in vs:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            stop.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(stop))
    return {name: statistics.median(samples[name]) for name, _, _ in vs}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase-ablation probe recipe (read-only / compute-only / full split). "
                    "Readings only; adapt variants() to the tree's kernel before relying "
                    "on the built-in self-checks.")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    args = ap.parse_args(argv)
    medians = measure(variants, args.rounds, args.warmup)
    face, verdict, detail = split_phases(medians.get("full"), medians.get("read"),
                                         medians.get("compute"))
    for line in report_lines(medians, face, verdict, detail):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
