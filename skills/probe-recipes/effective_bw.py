"""PROBE RECIPE: effective-BW triad — the bandwidth this NODE actually
reaches on a pure stream, for re-reading SOL percentages.

WP12 preset measurement recipe (skills/probe-recipes/). Operator-agnostic
and kernel-free: it never touches the operator's kernel. Copied into
`notes/<name>.py` and run through the verdict-probe channel
(`PROBE_REQUEST: notes/<name>.py` -> `cmd:python3 notes/<name>.py`, cwd =
repo root). Readings only — no gate.

The question it answers: NCU's `dram_pct` (memory SOL) is a percentage of the
THEORETICAL peak, but a node's reachable bandwidth is routinely 10–20% below
that. "dram 85%" can therefore sit far from the real wall. This recipe times
a same-scale pure streaming kernel (triad shape: read two arrays, write one —
the heaviest traffic per FLOP a DRAM can be asked for) at a working-set size
matching the operator's, and reports the bandwidth the node actually
delivers:

    achieved_GBps = (3 x bytes per element x elements) / median_time

Reading it against a digest/NCU reading (pure arithmetic, no extra job):

    effective_sol = kernel_achieved_GBps / triad_GBps x 100

i.e. the SOL denominator becomes MEASURED-REACHABLE instead of theoretical.
An effective SOL at or above ~90% means the kernel really is at the node's
wall (bandwidth-bound, stop pulling that thread); far below means the "85%"
was the theoretical-headroom illusion and the deficit is locality /
sectoring, not bandwidth.

Adaptation: usually none — set WORKING_SET_MIB to the operator's footprint
(the digest's dram_bytes_read+write per call) and optionally pass
--theoretical-gbps on a node whose spec sheet you trust. The default
triadic kernel is plain torch; it measures the node, not the operator.

Protocol: interleaved rounds in one job, CUDA events, median; one contract
line. Output lines:

    RECIPE effective-bw elements=<n> bytes_moved=<B> ms=<median> GBps=<bw>
    READ effective-bw rebase=<...> verdict=<...> detail=<...>
    PROBE PASS max_diff=n/a (node_bw=<GBps>GBps)  # node reachable bandwidth

`PROBE PASS max_diff=n/a` = "the instrument ran and produced its reading" —
citable via `via probe:<path>#<n>`, never a quality judgment.
"""
import argparse
import statistics
import sys

# --- CONFIG (edit to match the operator's working set) -------------------------
WORKING_SET_MIB = 1024       # buffer footprint per array; match the operator's scale
ROUNDS = 15
WARMUP = 5
THEORETICAL_GBPS = 0.0       # optional denominator override (0 = derive from the reading)

# --- reading (pure; no torch, unit-tested) --------------------------------------
AT_WALL_PCT = 90.0           # effective SOL at/above this = the kernel is at the node's wall


def triad_gbps(elements, median_ms, bytes_per_element=12):
    """Triad traffic per element: read b + read c + write a = 2x4 + 4 B (fp32).
    Returns GB/s (1 GB = 1e9 B, the convention NCU/nvidia-smi report in)."""
    if not median_ms or median_ms <= 0 or elements <= 0:
        return None
    return elements * bytes_per_element / (median_ms * 1e-3) / 1e9


def rebase(kernel_gbps, triad_gbps, theoretical_gbps=None):
    """The digest re-read: (value, verdict, detail).

    - with kernel_gbps (the operator's achieved DRAM GB/s, from the digest's
      dram_bytes / kernel time) and triad_gbps: effective SOL = kernel/triad.
    - with theoretical_gbps too: the node's reach as a fraction of its own
      spec sheet (triad/theoretical) — every dram_pct this run produced can
      be re-read against that denominator.
    """
    if triad_gbps and triad_gbps <= 0:
        triad_gbps = None
    if kernel_gbps and triad_gbps:
        eff = kernel_gbps / triad_gbps * 100.0
        verdict = ("at the node's wall: bandwidth-bound — stop pulling the bandwidth thread"
                   if eff >= AT_WALL_PCT else
                   "below the reachable wall: the deficit is locality/sectoring, not bandwidth")
        return eff, verdict, f"kernel {kernel_gbps:.0f} / triad {triad_gbps:.0f} GBps"
    if triad_gbps and theoretical_gbps:
        reach = triad_gbps / theoretical_gbps * 100.0
        return (reach, f"this node delivers {reach:.0f}% of its theoretical peak — "
                f"re-read every dram_pct against that denominator",
                f"triad {triad_gbps:.0f} / theoretical {theoretical_gbps:.0f} GBps")
    if triad_gbps:
        return (None, f"node reachable bandwidth {triad_gbps:.0f} GBps — divide the "
                f"operator's achieved GB/s (digest dram bytes / kernel time) by this "
                f"before calling a dram_pct a wall", "triad only")
    return None, "no reading", "n/a"


def report_lines(elements, median_ms, gbps, verdict, detail, rebase_val=None):
    out = [f"RECIPE effective-bw elements={elements} bytes_moved={elements * 12} "
           f"ms={median_ms:.4f} GBps={gbps:.1f}" if gbps else
           f"RECIPE effective-bw elements={elements} ms=n/a GBps=n/a"]
    rb = "n/a" if rebase_val is None else f"{rebase_val:.1f}%"
    out.append(f"READ effective-bw rebase={rb} verdict={verdict} detail=[{detail}]")
    if gbps:
        out.append(f"PROBE PASS max_diff=n/a (node_bw={gbps:.1f}GBps)  "
                   f"# node reachable bandwidth")
    return out


# --- GPU harness (lazy torch; runs only under the job) --------------------------

def triad(elements):
    """The streaming kernel: a = b + 0.5*c over `elements` fp32 elements —
    12 B of DRAM traffic per element, no reuse (the heaviest honest stream)."""
    import torch
    b = torch.ones(elements, device="cuda", dtype=torch.float32)
    c = torch.ones_like(b)
    a = torch.empty_like(b)

    def run():
        torch.add(b, c, alpha=0.5, out=a)
        torch.cuda.synchronize()
    return run


def measure(elements, rounds, warmup):
    """Interleaved-free: one kernel, many rounds, median ms (the node drift
    shows in the spread; the median carries the reading)."""
    import torch
    run = triad(elements)
    for _ in range(max(1, warmup)):
        run()
    samples = []
    for _ in range(max(1, rounds)):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Effective-BW triad probe recipe: the bandwidth this node actually "
                    "reaches on a pure stream, for re-reading SOL percentages against a "
                    "measured denominator. Readings only.")
    ap.add_argument("--mib", type=int, default=WORKING_SET_MIB,
                    help="per-array footprint in MiB (match the operator's working set)")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--theoretical-gbps", type=float, default=THEORETICAL_GBPS,
                    help="optional spec-sheet denominator; 0 = skip the reach reading")
    args = ap.parse_args(argv)
    elements = args.mib * 1024 * 1024 // 4
    median_ms = measure(elements, args.rounds, args.warmup)
    gbps = triad_gbps(elements, median_ms)
    theo = args.theoretical_gbps or 0.0
    _val, verdict, detail = rebase(None, gbps, theo or None)
    for line in report_lines(elements, median_ms, gbps, verdict, detail):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
