"""Backtest the WP11 SPRT early-stop rule against full sampling (task 6).

Runs the SAME window protocol the card's bench runs (burn-in elided — the
controller only sees post-burn-in window readings; A B B A over
``DEVICE_REPS`` device windows per side), once with the early-stop controller
armed and once without, over simulated paired measurements whose true ratio
and window noise are known. Each trial is PAIRED: the full-sampling verdict
and the early-stopped verdict read the SAME window sequence, so any verdict
difference is caused by truncation alone. Empirical gate (the acceptance
wording, verbatim):

* ZERO cases of "early stop says KEEP, full sampling says REVERT" (the
  false-win direction is intolerable). The rule is LOSS-ONLY and decides with
  the verdict's own ratio estimator on the candidate's own data, so this holds
  ARITHMETICALLY: a measurement that never stopped is the full measurement
  (identical ratio -> identical verdict), and a stopped one carries
  ratio < 1 - band <= 1 - floor, which no margin >= 0 scores KEEP;
* every other disagreement (full KEEP, early REVERT/loss-stop) stays < 2%.

The band here is the timing.py controller's own, copied for the offline run
(this script must run torch-free, so it cannot import the template): the
candidate pairs' MAD x 1.4826 x 3, clamped to [SPRT_BAND_FLOOR, SPRT_BAND_CAP]
== 0.005/0.015 — the same family and constants as loop/parse.py's
resolvable_diff (the _LEVER_NOISE_DEFAULT_PCT == lever_pricing_run.NOISE_FLOOR_PCT
drift-compare precedent). If timing.py's constants move, this file's must move
with them — grep SPRT_BAND_ there.

The verdict margin models the DRIVER's run-level resolvable_diff: one band per
(scenario, noise) cell — the run's recent paired ratios share a noise scale —
never a per-measurement fit (which would let a quiet draw keep what a run-level
margin would revert).

Usage: python scripts/backtest_early_stop.py [trials_per_cell]
"""

from __future__ import annotations

import random
import statistics
import sys

# --- timing.py's constants (drift pair — keep in sync, see docstring) ----------
DEVICE_REPS = 5           # windows per side in one paired measurement
SPRT_MIN_PAIRED = 4
SPRT_MIN_WINDOWS = 2
SPRT_BAND_FLOOR = 0.005
SPRT_BAND_CAP = 0.015


def sprt_stop_verdict(ratio, ratios, floor=SPRT_BAND_FLOOR, cap=SPRT_BAND_CAP):
    """The timing.py rule, verbatim: the candidate's gate ratio computed as the
    bench reports it; True when even the interval's TOP sits below parity."""
    ratios = [r for r in ratios if r > 0.0]
    if len(ratios) < SPRT_MIN_PAIRED:
        return False
    mad = statistics.median(abs(r - statistics.median(ratios)) for r in ratios)
    band = min(max(3.0 * 1.4826 * mad, floor), cap)
    return ratio + band < 1.0


class Controller:
    """The timing.py _SprtController's boundary logic, verbatim: decisions only
    at the second ref half's boundaries, on the candidate n-window protocol's
    own data; a decided count holds every side to exactly that count."""

    def __init__(self):
        self.sides: dict[str, list[float]] = {"r1": [], "k1": [], "k2": [], "r2": []}
        self.decided: int | None = None

    def _side(self, side, second_half):
        if side == "ref":
            return "r2" if second_half else "r1"
        return "k2" if second_half else "k1"

    def _candidate_ratio(self, n):
        meds = {}
        for key in ("r1", "k1", "k2", "r2"):
            readings = self.sides[key][:n]
            if len(readings) < n:
                return None
            meds[key] = statistics.median(readings)
        denom = meds["k1"] + meds["k2"]
        if not denom:
            return None
        return (meds["r1"] + meds["r2"]) / denom

    def _candidate_pairs(self, n):
        out = []
        for i in range(n):
            for ref_key, ker_key in (("r1", "k1"), ("r2", "k2")):
                if len(self.sides[ref_key]) > i and len(self.sides[ker_key]) > i:
                    out.append(self.sides[ref_key][i] / self.sides[ker_key][i])
        return out

    def on_window(self, side, second_half, index, ms):
        if self.decided is not None:
            return index + 1 < self.decided
        key = self._side(side, second_half)
        self.sides[key].append(ms)
        if not second_half or side != "ref":
            return True   # decisions happen only at the second ref half
        candidate = index + 1
        if candidate < SPRT_MIN_WINDOWS:
            return True   # the floors: never decide below a 2-window protocol
        ratio = self._candidate_ratio(candidate)
        if ratio is None:
            return True
        if sprt_stop_verdict(ratio, self._candidate_pairs(candidate)):
            self.decided = candidate    # clear loss on the candidate's own data
            return False
        return True


def _median_side(readings):
    return statistics.median(readings)


def trial(true_ratio, noise, margin_run, rng):
    """One simulated paired measurement, PAIRED: the full-sampling verdict and
    the early-stopped verdict read the SAME window sequence, so any verdict
    difference is caused by truncation alone — never by a different noise draw.

    The verdict margin is the RUN-level band (the same ``margin_run`` for both
    arms), modelling loop/parse.py's resolvable_diff: a property of the run's
    recent paired ratios, not of one measurement.

    Returns (full_verdict, early_verdict, n_full, n_early)."""
    ref_ms, ker_ms = 1.0, 1.0 / true_ratio
    seq = {
        ("ref", False): [ref_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
        ("kernel", False): [ker_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
        ("kernel", True): [ker_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
        ("ref", True): [ref_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
    }
    r1, k1, k2, r2 = (seq[("ref", False)], seq[("kernel", False)],
                      seq[("kernel", True)], seq[("ref", True)])

    def verdict(r1, k1, k2, r2):
        ratio = (_median_side(r1) + _median_side(r2)) / (_median_side(k1) + _median_side(k2))
        return "KEEP" if ratio - 1.0 > margin_run else "REVERT"

    v_full = verdict(r1, k1, k2, r2)
    ctl = Controller()
    e1 = _truncated(seq[("ref", False)], "ref", False, ctl)
    e2 = _truncated(seq[("kernel", False)], "kernel", False, ctl)
    e3 = _truncated(seq[("kernel", True)], "kernel", True, ctl)
    e4 = _truncated(seq[("ref", True)], "ref", True, ctl)
    if ctl.decided is not None:
        # symmetric trim: the decided early stop defines an n-window protocol —
        # every side is held to exactly that count before any median is taken,
        # exactly what measure_paired does (all sides commensurate)
        n = ctl.decided
        e1, e2, e3, e4 = (lst[:n] for lst in (e1, e2, e3, e4))
    v_early = verdict(e1, e2, e3, e4)
    n_full = 4 * DEVICE_REPS
    n_early = len(e1) + len(e2) + len(e3) + len(e4)
    return v_full, v_early, n_full, n_early


def _truncated(readings, side, second_half, ctl):
    """The side's window list as the controller takes it: stops at the first
    boundary the rule fires on; the rest of the sequence is never read."""
    out = []
    for index, ms in enumerate(readings):
        out.append(ms)
        if not ctl.on_window(side, second_half, index, ms):
            break
    return out


def main(trials: int = 4000) -> int:
    rng = random.Random(20261103)
    scenarios = [
        ("clear loss  x0.90", 0.90),
        ("loss        x0.97", 0.97),
        ("marginal    x0.995", 0.995),
        ("null        x1.000", 1.000),
        ("marginal    x1.006", 1.006),
        ("real win    x1.020", 1.020),
        ("big win     x1.080", 1.080),
    ]
    noises = [0.003, 0.010, 0.020]
    print(f"backtest_early_stop: {trials} paired trials per (scenario, noise) — "
          f"full and early read the SAME windows; "
          f"DEVICE_REPS={DEVICE_REPS}, band=[{SPRT_BAND_FLOOR}, {SPRT_BAND_CAP}], "
          f"rule: LOSS-ONLY, verdict-estimator ratio")
    print(f"{'scenario':<20} {'noise':>6} | {'full KEEP%':>10} {'early KEEP%':>11} "
          f"{'win-lost%':>9} {'false-win':>9} | {'early windows saved%':>20}")
    false_win_total = 0
    worst_win_lost = 0.0
    total_kept_windows = total_windows = 0
    for name, ratio in scenarios:
        for noise in noises:
            # the run-level verdict margin (resolvable_diff model): 3 sigma of
            # this cell's paired-ratio noise, clamped to the historical bounds
            margin_run = min(max(3.0 * 1.4826 * noise, SPRT_BAND_FLOOR), SPRT_BAND_CAP)
            false_win = win_lost = full_keep = early_keep = 0
            taken = full = 0
            for _ in range(trials):
                v_full, v_early, n_full, n_early = trial(ratio, noise, margin_run, rng)
                full += n_full
                taken += n_early
                full_keep += v_full == "KEEP"
                early_keep += v_early == "KEEP"
                if v_early == "KEEP" and v_full == "REVERT":
                    false_win += 1        # the intolerable direction
                if v_full == "KEEP" and v_early != "KEEP":
                    win_lost += 1         # a win the early stop threw away
            saved = 100.0 * (1.0 - taken / full) if full else 0.0
            lost_pct = 100.0 * win_lost / trials
            worst_win_lost = max(worst_win_lost, lost_pct)
            total_kept_windows += taken
            total_windows += full
            false_win_total += false_win
            print(f"{name:<20} {noise:>6.1%} | {100.0 * full_keep / trials:>9.1f}% "
                  f"{100.0 * early_keep / trials:>10.1f}% {lost_pct:>8.2f}% "
                  f"{false_win:>9} | {saved:>19.1f}%")
    print(f"\nTOTAL false-win (early KEEP / full REVERT): {false_win_total} "
          f"(gate: 0); worst win-lost rate: {worst_win_lost:.2f}% (gate: <2%); "
          f"windows taken: {total_kept_windows}/{total_windows} "
          f"({100.0 * (1.0 - total_kept_windows / total_windows):.1f}% saved)")
    ok = false_win_total == 0 and worst_win_lost < 2.0
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000))
