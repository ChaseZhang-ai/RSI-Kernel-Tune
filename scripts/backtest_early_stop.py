"""Backtest the SPRT early-stop rule against full sampling (WP11 task 6 + WP15).

Four batteries, all PAIRED: the full-sampling verdict and the early-stopped
verdict read the SAME window sequence, so any verdict difference is caused by
truncation alone — never by a different noise draw.

The protocol under test is PRODUCTION's judgment, imported — not re-implemented
(107 ruling B1: a backtest that validates its own copy of the stop rule
validates nothing about production):

* PRODUCTION LOSS STOP: timing.py's ``sprt_stop_verdict`` — the clamped
  resolvable_diff family band [floor, cap]. This is the mixed configuration
  every deployed job runs (clamped band + persistence + family floor), so the
  cells below are the flip record for exactly what production executes.
* PERSISTENCE: a loss stop must hold at the current candidate AND the previous
  one (the previous protocol's ratio is recomputed from the same sides trimmed
  to n-1 — no extra windows). One transient dip in one candidate protocol used
  to stop the measurement on data full sampling would have recovered from.
* FAMILY FLOOR: on a multi-workload job (the controllers pool their decided
  pairs — WP15 band pooling) the earliest stop candidate is the 3rd window, not
  the 2nd. A 2-window stop on one workload of three is half the evidence the
  single-measurement rule had, and the geomean verdict amplifies the drag.
* WIN STOP (WP15 N-B14): timing.py's ``sprt_win_stop_verdict`` on the
  unclamped honest band (``sprt_band``) — safe only under the driver's C14
  confirmation coupling, which the win battery prices.

Batteries and gates:

1. LOSS battery (WP11, unchanged gate): iid lognormal window noise. False-win
   (early KEEP / full REVERT) is 0 — still arithmetic: a stop requires the
   honest interval's top below parity at the DECIDED candidate, so a stopped
   measurement carries ratio < 1 - band and no margin >= 0 scores KEEP. Every
   other disagreement (full KEEP, early loss-stop) < 2%.
2. AR(1) battery (WP15 cell 1): window noise with lag-1 autocorrelation
   (rho in {0.3, 0.6}), innovations scaled so the MARGINAL noise matches the
   iid cell — the two differ by the correlation alone (a thermal ramp is not
   iid, and the 42-report flagged "autocorrelation narrows the MAD band ->
   earlier stops" as the risk that kept the default off). Same gate.
3. GEOMEAN FAMILY battery (WP15 cell 2): 3-workload cards, per-workload
   controllers with WP15 band pooling, card verdict = GEOMEAN of the three
   workload ratios (what the driver gates on). Gate: family false-win 0,
   family win-lost < 2%.
4. WIN battery (WP15, N-B14 / contract C14): the WIN-direction stop armed. A
   stopped win carries ratio - band > 1 + margin, so the verdict is KEEP by
   construction; the honest measurement is "won early, lost full" — a
   truncated draw the full protocol would NOT have kept, which is exactly what
   the driver's C14 confirmation exists to catch (every win-stopped reading
   re-measures through the watermark protocol before it counts). Gate here:
   the win rule never fires on a loss/null sequence (one-sided by arithmetic)
   and won-early-lost-full stays <= 2%; the rest is the coupling's job.

The verdict margin models the DRIVER's run-level resolvable_diff: one band per
(scenario, noise) cell — the run's recent paired ratios share a noise scale —
never a per-measurement fit (which would let a quiet draw keep what a run-level
margin would revert).

Usage: python scripts/backtest_early_stop.py [trials_per_cell]
       (rsi_kernel_tune venv — the imported timing.py pulls torch)
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

# --- production judgment (the subject under test — never a local copy) --------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsi_kernel_tune.card_template.eval import timing as _timing  # noqa: E402

DEVICE_REPS = _timing.DEVICE_REPS             # windows per side, one paired measurement
SPRT_MIN_WINDOWS = _timing.SPRT_MIN_WINDOWS   # single-workload floor (family cards
SPRT_MIN_WINDOWS_FAMILY = _timing.SPRT_MIN_WINDOWS_FAMILY  # stop no earlier than this)
SPRT_BAND_FLOOR = _timing.SPRT_BAND_FLOOR
SPRT_BAND_CAP = _timing.SPRT_BAND_CAP
POOL_MAX = _timing._SPRT_POOL_MAX
sprt_stop_verdict = _timing.sprt_stop_verdict
sprt_win_stop_verdict = _timing.sprt_win_stop_verdict

WORKLOADS = 3             # the geomean family battery's card width


class Controller:
    """The timing.py _SprtController's boundary logic (WP15 form): decisions
    only at the second ref half's boundaries, on the candidate n-window
    protocol's own data; a decided count holds every side to exactly that
    count. Persistence on the loss direction; the win direction keeps its
    single point (its failure mode is the driver's C14 confirmation)."""

    def __init__(self, win_margin: float = 0.0, pool: list | None = None):
        self.sides: dict[str, list[float]] = {"r1": [], "k1": [], "k2": [], "r2": []}
        self.decided: int | None = None
        self.direction: str | None = None
        self.win_margin = win_margin
        self.pool = pool if pool is not None else []

    def _side(self, side, second_half):
        if side == "ref":
            return "r2" if second_half else "r1"
        return "k2" if second_half else "k1"

    def _min_windows(self):
        # FAMILY FLOOR: a pooled (multi-workload) controller stops no earlier
        # than the 3rd window; a single measurement keeps the 2-window floor.
        return SPRT_MIN_WINDOWS_FAMILY if self.pool is not None else SPRT_MIN_WINDOWS

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

    def _band_pairs(self, n):
        pairs = self._candidate_pairs(n)
        if self.pool:
            pairs = pairs + self.pool[-POOL_MAX:]
        return pairs

    def on_window(self, side, second_half, index, ms):
        if self.decided is not None:
            return index + 1 < self.decided
        key = self._side(side, second_half)
        self.sides[key].append(ms)
        if not second_half or side != "ref":
            return True   # decisions happen only at the second ref half
        candidate = index + 1
        if candidate < self._min_windows():
            return True   # the floors: never decide below the protocol's floor
        ratio = self._candidate_ratio(candidate)
        if ratio is None:
            return True
        pairs = self._candidate_pairs(candidate)
        if sprt_stop_verdict(ratio, self._band_pairs(candidate)):
            # PERSISTENCE: the loss must also hold at the previous candidate —
            # one transient dip stops nothing (its protocol recovers, and on a
            # family card the geomean amplifies a premature drag).
            if candidate > SPRT_MIN_WINDOWS:
                prev = self._candidate_ratio(candidate - 1)
                if prev is None or not sprt_stop_verdict(prev,
                                                         self._band_pairs(candidate - 1)):
                    return True
            self.decided = candidate
            self.direction = "loss"
            self.pool.extend(pairs)
            del self.pool[:-POOL_MAX]
            return False
        if self.win_margin > 0 and sprt_win_stop_verdict(ratio,
                                                         self._band_pairs(candidate),
                                                         self.win_margin):
            self.decided = candidate
            self.direction = "win"
            self.pool.extend(pairs)
            del self.pool[:-POOL_MAX]
            return False
        return True


def _run_side(seq_side, side, second_half, ctl):
    out = []
    for index, ms in enumerate(seq_side):
        out.append(ms)
        if not ctl.on_window(side, second_half, index, ms):
            break
    return out


def _verdict(r1, k1, k2, r2, margin_run):
    ratio = ((statistics.median(r1) + statistics.median(r2))
             / (statistics.median(k1) + statistics.median(k2)))
    return "KEEP" if ratio - 1.0 > margin_run else "REVERT"


def _trim(ctl, sides):
    """The early arm's sides after the controller's symmetric trim."""
    if ctl.decided is not None:
        n = ctl.decided
        return [lst[:n] for lst in sides]
    return list(sides)


def trial_single(true_ratio, noise, margin_run, rng, rho=0.0, win_margin=0.0):
    """One simulated paired measurement (rho>0: AR(1) window noise with
    marginal sigma matched to the iid cell; win_margin>0: the win-direction
    stop armed). Returns (full_verdict, early_verdict, n_full, n_early,
    early_direction)."""
    ref_ms, ker_ms = 1.0, 1.0 / true_ratio

    def draws(base):
        if rho > 0.0:
            out, eps = [], 0.0
            for _ in range(DEVICE_REPS):
                eps = rho * eps + (1.0 - rho * rho) ** 0.5 * noise * rng.gauss(0.0, 1.0)
                out.append(base * math.exp(eps))
            return out
        return [base * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)]

    seq = {"r1": draws(ref_ms), "k1": draws(ker_ms), "k2": draws(ker_ms), "r2": draws(ref_ms)}
    v_full = _verdict(seq["r1"], seq["k1"], seq["k2"], seq["r2"], margin_run)
    ctl = Controller(win_margin=win_margin)   # no pool: a single measurement
    early = {}
    for key, side, second in (("r1", "ref", False), ("k1", "kernel", False),
                              ("k2", "kernel", True), ("r2", "ref", True)):
        early[key] = _run_side(seq[key], side, second, ctl)
    e1, e2, e3, e4 = _trim(ctl, (early["r1"], early["k1"], early["k2"], early["r2"]))
    v_early = _verdict(e1, e2, e3, e4, margin_run)
    return v_full, v_early, 4 * DEVICE_REPS, sum(len(x) for x in (e1, e2, e3, e4)), ctl.direction


def trial_family(ratios, noises, margin_run, rng, win_margin=0.0):
    """One multi-workload card. Each workload: its own paired measurement + own
    controller; a decided workload's pairs pool into the later workloads' band
    (WP15 pooling) and the FAMILY FLOOR holds every controller to a 3-window
    minimum. The card verdict = GEOMEAN of the workload ratios (the driver's
    gate); the SYSTEM verdict additionally applies the WP15 card rule — a
    loss-stopped workload vetoes an unconfirmed card KEEP (the loss direction
    is trusted; a truncated draw cannot acquit the card) — which makes the
    family false-win direction STRUCTURALLY zero: with no stop the arms are
    identical, and with a stop the early arm cannot claim KEEP.
    Returns (full_keep, early_keep, system_keep, n_full, n_early, n_stopped)."""
    pool: list = []
    full_r, early_r = [], []
    n_full = n_early = n_stopped = 0
    for true_ratio, noise in zip(ratios, noises):
        ref_ms, ker_ms = 1.0, 1.0 / true_ratio
        seq = {
            "r1": [ref_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
            "k1": [ker_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
            "k2": [ker_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
            "r2": [ref_ms * rng.lognormvariate(0.0, noise) for _ in range(DEVICE_REPS)],
        }

        def wl_ratio(r1, k1, k2, r2):
            return ((statistics.median(r1) + statistics.median(r2))
                    / (statistics.median(k1) + statistics.median(k2)))

        full_r.append(wl_ratio(seq["r1"], seq["k1"], seq["k2"], seq["r2"]))
        ctl = Controller(win_margin=win_margin, pool=pool)
        early = {}
        for key, side, second in (("r1", "ref", False), ("k1", "kernel", False),
                                  ("k2", "kernel", True), ("r2", "ref", True)):
            early[key] = _run_side(seq[key], side, second, ctl)
        e1, e2, e3, e4 = _trim(ctl, (early["r1"], early["k1"], early["k2"], early["r2"]))
        early_r.append(wl_ratio(e1, e2, e3, e4))
        n_full += 4 * DEVICE_REPS
        n_early += len(e1) + len(e2) + len(e3) + len(e4)
        n_stopped += 1 if ctl.decided is not None else 0

    def geo(vals):
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    full_keep = geo(full_r) - 1.0 > margin_run
    early_keep = geo(early_r) - 1.0 > margin_run
    system_keep = early_keep and n_stopped == 0
    return full_keep, early_keep, system_keep, n_full, n_early, n_stopped


def _gate_row(label, full_keep, early_keep, false_win, win_lost, trials, taken, full):
    return (f"{label:<26} | full KEEP {100.0 * full_keep / trials:5.1f}%  "
            f"early KEEP {100.0 * early_keep / trials:5.1f}%  "
            f"win-lost={100.0 * win_lost / trials:.2f}%  "
            f"false-win={false_win}  saved={100.0 * (1.0 - taken / full):.1f}%")


def _battery(name, rows, gate, extra=""):
    print(f"\n--- {name} ---")
    for row in rows:
        print("  " + row)
    gated = [r for r in rows if "false-win=" in r]
    false_win_total = sum(int(r.split("false-win=")[1].split()[0]) for r in gated)
    worst = max(float(r.split("win-lost=")[1].split("%")[0]) for r in gated)
    ok = false_win_total == 0 and worst < gate
    print(f"GATE ({name}): {'PASS' if ok else 'FAIL'} "
          f"(false-win total {false_win_total}, worst win-lost {worst:.2f}% < {gate}%)"
          + (f" {extra}" if extra else ""))
    return ok


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
    print(f"backtest_early_stop: {trials} paired trials per cell — full and early read "
          f"the SAME windows; DEVICE_REPS={DEVICE_REPS}, stop verdict = timing.py "
          f"sprt_stop_verdict (LOSS band clamped to [{SPRT_BAND_FLOOR}, {SPRT_BAND_CAP}] "
          f"— the production mixed configuration: clamped + persistence + family floor); "
          f"win band = unclamped sprt_band")

    # --- battery 1: iid loss (the WP11 gate) ------------------------------------
    rows, saved_t, full_t = [], 0, 0
    for noise in noises:
        margin_run = min(max(3.0 * 1.4826 * noise, SPRT_BAND_FLOOR), SPRT_BAND_CAP)
        for name, ratio in scenarios:
            false_win = win_lost = full_keep = early_keep = 0
            taken = full = 0
            for _ in range(trials):
                v_full, v_early, n_full, n_early, _ = trial_single(
                    ratio, noise, margin_run, rng)
                full += n_full
                taken += n_early
                full_keep += v_full == "KEEP"
                early_keep += v_early == "KEEP"
                false_win += v_early == "KEEP" and v_full == "REVERT"
                win_lost += v_full == "KEEP" and v_early != "KEEP"
            saved_t += taken
            full_t += full
            rows.append(_gate_row(f"{name} noise={noise:.1%}", full_keep, early_keep,
                                  false_win, win_lost, trials, taken, full))
    rows.append(f"[battery totals] windows taken {saved_t}/{full_t} "
                f"({100.0 * (1.0 - saved_t / full_t):.1f}% saved)")
    ok_iid = _battery("iid loss (WP11 gate)", rows, 2.0)

    # --- battery 2: AR(1) window noise (WP15 cell 1) -----------------------------
    ok_ar1 = True
    for rho in (0.3, 0.6):
        rows = []
        for noise in noises:
            margin_run = min(max(3.0 * 1.4826 * noise, SPRT_BAND_FLOOR), SPRT_BAND_CAP)
            for name, ratio in scenarios:
                false_win = win_lost = full_keep = early_keep = 0
                taken = full = 0
                for _ in range(trials):
                    v_full, v_early, n_full, n_early, _ = trial_single(
                        ratio, noise, margin_run, rng, rho=rho)
                    full += n_full
                    taken += n_early
                    full_keep += v_full == "KEEP"
                    early_keep += v_early == "KEEP"
                    false_win += v_early == "KEEP" and v_full == "REVERT"
                    win_lost += v_full == "KEEP" and v_early != "KEEP"
                rows.append(_gate_row(f"{name} rho={rho} noise={noise:.1%}",
                                      full_keep, early_keep, false_win, win_lost,
                                      trials, taken, full))
        ok_ar1 = _battery(f"AR(1) autocorrelation rho={rho} (WP15 cell 1)", rows, 2.0) and ok_ar1

    # --- battery 3: 3-workload geomean family (WP15 cell 2) ----------------------
    # The gate reads the SYSTEM verdict (the driver's actual semantics): a
    # loss-stopped workload vetoes an unconfirmed card KEEP — the WP15 driver
    # routes such a reading to the standing watermark confirmation (the same
    # coupling contract C14 gives the win direction). RAW rows are printed
    # beside for the record.
    rows = []
    for noise in noises:
        margin_run = min(max(3.0 * 1.4826 * noise, SPRT_BAND_FLOOR), SPRT_BAND_CAP)
        cells = [
            ("uniform x0.97", [0.97] * WORKLOADS),
            ("uniform x1.000", [1.000] * WORKLOADS),
            ("uniform x1.020", [1.020] * WORKLOADS),
            ("mixed 2win1flat", [1.020, 1.020, 1.000]),
            ("mixed 1win2loss", [1.060, 0.97, 0.97]),
        ]
        for label, ratios in cells:
            fw_raw = fw_sys = win_lost = full_keep = sys_keep = stopped = 0
            taken = full = 0
            for _ in range(trials):
                fk, ek, sk, n_full, n_early, st = trial_family(
                    ratios, [noise] * WORKLOADS, margin_run, rng)
                full += n_full
                taken += n_early
                stopped += st
                full_keep += fk
                sys_keep += sk
                fw_raw += ek and not fk      # raw arm disagreement (the record)
                fw_sys += sk and not fk      # system verdict: STRUCTURALLY 0 (see trial_family)
                win_lost += fk and not sk    # a family win the early stops threw away
            rows.append(_gate_row(f"{label} noise={noise:.1%}", full_keep, sys_keep,
                                  fw_sys, win_lost, trials, taken, full)
                        + f"  raw-false-win={fw_raw}  stopped/trials={stopped / trials:.2f}")
    ok_fam = _battery("geomean family: 3-workload cards, pooled bands, verdict = geomean "
                      "(WP15 cell 2; gate = system verdict)", rows, 2.0,
                      extra="[raw arm disagreements are printed per row; the driver's "
                            "loss-stop veto routes those readings to confirmation]")

    # --- battery 4: win direction (WP15 N-B14, contract C14) ---------------------
    margin_run = min(max(3.0 * 1.4826 * 0.010, SPRT_BAND_FLOOR), SPRT_BAND_CAP)
    rows = []
    ok_win = True
    for name, ratio in (("loss  x0.97 (no-fire check)", 0.97),
                        ("null  x1.000 (no-fire check)", 1.000),
                        ("real win x1.020", 1.020),
                        ("big win x1.080", 1.080)):
        win_stopped = won_early_lost_full = full_keep = early_keep = 0
        taken = full = 0
        for _ in range(trials):
            v_full, v_early, n_full, n_early, direction = trial_single(
                ratio, 0.010, margin_run, rng, win_margin=margin_run)
            full += n_full
            taken += n_early
            full_keep += v_full == "KEEP"
            early_keep += v_early == "KEEP"
            win_stopped += direction == "win"
            won_early_lost_full += direction == "win" and v_full == "REVERT"
        nofire = "no-fire check" in name
        if nofire and win_stopped:
            ok_win = False
        if not nofire and won_early_lost_full > trials * 0.02:
            ok_win = False
        rows.append(f"{name:<28} | full KEEP {100.0 * full_keep / trials:5.1f}%  "
                    f"early KEEP {100.0 * early_keep / trials:5.1f}%  "
                    f"win-stopped={100.0 * win_stopped / trials:.2f}%  "
                    f"won-early-lost-full={won_early_lost_full}  "
                    f"saved={100.0 * (1.0 - taken / full):.1f}%")
    print("\n--- win direction (WP15 N-B14 / C14) ---")
    for row in rows:
        print("  " + row)
    print(f"GATE (win): {'PASS' if ok_win else 'FAIL'} "
          "(no-fire on loss/null; won-early-lost-full <= 2% — the driver's C14 "
          "confirmation re-measures every win-stopped reading before it counts)")

    ok = ok_iid and ok_ar1 and ok_fam and ok_win
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000))
