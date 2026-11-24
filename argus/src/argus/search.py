"""Population search with regression-proof lineage (ARGUS contribution C3).

CudaForge's optimization loop is a **single greedy chain**::

    for round in range(G):
        ind = mutate(current_kernel)     # <-- always the LATEST candidate
        bench(ind)
        current_kernel = ind             # even if ind is worse!
        best = max(best, ind)            # best is only a checkpoint, never the parent

Two structural problems follow:

1. **It derails.**  After one bad optimisation the chain continues from the bad kernel and
   every later round is spent recovering, on a single lineage.  ``best`` is recorded but
   never used as the restart point.
2. **It burns evaluations on one bet.**  There is no way to test two structurally different
   ideas cheaply and keep the better one; the round budget is committed to whatever M0 the
   first seed happened to produce.

ARGUS keeps the two-agent loop but wraps it in a population:

* **K lineages**, each with its own architecture bet; proposals are drawn from a lineage's
  **head** (its best *verified* candidate), so a regression never becomes the parent -- the
  derail bug is structurally impossible;
* **UCB over (lineage, mechanism) arms**, so evaluations go where the expected marginal
  gain is highest, with the mechanism as the unit of mutation (tile / pipeline / vectorise
  / epilogue-fuse / tensor-core / warp-specialise / split-K);
* **multi-fidelity**: every candidate is screened cheaply and only confirmed with the full
  board protocol when it looks promising and *verifies*;
* **the objective is the board score**, not raw speedup, and a candidate is promoted only
  when it verifies (``verifier``) and survives the counter fingerprint (``fingerprint``).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import fingerprint as fp
from . import roofline as rl

#: Mechanism vocabulary used both by the model (:mod:`argus.roofline`) and the prompts.
#: The order is the default round-robin order a naive chain would follow; it deliberately
#: puts prerequisite-dependent mechanisms first, which is where an unordered chain locks
#: itself into a worse ceiling.
MECHANISMS = ["tensor_core", "warp_specialize", "vectorize", "pipeline",
              "split_k", "l2_tiling", "epilogue_fuse"]


@dataclass
class EvalResult:
    latency_s: float
    score: float                 # board score of the evaluated kernel (or a monotone proxy)
    counters: dict = field(default_factory=dict)
    accepted: bool = True
    detail: str = ""


@dataclass
class Candidate:
    cid: int
    lineage: int
    mechanism: str
    parent: Optional[int]
    code: str
    fidelity: int = 0
    result: Optional[EvalResult] = None

    @property
    def score(self) -> float:
        return self.result.score if self.result else float("-inf")


@dataclass
class SearchConfig:
    n_lineages: int = 4
    mechanisms: List[str] = field(default_factory=lambda: list(MECHANISMS))
    max_evals: int = 40
    screen_fidelity: int = 0
    confirm_fidelity: int = 1
    ucb_c: float = 1.0
    min_gain: float = 0.0            # score delta required to promote a candidate
    seed: int = 0
    confirm_when_promising: bool = True
    prune_margin: float = 0.10       # drop lineages this far (in score) below the best
    diversity_floor: int = 2         # ...but never keep fewer than this many lineages
    #: Optional model-based veto: ``arm_filter(mechanism, head_code) -> allowed``.  ARGUS
    #: installs a prerequisite-aware filter so a mechanism that would lock itself into a
    #: worse ceiling (applied before its prerequisite) is never pulled.
    arm_filter: Optional[Callable[[str, str], bool]] = None


@dataclass
class SearchResult:
    best: Optional[Candidate]
    candidates: List[Candidate]
    n_evals: int
    curve: List[float]              # best-so-far score after each evaluation
    lineage_heads: Dict[int, Candidate]

    def summary(self) -> str:
        b = self.best
        return (f"best score={b.score:.4f} (lineage {b.lineage}, mech {b.mechanism}) "
                f"in {self.n_evals} evals")


class PopulationSearch:
    """UCB-selected, multi-fidelity population search over kernel lineages."""

    def __init__(
        self,
        config: SearchConfig,
        seed_fn: Callable[[int], str],
        mutate_fn: Callable[[str, str, int], str],
        eval_fn: Callable[[str, int], EvalResult],
        *,
        transfer_prior: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.cfg = config
        self.seed_fn = seed_fn
        self.mutate_fn = mutate_fn
        self.eval_fn = eval_fn
        self.rng = random.Random(config.seed)
        self.candidates: Dict[int, Candidate] = {}
        self.heads: Dict[int, Optional[Candidate]] = {l: None for l in range(config.n_lineages)}
        self._cid = 0
        self.n_evals = 0
        self.curve: List[float] = []
        self.trajectory: List[dict] = []
        # (lineage, mechanism) -> [pulls, reward_sum]
        self.arms: Dict[Tuple[int, str], List[float]] = {}
        self.transfer = transfer_prior or {}

    # ---- bookkeeping ----
    def _next_cid(self) -> int:
        c = self._cid
        self._cid += 1
        return c

    def head(self, lineage: int) -> Optional[Candidate]:
        return self.heads[lineage]

    @property
    def best(self) -> Optional[Candidate]:
        heads = [h for h in self.heads.values() if h is not None]
        if not heads:
            return None
        return max(heads, key=lambda c: c.score)

    # ---- selection ----
    def _arm_stats(self, arm: Tuple[int, str]) -> Tuple[float, float]:
        pulls, reward = self.arms.get(arm, [0.0, 0.0])
        return pulls, reward

    def _active_lineages(self) -> List[int]:
        """Successive-halving over lineages: keep the best and the ones within margin.

        With a tight evaluation budget, sweeping every (lineage, mechanism) arm is wasteful;
        concentrating on the lineages that are still competitive is what lets the search
        reach a good configuration instead of spending its budget on hopeless bets.
        """
        heads = [(l, h) for l, h in self.heads.items() if h is not None]
        if not heads:
            return []
        best = max(h.score for _, h in heads)
        active = [l for l, h in heads if h.score >= best - self.cfg.prune_margin]
        if len(active) < self.cfg.diversity_floor:
            ranked = sorted(heads, key=lambda lh: lh[1].score, reverse=True)
            active = [l for l, _ in ranked[:self.cfg.diversity_floor]]
        return active

    def _allowed_mechanisms(self, lineage: int) -> List[str]:
        if self.cfg.arm_filter is None:
            return list(self.cfg.mechanisms)
        head = self.heads.get(lineage)
        code = head.code if head is not None else ""
        allowed = [m for m in self.cfg.mechanisms if self.cfg.arm_filter(m, code)]
        return allowed or list(self.cfg.mechanisms)

    def _select_arm(self) -> Tuple[int, str]:
        # Phase 1: seed every lineage once.
        for l in range(self.cfg.n_lineages):
            if self.heads[l] is None:
                return (l, "seed")
        active = self._active_lineages()
        # Phase 2: try every (active-lineage, allowed-mechanism) arm once, best prior first.
        untried = [(l, m) for l in active for m in self._allowed_mechanisms(l)
                   if (l, m) not in self.arms]
        if untried:
            def prior(arm):
                return self.transfer.get((str(arm[0]), arm[1]), 0.0)
            untried.sort(key=lambda a: (-prior(a), a[0], a[1]))
            return untried[0]
        # Phase 3: UCB1 within the active lineages, over allowed mechanisms only.
        total = sum(self.arms[a][0] for a in self.arms) or 1.0
        best_arm, best_ucb = None, -math.inf
        for l in active:
            for m in self._allowed_mechanisms(l):
                arm = (l, m)
                pulls, reward = self._arm_stats(arm)
                mean = reward / pulls if pulls else 0.0
                ucb = mean + self.cfg.ucb_c * math.sqrt(math.log(total + 1.0) / (pulls or 1.0))
                ucb += self.rng.uniform(0, 1e-6)       # deterministic tie-break
                if ucb > best_ucb:
                    best_ucb, best_arm = ucb, arm
        return best_arm  # type: ignore[return-value]

    # ---- one step ----
    def step(self) -> Optional[Candidate]:
        if self.n_evals >= self.cfg.max_evals:
            return None

        lineage, mechanism = self._select_arm()
        if mechanism == "seed" or self.heads[lineage] is None:
            code = self.seed_fn(lineage)
            cand = Candidate(self._next_cid(), lineage, "seed", None, code)
            parent_score = 0.0
        else:
            parent = self.heads[lineage] or self.best
            code = self.mutate_fn(parent.code, mechanism, lineage)
            cand = Candidate(self._next_cid(), lineage, mechanism, parent.cid, code)
            parent_score = parent.score

        # Multi-fidelity: screen cheap, then confirm only if promising & verified.
        result = self.eval_fn(code, self.cfg.screen_fidelity)
        self.n_evals += 1
        cand.result, cand.fidelity = result, self.cfg.screen_fidelity
        if (self.cfg.confirm_when_promising and result.accepted
                and result.score > parent_score + self.cfg.min_gain):
            conf = self.eval_fn(code, self.cfg.confirm_fidelity)
            self.n_evals += 1
            cand.result, cand.fidelity = conf, self.cfg.confirm_fidelity

        self.candidates[cand.cid] = cand
        if result.counters:
            self.trajectory.append(dict(result.counters, time_s=result.latency_s))

        # Promote only if it verifies and beats the head (regression-proof lineage).
        arm = (lineage, mechanism)
        pulls, reward = self.arms.get(arm, [0.0, 0.0])
        self.arms[arm] = [pulls + 1, reward]
        if cand.result.accepted:
            gain = cand.score - parent_score
            if self.heads[lineage] is None or cand.score > self.heads[lineage].score + self.cfg.min_gain:
                self.heads[lineage] = cand
                self.arms[arm] = [pulls + 1, reward + max(gain, 0.0)]

        best = self.best
        self.curve.append(best.score if best else float("-inf"))
        return cand

    def run(self) -> SearchResult:
        while self.step() is not None:
            pass
        return SearchResult(self.best, list(self.candidates.values()), self.n_evals,
                            list(self.curve), dict(self.heads))

    # ---- introspection ----
    def fingerprint_flags(self, peak: rl.GpuPeak, flops: float) -> List:
        return fp.fingerprint_audit(self.trajectory, peak, flops)

    def arm_table(self) -> List[Tuple[int, str, float, float]]:
        return [(l, m, v[0], v[1]) for (l, m), v in sorted(self.arms.items())]


# --------------------------------------------------------------------------- #
# Baseline: CudaForge's single greedy chain (for local A/B)
# --------------------------------------------------------------------------- #

def greedy_chain(
    seed_fn: Callable[[int], str],
    mutate_fn: Callable[[str, str, int], str],
    eval_fn: Callable[[str, int], EvalResult],
    *,
    mechanisms: Optional[List[str]] = None,
    max_evals: int = 40,
) -> SearchResult:
    """Faithful reproduction of CudaForge's loop: continue from the LATEST candidate.

    Kept deliberately un-improved so it can serve as the control in the local A/B.  The
    only change is that campaigns are counted in evaluations, like the population search.
    """
    mechanisms = mechanisms or MECHANISMS
    curve: List[float] = []
    candidates: List[Candidate] = []
    cur: Optional[Candidate] = None
    best: Optional[Candidate] = None
    for i in range(max_evals):
        if i == 0:
            code = seed_fn(0)
            cand = Candidate(0, 0, "seed", None, code)
        else:
            m = mechanisms[(i - 1) % len(mechanisms)]
            code = mutate_fn(cur.code, m, 0)          # parent = latest, not best
            cand = Candidate(i, 0, m, cur.cid if cur else None, code)
        cand.result = eval_fn(code, 0)
        candidates.append(cand)
        cur = cand                                    # derail: always advance
        if cand.result.accepted and (best is None or cand.score > best.score):
            best = cand
        curve.append(best.score if best else float("-inf"))
    return SearchResult(best, candidates, len(candidates), curve, {0: best})
