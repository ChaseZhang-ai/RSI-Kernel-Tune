"""Cross-task mechanism memory (ARGUS supporting component).

The search accumulates, per task, the *measured outcome* of each mechanism underneath a
concrete physical condition (the bottleneck class and the counters at the time it was
applied).  ARGUS persists that as a small JSONL table so a later task starts with priors
instead of uniforms: "(memory-bound RMSNorm, +l2_tiling) paid off 1.4x on three tasks" and,
just as importantly, "+tensor_core was refuted under a memory ceiling".

This is deliberately lighter than the sibling track's knowledge-graph work: ARGUS stores
*statistics keyed by executable conditions* and feeds them to the UCB selector, it does not
attempt a certified knowledge base.  Honest scope note in ``docs/design/findings.md``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MechanismRecord:
    op_class: str
    bottleneck: str
    mechanism: str
    gain: float                 # observed multiplicative gain (>=1) or <1 for a regression
    task: str = ""
    n: int = 1


class MechanismMemory:
    """Append-only table of mechanism outcomes, keyed by (op_class, bottleneck)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.records: List[MechanismRecord] = []
        if self.path and self.path.exists():
            self.load()

    def add(self, rec: MechanismRecord) -> None:
        self.records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")

    def mean_gain(self, op_class: str, bottleneck: str, mechanism: str) -> Tuple[float, int]:
        gains = [r.gain for r in self.records
                 if r.op_class == op_class and r.bottleneck == bottleneck
                 and r.mechanism == mechanism]
        if not gains:
            return 1.0, 0
        return float(sum(gains) / len(gains)), len(gains)

    def prior_for(self, op_class: str, bottleneck: str, mechanisms: List[str]) -> Dict[Tuple[str, str], float]:
        """UCB-friendly prior: ``(lineage-agnostic, mechanism) -> expected gain - 1``."""
        out: Dict[Tuple[str, str], float] = {}
        for m in mechanisms:
            mg, _ = self.mean_gain(op_class, bottleneck, m)
            out[(op_class, m)] = max(mg - 1.0, 0.0)
        return out

    def refuted(self, op_class: str, bottleneck: str, threshold: float = 0.98) -> List[str]:
        phys = op_class, bottleneck
        out = []
        for m in {r.mechanism for r in self.records
                  if r.op_class == phys[0] and r.bottleneck == phys[1]}:
            mg, n = self.mean_gain(op_class, bottleneck, m)
            if n >= 2 and mg < threshold:
                out.append(m)
        return sorted(out)

    def load(self) -> None:
        self.records = []
        if not self.path or not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.records.append(MechanismRecord(**json.loads(line)))
            except Exception:
                continue

    def to_lines(self) -> List[str]:
        return [json.dumps(asdict(r)) for r in self.records]
