"""Knowledge graph with *executable* applicability predicates and negative memory.

ForgeKernel stores hardware mechanisms in a knowledge base keyed by GPU architecture
and cites them from milestone plans.  Two weaknesses remain:

1. **Retrieval is qualitative.**  "Applicable when memory-bound" is a sentence the
   model interprets; it is not a checkable condition, so the wrong mechanism can be
   prescribed.
2. **Only successes are stored.**  The paper records failed changes' conditions in
   passing; they are not first-class retrievable knowledge, so later runs can repeat
   the same failed idea.

VULCAN replaces the prose entry with a *typed* entry:

* ``predicates`` -- a mapping ``feature -> (low, high)`` evaluated against the
  measured characteristics of the task (from NCU/SASS: ``dram_pct``, ``tensor_pct``,
  ``occupancy``, ``bytes_per_flop``, ``seq_len``, ...).  An entry matches only when
  *every* predicate holds, so retrieval is a filter, not a suggestion.
* ``status`` in ``{candidate, certified, refuted}`` -- only ``certified`` entries are
  prescribed; ``refuted`` entries are returned by ``negative_matches`` so the
  architecture agent can be told "this was tried and regressed under these
  conditions".
* ``analog`` -- a cross-architecture lift: an SM90 mechanism can be offered on SM100
  as a *hypothesis* (never as certified), with its predicate rewritten by the
  architecture agent and re-validated by a probe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

STATUS_CANDIDATE = "candidate"
STATUS_CERTIFIED = "certified"
STATUS_REFUTED = "refuted"

Interval = Tuple[float, float]


def _as_interval(v) -> Interval:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo, hi = float(v[0]), float(v[1])
        if lo > hi:
            raise ValueError(f"predicate interval low>high: {v!r}")
        return lo, hi
    x = float(v)
    return x, x


@dataclass
class KnowledgeEntry:
    id: str
    arch: str                       # sm90, sm100, ...
    mechanism: str                  # one-line name
    predicates: Dict[str, Interval] = field(default_factory=dict)
    effect: float = 0.0             # expected signed relative effect (+ = faster)
    status: str = STATUS_CANDIDATE
    evidence: str = ""              # probe id / job reference
    cost: str = ""                  # smem / registers / sync side effects
    recipe: str = ""                # minimal runnable example
    limitations: str = ""
    negative_of: Optional[str] = None
    analog_of: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in (STATUS_CANDIDATE, STATUS_CERTIFIED, STATUS_REFUTED):
            raise ValueError(f"bad status {self.status!r}")
        self.predicates = {k: _as_interval(v) for k, v in self.predicates.items()}

    def matches(self, features: Mapping[str, float]) -> bool:
        """True iff every predicate is satisfied by ``features``.

        A predicate on a feature that is absent from ``features`` does not match:
        unknown evidence is not evidence.
        """
        for name, (lo, hi) in self.predicates.items():
            if name not in features:
                return False
            x = float(features[name])
            if x < lo or x > hi:
                return False
        return True

    def specificity(self) -> int:
        return len(self.predicates)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "arch": self.arch,
            "mechanism": self.mechanism,
            "predicates": {k: list(v) for k, v in self.predicates.items()},
            "effect": self.effect,
            "status": self.status,
            "evidence": self.evidence,
            "cost": self.cost,
            "recipe": self.recipe,
            "limitations": self.limitations,
            "negative_of": self.negative_of,
            "analog_of": self.analog_of,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "KnowledgeEntry":
        return cls(
            id=d["id"], arch=d["arch"], mechanism=d["mechanism"],
            predicates=d.get("predicates", {}), effect=float(d.get("effect", 0.0)),
            status=d.get("status", STATUS_CANDIDATE), evidence=d.get("evidence", ""),
            cost=d.get("cost", ""), recipe=d.get("recipe", ""),
            limitations=d.get("limitations", ""), negative_of=d.get("negative_of"),
            analog_of=d.get("analog_of"), tags=list(d.get("tags", [])),
        )


class KnowledgeGraph:
    """In-memory graph; JSONL-persistable for cross-run reuse."""

    def __init__(self, entries: Optional[Iterable[KnowledgeEntry]] = None):
        self._by_id: Dict[str, KnowledgeEntry] = {}
        for e in entries or []:
            self.add(e)

    # -- mutation ----------------------------------------------------------- #
    def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        self._by_id[entry.id] = entry
        return entry

    def certify(self, entry_id: str, evidence: str = "") -> KnowledgeEntry:
        e = self._by_id[entry_id]
        e.status = STATUS_CERTIFIED
        if evidence:
            e.evidence = evidence
        return e

    def refute(self, entry_id: str, evidence: str = "") -> KnowledgeEntry:
        e = self._by_id[entry_id]
        e.status = STATUS_REFUTED
        if evidence:
            e.evidence = evidence
        return e

    def add_negative(self, target_id: str, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Record a *failed* variant of an existing certified mechanism."""
        entry.status = STATUS_REFUTED
        entry.negative_of = target_id
        return self.add(entry)

    # -- retrieval ---------------------------------------------------------- #
    def retrieve(
        self,
        features: Mapping[str, float],
        arch: str,
        top_k: int = 8,
    ) -> List[KnowledgeEntry]:
        """Certified, matching entries for ``arch``, best first.

        Same-architecture and certified-only: a mechanism is never prescribed for a
        GPU it was not probed on.  Cross-architecture material comes from
        :meth:`retrieve_analogs` and is always a *candidate*.
        """
        hits = [e for e in self._by_id.values()
                if e.status == STATUS_CERTIFIED and e.arch == arch and e.matches(features)]
        hits.sort(key=lambda e: (e.effect, e.specificity()), reverse=True)
        return hits[:top_k]

    def retrieve_analogs(
        self,
        features: Mapping[str, float],
        arch: str,
        top_k: int = 8,
    ) -> List[KnowledgeEntry]:
        """Candidate mechanisms lifted from other architectures that match.

        Returned as *hypotheses*: the caller must rewrite predicates for ``arch`` and
        certify with a probe before they can be prescribed.
        """
        hits = [e for e in self._by_id.values()
                if e.arch != arch and e.matches(features)
                and (e.analog_of is not None or "analog" in e.tags)]
        hits.sort(key=lambda e: e.effect, reverse=True)
        return hits[:top_k]

    def negative_matches(
        self,
        features: Mapping[str, float],
        arch: str,
        top_k: int = 8,
    ) -> List[KnowledgeEntry]:
        """Refuted entries whose failure conditions currently hold."""
        hits = [e for e in self._by_id.values()
                if e.status == STATUS_REFUTED and e.arch == arch and e.matches(features)]
        hits.sort(key=lambda e: -e.specificity())
        return hits[:top_k]

    def lift_analog(self, entry_id: str, target_arch: str, new_id: str) -> KnowledgeEntry:
        """Create a *candidate* entry for ``target_arch`` from a certified source.

        Predicates are inherited verbatim and must be re-validated on the target
        hardware; the status is ``candidate`` so it can never be prescribed before
        a probe certifies it.
        """
        src = self._by_id[entry_id]
        lifted = KnowledgeEntry(
            id=new_id,
            arch=target_arch,
            mechanism=src.mechanism,
            predicates=dict(src.predicates),
            effect=src.effect,
            status=STATUS_CANDIDATE,
            evidence=f"analog of {src.id} ({src.arch})",
            cost=src.cost,
            recipe=src.recipe,
            limitations=src.limitations,
            analog_of=src.id,
            tags=list(set(src.tags + ["analog"])),
        )
        return self.add(lifted)

    # -- persistence -------------------------------------------------------- #
    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), sort_keys=True)
                         for e in self._by_id.values()) + ("\n" if self._by_id else "")

    @classmethod
    def from_jsonl(cls, text: str) -> "KnowledgeGraph":
        entries = [KnowledgeEntry.from_dict(json.loads(line))
                   for line in text.splitlines() if line.strip()]
        return cls(entries)

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())
