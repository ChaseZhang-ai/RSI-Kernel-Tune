# sota-survey — the teacher's public-SOTA prior survey (teacher-side only)

**Owner: the teacher (meta agent) at the recon (blackbox) / distillation (source) step, before M0
freezes. A student never uses this skill** — the student is offline and never reads any prior
input; if you are the student, stop reading here.

## What it is

The M0 architecture freeze decides this run's performance ceiling, and until now its candidates
came from one place: back-solving (or distilling) the single reference this run was handed. The
world's best public implementations — and the literature explaining why they are fast — were not
on the table. This skill is the compliant channel: **you go online, survey the public state of
the art for this run's operator family, and write `notes/prior_art.md`** — decisions and
hypotheses only, each of which the loop then measures through its existing channels before
anything is believed.

Where each survey target comes from (license-permissive; read and describe freely, never copy):

- FlashAttention (FA3/FA4), FlashInfer, cuDNN-frontend docs — attention family;
- DeepGEMM, CUTLASS examples, ThunderKittens, Liger — GEMM / norm / elementwise families;
- the architecture sections of the papers behind them, and vendor docs (PTX ISA, CUDA C++
  programming guide, cuBLAS/cuDNN tuning guides).

## The four hard boundaries (all mechanical, none advisory)

1. **deny_urls stands.** The run's `[reference] deny_urls` (and, for a blackbox run, everything
   it implies) applies to this survey exactly as to any other lookup: never query the sealed
   reference implementation's own upstream repository or its mirrors, raw/api endpoints or
   package-manager downloads. The harness refuses them; trying alternate spellings is a leak.
2. **No implementation bodies anywhere.** `notes/prior_art.md`, the milestone, the KB and every
   probe carry mechanisms **in your own words** — a copied or transliterated function, a renamed
   snippet, a pseudo-code transcription of someone's kernel: all forbidden. "Decision + parameter
   values + why" is the currency (L0, the same rule the milestone already runs under).
3. **Borrowed authority does not count.** A survey conclusion — a skeleton hypothesis, a config
   table, a reported speedup — **changes no gate number, no M0 freeze, no stopping judgment until
   THIS run has measured it** (ARCH_PROBE / SWEEP / probe / lever pricing). Unmeasured claims may
   enter evidence only as `[ESTIMATE]` candidates. A README benchmark is not a measurement.
4. **Lane labeling.** This run is a **survey-lane** run; its report and any comparison against a
   non-survey run must say so (see `docs/prior-supply.md`). A gain the survey lane posts is
   reported as "survey-lane", never merged into a goal-only/source baseline's column.

## The deliverable: `notes/prior_art.md`

One candidate decision per entry, in exactly this shape (repeat per entry; 5–15 entries is
normal, keep it under one page of mechanism prose per entry):

```
### <n>. <short decision name>
- Decision: <skeleton hypothesis | config starting values — concrete values, e.g.
  "cluster 2x1x1, A multicast; STAGE {3,4} sweep start 3">
- Expected axis: <which decision/phase of THIS run's operator it moves, and the mechanism's
  expected direction — e.g. "epilogue store pressure; fewer global round-trips on large K">
- Mechanism (own words): <why this is believed to work; cite the public source by name and
  what IT reports, not what its code does line by line>
- License/provenance: <repo or paper, license name (permissive only — GPL/AGPL excluded),
  and which version you read>
- Verification: <one of>
    - ARCH_PROBE: recipe notes/arch_<n>.json — skeleton timing this hypothesis
      (`ARCH_CANDIDATES: dual` divergence line in M0; skills/arch-probe protocol)
    - SWEEP: `sweep:` field on decision <id> — config table as starting values
      (skills/sweep-decision protocol, ≤ 8 variants)
    - PROBE: notes/probe_<x>.py — mechanism-verification probe on the ref or best tree
      (the verdict-probe protocol)
```

**Skeleton hypotheses: 2–3, and each must differ from THIS run's reference architecture** (a
hypothesis that restates the reference teaches nothing new; the probe bracket is where a
different topology gets its price tag). Config tables ride decisions that already exist in the
milestone skeleton — a starting value with no decision to attach to is dropped.

If an offline dossier was already declared for this run (`[agent.teacher] prior_art`), read it
first and reconcile: keep its entries, mark what your survey adds or contradicts.

## What happens next (so you write the right thing)

The driver consumes the milestone's `ARCH_PROBE_REQUEST` lines and the decisions' `sweep:` fields
through the standard channels; the winner of a probe bracket is what M0 freezes. Nothing in
`notes/prior_art.md` is read by the student, and nothing in it is true until the loop says so.

## Map every uncovered mechanism to a candidate row (WP18)

Every mechanism the survey surfaces that this repo's knowledge base does not already cover must end
the survey as one more line at the bottom of `notes/prior_art.md` (one line per mechanism):

    CANDIDATE-DRAFT: <mechanism-name> | <tree(s) or "any tree a task demands"> | <the axis it moves> | <why it is worth a probe>

The end-of-run harvest lands each such line as a `DRAFT` row in the candidate queue
(`hw_kb/CANDIDATES.md`) after a mechanical dedup against the three trees' indexes — a mechanism a
tree already covers is a dup and is not landed. A draft row certifies nothing: it states only where
knowledge is missing, and it becomes knowledge only when a run's GPU probe passes.
