# Prior supply lane discipline (WP8)

The prior-supply line — the teacher-side SOTA survey (`skills/sota-survey/`),
offline dossiers (`docs/prior-art/`), the repo prior library (`priors/`) and
the cross-generation carry (`tasks/<op>-g<k>/carry/`) — feeds hypotheses to the
teacher before M0 freezes. This page is the discipline that keeps that line
from contaminating everything the framework measures. The mechanics live in the
skill and the driver; this page is the three rules reviewers hold the line on.

## 1. Lane separation (contamination defense)

- A run with any prior lane on (`[agent.teacher] prior_survey` / `prior_art`,
  or a `carry/` directory) is a **survey-lane** run. Its report says so, and it
  is never pooled with goal-only / source baseline runs: the comparison a paper
  or a board post claims must be within one lane.
- FA4, DeepGEMM, CUTLASS et al. are public code — a teacher that has read a
  mechanism description cannot be distinguished from one that re-derived it.
  Lane labeling is the honest answer: the number stands, its provenance is
  declared. A baseline column implicitly claims "no mechanism descriptions were
  read"; mixing lanes breaks that claim for the whole column.
- Carry is a lane of its own strength: parent evidence on a child's reference
  tree is precise (the child's reference IS the parent's best), but it is still
  yesterday's measurement — the briefs inject it as hypothesis-reuse material
  only, and the pricing discipline re-prices on the current tree as usual.

## 2. Reference replacement (a full ref transfusion) and "strongest legal reference"

- A self-evolution chain starts from the strongest **legal** reference available
  in its lane: expert-seeded starts (a competition-grade kernel as generation 0's
  reference) are a legal lane of their own — declared as such, never blended into
  a from-scratch chain's numbers.
- "Strongest on the market" is decided by **paired timing measured on the machine
  that will run the chain** (the harness's own A-B-B-A window), not by a README
  or paper number: different clocks, inputs and windows make published ratios
  incomparable at the resolution this framework works at.
- On every generation boundary, re-check the market: a stronger legal reference
  that appeared mid-chain joins as the next generation's reference (its own lane
  note says a swap happened), rather than letting the chain optimize against a
  reference the world has already surpassed.
- The one non-negotiable: whatever the reference, the student still never sees
  an equivalent implementation — the supply line ends at the teacher.

## 3. License and provenance

- Survey targets are **permissive-licensed only**: MIT / BSD / Apache-2.0 /
  CUDA-TensorRT-style licenses fine; **GPL / AGPL excluded** (a mechanism is
  independent of license, but the workflow reads code, and clean-room hygiene
  is cheaper to assert ex ante than to argue ex post). Public papers and vendor
  docs are fine regardless of the implementation's license.
- Every survey entry carries `License/provenance` (repo or paper, license,
  version read) — see the skill's format. Dossier entries: same field.
- Mechanism conclusions that survive measurement graduate into the KB through
  the normal certification; their provenance lands in `docs/kb-provenance/`
  like every other certified entry (the survey is where a reading came *from*,
  the probe is why anyone believes it).
