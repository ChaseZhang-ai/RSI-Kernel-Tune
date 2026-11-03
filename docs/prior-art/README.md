# docs/prior-art/ — offline prior-art dossiers (teacher read-set only)

An offline, human-curated alternative to the run-time SOTA survey
(`skills/sota-survey/`): a dossier a person wrote once, a task declares with

```toml
[agent.teacher]
prior_art = "prior-art/<op>.md"     # relative to the task directory
```

and the driver injects into the distill/recon briefs' PRIOR-INPUT section
(`{PRIOR_ART}` face). Nothing here is read by a student; nothing here is
measured until a run measures it.

## The one rule that shapes the format

**Decisions, not implementations.** A dossier may say "persistent scheduler,
raster order N-major with group-16 zigzag, cluster 2×1×1 with A multicast" —
it may not carry a function, a kernel body, or a line-by-line transcription of
one, from any source. Same rule as the milestone's L0 boundary, same reason:
the student implements from decisions, and a code body in any teacher-facing
file is one careless diff away from the student's tree.

## File format

One `<op>.md` per operator family, entries in the survey skill's shape
(`skills/sota-survey/SKILL.md`): decision + expected axis + mechanism in your
own words + license/provenance + verification route (ARCH_PROBE recipe /
`sweep:` starting values / probe). Skeleton hypotheses: 2–3, each different
from the reference the target task starts from. `template-op.md` is the blank.

## Provenance

License notes live in the entries (permissive licenses only — GPL/AGPL
excluded, see `docs/prior-supply.md`); run-facing provenance (what a run did
with an entry, what it measured) belongs to that run's evidence and to
`docs/kb-provenance/` once it is certified knowledge.
