# Triton hardware / language knowledge base (hw_kb/triton)

> This is the knowledge base for **Triton mode** (loop language `triton`: the student writes one
> Python `kernel.py` whose compute is `@triton.jit` kernels and whose `run()` launches them — the
> leaderboard's `spec.languages == ["triton"]`). The CuTe DSL mode reads `hw_kb/cutedsl/`, the
> CUDA mode `hw_kb/cuda/`; the three trees are independent (`src/rsi_kernel_tune/loop/lang_mode.py`).

An **operator-agnostic** library of language-mechanics and hardware-interface entries. It is the
**only** artifact of the loop allowed to contain code — because "how a `tl.dot` lowers / how a
`tl.load` pays for its mask / how a launch's grid axes map to CTA order" is a mechanism, not some
operator's algorithm.

Triton is a compiler, not an assembly language: most of what a Triton student must know is
*what the compiler does with what you wrote* (which MMA the tile picker chooses, when the
pipeliner allocates a stage, where the accumulator lives) plus the handful of knobs that are
actually yours (`num_warps`, `num_stages`, `BLOCK` sizes, grid shape, `tl.dot` operand dtypes,
`input_precision`, cache modifiers).

## Invariants

- **Mechanisms only, never operator algorithms.** "How `num_warps` changes the MMA the tile picker
  selects" ✅; "how this operator's three matmuls are wired" ❌ (that is the per-op milestone's).
- Every entry is self-contained, minimal and runnable against a real Triton install, and carries
  its own sources. Example code comes from a **verifiable source** (official Triton
  documentation / upstream API symbols / a minimal example run in this repo); nothing is made up.
- **The evidence chain is split into two layers by reader**: the entry's "Sources" is read by the
  student and may cite only official docs, public-infra API symbols and this repo's probes; the
  narrated provenance (which public guide a passage was derived from, what was measured where)
  goes to `docs/kb-provenance/triton/<entry name>.md`, which is in no agent's read set. The
  mechanical gate is `src/rsi_kernel_tune/loop/kb_lint.py`.

## Certification status

This tree starts the way the CUDA tree did: from **methodology entries**. Two statuses appear:

| status | meaning |
|---|---|
| ✅ Verified | a GPU probe in this repository certified the claim (`PROBE PASS ...`), and the entry's `Certification source` line says which |
| **[PENDING-CERT]** | written from public documentation / a public MIT-licensed Triton guide; **not** certified by a probe here. Treat its numbers as a starting point to verify, never as a reading |

An entry marked `[PENDING-CERT]` is *usable* (the mechanism is right or it would not ship) but every
quantitative claim in it must be re-earned on the target GPU before a milestone leans on it. The
next run that probes the mechanism rewrites the line to ✅ Verified with the probe's verdict, or
retires the entry. Do not cite a `[PENDING-CERT]` number as measured evidence.

## Layout

```
hw_kb/triton/
  TEMPLATE.md            # entry template
  sm100/                 # Blackwell (B200, CC 10.0) — the board's target
    index.md             # "look up by symptom first" table + completion status
    <entry>.md
  sm103/                 # Blackwell Ultra (B300, CC 10.3) — the development GPU
    index.md
  sm90/                  # Hopper
    index.md
  sm86/                  # Ampere (RTX 3060) — a DEVELOPMENT / SMOKE target, never a production one
    index.md
```

**One tree per architecture; there is no shared tree.** An entry belongs to a tree by **what it
depends on**: as soon as it names a mechanism specific to one architecture (the tcgen05/TMEM
accumulator, Hopper's wgmma, Ampere's `mma.sync` family or its absence of TMA), it belongs to that
tree only. An entry that names no hardware unit (grid dispatch order, the shape-keyed knob table,
the deterministic-concatenation trick) is **copied into a tree when that tree's runs need it** — a
deliberate duplication, demand-driven like everything else here: this release ships the full set in
`sm100/` (the target) and the smaller sets the Hopper, Ampere and development-GPU trees carry. A
transcribed copy carries a `Certification source` line naming the architecture its readings came
from; **location is not provenance — that line is.**

**`sm86` is a development / smoke target, not a production target.** It exists so the whole loop can
be exercised end to end on a small card: a kernel that compiles, passes the card's correctness check
and times there has a working call path, but its *numbers* say nothing about the target (Ampere has
no TMA, no clusters, no tcgen05, a different MMA family and a different SM count). The same holds
for `sm103` in the other direction: it is the development GPU, its readings are close to but not the
board's, and a kernel must never be compiled *for* it (see the arch-pin finding in
`src/rsi_kernel_tune/triton_check.py`).

**The read set is enforced by the loop.** A run receives exactly one tree, `hw_kb/triton/<kb_arch>/`:
`KB:<entry>` references resolve only inside it, the leak lints only scan it, and the write-side
prompts register new recipes in that tree's `index.md`. `kb_arch` comes from `[loop] kb_arch` in
`operators/<op>/loop.toml` (or `forge loop --kb-arch`); the tree must exist with an `index.md` or the
run is refused at start (exit 14). Opening a tree for a new architecture = create the directory +
write an `index.md`; no code change.

## What Triton mode does not have

Two harness tools have no channel into a Triton kernel, and the loop says so instead of running them
(`lang_mode.TOOL_UNAVAILABLE`, `docs/triton-mode.md`):

- the **timeline probe** (`_seg` / `FK_SEG` markers + a device clock) — no marker channel for
  `@triton.jit`;
- the **automated orchestration census** (the per-iteration SASS digest that reads the cubin the
  operator tree's own build dumps) — a Triton kernel's cubin is emitted by the Triton compiler into
  its own cache, so the run profiles with plain NCU instead. Its consequence for a milestone: the
  census's contract axes (`cache_ops`, `commit`, `supply`, the NamedBarrier verdicts) are not
  re-read automatically, so a gate row must not be anchored on them.

What *is* available, and worth knowing: NCU profiling (the kernel name is the `@triton.jit`
function's name), stall pricing (NCU PC sampling), lever pricing (source-text edits + the card's own
bench), and **the SASS text itself** — `ncu --import <rep> --page source --print-source sass` on the
profiled kernel, which the `sass_tools` / `sched_live_tools` / `struct_tools` family parses exactly
as it does for any other kernel. A Triton run that wants the SASS/SCHED/LIVE/STRUCT channels writes
its own probe for them. The compiler's own artefacts (`*.ptx` / `*.ttgir` / `*.cubin` under a pinned
`TRITON_CACHE_DIR`) are the other readable channel, and the one that answers "which MMA did it
pick".

## Fill strategy

Demand-driven, like the other trees: the student runs an existing milestone and whichever
mechanism it gets stuck on is the entry filled first. Each `index.md` keeps its own to-be-filled
list; do not pre-fill from an imagined operator.
