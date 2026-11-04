# NCU-trace skill — let the meta agent see "how the performance metrics change" in hardware ground truth

> **Operator-agnostic.** Gives the meta agent a **cross-iteration evolution of NCU metrics**, so it judges the
> binding constraint from **measured hardware ground truth** instead of trusting the student's self-reported
> `binding_constraint` in its prose. Student prose turns "what wall it hit" into one sentence; behind the same
> sentence there may be a bank conflict, a register spill, or a memory-bound kernel — **only NCU can tell them apart**.

## Invariant: the student records NCU every round (no more `profiler={}`)

In every round's GPU eval the inner driver **must, besides correctness/perf, also run one NCU profile**
(the operator's `gpu_entry.sh ncu` task) and parse it with `peregrine_kernel.inner.ncu.profiler_from_ncu(csv)` into a
compact key-metrics dict written to that round's `IterationRecord.profiler`. **Without this step the meta agent
can only aggregate student prose — the root cause of past diagnoses leaning too heavily on student signals.**

Two kinds of empty profiler are **expected** and are not missing data: correctness-broken rounds (the bench
crashed; NCU legitimately has nothing to collect); and build-phase `NEUTRAL` rounds (the whole machine has not
had its first PASS yet; under the build-phase semantics of POLICY §3 the driver skips the whole-machine eval
and defers verification to the gate row).

`profiler_from_ncu` automatically picks the **slowest kernel** (largest `gpu__time_active.sum`) as the bottleneck
kernel and keeps only the decision-relevant metrics (below).

> **Reading regime (WP9)**: the collection runs `--cache-control none --clock-control none`, so these
> readings describe the timing protocol's regime — warm L2, the board's own clock state — not a flushed,
> base-locked machine. Historical records (pre-WP9) were collected flushed + base-locked: the two regimes
> do not mix, and `l2_hit_rate_pct` / SOL pct especially are only comparable within one.

## How to use it (the meta agent calls it directly)

```bash
peregrine <op> ncu-report       # a direct summary: one bottleneck verdict per round + key metrics tagged by severity
peregrine <op> ncu-trace        # the raw per-iteration metric evolution table (when you need to see how values change)
```

- **Look at `ncu-report` first**: it compresses dozens of NCU metrics into "one verdict per round (e.g. SMEM
  bank-conflict bound / register-spill / memory-bandwidth bound / latency bound) + a few key metrics tagged
  `severe`/`elevated`". This is the layer that lets you **see the key parts directly without studying thresholds**.
- When you need to see **how values move across iterations** (whether a decision really moved the metric it was
  supposed to move), then look at the raw `ncu-trace` table.
- You can also read `profiler[]` in `operators/<op>/student/digest.json` (same source, includes KEEP and REVERT,
  tagged with the verdict).

## Decision-relevant metrics → diagnostic meaning

| Column | Meaning | A high value means |
|----|------|-----------|
| `bankconf_st` / `bankconf_ld` | shared-mem store/load bank conflicts | **A degenerate / conflicting smem layout.** If the milestone names a "conflict-free" decision but this value is still large ⇒ the student delivered a **degenerate fallback** (knows what to do, cannot use that hardware interface) → **G3: add an hw_kb recipe, do not change the milestone** |
| `smem_wavefronts_st` / `smem_wavefronts_ld` | shared-mem store/load wavefront **volume** (≠ conflicts) | No absolute threshold; **only comparable against the reference signature**: a total far above the reference ⇒ a **scalar / per-element access path** (wrong copy atom, not a swizzle problem — conflicts can be zero while the volume tax is still paid, and the read side in particular is invisible to every conflict metric). The fix = change the atom (STSM/LDSM/direct descriptor reads): G1/R1 pointing at the milestone's path contract, **not** a G3 swizzle tweak |
| `spill_st_sectors` / `spill_ld_sectors` | local-mem store/load (= register spill traffic) | **Register pressure.** Read together with `regs_per_thread`; if a foundation choice pins the footprint above the spill cliff and the in-phase levers are exhausted ⇒ **R1 foundation ceiling** |
| `regs_per_thread` | registers per thread | near the limit + high spill = register-limited |
| `tensor_pct` | tensor-pipe active share | low = the tensor cores are not fed (a latency/scheduling problem, not a compute problem) |
| `warp_active_pct` | warp activity | low = latency-limited (held back by stalls) |
| `stall_long_sb` | long-scoreboard stall | ⚠ **In an mbarrier-pipelined kernel this bucket is dominated by mbarrier/TMA spin-waits (NANOSLEEP)** (measured: almost all NANOSLEEP samples land here), not "waiting for DRAM" — optimizing for memory latency goes the wrong way; the real memory-access criterion is `dram_pct` and spill |
| `stall_barrier` | barrier stall | **In a WGMMA kernel the bulk is `wgmma wait_group` waiting** (the vast majority of DEPBAR waits land here) + named barriers — read it as a "wait depth / overlap quality" signal, not only as CTA synchronization imbalance |
| `stall_mio` | MIO throttle (smem / special function units) | high = congestion on the smem path or the SFU |
| `dram_pct` | DRAM throughput as a share of peak | high = memory-bound; low = not a bandwidth problem |
| `compute_sol_pct` | compute-side SOL (share of peak sustained compute throughput) | high = the compute side is near its ceiling — this axis is done; low together with low `tensor_pct` = latency-bound, go to the stall ledger; read FIRST in the digest's `sol` header, before the stall accounting |
| `memory_sol_pct` | memory-side SOL | high = the memory path saturates (check `dram_pct` beside it: a high memory SOL with a low `dram_pct` is L1/L2 pressure, not DRAM); low with low `compute_sol_pct` = latency-bound |
| `l2_hit_rate_pct` | L2 hit rate | **low** = a working set that should sit in L2 keeps missing ⇒ tiling / ordering defect (a reuse problem), not a bandwidth wall — raising bandwidth is the wrong fix |
| `waves_per_sm` | waves per SM (tail-wave quantization) | **< 1** = most SMs never saw work — a grid/launch problem before any inner-loop tuning; a fractional last wave costs its whole slot (~148-SM parts: x.9x ≈ 10–20% tail loss) |
| `global_ld_sectors` ÷ `global_ld_requests` | sectors per global-load request = the coalescing factor (32 = fully merged) | **low** = an uncoalesced / scalar global access path; L2 can absorb the re-reads so `dram_pct` stays low while the kernel still pays the request volume |
| `stall_lg_throttle` | LG throttle stall (local/global LSU queue full) | high = the long-global-queue path congested (long-tail sibling of `stall_mio`); absent on some archs — collection filters the request per device |
| `resusage_regs` / `resusage_smem` / `resusage_cstack` | build-time readings off the shipped binary (`cuobjdump -res-usage`): registers / STATIC smem / per-thread stack | `cstack > 0` = the compile already committed to a local frame (spill cliff) — visible a full iteration BEFORE NCU confirms it; `smem` is static only (dynamic smem is a launch attribute: `smem_dynamic`) |

**Arch note:** the metric request is filtered per device before collection (the card's gpu_entry.sh intersects the desired set with
`ncu --query-metrics`; `launch__*` names pass unconditionally — they are launch attributes, never listed by the query). A column missing
from the trace means the arch does not answer to it (or the profile predates the expansion), not that it was clean.

## How to read "the course of change" (this is where the skill earns its keep)

Do not look at single points; look at **whether, after a decision lands, the metric it was supposed to move actually moved**:

- After an M5 decision that claims "drop a fragment to reduce spill" lands, if `spill_*` / `regs_per_thread`
  **did not go down** ⇒ that lever does not bind: spill is not the current real bottleneck, or this cut is below the cliff.
- After a decision that claims a "conflict-free layout" lands, if `bankconf_*` **is still large** ⇒ the student did
  not produce the conflict-free version (degenerate fallback) ⇒ **G3 / KB**, not a milestone defect.
- The whole perf curve is flat while `bankconf_st` and `spill_*` are **both** high ⇒ do not conclude from one number:
  when two real bottlenecks coexist, fix the **more foundational / cheaper** one first (foundation first), and use the
  next round's NCU to verify it really lowered the corresponding metric.

## Hard rules

- **NCU first, student prose second**: the self-reported `binding_constraint` is only a "navigation hypothesis";
  load-bearing conclusions must be confirmed by NCU metrics (matching the evidence binding in `meta_milestone.md` /
  `diagnose_prompt.md`).
- **Metrics are cited only from the trace/digest**: do not assume from memory what bottleneck an operator "should"
  have (operator-agnostic L1).
- **Missing NCU data**: if a round's `profiler` is empty, the trace skips it; in the diagnosis say "this round has no
  hardware ground truth; the conclusion is downgraded to a prose inference" — do not pretend to have evidence.
