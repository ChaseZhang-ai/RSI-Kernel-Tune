# KB candidate queue (pending certification)

The knowledge base is demand-driven: a recipe is written when a run hits the gap, so the gaps nobody
has hit yet do not exist in any tree — even when they are known to be expensive. This file is the
queue of those known gaps. It is **not** the KB, and it is **not** a plan that certifies anything:

- **A queued row certifies nothing.** The only entrance into a tree stays what the per-tree README and
  `TEMPLATE.md` already require — a GPU probe returning `PROBE PASS`, or, for numerically invisible
  knobs, a shipped KEEP whose own NCU census shows the contract axis moved. Nothing in this file may
  be read as a mechanism conclusion: a row states where knowledge is **missing** and why closing it is
  believed worth a probe, never how the hardware behaves.
- **How a row clears**: a run lands / certifies the mechanism and its finish-time KB write-up covers
  it — the harvest then flips that row's `Status` to `DONE` and appends the new entry's link. A row
  may also be rewritten or dropped when a probe shows the gap was mispriced; say why on the row.
- **Who edits**: the harvest round (queue-hit), and people. The student never reads this file; it is
  not in any read set, and it is not licensed to break the KB rules — the leak lint scans it like
  every other file here.

| Candidate entry | Tree(s) | Priority | Basis | Status |
|---|---|---|---|---|
| `tcgen05_blockscaled_mma_sf_tmem` — certify the fp4 (e2m1) operand face and the CTA-pair cells | `cuda/sm100` | High | The same-named entry in `cutedsl/sm100` is fully certified (fp4 + pair cells), while the `cuda/sm100` entry reads "fp4 / pair uncertified" (`cuda/sm100/index.md`); transcribing from the certified tree and re-certifying in CUDA form halves the work a from-scratch probe would pay | TODO |
| `tma_multicast` — cluster TMA multicast operand load | `cutedsl/sm100`, `cuda/sm100` | High | Cluster multicast is a first-order supply lever at B200 scale, and neither sm100 tree has the entry: `cutedsl/sm90/tma_multicast_cluster.md` is a draft (sources checked, GPU probe pending), and the sm100 trees carry only the **non**-multicast pair-load entry (`*/tma_2cta_nonmulticast_operand_load.md`) | TODO |
| Grid-level split-K / stream-K work decomposition | `cutedsl/sm90`, `cutedsl/sm100`, `cuda/sm100` | High | Only a warp-level split exists (`cutedsl/sm90/mma_sync_subtile_broadcast_splitk.md`); the grid-level decomposition (K-slices per CTA, semaphore-reduced partial epilogues, leftover-SM absorption) is the standard machinery of the public grouped-GEMM stacks (DeepGEMM, CUTLASS stream-K) and is unrepresented in all three trees | TODO |
| Wave quantization / tail waves (persistent scheduling + CTA tile height vs the SM count) | `cutedsl/sm90`, `cutedsl/sm100`, `cuda/sm100` | Medium | No tree has an entry reading the tail-wave occupancy cliff — persistent scheduling and tile-height choice against the 132-SM (sm90) / 148-SM (sm100) divisibility are decided blind today; symptom: last-wave-linger shapes where NCU shows SMs parked past the tensor-busy plateau | TODO |
| L2 access policy window / persisting set-aside (`cudaAccessPolicyWindow`, persisting-L2 carve-out) | `cuda/sm100` | Medium | The sm100 entries only touch inline-PTX `.L2::cache_hint` spellings (`cuda/sm100/cp_async_bulk_1d_mbarrier_gather.md`, `cutedsl/sm100/tma_param_desc_inline_ptx.md`); the window/set-aside API surface (`cudaStreamSetAttribute` + `cudaLimitPersistingL2CacheSize`) has no entry. A `cutedsl` row joins only if the DSL exposes an API — check before queuing it there | TODO |
| PDL launch side (`griddepcontrol.launch_dependents` + the launch attribute) | `cutedsl/sm100`, `cutedsl/sm90` | Medium | Both cutedsl trees miss the launch-side spelling: `cutedsl/sm100` has only the wait-side reading (`post_wait_gather_single_trip.md`), and the `cutedsl/sm90` index row `griddepcontrol_pdl` is a TODO; the `cuda/sm100` tree's `pdl_griddepcontrol_launch.md` can seed the transcription + re-certification | TODO |
| 2:4 structured sparsity MMA (sparse `mma` / `tcgen05` sparse paths, metadata layout) | any tree a task demands | Low | No entry anywhere (grep-clean across all three trees); queue it only when a task actually needs sparsity — an unhit gap stays cheaper as a row than as an unprobed entry | TODO |
