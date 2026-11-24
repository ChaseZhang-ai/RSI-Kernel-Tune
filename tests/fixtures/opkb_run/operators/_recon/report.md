# Recon report -- 069_rms_norm

Every observation below is marked with the job it came from. What could not be measured says
so instead of a number.

## 1. Launch configuration [MEASURED, job 1]

LAUNCH-OK kernel=rmsnorm_kernel grid=(2,1,66) block=(384,1,1) ctas=132 cluster=(2,1,1) cluster_max_active=2 smem_dynamic=188628

## 2. SASS [MEASURED, job 1]

SASS-OK kernel=rmsnorm_kernel instructions=2615 opcodes=41 tensor_family=HGMMA

## 3. Schedule and live ranges [MEASURED, job 2]

SCHED-OK isa=mma_sync loop=1 mma_runs=4 barriers=3 ldsm=12 handoff_before_each_ew=[DEPBAR.LE SB 1, BAR.SYNC 2]

LIVE-OK peak_live=168 non_accumulator=40 peak_at=mainloop

## 4. Interpretation (prose -- not a machine verdict)

The reference keeps the weight row resident and re-reads the input rows in a strided pattern;
it appears to use a 256-wide reduction tree with a deferred merge in the epilogue.
DECOY_TOKEN_REFERENCE_PROSE_9f3a
