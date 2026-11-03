# lever-pricing

Measured prices of candidate levers on the current best kernel, in one GPU job: each lever
added alone, removed alone from the full stack, and all of them on. It is the only reading
that also shows a lever worthless alone but required by the others (about zero when added,
negative when removed), or one visible only once the others are subtracted.

The teacher writes the candidates into `notes/levers.json`; the driver adds the kernel file
and the number of rounds and runs the frozen recipe on the best tree before it asks for a
diagnosis (`python3 notes/lever_pricing_run.py notes/levers.json`):

```json
{"kernel_file": "kernel.py", "rounds": 3,
 "levers": [{"name": "ring_per_stage",
             "edits": [{"old": "<exact text in the kernel file>", "new": "<replacement>"}],
             "expect": "producer refills each stage as its consumer releases it",
             "in_stack": true}],
 "variants": [{"name": "ring+lane", "levers": ["ring_per_stage", "lane_owned"],
               "question": "is lane_owned a dependency on the ring?"}]}
```

The full format (`in_stack`, `variants`, `ablations`) is in the script's docstring.

An ablation can be requested the same way with `PROBE_REQUEST: notes/<name>.json`
(`ablations` only). Not a student tool.

## WP17: in-process bench switching (default on, `FK_SCREENING_INPROC=0` to kill)

Every timed reading this tool takes (the pricing pass, the ablation bench, and the sweep /
prescreen jobs that import this file) routes through one long-lived bench worker process:
torch and the card's eval modules import once, and each call re-executes `eval/bench.py`
in that process — the subprocess's protocol per call, minus the per-call cold start. A hung
or crashed call is killed as a process group and reads exactly as the subprocess timeout
always read (NODE-DIRTY, one driver re-buy). The cubin-dump calls (`keep_cubin=True`) stay
per-call subprocesses. Set `FK_SCREENING_INPROC=0` in the job env for the historical
behavior, byte for byte.
