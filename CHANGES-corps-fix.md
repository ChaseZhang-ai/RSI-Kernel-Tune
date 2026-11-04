# CHANGES — corps fix batch B (guards / adversarial surface)

Branch `feat/corps-fix-b`, base 8977b63. Six B-class bugs from the audit-r2
triage, each reproduced against the pre-fix code in /tmp before the fix and
re-run clean after; regression tests ported from the archived upstream fix
batch as `tests/test_guard_fixes.py`.

## Fixes (one commit per bug)

### BUG-SEED-07 (P0) — tree-root shadow modules hijack the harness — b2bfdff

A student delivery consisting only of `sitecustomize.py` / `json.py` at the
tree root passed all three guard layers (the delivery diff, the per-iteration
hygiene, the job overlay), rode every job tree, and was imported by the bench
itself: `gpu_entry.sh` put `$WORKSPACE` FIRST on PYTHONPATH, and the monitor
hook's `_chain()` chained whatever `sitecustomize` came next — the student's.

* `loop/harness_guard.py`: `ROOT_PY_PATHSPEC = "*.py"` joins
  `HARNESS_PATHSPECS` for the git calls (diff / ls-files take the wildcard;
  only the depth is narrowed, by `is_harness_path` — a root-level module is
  harness, the kernel file and `notes/` stay exempt, deep modules stay the
  student's). `tree_harness_files` lists the whole tree and narrows in Python:
  `git ls-tree` takes no wildcard pathspec.
* `cardkit.py`: the generated entry appends `$WORKSPACE/eval` and `$WORKSPACE`
  behind the inherited PYTHONPATH, so the monitor's hook directory (prepended
  by the runner) stays in front and a surviving root sitecustomize cannot
  overtake it. `-I` is NOT usable here — it would kill the sitecustomize
  injection mechanism itself.
* `benchkit.py`: the fresh-recheck / replay-probe children inherit the same
  order.
* `monitor/runtime/hook/sitecustomize.py`: `_chain()` drops student-owned
  sys.path entries (the deployed `<ws>/.pk_monitor/runtime/hook` layout derives
  the workspace) before chaining — only an image-shipped sitecustomize is
  handed over.

Reproduction: `/tmp/seedexp/exp_seed07.py` (root shadows passed all three
layers and rode the job tree; after the fix the delivery diff and the drift
name them and the overlay drops them), `/tmp/seedexp/chaindemo` (with the
workspace first on PYTHONPATH a bare `python3` ran the student's
sitecustomize; with the hook dir first and the fixed `_chain`, the hook
installed and an image-shipped sitecustomize was chained instead).

### BUG-SEED-08 (P1) — library scan escapes — 013d5a1

The scan matched one name after a line-anchored `(?:from|import)`;
`import os, flashinfer`, `a = 1;import flashinfer`, `importlib.import_module`,
`__import__`, `exec('import flashinfer')` and a `load_inline` cuda_sources
string all screened as library-free — a library-call kernel could become the
oracle. The import rules now walk an AST over the source as-is (every alias,
dynamic imports with literal targets, exec/eval recursed one level,
source-carrying keyword arguments scanned); an unparseable source falls back
to a statement-splitting regex (semicolons, continuations, parenthesized
from-imports) that checks every name. A variable target stays invisible.

Reproduction: `/tmp/seedexp/exp_seed08_09_c3.py` (all five escapes returned
`[]` pre-fix; all flagged post-fix; the innocent kernel's finding unchanged).

### BUG-C3-01 (P1) — torch C++ API forms unscreened — aee7adf

`kernel.cu` candidates computing through `torch::matmul` / `at::mm` /
`at::softmax` / `torch::ops::x` screened as library-free. The torch compute
and nn op lists moved to shared constants; `torch::/at::/c10::` and
`torch::ops::` rules join `_LIB_PATTERNS`, mirroring the python ones.

Reproduction: same script (all four C++ shapes returned `[]` pre-fix; flagged
post-fix; an innocent CUDA kernel stays clean).

### BUG-SEED-09 (P1) — load_entry shadowed the reference — c05a3f4

`load_entry` put the kernel's directory at `sys.path[0]`; the reference loads
first and runs later in the same process, so a kernel-directory `json.py`
resolved ahead of the stdlib for the reference's own lazy imports (measured:
a forged `helper` module resolved to `'kernel-forged'` instead of
`'reference'`). The directory is now appended — siblings still resolve, the
stdlib and site-packages stay in front.

Reproduction: `/tmp/seedexp/exp_seed08_09_c3.py` (shadowed True → False,
`ref_dir` now precedes `k_dir` on sys.path).

### BUG-C5-02 (P2) — python load failure: bare traceback, no JSON — e3d5a53

Only a `.cu` suffix failure was structured; a python kernel's module-level
syntax error / missing entry / broken sibling was re-raised and the generated
bench.py has no catch — the job died with a traceback and no `=== JSON ===`
block. `_impl(problem, "ref")` sat outside the try, so a broken reference
killed every mode the same way. Both load sites now land in the structured
channel: `.cu` keeps COMPILE_ERROR, python becomes IMPORT_ERROR, the
reference becomes REFERENCE_ERROR (pass=False, why truncated, rc 0).

Reproduction: `/tmp/seedexp/exp_c5_02_03.py` (pre-fix `RAW SyntaxError … NO
JSON AT ALL`; post-fix rc=0, json emitted, status IMPORT_ERROR; a reference
without its `make_inputs` yields REFERENCE_ERROR).

### BUG-C5-03 (P2) — empty input_groups judged PASSED — d46c156

`task_correctness` never entered its loop for an empty `input_groups` and
fell through to pass=True with 0 checks; `--mode all` handed timing an empty
set list (ValueError), and `ncu_window` raised ZeroDivisionError on
`calls // len(pairs)` before `max()` could mask it. Empty groups are now an
INPUT_ERROR verdict and the profiler exits rc 3. `timing.py` itself is
untouched — its explicit "needs at least one workload" ValueError is the
precedent this fix makes unreachable (`measure_paired` stays frozen).

Reproduction: `/tmp/seedexp/exp_c5_02_03.py` (pre-fix `pass=True, checks=0`;
post-fix `pass=False, INPUT_ERROR`; ncu window exits 3 on the same card).

## Test port — tests/test_guard_fixes.py — d0adef9

Ported (18 tests, all green): the four tree-root guard tests; the import-scan
tests (commas / semicolons / importlib / `__import__` / exec / unparseable
source / quiet-on-clean-and-CUDA); the C++ form and load_inline-string tests
(new, this batch); the load_entry append test; the IMPORT_ERROR /
REFERENCE_ERROR verdict-JSON tests and the empty-groups INPUT_ERROR / NCU
refusal tests (new); the gpu_entry PYTHONPATH order; the two monitor-hook
chain tests (new); the precision stress record (ported as-is).

Dropped from the archived file, with reasons:

* `test_fib_judge_refuses_a_non_finite_reference` — the board lane
  (`fib_judge`, `FIB_DEFAULTS`, `SUITE_TASK`) was deleted with the board
  (WP24); `task_error` covers the non-finite reference.
* `test_host_path_note_names_both_directions` — SEED-23, assigned to another
  engineer; porting it here would stay red. `timing_audit.py` untouched.
* The timing-aggregate trio (`_geomean` degrade, workload/ref aggregate
  degrade) — NOT ported: this tree's `_geomean` skips None entries and
  `workloads_performance` crashes on a 0.0 host reading
  (`round(None)`). That is the archived aggregate fix not carried over,
  inside the frozen `timing.py` surface (only the ncu divide point may move
  in this batch) — it needs its own task.
* `test_measure_with_mismatched_output_counts_is_unavailable` — NOT ported:
  this tree's `precision.measure` silently zip-truncates a length mismatch to
  `pass`. Same situation: the guard was never carried, precision constants
  are a frozen surface, and the fix is outside this batch's list.
* `test_task_stress_rows_draw_make_inputs_not_the_sol_heuristics` — pins
  `gen_inputs` routing the task's `make_inputs`; peregrine dropped that
  channel BY DESIGN (docstring: custom_inputs_fn is accepted and ignored; the
  task draws through `task_inputs`).

Also updated: `tests/test_harness_guard.py` — `evaluation.py` moves to the
harness side of the parametrize (a tree-root module), plus
`sitecustomize.py` / `json.py` cases.

## Intentional behavior changes to be aware of

* Students may no longer ADD tree-root `.py` files: the delivery check
  REVERTs them, the hygiene restore removes them, and the job overlay drops
  them. A card's own tracked root modules ride as before (byte-identical
  overlay). Sibling imports of the kernel still work (the kernel's directory
  is appended to sys.path by `load_entry`).
* gpu_entry.sh PYTHONPATH order: inherited front (monitor hook, image paths)
  → `$WORKSPACE/eval` → `$WORKSPACE`. A card relying on its root shadowing an
  image module would now lose that — no generated card does.
* New verdict statuses: IMPORT_ERROR (python kernel load failure),
  REFERENCE_ERROR (reference load failure) beside the existing COMPILE_ERROR.
