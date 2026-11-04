# eval-export — external evaluation submission packager

`pack.py` packages a gated kernel into an external evaluation suite's submission
format. **Manual, opt-in, human-run, offline**: zero network I/O, zero imports from
`src/peregrine_kernel/` (stdlib only — pinned by `tests/test_eval_export.py`), never
part of any automatic loop. Uploading the package is a separate human step.

```bash
python3 scripts/eval-export/pack.py pkg/ \
    --run-dir .artifacts/<task>_<id> --op <op> \   # best KEEP, auto-located
    --definition <suite-dataset-definition-name> --target sol|fib|both
```

Also accepts `--tree <operator tree>` (contract from the card's
`eval/problem/problem.json` or `window.toml`) and `--kernel <file>` (signature via
`--inputs/--outputs/--entry`). Package: the suite's solution JSON(s), the kernel
verbatim, `preflight.json` (lint / compile / correctness report), and
`artifact_sha256.json` (hash manifest + run provenance anchoring the gated bytes).

Protocol, suite documentation, the honesty rules and the full story:
[`docs/external-eval.md`](../../docs/external-eval.md).

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
