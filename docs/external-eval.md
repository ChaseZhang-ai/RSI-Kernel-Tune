# External evaluation export (placeholder)

Placeholder — WP27 fills this document. Planned content, per the WP25-era adjudications:

- the SOL-ExecBench free B200 evaluation protocol, recorded faithfully as its documentation
  states it (submission format, judgment criteria; its clock/measurement protocol cited from
  its docs);
- the user-leased-card self-evaluation protocol (the framework's own strict measurement:
  locked clocks + paired timing);
- the boundary red lines stay as they are today: the export tooling is standalone, opt-in,
  human-run, never inside any automatic loop; the `src/` zero-network-egress audit scope is
  unchanged (`scripts/eval-export`, when it lands, is an explicitly-exempt, labeled
  "manual evaluation export").

Note: the free part is the **evaluation**, not the optimization process — the search loop
needs interactive timing (paired A-B/A, NCU), which a submission-based evaluator cannot
serve as a timing oracle.

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
