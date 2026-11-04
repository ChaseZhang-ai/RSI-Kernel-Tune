#!/usr/bin/env python3
"""A `claude -p` stand-in agent (stub) for the WP21 mini smoke.

Same protocol surface as scripts/agent-claude.sh: prompt on stdin, reply on stdout, exit code 0.
The role is told from markers in the prompt; the student turn writes kernel.cu + a record
block on its first call, then EXHAUSTED; the teacher turn gives the minimal legal reply (the
patch turn really appends an entry to milestone.md before reporting the patch protocol line).

Serves only the tasks/wp21mini smoke (chain and dataflow evidence); it does NOT mean agent
behavior is verified — a real-agent smoke needs the machine to configure agent credentials
(smoke defect list, item 1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STATE = Path(".wp21_stub_state.json")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"student_turns": 0, "records": 0}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s))


KERNEL_CU = """// wp21mini stub kernel: y = a + b, one elementwise CUDA kernel launch (at::add).
// fk-cuda-cflags: -gencode=arch=compute_86,code=sm_86
#include <torch/extension.h>

torch::Tensor run(torch::Tensor a, torch::Tensor b) {
    return at::add(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
"""

RECORD = """decision_id:        D0.1
change:             D0.1 single-launch elementwise add via at::add, coalesced fp32 reads
files_touched:      kernel.cu
"""


JUDGE_PY = '''"""WP21 mini judge: y = a + b (float32, exact IEEE add)."""

import torch


def run(a, b):
    return a + b


def make_inputs(axes, group, seed, device):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(axes["n"], generator=g, dtype=torch.float32).to(device)
    b = torch.randn(axes["n"], generator=g, dtype=torch.float32).to(device)
    return {"a": a, "b": b}
'''

DEFINITION = """{
  "inputs": {
    "a": {"shape": ["n"], "dtype": "float32"},
    "b": {"shape": ["n"], "dtype": "float32"}
  },
  "outputs": {
    "y": {"shape": ["n"], "dtype": "float32"}
  }
}
"""

GROUPS = """{
  "tiny": [
    {"name": "std-1", "seed": 1001, "intent": "standard normal draw"},
    {"name": "std-2", "seed": 1002, "intent": "standard normal draw"},
    {"name": "std-3", "seed": 1003, "intent": "standard normal draw"}
  ]
}
"""


def write_judge_files() -> None:
    Path("definition.json").write_text(DEFINITION)
    Path("judge.py").write_text(JUDGE_PY)
    Path("groups.json").write_text(GROUPS)


def main() -> int:
    prompt = sys.stdin.read()
    state = load_state()
    out = ""

    # Smoke evidence: persist the mechanism markers seen in each received prompt (kbmenu render hits, the SOL header line, etc.).
    markers = [m for m in ("KB-MENU", "KB-HINTS", "LEVER-MENU", "CALIBRATION",
                           "compute_sol_pct", "STAGNATION", "LEDGER")
               if m in prompt]
    if markers:
        with open(".wp21_stub_prompt_log.txt", "a") as fh:
            fh.write(f"markers={markers} len={len(prompt)}\n")

    # Role detection (markers from each prompt/command brief's headings and protocol-specific words)
    if "You are writing the correctness judge" in prompt:
        # judge turn: cwd=check/, writes the three-piece set
        write_judge_files()
        out = "judge: wrote definition.json, judge.py, groups.json for wp21mini.\n"
    elif "Initial distillation authoring step" in prompt:
        # distill turn: the milestone lands at the {MILESTONE_PATH} absolute path bound
        # in the driver's prompt (loop/cli.py defaults to operators/<op>/milestone.md while
        # the agent cwd=ws/); ws/milestone.md is written too as a fallback.
        import re as _re
        mm = _re.search(r"(/[^\s`']+/milestone\.md)", prompt)
        targets = ([Path(mm.group(1))] if mm
                   else [Path("../operators/wp21mini/milestone.md")])
        targets.append(Path("milestone.md"))
        body = ("""# wp21mini milestone

## 0. Problem statement and target

- op: wp21mini — elementwise add y = a + b (float32, n = 1048576), one shape `tiny`
  (axes n = 1048576); reference: torch a + b (the architecture reference); hardware: sm86.
- target_ms: 0.0522
- target_window: device
- equivalence: correctness on every input group plus the ratio gate (1.2x the reference).
The target is the reference's device reading divided by the ratio; the reading is anchored
at run start by the ref baseline job (machine-anchored, see the run log).

## 1. Problem contract (Math Spec)

For every i: y[i] = a[i] + b[i], an exact IEEE-754 float32 add. Inputs are arbitrary finite
float32; no reduction, no epsilon, no dtype round-trip.

## 2. Decision DAG + phase order

Phase order: M0 → M1. The ref baseline is an authoring input, not a phase.
[SYNC: none] (single kernel, no cross-role handoff)
[GATHER: none] (no producer/consumer split)
[PHASE: none] (no overlapping milestones)

## 3. Decision details

### M0 — baseline architecture (smoke)

- D0.1 single-launch elementwise add
  - Action: deliver run(a, b) as one CUDA elementwise kernel launch (at::add); one
    kernel in the timed window, coalesced reads, no smem, no extra sync.
  - Purpose: the minimal correct baseline the WP21 smoke evaluates once end-to-end.
  - Trigger: milestone authored; the ref baseline job anchored the target.
  - Cost: one merged eval+NCU job; no measurable resource budget beyond the default
    launch [ESTIMATE].
  - Depends on: none
  - [VERIFY: ncu-direct] kernel_count = 1 in the student's timed window.

### M0 Gate

Correctness pass on every input group. Mechanism-verification line:
  kernel_count = 1 in the student's window vs the reference census
  (vectorized_elementwise_kernel, 6 launches) — expected direction: fewer launches,
  same or smaller device time. [SYNC: none] [GATHER: none] [PHASE: none]

### M1 — smoke termination (non-structural)

- D1.1 hold the kernel and exhaust
  - Action: keep D0.1's kernel; the student declares phase exhaustion after one
    evaluated iteration (the smoke's bounded budget).
  - Purpose: the smoke verifies link & dataflow, not optimization depth.
  - Trigger: D0.1 evaluated (KEEP or REVERT either way the flow is exercised).
  - Cost: zero additional jobs [ESTIMATE].
  - Depends on: D0.1
  - [VERIFY: correctness] y == judge(a, b) on every input group.

### M1 Gate

Mechanism-verification line: inherits M0's kernel_count = 1 (no structural
change is proposed in M1; nothing to separate). [SYNC: none] [GATHER: none]
  [PHASE: none]

## 4. Known dead ends

(none — first version, written by the smoke's stub agent)

## 5. Global correctness / performance gate

- correctness: the judge's calibrated tolerance; any FAIL reverts the iteration.
- performance: stop target = the section 0 target in the device window; plateau and
  confirmation follow the run's resolvable_diff band.
- health: fresh_recheck must not flag; gpu_state clock drift within the driver band.

## 6. Rules

Delivery granularity and the three disciplines, the marking discipline, anti-plagiarism
and the INFEASIBLE protocol are inherited from meta_milestone.md template A.6 verbatim;
the smoke adds none of its own.

## Worked Example

(none — the smoke shape is trivial and fully specified by section 1)
""")
        for tgt in targets:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(body)
        out = ("milestone.md authored (v1): M0 baseline D0.1 -> M1 termination D1.1; "
               "target anchored to the ref baseline job; [SYNC: none]/[GATHER: none]/"
               "[PHASE: none] adjudications present; mechanism-verification lines in "
               "both gates.\n")
    elif "Mandatory M0 finish audit" in prompt:
        # M0 audit: OK closes the audit (enum at driver._CAUSE_ENUM_AUDIT)
        out = ("cause_class: OK\n"
               "diagnosis: stub — milestone consistent with the smoke trajectory; "
               "no lever citations.\n")
    elif "Diagnosis step" in prompt:
        # Platform diagnose: legal enum G1-G5/R1/R2/INFEASIBLE; gives G1 + a no-op prose
        out = ("cause_class: G1\n"
               "reachability: floor_opt=0.0001 ms -> OPEN (the floor sits far inside the "
               "target; keep walking)\n"
               "diagnosis: stub — single-iteration smoke; no lever citations, "
               "no probe requests.\n")
    elif "Patch step" in prompt or "patch_class" in prompt:
        # patch turn: append to the milestone absolute path bound in the driver's prompt
        # (the patch gate verifies that file changed; cwd is ws, ctx.milestone sits in operators/<op>/)
        import re as _re
        mm = _re.search(r"(/[^\s`']+/milestone\.md)", prompt)
        ms = Path(mm.group(1)) if mm else Path("milestone.md")
        if ms.is_file():
            with ms.open("a", encoding="utf-8") as fh:
                fh.write("\n### P1.1 (wp21 stub)\nsmoke patch: no-op additive entry.\n")
        out = ("patch_class: additive\n"
               "changed_entries: [P1.1]\n"
               "patch: wp21 stub no-op additive entry appended to milestone.md\n")
    elif "KB harvest" in prompt:
        out = "KB-HARVEST: stub — nothing to certify this smoke.\n"
    elif "RECON_REQUEST" in prompt and "student" not in prompt[:200].lower():
        out = "recon: stub — no requests.\n"
    else:
        # student turn
        state["student_turns"] += 1
        save_state(state)
        if state["student_turns"] == 1:
            Path("kernel.cu").write_text(KERNEL_CU)
            out = ("Stub delivered the baseline kernel.\n\n" + RECORD)
        else:
            out = "EXHAUSTED: wp21 mini smoke — one evaluated iteration is the budget.\n"

    # Truncation-detection guard: a reply under 400 chars with no on-disk trace would be
    # judged a "gateway truncation" and retried for free (driver.TRUNCATION_MAX_CHARS) —
    # the stub pads every reply to a safe length.
    if len(out.strip()) < 400:
        out = out.rstrip("\n") + (
            "\n\n[wp21 stub context] This is a scripted stand-in agent for the WP21 mini "
            "smoke run (tasks/wp21mini/agent_stub.py), not a language model. The machine "
            "mechanisms under test are the eval/NCU job path, the trajectory records, the "
            "KB-MENU synthesis, the floor backtest --from-run channel and the wall-clock "
            "composition telemetry; the agent turns themselves are out of scope until this "
            "machine has real agent credentials (see CHANGES-WP21.md defect list).\n")
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
