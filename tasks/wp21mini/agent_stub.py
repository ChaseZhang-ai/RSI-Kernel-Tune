#!/usr/bin/env python3
"""WP21 mini 冒烟用 `claude -p` 替身 agent（stub）。

协议面与 scripts/agent-claude.sh 相同：prompt 走 stdin、回复走 stdout，退出码 0。
角色按 prompt 标记判别；学生回合第一次写 kernel.cu + 记录块，之后 EXHAUSTED；
教师回合给最小合法回复（patch 回合会真的在 milestone.md 上追加条目再报 patch 协议行）。

只服务 tasks/wp21mini 冒烟（链路与数据流证据），不代表 agent 行为已验证——
真 agent 冒烟需机器配置 agent 凭据（冒烟 defect 清单第 1 条）。
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

    # 冒烟证据：把每次收到的提示词里的机制标记落盘（kbmenu 渲染命中、SOL 头行等）。
    markers = [m for m in ("KB-MENU", "KB-HINTS", "LEVER-MENU", "CALIBRATION",
                           "compute_sol_pct", "STAGNATION", "LEDGER")
               if m in prompt]
    if markers:
        with open(".wp21_stub_prompt_log.txt", "a") as fh:
            fh.write(f"markers={markers} len={len(prompt)}\n")

    # 角色判别（标记取自各提示词/命令 brief 的题头与专有协议词）
    if "You are writing the correctness judge" in prompt:
        # judge 回合：cwd=check/，写三件套
        write_judge_files()
        out = "judge: wrote definition.json, judge.py, groups.json for wp21mini.\n"
    elif "Initial distillation authoring step" in prompt:
        # 蒸馏回合：把合规 milestone 写进 cwd（工作区根），之后教师轮只补丁
        ms = Path("milestone.md")
        ms.write_text("""# wp21mini milestone

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
        out = ("milestone.md authored (v1): M0 baseline D0.1 -> M1 termination D1.1; "
               "target anchored to the ref baseline job; [SYNC: none]/[GATHER: none]/"
               "[PHASE: none] adjudications present; mechanism-verification lines in "
               "both gates.\n")
    elif "Mandatory M0 finish audit" in prompt:
        # M0 审计：OK 关审计（枚举见 driver._CAUSE_ENUM_AUDIT）
        out = ("cause_class: OK\n"
               "diagnosis: stub — milestone consistent with the smoke trajectory; "
               "no lever citations.\n")
    elif "Diagnosis step" in prompt:
        # 平台诊断：合法枚举 G1-G5/R1/R2/INFEASIBLE；给 G1 + 无操作 prose
        out = ("cause_class: G1\n"
               "diagnosis: stub — single-iteration smoke; no lever citations, "
               "no PROBE_REQUEST, no INFEASIBLE request.\n")
    elif "Patch step" in prompt or "patch_class" in prompt:
        # 补丁回合：真的在 milestone.md 上追加条目（patch gate 校验文件被改过）
        ms = Path("milestone.md")
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
        # 学生回合
        state["student_turns"] += 1
        save_state(state)
        if state["student_turns"] == 1:
            Path("kernel.cu").write_text(KERNEL_CU)
            out = ("Stub delivered the baseline kernel.\n\n" + RECORD)
        else:
            out = "EXHAUSTED: wp21 mini smoke — one evaluated iteration is the budget.\n"

    # 截断检测护栏：回复 <400 字符且无盘上痕迹会被判"网关截断"而免费重试
    # （driver.TRUNCATION_MAX_CHARS）——stub 回统一垫足长度。
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
