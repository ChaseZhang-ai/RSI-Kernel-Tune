# CHANGES-WP2 — 每轮遥测扩容：SOL/L2/尾波/合并度 + build 资源遥测 + roofline 诊断入口

分支 `feat/wp2-telemetry`（基于 main @ 5030503）。四个独立 commit：任务 1（采集集 + 架构过滤）、任务 2（resusage）、任务 3（digest 头行 + 诊断读数顺序）、本文档。

## 1. 新增 NCU 指标（17 → 25 项；brief 写 17，实际原集 18 项——wavefronts_st/ld 已在集内）

| NCU 指标名 | 字段名 | 诊断含义 | pass 成本 |
|---|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | `compute_sol_pct` | 计算 SOL%——离峰值多远；高 ⇒ 计算侧已尽，低+低 tensor_pct ⇒ latency-bound | 与 `sm__warps_active` 同 SM throughput 组，复用既有 pass，≈0 增 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | `memory_sol_pct` | 内存 SOL%；高+低 dram_pct ⇒ L1/L2 压力而非 DRAM | 与 dram 同属 memory 组族，≈0 增 |
| `lts__t_sector_hit_rate.pct` | `l2_hit_rate_pct` | L2 命中率；该进 L2 的工作集持续 miss ⇒ tiling/序问题 | **新增 lts 组，约 +1 pass** |
| `launch__waves_per_multiprocessor` | `waves_per_sm` | 尾波量化；<1 ⇒ 多数 SM 没活干，x.9x ⇒ 10-20% 尾损 | launch 属性，0 pass |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | `global_ld_sectors` | 全局读流量（可加计数，按 launch 数取每 launch 均值） | l1tex 组内加列，≈0 增 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | `global_ld_requests` | 与上项相除 = sectors/request（合并度，32=全合并） | 同上 |
| `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` | `stall_lg_throttle` | lg-throttle 长尾 stall（mio 的兄弟桶）；部分架构无此名，过滤机制自动裁剪 | smsp 组内加列，≈0 增 |

单 kernel replay 次数增幅估算：**≤2 次**（主要来自 lts 组 +1；其余落进既有 dram/l1tex/sm/smsp 组的同一 pass；launch 指标不占 pass）。`ncu --query-metrics` 每次 ncu 任务前多一次查询（~1s，仅 ncu 任务）。

键名/语义纪律：既有 18 项键名与语义零改动，只增不改；`_TRACE_COLUMNS`、`_NOTES` 同步补行；`_is_additive` 规则自动覆盖新 `.sum`（per-launch 均值），pct/ratio/launch 属性不参与求和——`tests/test_ncu.py` 断言。

## 2. 架构自适配过滤（`filter_metrics_for_device`）

- 参考实现：`src/forgekernel/inner/ncu.py`（`query_metric_names` / `filter_metrics_for_device`，单测覆盖）。
- 落地：卡的 `gpu_entry.sh` 由 `src/forgekernel/leaderboard.py` 的 `GPU_ENTRY_TEMPLATE` 渲染（**这就是卡模板里 NCU 调用处的唯一落点**——围栏写的"card_template/ 内 NCU 调用处"经核实实际生成自该文件；`card_template/eval/` 内无 NCU 调用）。ncu 任务执行前 `ncu_filter_metrics` 用 bash 完成同逻辑：`ncu --query-metrics` 输出取 base 名（首个 `.` 之前）与期望集取交集；`launch__*` 无条件保留（launch 属性在任何架构的 query 输出里都不出现，朴素交集会静默丢掉 regs/smem/waves）；query 为空/失败时原样透传（解析端对缺键本就容错，过滤器失效不得静默整个采集）。
- **顺手修复一处既有漂移**：模板里硬编码的 `NCU_METRICS` 串原本缺 wavefronts 两项（与 `KEY_METRICS` 不同步 26 项 vs 18 项）。新增同步单测 `test_gpu_entry_template_requests_exactly_what_the_parser_reads` + bash 过滤器等价性测试（用 stub ncu 实跑模板函数，与 Python 参考实现对拍，`set -euo pipefail` 环境下）。
- sealer/binref 的 recon-tree `gpu_entry.sh` 复用卡文本（`sealref.ref_tree_gpu_entry` / `binref.ref_tree_gpu_entry`），自动继承新串与过滤器，无需另改。
- 解析端容错：`profiler_from_ncu` 对缺键跳过不报错（现状已满足，单测 `test_a_profile_without_the_new_metrics_parses_without_them` 固化）。

## 3. build 资源遥测（cuobjdump -res-usage）

- 行协议（对齐 `SASS-OPERANDS` 行约定，进 job log 供人工 grep）：

  ```
  RES-USAGE kernel=<name> regs=<n> smem=<n> cstack=<n>
  ```

  `smem` = 静态 smem（动态 smem 是 launch 参数，二进制里没有，由 NCU `smem_dynamic` 覆盖）；`cstack` = per-thread STACK 帧（**spill 悬崖，编译后立即可见**）；缺字段则不打印该字段。
- eval 侧（`card_template/eval/leaderboard.py`）：CUDA 卡在 `load_cuda_entry` 构建完成后对 `fk_kernel_*.so` 跑 `cuobjdump -res-usage`（先找构建该 kernel 的 toolkit，PATH 兜底），每 kernel 打一行协议行，全表嵌入 eval JSON **顶层新键 `resusage`**（列表，`[{kernel, regs?, smem?, cstack?}]`；correctness FAIL 也带——spill 悬崖先于任何计时可见）。`loop/parse.py` 未动（WP1 围栏）：其 `parse_eval` 只读已知键，未知顶层键天然透传。
- driver 侧（围栏内 `_ncu_profiler` 区，本分支 task 2b 提交）：从 **NCU job log 的 RES-USAGE 协议行**（eval 进程每次构建/加载 CUDA kernel 都会打印，本任务必经）解析全表，取**主导 kernel**（`kernels_from_ncu` 首行）的读数以 `resusage_kernel / resusage_regs / resusage_smem / resusage_cstack` 并入 profiler dict，并打一行 `[loop] RES-USAGE ... [dominant kernel, cuobjdump -res-usage]` 日志。
  - 与 brief 的偏差说明：brief 写"读取 `ev["resusage"]`"，但 `_ncu_profiler` 拿不到 `ev`（它只收 NCU job 输出）；改从同一 NCU job log 的协议行读取，语义等价（同一 job、同一构建、同一协议），且不越围栏拿 ev。eval JSON 的 `resusage` 键仍按 brief 嵌入，供 job log 之外的消费方使用。
  - 匹配规则 `match_resusage`：精确符号名 → `_Z<len>` 前缀剥参后缀的去 mangle 包含匹配（cuobjdump 报符号名，NCU CSV 报去 mangle 名）→ 都不中则**不加字段**（错误读数不得冒充主导 kernel）。
- **CuteDSL 侧：未落地**。DSL 卡（kernel_file=kernel.py）走 `dsl_source.py` 编译，无稳定 .so/.cubin 产物路径可指给 cuobjdump；按 brief"不许为凑数硬造"，只落 CUDA，DSL 待办记录于此。DSL 卡无 RES-USAGE 行/无 `resusage` 键 → driver 侧静默无字段，零影响。
- fixtures：`tests/fixtures/cuobjdump_resusage.txt`（真实 CUDA 13.4 dump 格式 + fatbin 双架构重复块，验证 last-arch-wins）。

## 4. roofline 头行进 digest 与诊断入口

- `build_digest`（trajectory.py）顶部新增 `"sol"` 键：`sol_header()` 取**最新一个测得这些字段的迭代**（KEEP/REVERT 均可）的 `compute_sol_pct / memory_sol_pct / l2_hit_rate_pct / waves_per_sm / tensor_pct / dram_pct`，只输出在场字段；扩容前的记录 → `{}`，不缺不炸。≤3 行纪律：单行紧凑 JSON；现有压缩键逻辑未动。示例：

  ```json
  "sol": {"iter": 7, "compute_sol_pct": 42.13, "memory_sol_pct": 18.76,
          "l2_hit_rate_pct": 71.29, "waves_per_sm": 340.28, "tensor_pct": 37.44, "dram_pct": 8.99}
  ```

- `meta/diagnose_prompt.md`（仅诊断读数顺序段）：1b 第一步改为**先读 digest 的 `sol` 头行**（compute/memory 谁先饱和、waves<1、L2 命中率），再进 stall 账本；1c 的 hard anchor 处注明"用 `compute_sol_pct` 回填/校验 MFU 项（差异大 = 读数缺陷，先解决再算地板）"。**floor_opt 公式与 BLOCKED 门槛（driver.py `_REACH_BLOCKED_MARGIN`=1.20 机器语义）未动**——只加校验提示。
- `skills/ncu-trace/SKILL.md`：症状表补 8 行（新 7 指标 + resusage 三兄弟一行）+ 架构过滤说明段，格式对齐现有表。

## 5. 任务 4（traffic_amp）：跳过

`IterationRecord`/`EvalResult` 均无 input/output 字节数；从 shapes 推字节数需改 shapes.json 契约（brief 明令不许）。`dram_bytes_read/write` 有了，分母没有——跳过，待上游 record 增补字节字段后一行即可接上。

## 6. 交付拓扑（重构后）

**最终拓扑**：`feat/wp2-telemetry` 从 main @ 5030503 独立开出，恰 5 个提交、只含 WP2 内容：

```
main @ 5030503
└─ WP2 task 1   采集集 18→25 + filter_metrics_for_device + gpu_entry 模板 bash 过滤器 + 同步单测 + 新指标 fixture
   └─ WP2 task 2a  resusage 参考实现（inner/ncu.py）+ eval 侧采集（card_template/eval/leaderboard.py）+ fixture
      └─ WP2 task 2b  driver fold（_ncu_profiler 围栏区：import 块 + RES-USAGE fold，+ fold 测试）
         └─ WP2 task 3   digest SOL 头行 + diagnose_prompt 读数顺序 + SKILL.md 症状表
            └─ WP2 CHANGES  本文档
```

**重构缘由（如实记录）**：初版交付在六路并行的共享物理工作树上进行，git index 被多会话轮流持有，产生了两处竞态——driver.py 的 resusage fold 被 wp5 会话的全量 add 卷入其 commit（原 `a749131`）；task 3 的 commit（原 `9416ae6`）卷入了 WP1 会话当时在途的 `timing.py`/`parse.py`/`driver.py`/`test_wp1_measurement.py` 改动，使初版分支与 `feat/wp1-measurement` 共享 lineage、混入他包提交，不满足六路并行的交付契约。

**重构方式**：所有代码以初版提交为准、内容零改动——在独立 worktree（`wk-wp2`）从 5030503 依序 cherry-pick（task 3 按"保留我方/丢弃 WP1"解决三处冲突；WP1 在途的 `timing.py`/`parse.py`/`driver.py`/`test_wp1_measurement.py` 全部剥离，WP1 原提交不受影响，仍被 `feat/wp1-measurement` 完整引用）。除本文档 §3/§6/§7 外，无任何手改内容。

**与其他包的关系**：`sol` 头行只读既有 profiler dict，与 WP1 的新字段（gpu_state/gate_spread_pct 等）零耦合——其字段缺失时头行照常工作。范围外文件（`loop/parse.py`、`card_template/eval/timing.py` 等）在本分支上一概维持 main @ 5030503 原样。

## 7. 验证（重构后分支，wk-wp2 树实测）

- `pytest tests/ -q`（~/venvs/forgekernel，py3.14 无 torch）：**2079 passed**（0 failed）。
- torch 侧（~/venvs/ml）：`test_correctness_monitor.py + test_launch.py + test_sass_census.py` **61 passed**。
- `git diff main feat/wp2-telemetry --stat` 仅含：`inner/ncu.py`、`leaderboard.py`（gpu_entry 模板）、`card_template/eval/leaderboard.py`、`inner/trajectory.py`、`loop/driver.py`（仅 import 块 + `_ncu_profiler` 区）、`meta/diagnose_prompt.md`、`skills/ncu-trace/SKILL.md`、`tests/test_ncu.py`、`tests/test_loop_driver.py`、`tests/test_inner_trajectory.py`、fixtures 两件、`CHANGES-WP2.md`——绝无 `timing.py`/`parse.py`/`test_wp1_measurement.py`。
- fixtures：`ncu_sample_new_metrics.csv`（35 行，含全部新指标的双 kernel CSV）+ `cuobjdump_resusage.txt`；本机 CUDA 13.4 实测 `cuobjdump -res-usage` 与 `ncu --query-metrics` 输出格式作为 fixtures 蓝本（RTX 3060 GA106 实查）。
