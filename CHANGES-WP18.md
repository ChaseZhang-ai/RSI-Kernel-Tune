# CHANGES-WP18 — 知识闭环与超越型供给

基线：`origin/main` @`359696b`（二代集成终态）。分支 `feat/wp18-knowledge-supply`。
基线 pytest：**2437 passed**（326s，全绿）。本包合并前回归见文末。

---

## 任务 1（P1）：KB→决策闭环（甲G21/G22/G23）

**`kb_used:` 字段 + 按条目台账**（G21）——`meta/student_prompt.md` 与 `meta/cuda/student_prompt.md`
的 record block 增可选 `kb_used:` 行（逗号分隔的条目名），客观轨迹段补一句"KB 引用属客观记录"。
解析在 driver 侧完成（`_wp18_kb_used_entries`，driver.py；`parse.py` 是 WP24 领地未动）：路径/`.md`
后缀剥离、跨行去重；学生 record 被接受的两处（主循环 + 预算耗尽的 deliver-only 轮）落 **PENDING 行**
进 `evidence/kb_usage.jsonl`（断电安全），verdict 落地时（全部 8 处 `append_record` 的 IterationRecord
路径前挂 `_wp18_kb_usage_flow`）追加终判行。台账是**学生自产客观记录**（红线豁免类）：只有条目名与
KEEP/REVERT，无任何教师 curated 内容。`_wp18_kb_usage_summary` 按条目聚 KEEP/REVERT。

**带宽配方反弹**（G22）——plateau 诊断的 INFEASIBLE 分支（`_infeasible_refusal` 通过后的
would-be-accept 路径）：BLOCKED 且 `_wp18_bw_recipe_missing`（本步 probe readings 与诊断文本均无
`effective_bw` 配方引用、无带宽数字 TB/s·GB/s 实测引用；纯散文"bandwidth"不算——散文正是反弹要抓的）
→ 反弹一次（`_WP18_BW_BOUNCE_BLOCK`：跑配方 / 引用同族实测 / 论证为何不适用，三选一），重发后照常
接受。refusal 已在失败路径（OPEN/数字不足）时反弹权归 refusal，不叠加本检查（现有 5 个 INFEASIBLE
fixture 补一发重发；`test_infeasible_on_open_*` 系列不动即绿）。

**KB-MENU 读最近 2 条**（G22）——`_wp18_recent_readings(records, 2)`（driver.py），`build_kb_menu`
增 keyword-only `prev_readings`：只在前一条读数命中、最新一条未命中的规则渲染一行 `[prev reading]`
标记（瓶颈中途换相可见）；双命中照旧无标记；`None` 时渲染与改动前逐字节一致（C7 纯度与快照保持）。

**决策行 `priced_null:` 注解 + 价格矩阵**（G23）——`_persist_lever_table` 落表后（仅教师自产表，
reprice 表走 SUSPENDED 通道不变）：`_wp18_stamp_priced_null` 对 cls∈{null, noop} 的杠杆，把
`[PRICED_NULL: measured <cls> (<Δ>) …]` 盖到 milestone 中引用它的 `lever:<name>` 行（幂等：换新
verdict 时旧标记替换，复用 `_mark_lever_citations` 的"一处一判"语义）；harmful/faded 仍归
REMOVE_CANDIDATES/reprice 通道不在此盖。`_wp18_write_lever_shape_matrix` 每次定价落一行
`evidence/lever_shape_matrix.json`（lever×workload 矩阵；接缝兑现：WP14 未合入，按 workload 维度建，
`workloads`/`per_shape` 字段来了自动升级）。

## 任务 2（P1）：超越型机制供给二批（甲A6+丙WP17-T1 合并）

`hw_kb/CANDIDATES.md` **只追加** 9 行（kb_lint 扫描 0 finding；现有 13 行一字未动）：
`tcgen05_tmem_double_buffer_epilogue`、`clc_persistent_scheduling_supplement`、
`warp_spec_pipeline_stages`、`fa4_mma_sfu_cross_tile_overlap`、`l2_access_policy_window_cutedsl`
（带前提条件：仅流式与持久数据竞争 L2；且 DSL 面先查 API 再排队）、
`launch_bounds_controlled_spill_variant`（注 MC 菜单用法）、`cuda_graph_device_time_negative`
（C16：只写"为什么无效"——graph 省的是 host launch 税，device-time 配对窗从不含它，量不出差异；
防预算烧假轴）、`shape_keyed_jit_specialization`（带 DeepGEMM 反例：跨 GPU 型号/形状分档，不保证赢）、
`software_exp_blackwell`。TMA multicast（既有 High 行）、wave-tail 治理（既有行）、
cp.async.bulk.prefetch-L2（既有行）维持不加行。全部 TODO，全部待本 run 实测（二代红线④）。
配套：kbmenu 匹配表**只追加** 6 条症状规则（`wave-tail`/`tma-multicast`/`warp-spec-pipeline`/
`prefetch-lead`/`launch-bounds`/`l2-policy-window`），指针关键词对三树实测可解析（`persistent`/
`multicast`/`producer`/`setmaxnreg` 三条全树命中，`prefetch`/`L2` 部分树命中时按 C10 只渲染 lever-form
不造指针）；既有 10 条规则与阈值一字未动（`test_rules_fire_on_the_documented_thresholds` 追加 WP18
用例、改 2 个受新规则影响的 pin）。

## 任务 3（P1）：regime 字段 + 批量认证 + survey 强制映射 + 存活升权（丙WP17-T2/T3/T4）

**regime + 存活升权 + DRAFT/BATCH 记法**：`meta/kb_harvest_prompt.md` 与 cuda overlay 同步加
"Regime field, survival weighting, and the queue's notations (WP18)" 段（4 条，≤20 行）：新条目可选
`regime:` 行（适用 shape 域/前提/反例；JIT 特化跨型号反例为指导正身）；`survival: <n operators>`
存活升权 + 反复反转降权复审（与 WP12 STALE-REVIEW 衔接）；`DRAFT` 行语义；`BATCH:<job>` 行语义。

**批量认证**：driver 侧排程 + 批协议——LoopContext 新键 `kb_cert_batch_size`（默认 4；1=逐条旧路径；
0=关）与 `kb_cert_every_meta`（默认 2），经 `_wp18_apply_cfg` 读卡层（契约 C17，不经 cli）；
`_wp18_kb_cert_duty_block` 每 N 个 meta 步在 diagnose 注入区（KB-MENU 之后、非 C10 固定小节）给一次
"队首认证"职责块；教师写 `notes/kb_cert_batch.py`（每候选独立计时块+对照块、逐条 PROBE 行、单条失败
不中断）→ `PROBE_REQUEST` 走既有 verdict-probe 的 `cmd:` 单 job 模式 → `_wp18_parse_kb_cert_log`
按 trailing note 解析逐候选 verdict → verdicts 以 `kb_cert:<name>` contract 行并入该步
`MetaRecord.probe_readings`（meta record 落账，复用既有字段不改 schema）；PASS 行由
`_wp18_stamp_kb_cert` 把 CANDIDATES 对应行 Status 原位改 `BATCH:<job>`（仅 Status 单元格，harvest
再按既有 queue-hit 流程写条目翻 DONE）；接缝：WP17 screening 原语/WP19 manifest 合入后由集成步收编
生成侧（CHANGES 预留）。

**survey 强制映射**：`skills/sota-survey/SKILL.md` 加 "Map every uncovered mechanism" 段（≤8 行）：
每条 KB 未覆盖机制必落一行 `CANDIDATE-DRAFT: <名> | <树> | <轴> | <为何值得 probe>`；harvest 轮
`_wp18_land_candidate_drafts` 机械落 `Status: DRAFT` 行——查重 `_wp18_kb_covers` 对三树
`*/*/index.md` 与条目文件名做纯字符串包含（coarse 防呆：误"已覆盖"只丢一行草稿，误"新"只费一眼），
**dup 不落、读失败 fail-closed 不落**。

## 任务 4（P1）：kernels/ 机械先验工件库（乙A9）

- `scripts/collect_kernel_artifacts.py`（新）：26→实际 25 条 kernel 逐条两车道。**ncu 车道**：arch 门
  （h100→sm90 / b200→sm100 对本机 compute capability）+ 每算子 `ncu_runner.py`（GPU 机上写，形状用
  发布形状）→ NCU 关键指标（双 SOL / DRAM / L2 hit / occupancy / launch stats）→ `ncu.json`；
  **sass 车道**：`.cu` kernel `nvcc -cubin -arch=sm_XXa`（torch include paths 补齐；纯编译不需 GPU）+
  `nvdisasm -c` → 指令族 census（mma/tma/cp_async/mbarrier/reg_role/st_matrix/…）→
  `sass_census.json`；DSL kernel 的 census 记 skipped+原因（runtime 编译需同 arch GPU）。一切落
  `priors/artifacts/<family>/<op>_<gpu>/manifest.json`（标注 regime=README 发布行）——**本机（RTX 3060,
  sm86）诚实全 skip：车道与解析可单测，采集在总工程师的 H100/B200 机上补**。
- `scripts/build_priors.py` 加 `--from-artifacts`：骨架六节按工件落 `Measured anchor (<source>):`
  机器事实行（无解释、无叙事、无实现体），六节全锚→`Status: filled (auto from measured artifacts…)`，
  部分锚→`Status: auto-anchored`（`--force` 不碰；filled/human 文件永不覆写）。fixture 测试钉住
  骨架→实条目 diff（6/6 节 anchor、状态翻转、二次运行不动、部分锚保护）。
- distill/recon 的 PRIOR-INPUT 通道：`meta/recon_prompt.md` Inputs 增 `KB_DIR` binding 行（只读本
  arch 树、与 distill 同权；渲染样例测试钉住 `{KB_READ_DIR}` 已解析）；`meta/cuda/recon_prompt.md`
  overlay 同步行（test_cuda_mode 的 overlay-parity 门要求两面对应）。工件被教师侧引用即实测签名
  （红线④天然合规）。

## 任务 5（P2）：kbmenu 命中遥测 + recon KB binding（乙A12/F16）

- **命中遥测**：kbmenu.py 增纯函数 `fired_rule_names` / `hit_record`（C7 不破坏：合成器零副作用）；
  driver 在每次 KB-MENU 渲染后 `_wp18_kbmenu_hits_note` 落一行 `evidence/kbmenu_hits.jsonl`
  （ts/kb_label/op/rules/各段行数）；run 末（harvest 入口，cli.py 未动）`_wp18_kbmenu_hits_summary`
  聚合 renders + rule_fires + 行数 → `evidence/kbmenu_hits_summary.json` + 日志行。这是 KB-MENU
  收益的第一手数据，只记账不进决策。
- **recon KB binding**：见任务 4 末条。

## 配置（契约 C17）

`global_config/loop.toml` 追加 `# --- WP18 ---` 注释键块（全局表封闭词汇不扩键；键在卡层 free-form
读取）：`kb_cert_batch_size` / `kb_cert_every_meta` 语义与默认值，及常开的四项台账（kb_usage /
kbmenu_hits / PRICED_NULL / lever_shape_matrix）说明。

## 新增候选行 Basis 来源表

| 行 | 来源 |
|---|---|
| `tcgen05_tmem_double_buffer_epilogue` | 44 §二.3（CUTLASS Blackwell tcgen05/TMEM 教程、Colfax TMEM write-ups）；本地 grep：两 sm100 树均只有单缓冲 epilogue 条目（`*/tcgen05_epilogue_t2r_stsm_tma_store.md`） |
| `clc_persistent_scheduling_supplement` | 28 §一 A6（CLC 动态持久调度）；本地 grep：`*/cluster_launch_control_work_queue.md` 与 `cutedsl/sm90/persistent_tile_loop_ring_continuity.md` 各自存在、组合条目无 |
| `warp_spec_pipeline_stages` | 44 §二.1（FA4 producer/consumer staging）；本地 grep：三树 `warp_specialization.md` 均为双角色，无 per-stage 分级 |
| `fa4_mma_sfu_cross_tile_overlap` | 44 §二.1（FA4 MMA/SFU 跨 tile 重叠先验，28 §一 A6 同）；本地 grep：`*/softmax_segment_ilp.md` 无跨 tile 形态 |
| `l2_access_policy_window_cutedsl` | 44 §二.8（**仅流式与持久竞争 L2 时显著**——前提进 Basis）；41 面5（sm100 只有 `.L2::cache_hint` 拼写）；既有 cuda/sm100 行自注 cutedsl 需查 API |
| `launch_bounds_controlled_spill_variant` | 28 §一 A6（ptxas 静默忽略 maxrregcount 的证据）；本地 grep：三树 setmaxnreg/spill 条目均以"除 spill"为方向，无受控溢出变体 |
| `cuda_graph_device_time_negative` | 44 §二.9 + 总师复核注记；本地红线一致：device 窗=流水化 batch 窗，host launch 税不在窗内（timing 侧既有口径；41 面4 同） |
| `shape_keyed_jit_specialization` | 44 §二.2（DeepGEMM JIT 及反例：H20 慢于静态特化栈、小 batch 不达预期）；28 §一 A6（shape 键控 JIT 特化）；本地 grep：`*/cute_dsl_const_expr_if.md` 无 shape 分档键控 |
| `software_exp_blackwell` | 44 §二.1（FA4 软件 exp；Blackwell 无硬件 exp）；本地 grep：`cuda/sm100/mufu_f2fp_ffma2_issue_overlap.md` 为 SFU 相邻读数，无 exp 软件路径 |

（44=三代外部 SOTA 调研，28=三代研究报告甲线扫描，41=三代性能天花板——均在 handoff/。全部情报只进
待认证行与教师先验，未经本 run GPU probe 实测不影响 gate/M0/停止。）

## 知识注入 A/B 注记（红线⑤）

新知识流（kb_used 台账、带宽反弹、KB-MENU prev 行与 6 条新规则、priced_null 盖章、批量认证职责块、
CANDIDATE-DRAFT 落地、recon KB binding、student `kb_used` 字段）上线路径：
1. **上线前后同 job A/B**：同一算子卡、同基座、同预算，先跑关闭形态（卡层 `kb_cert_batch_size=0`、
   `kb_menu=false` 即回到改动前逐字节行为——`build_kb_menu(prev_readings=None)` 与既有快照逐字节一致，
   反弹/职责块/遥测均随宿主开关归零）作对照，再开全量；对比 stop 分布、meta 步数、KB 引用回流率、
   认证吞吐（BATCH 翻转数/千 GPU 秒）。
2. **换基座重测**：对照组与实验组各在另一基座模型上复跑一遍，确认效应不依赖单一基座的风格（教师
   弹回顺应性与学生字段依从性各验一次）。
3. 命中遥测（kbmenu_hits）即本注记的度量件：rule_fires 分布与 KEEP/REVERT 台账是"菜单是否拉动决策"
   的第一手证据，harvest 自动落盘。

## 范围与红线自查

- 只动围栏内文件 + 任务 4 显式点名的 `scripts/build_priors.py`、新文件 `scripts/collect_kernel_artifacts.py`、
  `tests/test_wp18_knowledge_closure.py`（新），以及既有测试中本包行为变化处的 pins
  （`test_kb_candidates`/`test_kbmenu`/`test_loop_driver` 的 INFEASIBLE fixture）；WP24 禁碰清单零触碰
  （`parse.py`/`timing.py`/`cli.py`/leaderboard 面未动——kb_used 解析与 run 末汇总均在 driver.py 内完成）。
- 三树既有条目正文与 index、kb_lint、falsified_ledger 写入方、lever_pool.py、screening.py、
  analytic_floor.py、注入区其他小节：零改动。
- 测试 pins 更新（行为即本包目标处）：`test_kb_candidates` 队列行数 13→22、`test_kbmenu` 阈值表
  2 处 pin + 新用例、5 个 INFEASIBLE scripted fixture 各补一发重发（WP18 反弹的一次预算）。
- `meta/` 提示词面过 `lint_exam_surfaces`（无日期/无字面 job id/无行号引用）；`hw_kb/` 过 kb_lint
  （0 finding）。

## 回归

`~/venvs/forgekernel/bin/python -m pytest tests/ -q`：**2472 passed**（基线 2437 + 新增/更新 35），
0 failed（266s）。新增专项文件 `tests/test_wp18_knowledge_closure.py`（27 例）覆盖全部验收点：
kb_used 引用→KEEP/REVERT 回流、反弹一次/有实测不弹/OPEN 路径不叠加、KB-MENU prev-only 标记、
批量认证 N=4 单 job/逐条 verdict/单条失败不连坐/batch=1 逐条回归/batch=0 关闭、三树各 ≥3 条追加规则
指针逐字验证、命中计数格式、CANDIDATE-DRAFT dup 不落、priced_null 幂等、价格矩阵落盘、build_priors
骨架→实 diff、recon 渲染样例、提示词段与 loop.toml 键块、采集器双解析器。
