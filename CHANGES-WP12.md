# CHANGES — WP12 知识注入与 KB 生命周期

分支 `feat/wp12-knowledge-lifecycle`（基线：kernel-ZXtune main `c1971f1` = ForgeKernel @5030503 + WP1–WP6 集成）。
基线回归 **2222 passed**；交付后 **2287 passed**（新增 65 个用例，全绿）。全部教师侧、零模型调用、零 GPU 判定。

## 任务 1 — 确定性合成器 `src/forgekernel/loop/kbmenu.py`（新，契约 C7 单一实现）

**纯函数、无 I/O（输入全由 driver 备好）、同输入同输出**。入口 `build_kb_menu(readings, resusage, ledger_rows, meta_records, index_text, kb_label, op, recipe_dir)` 渲染 ≤30 行的 `KB-MENU` 块（实测样例 22 行），三段独立留空、绝不渲染空壳标题、全空返回 `""`（driver 不注入）：

- **KB-HINTS**（≤5 行）：10 条规则的匹配表 `kbmenu.RULES`（结构可扩展：`Rule(name, when, label, keywords, lever_hint)`，阈值常量具名 `L2_LOW_PCT/WAVES_LOW/SOL_WARM_PCT/SMEM_CLIFF_BYTES`）。触发条件全部映射任务指定的读数：`spill_*_sectors>0 或 cstack>0 → 找 STL/spill 行`、`regs≥128 且 spill → setmaxnreg 行`、`bankconf_*>0 → bankconf 行`、`l2_hit_rate_pct<40 → swizzle+reference 行`、`waves_per_sm<1 → persistent 行`、`resusage smem≥200000 → ring-depth 行`，另 4 条无指针规则（latency-bound / unbound-attribution / wave-quantization / dram-high）只进 LEVER-MENU。指针 = `hw_kb/<lang>/<arch>/<entry>.md — "<症状行摘引>"`：**关键词只匹配行的症状/用途单元格**（`SymptomRow.probe`，非截断），摘引 `snippet`（≤96 字符）是原始行逐字子串；同指针行去重。索引解析兼容两种树形：症状优先表（cuda/sm100、cutedsl/sm100）与 sm90 平面接口表（无症状节 → 全表，入口列链接时摘用途格）。
- **LEVER-MENU**（≤10 行）：触发症状 → 杠杆**形态**（"register-frame levers (setmaxnreg / …)"），不带任何未实测数字；falsified_ledger 同 arch 行（同 op 优先，≤4 行）渲染为 `LEDGER <lever> <class> Δ<..>% — do not re-propose unless you name what changed`（已实测的排除证据，合法）。配方指针由 driver 传入仓库**绝对路径** `recipe_dir` 重写（教师 cwd = 学生工作区，仓库相对路径不可解析——与 `_bindings` 的绝对路径约定一致）。
- **CALIBRATION**（≤5 行）：`meta_status` 尾聚合——近 5 次 reopen 回本率（`reopen_price` 非空的记录按 `KEEP_PATCH` 计回本）、账本尾 10 行 faded/reversed 率（`class=="faded"` 或 condition 含 "the sign flipped"，即 `_reprice_verdict` reversed 的落账形态）、连续 REVERT_PATCH 方向列表（≤3 条、最新在前，对照"死路禁重试"）、KB 条目失效预警（任务 2）。

**driver 接线（三处，均 `# --- WP12 ---` 标注）**：
1. `LoopContext` 新字段 `kb_menu: bool | None = None`（契约 C7：按 WP1/WP5 先例走 LoopContext 默认区；`[loop]` 键由卡片层 loop.toml 解析——`_wp12_kb_menu_on` 复用 WP5 的两层读取器 `_wp5_cfg_sections`，显式赋值永不被覆盖）。
2. 输入采集小段（`_wp12_readings/_wp12_resusage/_wp12_ledger_rows/_wp12_index_text/_wp12_kb_menu_block`）：readings 取最新带 profiler 的 live 迭代（KEEP/REVERT 同读，与 `_falsified_ledger_block` 同面），resusage 从 profiler 的 `resusage_*` 名空间展开，ledger 复用 `_falsified_ledger_path` 同文件同容忍读法，index 取 `kb_read_dir(ctx)/index.md`。这是"注入区 + via KB: 引用聚合小段"围栏内的胶水，不含第二个合成器实现。
3. **注入区一个块**（`run_meta_step`，REMOVE_CANDIDATES / phase-exhaustion 之后、`prompt_file` 之前，即契约 C3 渲染序尾部；WP10 STAGNATION 由集成步插在其前）：`if _wp12_kb_menu_on(ctx): diagnose_extra += _wp12_kb_menu_block(...)` —— 全 driver 仅此一处调用。

**消费段**（`meta/diagnose_prompt.md`，新增 7 行 ≤8）：三段各一句语义 + "The menu is not a price, a pointer is not an answer"。

**确定性证明**：`test_kbmenu.py::test_snapshot_same_input_same_output` 固定输入 → 金样前缀快照 + 两次调用逐字节相等；渲染只用有序结构（RULES 顺序、文件序、dict 插入序），无集合遍历出渲染文本。

**空输入健壮性**（任务要求逐条落测）：ledger 无行/行含脏数据（非 dict 行跳过）、meta 很短、index 零匹配 → 各段独立缩短；`arch` 不匹配的 ledger 行整行不渲染；全空 → `""`。注：`falsified_ledger` 全空时 LEVER-MENU 仍可由读数规则单独供行，LEDGER 排除行段自然缺席。

## 任务 2 — KB 条目死亡率 + 死路跨 run 化

**`meta/kb_harvest_prompt.md` 加段（5 行 ≤10；`meta/cuda/kb_harvest_prompt.md` overlay 同步加同段 5 行——`test_cuda_mode` 的 binding 同步测试强制 overlay 携带相同 `{...}` token 集，`{META_TRAJECTORY_PATH}` 是本段新增 binding）**：
1. **时效字段**：写入/更新的条目在 evidence 括号旁追加 `last_confirmed: job <number>`（该读数最近一次被哪个实测 job 确认）。**与任务书的 `<run>/<date>` 形态有一处有意偏差**：kb_lint 的 `NARRATIVE_RULES` 对 KB 条目正文禁日历日期（`20XX-XX-XX` 与 MMDD 均命中）而 job id 显式放行（`lint_text → lint_narrative_text(allow_job_ids=True)`），kb_lint.py 本包不许动——故时效货币取 job 号（单调可比、与 harvest 现行 evidence 括号同币种），CHANGES 存照。
2. **死路跨 run 化**（报告 2 A5）：harvest 轮末把 milestone §4 Known dead ends 逐条转成 `hw_kb/falsified_ledger.jsonl` 追加行，`class: structural-dead-end` + `op_family` + `arch`（WP3 schema 已有字段，只追加不改旧行）；下个同族 run 的教师经现有 LEDGER/LEVER-MENU 通道读到。纯提示词指导，driver 零改动。
3. **条目失效复审**：扫描 `{META_TRAJECTORY_PATH}` 的 `reopen_price:` 行，`via KB:<name>` 引用的条目若最近 2 条引用记录均为 `REVERT_PATCH` → 向 `hw_kb/CANDIDATES.md` 追加一行（表格 schema 不变，`Status` = `STALE-REVIEW`，行内点名条目与两次回撤）；条目本体不动，留给复审裁决。

**driver 侧聚合**（诊断时预警）：`kbmenu.stale_kb_entries(meta_records, streak=2)`（纯函数；按 meta_step 序归集每条目 `via KB:` 引用的 outcome 序列，末 streak 条全 REVERT_PATCH 即判疑似失效），由注入块喂给 CALIBRATION 渲染 `- KB entry <name> possibly stale (last 2 reopens citing it REVERTed) — schedule a harvest review`。`kb_cite_name` 与价格门 `_REOPEN_PRICE_SRC` 同形（含全角冒容、`.md`/路径剥离）。**未做** lever↔条目 prescribed 联动（任务书明令：现有记录无此层数据）。STALE 行由 harvest 运行时按 prompt 追加，本包代码不碰 CANDIDATES.md；kb_lint.py 零改动。

## 任务 3 — 预置探针配方库 `skills/probe-recipes/`（新）

三个配方 = 脚本模板（自包含：**无 forgekernel 可导入**、torch 只在惰性 GPU 路径，`--help` 离线可跑）+ README（适用症状/判读规则/额度占用），统一走 verdict probe 通道：教师拷入 `notes/<name>.py` 后 `PROBE_REQUEST`（driver 按 `cmd:python3 notes/<name>.py` 单 job 执行）。**只产读数**：收尾 `PROBE PASS max_diff=n/a (<contract>)  # <contract name>` 是探针协议的"仪表出了读数"行（非数值契约，照 probe brief 惯例），可被 `via probe:<path>#<n>` 引用，不是质量判定。

| 配方 | 测什么 | 判读（脚本 READ 行预注册） |
|---|---|---|
| `occupancy_scaling.py` | 同 kernel grid×{0.25,0.5,1.0} 交错计时（persistent 用 num-SM 等价旋钮），CUDA events、中位数、对 s=1.0 取比值 | `flat`=latency-bound（±0.15 带）；`linear`=throughput-bound（ratio≈s ±0.25）；相邻档 ≥1.30× 跳变=`stepped`（wave quantization）；否则 `mixed` 不归因 |
| `phase_ablation.py` | 只读不算 / 只算不读 / 全量三版交错计时 | read≈full→memory-face（supply 杠杆）；compute≈full→compute-face（scheduling 杠杆）；read+compute≪full→overlap（时间线问题）；引用解析地板但**不依赖** |
| `effective_bw.py` | 同规模纯流式 triad（12 B/元素）测节点实际可达 GB/s | 教师用 digest dram bytes/kernel time ÷ triad GB/s 得 effective SOL（≥90%=真墙）；`--theoretical-gbps` 给出节点可达/理论比，换 SOL% 分母 |

每个配方的**算子无关骨架 + 可测纯函数**（`read_curve`/`split_phases`/`triad_gbps`/`rebase`），默认 hook 是内置 torch 流式自检（README 明示：默认 hook 的读数与算子无关，适配 hook 后才可作证据）。额度：每配方 1 个 verdict-probe 轮（`[loop] max_verdict_probe_rounds` 默认 4 内）。`meta/diagnose_prompt.md` 的 `{VERDICT_PROBE_BRIEF}` 旁加引用行（与消费段合计 7 行 ≤10）。

**教师侧 only 的落地形态**：库目录**不带 SKILL.md**——`skills.mirror` 只镜像含 SKILL.md 的目录，无 SKILL.md 即不进任何 agent 工作区镜像，学生 read set 零改动；教师经 KB-MENU 的绝对路径指针 + diagnose_prompt 引用行读取。由此 `tests/test_skills.py` 的"skills/ 下每个目录都是 skill"布局不变量按例外放行（`NON_SKILL_DIRS = {"probe-recipes"}`，并在 `skills/README.md` 加 1 行说明——两处为全绿强制同步改动，存照）。

单测 `tests/test_probe_recipes.py`（20 个）：mock 计时驱动纯函数（平/线性/台阶/混合曲线、四面判读、带宽算术与 rebase）、输出格式（RECIPE/READ/PROBE 行语法）、三配方判读行文法一致、`--help` 子进程全跑通、无 forgekernel 导入 + torch 惰性。

## 配置（契约 C7 词汇表选择，CHANGES 存照）

- **选择**：走 LoopContext 默认区（WP1 先例）。`[loop] kb_menu` 从卡片层 `operators/<op>/loop.toml` 解析（`[loop]` 表在该层自由读取，`_wp5_cfg_sections` 已含全局+卡片两层），默认 `true`，显式赋值优先，坏类型落默认。
- **原因**：`global_config/loop.toml` 的表词汇表封闭（`forgekernel.task._KEYS`，无 `[loop]` 表），扩词汇表需改 `task.py`——不在本包围栏内。故 `global_config/loop.toml` 只**追加 WP12 注释块**（12 行）记载键名、默认与卡片层设置方式；追加后 `read_global_config` 校验零问题（已验证）。

## 验收对照

1. **pytest 全绿**：2287 passed（基线 2222 + 新增 test_kbmenu.py 45 例 + test_probe_recipes.py 20 例）；快照测试锁定确定性。
2. **dry-run 证据**（fixtures：WP2 全量读数 + resusage(cstack=8, smem=210000) + 2 行 ledger + 3 条 meta 记录，cuda/sm100 树，22 行 ≤30）：

```
## KB-MENU (deterministic retrieval aid — a pointer is not an answer, the menu is not a price)
KB-HINTS:
- spill/cstack>0 (spill_st=64000, spill_ld=0, cstack=8): hw_kb/cuda/sm100/lowtier_role_tmem_readout_spill_ladder.md — "**TMEM readback spills** in a `setmaxnreg.dec` role (`STL` / `LDL`, non-zero `spill stores`)"
- bank conflicts (bankconf_st=1200000, bankconf_ld=0): hw_kb/cuda/sm100/tcgen05_epilogue_t2r_stsm_tma_store.md — "**Accumulator out of TMEM → bf16 → TMA store**; the store writes stale data / half a tile / 16 B"
- l2_hit_rate_pct=35 < 40: hw_kb/cuda/sm100/smem_traffic_triage.md — "**smem traffic over the line** (`smem_wavefronts_st/ld` above the reference) — unsure whether to"
- waves_per_sm=0.7 < 1: hw_kb/cuda/sm100/cluster_launch_control_work_queue.md — "**A persistent kernel wants resident CTAs to claim work items dynamically** (grid declared as th"
- smem=210000 B >= 200000 (stage-count cliff): hw_kb/cuda/sm100/umma_issue_chain_precomputed_desc.md — "The MMA warp's per-beat issue chain; **unrolled vs runtime stage**; ring depth does not divide t"
LEVER-MENU:
- spill/cstack>0 (spill_st=64000, spill_ld=0, cstack=8) -> register-frame levers (setmaxnreg / maxrregcount / cursor-vs-array / role-local scalars); KB: lowtier_role_tmem_readout_spill_ladder.md
- spilling at high regs (regs_per_thread=168) -> register-tier levers (warp_specialization caps / two-pass readout); KB: lowtier_role_tmem_readout_spill_ladder.md
- bank conflicts (bankconf_st=1200000, bankconf_ld=0) -> smem layout / face width / read-mapping levers; KB: tcgen05_epilogue_t2r_stsm_tma_store.md
- l2_hit_rate_pct=35 < 40 -> tiling-order / swizzle / raster levers (check the working set against L2 first); KB: smem_traffic_triage.md
- waves_per_sm=0.7 < 1 -> grid-shape / tail / persistent-scheduling levers (settle the grid before inner-loop tuning); KB: cluster_launch_control_work_queue.md
- smem=210000 B >= 200000 (stage-count cliff) -> stage_count / ring-depth levers (depth pays only to saturation; smem is the cap); KB: umma_issue_chain_precomputed_desc.md
- both SOL low (compute=45%, memory=50%) -> latency-bound reading: price per-instruction stall-ledger rows, not aggregate ratios
- tensor_pct=55% with both SOL low -> attribution recipe: /repo/skills/probe-recipes/phase_ablation.py (read/compute split)
- waves_per_sm=0.7 (few waves: check the tail) -> curve recipe: /repo/skills/probe-recipes/occupancy_scaling.py (flat/linear/step reading)
- LEDGER warp_split harmful Δ-11.63% — do not re-propose unless you name what changed on the current tree
CALIBRATION:
- reopens (last 2): 0/2 paid back (KEEP_PATCH); use a cheap probe on the next one first
- ledger tail (last 2): 1/2 rows faded/reversed — old prices do not carry across trees; re-price before citing
- KB entry wgedge possibly stale (last 2 reopens citing it REVERTed) — schedule a harvest review
```

   注入区 diff：driver.py 仅三处 `# --- WP12 ---` 标注（LoopContext 键块 / 输入采集小段 / 注入区单块），无其他区域改动。空输入样例：`build_kb_menu(readings={}, index_text="") == ""`（不注入）。
3. **匹配表自查**：`test_every_symptom_row_parses_with_a_real_quote` 对三棵树**全部**症状行（超出每树抽 10 条的要求：cuda/sm100 与 cutedsl/sm100 各 ~50-60 行、cutedsl/sm90 全表）断言摘引为原始行逐字子串、入口文件存在于树内；`test_rendered_pointers_quote_the_index_verbatim` 再对强触发渲染出的每条指针验证同一性质。
4. **配方库**：三个脚本 `--help` 子进程跑通（无 GPU/torch 依赖）；20 个单测绿；判读行文法一致（`RECIPE`/单条 `READ`/单条 `PROBE PASS max_diff=n/a (…)  # <contract>`，`test_shared_line_vocabulary` 锁定）。
5. 本文件即 CHANGES；A/B 注记见下。

## 知识注入 A/B 度量方案（总览红线 5，JustGoal 教训）

KB-MENU 是知识注入类改动，上线前后必须对照同基座 goal-only 基线，基座模型更换后重测：

- **A/B 协议**：同一 operator 卡（建议 grouped-gemm 或 fp8_gemm，KB 命中面大）× 同基座 teacher/student 模型 × 同 loop.toml，两组各跑 ≥2 个同构 run：A 组 `[loop] kb_menu = false`（kill switch 即本包开关，学生侧两组完全一致），B 组默认开。教师/学生随机种子与温度按部署层现状固定不了，以 run 数对冲。
- **主指标**（教师决策质量，本包的靶）：① reopen 回本率（meta_status 聚合，KB-MENU 自己就在度量它）；② 每 meta 步的 verdict-probe 消耗与"重测已测读数"次数（probe_readings 重复 path#contract 计数）；③ falsified_ledger 命中——设计轮提出 ∩ 账本已有同向死路的 lever 数（下降=死路记忆生效）；④ G3/KB_CERTIFIED 轮次里 cite 真实 index 条目的比例（KB-HINTS 命中率，总览裁决"先上指针，命中率数据说话"的直接读数）。
- **护栏指标**（不许变差）：全部 gate 数字、正确性、precision、防造假事件数；学生侧指标应两组无差（学生 read set 零改动，若见差即实现泄漏，先修再谈收益）。
- **换基座重测**：换 teacher/student 模型后上述 A/B 重跑（JustGoal 消融显示机制收益随基座漂移）；匹配表阈值均为读数语义阈值（不含模型行为假设），预期可迁移，但 LEVER-MENU 对教师行为的引导度必须重测。
- **配方库对照**：三配方另计"误判纠正率"——上线前人工复盘历史 run 中"SOL 85% 判带宽墙 / 波尾误判"类诊断，上线后同形态症状走配方的诊断修正数（occupancy/phase/BW 各自计）。

## 红线自查

- 教师侧 only：KB-MENU 只进 `_meta_prompt` 注入区；学生 read set/prompt 零改动（配方库无 SKILL.md 不镜像；trajectory.py/timing.py/parse.py 零改动）。
- 菜单不是价格：KB-MENU 除 ledger 已实测排除行外不带任何数字；无 gate 读它。
- 契约 C7：`kbmenu.py` 唯一合成器实现，纯函数、无模型调用；driver 无第二份渲染。
- 契约 C3：注入区仅一个 WP12 块（≤30 行）；消费段与配方引用行合计 7 行（≤10）；未触碰 WP10 STAGNATION/WP9 领地/`kb_lint.py`/`hw_kb` 三树正文/`CANDIDATES.md`。
- 围栏外的最小强制同步（全绿所需，存照）：`meta/cuda/kb_harvest_prompt.md`（binding 同步测试）、`tests/test_skills.py` + `skills/README.md`（无 SKILL.md 库目录的布局不变量）、未动 `student_prompt.md`。

## 复查修复记录 — occupancy_scaling 台阶判读物理方向颠倒（复查后追加）

**旧判据为何反向（一句）**：`r0 / max(r1, 1e-9) >= 1.30` 只在"grid 越大、时间越短"（t 随 s 递减 ≥1.3×）时触发，而波模型下 t(s) 单调不减、真实波界签名是"缩小 grid 跨档时时间**下降不足线性比例**"（尾波释放不掉），方向恰好相反；且旧 stepped 排在 linear 之后，W=1.2 尾波曲线（|0.5−0.25| = 0.25 恰压 LINEAR_BAND 线）先被判 linear，配方对专抓的尾波形态失明。

**新判据公式**（`read_curve`，检查顺序改为 **flat → stepped → linear → mixed**，台阶扫描含"最大 sub 档 → s=1.0"一跳）：对相邻对 (s0 < s1) 取线性期望 `expected = r1 × (s0/s1)`，`r0 ≥ STEP_JUMP(1.30) × expected` 判跨波界——完美线性时该比值恰为 1.0，不受 scale 间距影响；同时要求曲线时间随 s 单调不减（容差 `MONOTONE_BAND = 0.15`，波模型不可能出现反向台阶，超出容差即非波形落 mixed，容差内视为中位数噪声回摆）。抗噪取舍：读数本身是 15 轮交错中位数，单点噪声难成反向台阶，故不另加"超额须多对稳定成立"的条件（W=1.2 形态只有一对触发，多对要求反而漏判）。

**fixture 预期判读**（`tests/test_probe_recipes.py`，20 → 23 例，回归 **2290 passed**）：

| 曲线 | 判读 |
|---|---|
| W=1.2 台阶 `[(0.25,0.50),(0.5,0.50),(1.0,1.00)]` | `stepped`，detail `step between s=0.25 and s=0.5: r=0.500 vs linear 0.250` |
| 线性 `[(0.25,0.26),(0.5,0.52),(1.0,1.00)]` | 仍 `linear`（相邻对比值 1.00/1.04；新增反例 `test_linear_curve_is_not_misread_as_stepped` 钉死不被误触） |
| flat `[(0.25,1.02),(0.5,1.01),(1.0,1.00)]` 与无基线用例 | 行为不变（flat 仍最前；无基线仍 mixed） |

新增/换值说明：`test_wave_jump_flavor_reads_stepped`（W=2.2 波形 1/2/3，台阶点全在 LINEAR_BAND 内，钉死 stepped 先于 linear 的顺序）；`test_small_backward_wiggle_on_plateau_stays_stepped`（0.02 噪声回摆不丢判，钉住 MONOTONE_BAND 语义）；**mixed fixture 换值**：旧 `[(0.25,0.65),(0.5,0.80),(1.0,1.00)]` 在新判据下两对均为"缩小不按比例省时"（比值 1.63/1.60），恰是"每一档都付尾波"的真台阶形态 → `stepped`，不再是"无干净形态"样例；换成非单调摆动 `[(0.25,0.90),(0.5,0.70),(1.0,1.00)]`（缩小 grid 反而变慢，回摆 0.20 > MONOTONE_BAND，波模型不可能）。

**文档同步**：脚本模块 docstring 判读表、`read_curve` docstring、`occupancy_scaling.md` 判读表行统一为方向正确表述（"相邻档小档时间 ≥1.30× 线性期望——缩小 grid 跨过波界，时间下降不足线性比例"）；`skills/probe-recipes/README.md` 无方向性措辞，未动；`phase_ablation.py` / `effective_bw.py` 判读方向复查无误，未动；READ/RECIPE/PROBE 行文法未变（`test_shared_line_vocabulary` 继续过）。改动仅触及围栏内四文件 + 本文件追加节。
