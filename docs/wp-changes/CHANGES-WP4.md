# CHANGES-WP4 — 搜索机制升级

分支 `feat/wp4-search`（基于 main @ 5030503），worktree `/home/zcz/forgeKernelOptimize/wp4-search`。
代码三个任务（LANDABLE / SWEEP_REQUEST / 慢决策记账）已经总工程师审查通过；本文件是补交的
交付文档，一个 commit（`wp4: CHANGES-WP4.md`），**不含任何代码改动**。分支留在本地，不合并
main、不 push。

---

## 0. 测试证据（本 worktree 实测）

```
$ ~/venvs/forgekernel/bin/python -m pytest tests/ -q
................................................................         [100%]
2080 passed in 225.94s
```

- 基线（main @ 5030503，同环境）2054 passed / 0 failed；本包新增 **26** 个测试
  （`tests/test_wp4_search.py` 20 个 + `tests/test_sweep_run.py` 6 个），全部通过。
- 新增 fixtures（3 个，`tests/fixtures/wp4/`）：`lever_price_stack.json`（含 all_pos 的价格表）、
  `sweep_recipe.json`（sweep 配方样本）、`sweep_job_output.txt`（sweep job 输出样本）。
- 环境说明：解释器为 `~/venvs/forgekernel/bin/python`（复用项目 venv）。基线阶段该 venv 缺
  torch/numpy 时有 2 个环境性 FAIL（`leaderboard.py` 顶层 `import torch`），按守则从
  `~/tools/wheelhouse` 本地缓存补装 torch 2.15.0+cpu + numpy 2.5.3 后基线全绿（未重复下载）。

## 1. 任务 1 — LANDABLE：定价组合自动落地（teacher 触发、学生落地、标准 gate 兜底）

**触发条件**：定价 job 的 STACK 行里 `all_pos_pct` 非空、`all_pos_levers` 非空，且
`all_pos_pct` **严格大于该定价 job 自己的噪声底**（`PRICE noise_floor=` 行，解析进
`stack["noise_pct"]`；早于该字段的旧表回退 `_LEVER_NOISE_DEFAULT_PCT = 0.5`）。

**渲染格式**（价格块顶部一行）：

```
LANDABLE: all-pos levers=<L1,L3,L4> expected=+12.30% (floor=0.50%) priced_on=<best tree sha> ——
teacher 可用一个 milestone patch 把这些 lever 合并为单一决策行落地；学生按 rung 一次交付（普通
bundle 交付，标准 eval gate 判定）。
```

**落点**：`_landable_line`（driver.py:4443），由 `_lever_price_block`（:4469）在块顶调用
（:4483，`## Pricing table` 头之后、lever 表之前）。纯渲染、无副作用。

**teacher 消费路径**：`meta/diagnose_prompt.md` 的 **Pricing-table alignment** 条目末尾
（价格块消费段，WP4 拥有的段落）新增 "LANDABLE first" 规则：LANDABLE 优先于逐 lever 提案；
teacher 把列出的 lever 合并为**单一决策 rung** 写进 milestone patch；rung 的 edit 文本逐字抄
设计回合 levers.json 的 old/new 对（改写文本会被当作不同 lever 重新定价）；落地后既有
re-price 机制照旧在新 best 上复验。

**落地走标准 eval gate 的保证**（driver 不应用任何 edit，"student line stays teacher-free"）：

- `test_landable_keeps_the_student_line_teacher_free` —— 含 LANDABLE 表的完整 meta step 之后，
  kernel 文件逐字节未变（driver 没有落地路径）；
- `test_inner_iteration_correctness_fail_reverts_and_resets`（既有）—— 落地即普通 record →
  eval，correctness FAIL 是硬 REVERT；
- `test_sweep_winner_delivery_is_judged_by_the_standard_gate` —— 同一性质在 sweep 通道的锁死。

## 2. 任务 2 — SWEEP_REQUEST：扫描型决策（teacher 出题、driver 执行、学生落地）

**决策行 schema**（milestone §3 决策块新增可选字段，teacher 预写；学生 prompt 的 sweep 禁令
原文保留，仅在其后追加一句 "sweep decisions are teacher-authored: trigger with SWEEP_REQUEST,
then land only the winner"）：

```json
sweep: {"variants": [{"name": "tile64", "edits": [{"old": "...", "new": "..."}]}, ...],
        "criterion": "min same-job ratio above floor"}
```

- ≤ 8 变体（`_SWEEP_MAX_VARIANTS = 8`，driver.py:4117）；criterion 唯一合法值
  （`_SWEEP_CRITERION`，:4118）；每个变体 = exact-text edit 对列表（与定价 lever 同纪律）。

**执行流程（文字版）**：

1. 学生到达该决策，把配方**逐字**复制到 `notes/<decision_id>.json`，回合以一行
   `SWEEP_REQUEST: notes/<decision_id>.json` 结束（student_prompt "Ending this turn" 新增 A2）。
2. driver 在 probe 循环内先于 `parse_student_turn` 识别该标记（driver.py:2159，`_sweep_request_path`）。
3. **防夹带校验**：从 milestone §3 决策块提取预写配方（`_milestone_sweep_recipe`，:4926，
   `_decision_blocks` + `json.JSONDecoder().raw_decode`），对提交文件与配方各算
   **canonical-JSON sha256**（`_canonical_recipe_hash`，:4918：sort_keys + 紧凑分隔符——
   格式与键序无关、内容敏感）比对；形状校验 `_sweep_spec_refused`（:4953）。
   无配方 / 超 8 变体 / 坏形状 / **hash 不一致 → 拒绝且不买任何 GPU job**（回合照耗）。
4. 校验通过：`_seed_sweep_toolkits`（:4979）把 sweep_run + lever_pricing_run vendor 进 notes/
   并 commit → driver 自己的 spec 副本注入 `kernel_file`（**绝不改学生文件**，学生文件保持
   milestone hash）→ commit + push `-i<n>sweep` 分支 → `_sweep_job`（:5050）提交单个 job。
5. `_sweep_job` 的 settle/hygiene 与定价同款：归档重读（`_sweep_log_settled`）、NODE-DIRTY
   重买一次（复用 lever_pricing_run 的 node hygiene 与 NODE-DIRTY 标记）、SWEEP verdict 识别
   （定价的 PRICE/ABLATE 检查不认 SWEEP 行，故单独实现，避免把成功 sweep 误判"无 verdict"重买）。
6. `parse_sweep_results`（:5000）解析行；反馈 `_sweep_feedback`（:5071）把测量表 + 赢家交还学生。
7. 学生下一轮**只落地赢家**（普通 record block → 标准 eval gate，correctness FAIL 仍硬 REVERT）。

**机器行**：`SWEEP name=<n> ratio=<r> floor=<f%> delta=<±x.xx%> ms=<m> ok=<pass|fail>` /
`SWEEP-WINNER name=<n>`（无赢家时 `name=none`）/ `SWEEP-OK variants=<n> rounds=<r> base_ms=<..>` /
`SWEEP-FAIL reason=<...>`。

**超时与预算**（`skills/sweep-decision/sweep_run.py`）：`SWEEP_BENCH_TIMEOUT = 900`
（对齐定价工具 `NCU_TIME_TIMEOUT`）、`SWEEP_BUDGET_S = 3600`（对齐 `ABLATION_BUDGET_S`），
并 clamp 到 job 剩余时钟（`TASK_TIMEOUT` − `ROLE_PASS_MARGIN_S`），保证收尾行先于 gpu_entry
的 kill 落进日志。预算归类：每个 SWEEP_REQUEST 占用 probe 循环的一个回合（复用
`max_probe_rounds` 语义），**不新增预算维度**。

**参数化复用**：sweep_run.py bare-`import lever_pricing_run`（notes/ 平铺 vendored），复用
`run_bench` / `apply_lever` / `noise_floor` / `make_scorer` / `ratio_reading` / node hygiene；
**lever_pricing_run.py 一字未改，CLI 完全兼容**。

## 3. 任务 3 — 慢决策边际记账 + REMOVE_CANDIDATES

- **新字段**（契约 C2：`inner/trajectory.py` **恰好追加一行**，:115）：
  `regression_vs_best_pct: float | None = None`
- **helper**（全部在 driver.py 自有区域 :5172-5256）：
  - **C1 垫片**（:5185-5190）：`try: from forgekernel.loop.parse import resolvable_diff` /
    `except: return 0.015`（WP1 合入前的 legacy 回退）。全仓唯一一份，无第二实现；
    `test_the_shim_tracks_wp1_whenever_it_lands` 动态锁定两个状态；
    `test_the_fallback_engages_when_the_symbol_is_missing` 用子进程（删名后重导入 driver）
    锁 import 期回退分支。
  - 账本 `_WP4_SLOW_LEDGER`（:5193，进程内 `{decision_id: [(iter, delta_pct, workload)]}`）
    + 只读访问器 `wp4_slow_ledger()`（:5196）。
  - `_wp4_note_regression`（:5201）：`regression_vs_best_pct = (gate_ms − best_ms_before)
    / best_ms_before × 100`；显著判据 `delta_pct/100 > resolvable_diff(records, gate_window)`
    （仅正回归时才 load records，开销可忽略）。
  - `_wp4_remove_candidates_block`（:5227）：**从 records 渲染**（重启安全），为每个带显著
    回归的决策列出 known_levers 中 `kind=="lever"` 且 `decisions` 含它的 lever 候选。
- **KEEP 段插入（:2439，恰好 1 行，满足 ≤2 限制）**：`_wp4_note_regression(ctx, rec, iter_no,
  cluster)`，位于 KEEP rec 构造与 `append_record` 之间。
- **REMOVE_CANDIDATES 渲染**（meta 诊断注入，见 §4）：
  `REMOVE_CANDIDATES: <decision_id> Δ+x.xx% @<workload> (lever 候选: <L1, L2> | no priced lever)`
- **红线**：无自动回退——relaxed-revert 原样（`test_a_keep_slower_than_the_best_is_booked_
  and_rendered` 断言 verdict 仍 KEEP、无 reset 调用；`test_bookkeeping_never_touches_the_gate`
  端到端锁"记账不进 gate"）；渲染块文本明示 "no automatic revert"。

## 4. 审查落点说明（供集成步使用，如实记录）

1. **REMOVE_CANDIDATES 注入点在 `run_meta_step` 内原 6373 行一带**
   （现分支 :6810-6812，插在 `known_levers = _merge_lever_tables(...)` 之后）——位于 brief 命名
   区间 6169-6254 **之外**、同一函数（diagnose_extra 组装区）更靠后的位置。它与 WP3 的 ledger
   注入（原 6214 一带）**不同行、不同小节**：WP4 小节只在 lever 表注入（原 6352，`_lever_price_block`
   调用处）与 revisions 块之后追加，不动任何他人小节，预期可干净合并。**建议注入顺序**：
   SOL 头 → NCU 报表 → lever 价格块（含 LANDABLE）→ **WP3 LEDGER → REMOVE_CANDIDATES**
   （REMOVE_CANDIDATES 放最后：它是基于全量 ledger 的收尾提示，读在所有测量证据之后）。
2. **`skills/README.md` +1 行**（sweep-decision 索引条目）——超出原围栏清单的一行，声明为
   **文档索引性质**：`tests/test_skills.py::test_every_skill_has_a_skill_md_and_the_readme_lists_it`
   硬性要求 README 列出每个 skill 目录，新增 `skills/sweep-decision/` 不补此行则全量不绿。
   未改任何他人条目，仅插入一行表格。

## 5. 围栏自查表（`git diff main --name-only` 全量 14 文件）

| 文件 | 围栏依据 | 说明 |
|---|---|---|
| `src/forgekernel/loop/driver.py` | lever pricing 全区 / probe-SWEEP 分支（2157-2214）/ KEEP 段（仅 1 行）/ :4086 附近常量 / meta 注入区自有小节 | 8 个 hunk 全部在授权区；**默认参数区（340-435）未动**（`# --- WP4 ---` 常量块按 brief 落在 :4114-4119）；KEEP 段恰好 1 行（:2439）；probe 分支只加 SWEEP 识别（:2159，分支内部，未动 WP1 的 watermark / NCU 区域）；sweep 机制与记账 helper 为新函数块（:4886-5256，插在 `_lever_spec_refused` 与 `_banked_lever_table` 之间，行号因前文插入越过 5000 快照线——内容全为新增，见 §4.1 同类说明） |
| `src/forgekernel/inner/trajectory.py` | 契约 C2 | 恰好 1 行（:115），schema 其余未动；`build_digest` 函数体未动 |
| `meta/diagnose_prompt.md` | 仅价格块消费段 | Pricing-table alignment 条目内追加 "LANDABLE first"（+7 行）；读数顺序段（WP2）/ R2 段（WP6）未动 |
| `meta/student_prompt.md` | :41-42 与 :144-157 附近 | sweep 禁令原文保留 + 追加一句（:43）；"Ending this turn" 新增 A2 选项（+4 行） |
| `meta/verdict_probe_brief.md` | 允许（probe 协议措辞对齐） | 末尾 +8 行：SWEEP_REQUEST 与 teacher verdict probe 的通道区分 |
| `skills/sweep-decision/`（SKILL.md, sweep_run.py） | 允许（新增） | 新 skill；执行器见 §2 |
| `skills/README.md` | **超出清单，见 §4.2** | +1 行索引，文档性质 |
| `tests/test_wp4_search.py`, `tests/test_sweep_run.py`, `tests/fixtures/wp4/*` | 允许（tests/） | 本包测试与 fixtures |
| `CHANGES-WP4.md` | 交付物 | 本文件 |

**明确未触碰**：`loop/parse.py`（resolvable_diff 仅经 C1 垫片消费）、`timing.py`、
`_confirm_suspicious_watermark`（原 1457-1558）、`build_digest` 函数体、run_two_layer 的
meta/预算/蒸馏段（WP5/WP6）、`_ncu_profiler`（原 1748-1796）、`_persist_lever_table`
（WP3 归属，函数体未动）、probe 分支以外的全部 WP1 区域。

## 6. 提交清单（分支 tip 起回溯）

```
wp4: CHANGES-WP4.md                       ← 本 commit（仅文档）
WP4: subprocess lock for the C1 shim fallback (parse import without the symbol)
WP4: CHANGES-WP4.md — protocols, schema, line map, package seams, fence notes   ← 上一轮的文档初版（被本文件取代）
2f494dc WP4 task 3: slow-decision regression bookkeeping + REMOVE_CANDIDATES
27fa714 WP4 task 2: SWEEP_REQUEST — teacher-authored sweeps, driver-executed, student-landed
305726d WP4 task 1: LANDABLE notice on the price table (all-pos above the job floor)
```

代码审查通过后的本轮**零代码改动**；分支停在本地等待集成。
