# CHANGES-WP16 — 主动搜索与种群治理

分支 `feat/wp16-search-population`，基线 `origin/main @ 359696b`（二代集成终态）。
基线 pytest：**2437 passed**；完成后：**2463 passed**（+26 个新 WP16 测试，全绿）。
工作区 `wk3-wp16`；测试环境 `~/venvs/forgekernel/bin/python -m pytest tests/ -q`。

dry-run 样例：`dry-run-wp16.md`（由 `dry_run_wp16.py` 确定性生成：COMBO-POOL 注入 /
pattern search 读数表 / 重启靶子行 / EXPIRED 行 / 耦合图边表，五类齐全）。

---

## 任务 1（P0）：收缩式 pattern search（乙A3）

**触发**：`_meta` 注入区在 STAGNATION 或 phase-exhaustion 触发时（`driver.py` 注入区，
COMBO-POOL 之后、STAGNATION 块之前渲染）调用 `_run_pattern_search(ctx, be, trigger)`。

- **教师指定轴的通道**：milestone 决策行 `pattern_search:` 字段（与 `sweep:` 同一通道与
  解析纪律：`_milestone_pattern_search_recipe` 从 milestone 文件本身读取，raw-decode JSON；
  学生不可走私）。格式：
  `pattern_search: {"axes": [{"name": "tile", "old": "TILE = 64", "new": "TILE = {v}", "values": [16,32,64,128]}], "stages": 3}`
  ≤2 轴（`_PS_MAX_AXES`），每轴 2..7 个数值（`_PS_MAX_VALUES`），`new` 模板恰含一个
  `{v}` 占位符，`old` 为精确现存文本（exact-once 编辑纪律同 sweep/pricing）。
- **收缩网格**：stage 1 计时教师全部取值；stage>1 只计时 `ps_next_points`（纯函数）给出的
  中点——赢家与相邻已测点的中点，整数收缩为整数、相邻整数不再产生新点（半径因点距减半而
  减半，compass/pattern search 语义）。`ps_stage_variants`（纯函数）产出变体与
  name→(axis,value) 映射，重名追加 `_r<stage>`。
- **计时内核复用**：每 stage 一个 sweep 风格 job——`_seed_sweep_toolkits` + `_sweep_job`
  （`_SWEEP_CMD`，rounds=2），复用 sweep_run 交错计时与噪声地板读法。**接缝注记**：WP17 的
  `screening.py` 不在本基线（screening.py 未出现在 loop/），按发包指令走 sweep_run 直调过渡，
  **不自建第二份筛选基建**；集成步按契约 C9 收编到 screening 原语。
- **LANDABLE 渲染**：`_pattern_search_block` 渲染读数表 + 赢家行
  `LANDABLE: pattern-search <name> (axis = value) expected=…`，块内明写落地唯一走
  教师 → 学生 patch → 全量 gate（契约 C9）。无赢家时块明说"邻域测平，触发器的结构化诉求
  仍然成立"。
- **独立预算类**：`[loop] pattern_search_budget`（默认每 run ≤2 次，0=关），**不占** probe
  轮 / 学生轮 / 确认预算；状态持久化 `notes/pattern_search.json`（重启安全）。每搜索 ≤
  `pattern_search_stages` 个 job（默认 3，clamp 1..3）。截止前 60 分钟内不点火（与
  re-price 同闸）。

## 任务 2（P1）：SWEEP 轮间 Successive-Halving（乙A4）

- `skills/sweep-decision/sweep_run.py`：spec 新键 `"halving": true`。每轮完成后淘汰存活者
  中**最差的一半**（`halving_cut`，保 ceil(n/2)），被淘汰者不再计时；配方轮数照跑，释放的
  预算给幸存者加轮（上限 2×配方轮，pass 预算闸不变）。每条淘汰打印
  `SWEEP HALVING round=<r> kept=… eliminated=…`（日志行，非 verdict）。
- **噪声门控**（纪律同 MC 防线）：存活者 <4 不淘汰；每变体只有 1 轮读数不淘汰（单次抽签
  定不了名次）；与淘汰线的分差 ≤ job 自身噪声地板（`lpr.noise_floor`，终表的同一规则）的
  近差不淘汰——噪声大的节点上淘汰自动收敛为全网格（安全方向）。
- **淘汰只作用"下一轮还测谁"，不产 verdict**：终表与赢家仍读全部测过的变体（含被淘汰者的
  部分读数），与无 halving 完全同一规则。
- **教师 recipe 只给值域**：`halving` 配方可写 `axes`（`values` 2..7 点或
  `range:[lo,hi]+points` 2..16 点，≤2 轴）替代枚举 `variants`；驱动在 submit 时展开
  （`_expand_sweep_axes`，确定性 linspace、整数收缩），学生的 hash 校验文件仍是 milestone
  原文。变体上限：普通 8（`_SWEEP_MAX_VARIANTS` 不动），halving 32
  （`_SWEEP_MAX_VARIANTS_HALVING`，== sweep_run.MAX_VARIANTS_HALVING）。
- **回测**：`scripts/backtest_sh.py`（沿 `backtest_early_stop.py` 先例，torch-free、stdlib）。
  口径与门槛见下文"SH 回测粘贴"。协议语义不动：SWEEP 学生出题教师 recipe、单 job 交错计
  时、赢家走普通落地 + 全量 gate，全部保持。

### SH 回测赢家恢复率（粘贴）

```
halving backtest: 4000 trials/cell, recipe rounds=3, equal bench budget per trial

      grid   sigma  budget  fullV obj   SH obj full2V obj  rule loss  gates
   8 vs 16      0.5%      24     0.8009   0.9980     0.9980    1/8       rule:PASS bonus:PASS  (rec: fullV 91.9%, SH 82.7%, full2V 81.3%)
   8 vs 16      2.0%      24     0.8022   0.9811     0.9813  341/355     rule:PASS bonus:PASS  (rec: fullV 72.4%, SH 46.5%, full2V 47.0%)
   8 vs 16      5.0%      24     0.8025   0.9533     0.9539  776/770     rule:PASS bonus:PASS  (rec: fullV 50.4%, SH 26.5%, full2V 27.9%)
  16 vs 32      0.5%      48     0.7986   0.9971     0.9972    1/1       rule:PASS bonus:PASS  (rec: fullV 86.7%, SH 71.9%, full2V 72.3%)
  16 vs 32      2.0%      48     0.7955   0.9777     0.9771  354/351     rule:PASS bonus:PASS  (rec: fullV 57.5%, SH 34.0%, full2V 34.2%)
  16 vs 32      5.0%      48     0.7930   0.9484     0.9490  777/780     rule:PASS bonus:PASS  (rec: fullV 33.6%, SH 17.2%, full2V 18.4%)

fullV = the current sweep on the V-point grid; SH = the WP16 halving sweep on the 2V-point
grid; full2V = a plain full sweep over the same 2V points at the same budget (fewer rounds).
obj = the winner's mean true objective normalized to the 2V grid's range (1.0 = its true
best); rec = winner-is-true-argmax rate (reported).
GATE rule: at equal budget and equal grid, SH's mean objective is within 0.01 of the full
grid's, and its paired tail is symmetric (|wins - losses| <= 2% of trials; the swings' rate
tracks the cell's noise — elimination only routes the budget, it never decides).
GATE bonus: SH(2V) beats the current full(V) sweep in mean true objective — the freed budget
affords the wider grid.
RESULT: PASS
```

读法：等预算下 SH 的规则成本 ≈ 0（同网格均值差 ≤0.001、尾部两侧对称，>5% 幅度的摆动频
率由格子噪声决定而非规则——`wins≈losses`），而等预算换来 2× 网格后真实目标值从 ~0.80 抬
到 0.95–0.998（bonus 门槛全 PASS）。argmax 恢复率按报告口径打印：宽网格的 argmax 本身是
更难的目标（点距更细、噪声不变），恢复率下降正是 halving 用它换期望目标值的公开交易。

## 任务 3（P1）：MC 预筛噪声防线（乙A6）

- `_mc_top2_gap_pct(rows)`（纯函数）：预筛行表中两名 smoke-pass 且有 device 读数的幸存者，
  winner=更低 ms，gap=快多少 %。
- `_mc_deliver` 内：赢家存在且 `confirm_budget_per_iter > 0` 且 top-2 差距 <
  `resolvable_diff`（契约 C8 直接导入，`floor/cap` 同 run 配置）→ 追加一次预筛
  （`_mc_revote`）：top-2 两变体、rounds 2→3（`_MC_ROUNDS + 1`）、同一 confirm 预算类、同
  一 job 通道（`_mc_job`）与新 branch。revote 读数替换同名行；新赢家只在 top-2 内产生
  （不扩大选择面）；job 失败/无 verdict → 首轮读数照旧。**全 DNF 路径逐字节不变**。

## 任务 4（P1）：亚噪声复利池（丙WP16-T1/T2）

- **唯一实现**：`src/forgekernel/loop/lever_pool.py`（纯函数，210 行）：admission /
  should_trigger / pool_age_steps / consume / combo_null_ledger_rows / bundle_levers /
  combo_pool_block / parse_pool / serialize_pool。driver 是唯一写者与 job 执行者。
- **入池**：`_persist_lever_table` 落表时旁路写 `_pool_admit_table`——`null` 定价、带 banked
  recipe、无同名 pending 的 lever 入 `notes/lever_pool.jsonl`（`harmful/noop/faded/coupled/
  independent` 不入池；`reprice` 表不触发入池；无 recipe 不入池）。regime 字段只记录不判定
  （WP14 领地不越界）。
- **触发**：`_meta` 注入区（契约 C10 位置：REMOVE_CANDIDATES 之后、STAGNATION 之前）：
  pending ≥2 且（`detect_plateau` 同参为真 或 池龄 ≥ `[loop] pool_age_trigger`（默认 2，
  0=关；步钟 = `_plateau_diagnoses(meta records)+1`，与诊断步同源））→
  `_run_pool_bundle`：**复用 STACK 基建**——一个常规 pricing job（`_lever_price_submit`，
  pooled levers 即表，工具自己的 all-on/STACK pass 出 bundle 读数），driver 不建第二份堆叠
  实现；bundle 表以 `reprice=True` 落账（不进教师表、不触发入池）。
- **bundle null** → `combo_null_ledger_rows` 写 `hw_kb/falsified_ledger.jsonl`
  （`class: combo-null`，幂等键 (run, lever, combo-null)）+ 池清空（consume → combo-null）。
  **bundle pays** → 读数进 COMBO-POOL 块，池条目标 `priced`（防重复触发），落地仍是教师的
  （块内明写 normal gate 通道）。job 失败/拒绝 → 池原样保留（缺测量不机器改判）。
- **null 规则改写**（`diagnose_prompt.md:241-242`，净 +9 行，≤30 行红线内）：null 只禁止作
  **standalone** 方向重提；组合开放，由 COMBO-POOL 块承载，combo-null 才关闭组合。
- **COMBO-POOL 注入块 ≤10 行**（`lever_pool.combo_pool_block` 硬截断，测试钉住）；空池渲染
  ""（回归测试 + WP10 集成测试双重钉住）。
- **STAGNATION 交互**：池有未测组合且 bundle 未在本步跑过 → STAGNATION 块延后一轮
  （`ctx.wp16_pool_deferred` 一次性；延后步 override 不被 honored——`stagnation is not None`
  守卫自然覆盖）；池空 → 一切照旧。**池空回归现状逐字节一致**：
  `test_wp10_search_depth.py` 全部集成级测试（精确 prompt 断言）+ 新增组合测试双保险。

## 任务 5（P1）：k-best runner-up 池 + STAGNATION 重启 + 骨架墓园（甲G16+丙合并）

- **池**：`_runner_up_pool(records, gate_window, k)`（纯函数）：top-k KEEP 树按 gate 读数排
  名（rank 1 = 现任最佳，身份规则同 `_best_keep_record`；`reading_of` 复用行排除——池条目
  必是买过自己 eval 的树；sha 去重、iter 破平）。快照 `notes/runner_up.jsonl`（确定性字
  节）在每个 KEEP 落账后与 REVERT reset 点前刷新——REVERT 不再让 runner-up 的账目失联：
  sha+读数+已购价格持续可寻址（`_write_runner_up_pool`，纯文件写，零后端调用——KEEP 树本
  就是 reset 目标的祖先，commit 不会失联，文件补的是账本不是可达性）。
- **重启靶子行**（契约 C10 "STAGNATION +重启靶子行 ≤6 行"）：`_restart_target_lines` 在
  STAGNATION 块**之后**追加（块格式与 override 通道一字不动，只追加）：rank-2 的决策、
  sha、gate 读数、与 best 的差距、"its eval and NCU are already paid"、复活是合法结构路
  线、复活必重验。无 #2 / 池空 → 空串。
- **复活必重验（红线追加②）**：`_revival_note`——投递 base_sha 命中池条目时打审计行，明写
  复活的读数只经该投递自身的全量 gate（watermark / 时钟漂移 / fresh-recheck 如常全开）重
  新进入 best/plateau/pricing；复活本身 = 教师 milestone patch + 学生普通投递，无旁路。
  三态：重验过（KEEP 记账）/ 不过（REVERT 记账）/ 池空（无行），单测钉住。
- **骨架墓园**：`_run_arch_probes` 落选者结构化存档 `notes/arch_graveyard.jsonl`——只有
  **decisive**（top-2 差距 > 分辨率地板）时败者入墓（inconclusive 对都活着）；无读数候选
  也入墓（买过 job 没出声的下批教师该知道）。行含：候选、读数 ms、死因（含差距与地板）、
  `tunables_seen`、`hybrid_of`。reset 前读 recipe 元数据、reset 后落盘（untracked 存活）。
- **hybrid_of**：recipe JSON 可带 `"hybrid_of": ["arch_0", "arch_1"]`——probe 批次照常占预
  算槽位，血缘进日志与墓园行（记录不判定）；`ARCH_CANDIDATES: hybrid_of:<a>+<b>` 行由教师
  在 milestone 书写（`_ARCH_CANDIDATES_RE` 的 `\S+` 捕获天然容纳），机器件按 recipe 字段
  消费。
- **carry 墓园件**：`scripts/self_evolve.py` `_write_carry` 增第 5 件
  `arch_graveyard.jsonl`（父代墓园直拷，schema 独立成件——与 WP19 的 carry 扩件并行落盘
  互不覆盖）。

## 任务 6（P1）：重定价覆盖面 + lever 耦合图（甲G12/G17）

- **EXPIRED**：`_mark_expired_levers(ctx, rec)` 在新最佳落账、cited-lever re-price 之后运
  行：最近一张教师表里 add 超噪声、milestone 未引用、且新树上无新鲜读数覆盖的赢家 → 写
  `verdict: expired` 修订记录（复用 price-revision 存储与渲染——`_REVISION_FLAGGED` 增
  "expired"，Price-tag review 块自动承载，教师读一次即弃静默）。已引用者归既有 re-price
  管辖；本树已重读者不标；截止前 60 分钟不标（与 re-price 同闸）。
- **爬升段提前重价**：`_climb_streak`（纯函数）——最新起连续 1-3% KEEP 计数；≥
  `[loop] reprice_climb_streak`（默认 2，0=关）时该步 re-price 门槛降为 0（`min_gain_pct`，
  skip 日志同步用实际阈值）。连续小步把 best 挪出引用价格管辖树的漂移，一次 new-best 步即
  被重价对齐。
- **耦合图**：教师 recipe 可带 `requires:` / `conflicts:`（键入 `_persist_lever_table` 白
  名单，与 `regime` 一样只记录；pricing 工具忽略未知键，安全）。`_lever_edges_block` 渲进
  **现有价格块**尾部：requires 边（`a → b`：先价先落 a）+ conflicts 边（`a >< b`：分开交
  付、绝不捆绑；all-on 已含两者 = 边过期）+ 未知目标点名丢弃；每类 ≤3 行封顶（每包增量
  ≤30 行纪律）。边是排序与交付建议，report-only（红线追加①：永不替代实测）；`_lever_
  price_block(table, specs=None)` 缺省参数——无 specs 时价格块逐字节不变。
- **重定价自动排序**：`requires` 边随 spec_levers 落账，`_reprice_cited_levers` 的 todo 按
  spec 顺序天然保持教师书写序（前置先价），边表在块内给教师同序建议。

## 任务 7（P1）：floor_progress 进展度 advisory（丙）

- `_floor_progress_line(ctx, records)`：`floor_progress = floor_ms / best_ms`——本 run 解析
  地板（WP9 `floor_for`，`_analytic_floor_ms` 单一来源，不建第二份地板）÷ 自身 best。
  渲染在教师 NCU 段尾 floor 行旁（契约 C10 floor/floor_progress 位置），report-only 行明
  写 "no stop or verdict reads it"；默认 target 1.2× 停止语义不动。无地板 / 无 best →
  空串（卡片无 problem.json 的 run 逐字节现状）。**不使用任何外部基准的评分概念**。
- `scripts/self_evolve.py`：每代门槛打印行旁一行 `[self-evolve] generation k:
  floor_progress=… (analytic floor … ms / best … ms) — report-only; the generation gate is
  unchanged`；链停规则（增益 vs 噪声地板）与 `--step` 目标不动。

---

## 配置词汇（契约 C17）

新键走 LoopContext 字段（None 哨兵）+ `_wp16_apply_cfg`（`_wp5_cfg_sections` 读卡层
loop.toml [loop] 自由表，global 层在下），**不经 cli.py**（WP24 禁碰清单）。五个键：
`pattern_search_budget=2`、`pattern_search_stages=3`、`pool_age_trigger=2`、
`reprice_climb_streak=2`、`runner_up_pool_k=3`；显式赋值（测试/调用方）永不被覆盖，坏值
落回文档默认。`global_config/loop.toml` 只加注释文档块（`# --- WP16 ---`，全局层词汇保持
封闭——`forgekernel.task._KEYS` 未动）。

## 触碰面（对照围栏）

| 文件 | 改动 |
|---|---|
| `src/forgekernel/loop/lever_pool.py` | 新（纯函数库，唯一实现） |
| `src/forgekernel/loop/driver.py` | 任务 1/3/4/5/6/7 全部 driver 侧（WP16 独立节段 + 注入区 + `_persist_lever_table`/`_reprice_cited_levers`/`_mc_deliver`/`_run_arch_probes`/KEEP 与 reset 点接线/`_wp16_apply_cfg`） |
| `skills/sweep-decision/sweep_run.py` | 仅轮间淘汰与 STACK 复用入口（`halving`/`halving_cut`/上限），协议语义不动 |
| `meta/diagnose_prompt.md` | null 改写 + COMBO-POOL 消费 + STAGNATION 靶子/pattern search 消费，净 +9 行（≤30 ✓） |
| `meta/verdict_probe_brief.md` | pattern search/池 bundle 注记 6 行（≤6 ✓） |
| `scripts/self_evolve.py` | carry 墓园件 + floor_progress 打印行 |
| `global_config/loop.toml` | `# --- WP16 ---` 注释键块（词汇封闭不动） |
| `scripts/backtest_sh.py` | 新（SH 回测） |
| `dry_run_wp16.py` / `dry-run-wp16.md` | 新（dry-run 交付物） |
| `tests/test_wp16_lever_pool.py` / `tests/test_wp16_search.py` | 新（26 测试） |
| `tests/test_price_staleness.py` | 一处断言更新（见下"行为变更记账"） |

**未碰**：screening.py（不存在于本基线）、parse.py、timing.py、`_confirm_suspicious_watermark`、
meta_milestone.md、student_prompt.md、cli.py、leaderboard.py、card_template、task.py、
taskcard.py、window_contract.py、scaffold.py、launch.py、__main__.py、binref/sealref tool、
NOTICE、README.md、docs/。

## 行为变更记账（红线 9）

1. **`test_price_staleness.py` 的 uncited 用例**：旧行为=未引用的过期赢家静默（无修订行）；
   新行为=按 甲G12 写 `expired` 修订行（仍不买 re-price job，gpu 调用断言不变）。该测试
   的 budget 用例断言同步改为"本用例不新增修订行"（用例间共享 evidence 目录，uncited 用例
   的 EXPIRED 行是设计产物）。band 口径 / best 数字语义 / gate 协议零变化。
2. **STAGNATION 块字节**：块本体与 override 通道一字未动（`_stagnation_block` 无 diff 语义
   变化；新增内容全部为块外追加：靶子行 ≤6 行、pattern search 表、COMBO-POOL 块）。
3. **MC/CON FIRM 路径**：近差防线只在 `confirm_budget_per_iter > 0` 时加一个 job；DNF 路
   径、赢家应用、标准 gate 全部不动。

## A/B 注记（红线⑤：复利池与 pattern search）

- **复利池**：上线前后同 job A/B 方案——同一任务卡（建议 rmsnorm 双 regime 卡）先跑对照
  （`pool_age_trigger=0` 且 COMBO-POOL 块禁渲染，等价池永久为空），再跑开池 run；比较（a）
  进入 bundle pricing 的池触发次数与 job 数，(b) bundle pays / combo-null 的实测分布，
  (c) plateau 步数与终值 geomean。换基座模型须重测（教师对池块的消费行为是模型侧变量）。
- **pattern search**：同上双 run——对照（`pattern_search_budget=0`）vs 开启（=2）；比较
  （a）STAGNATION 触发步数与 override 次数，(b) pattern search 读数表赢家最终经全量 gate
  的 KEEP 率（读数表 Δ 与 gate Δ 的吻合度即噪声防线有效性的直接度量），(c) 每 run GPU
  job 总数（预算类是否真的自付）。
- 两者均默认口径不改 gate：A/B 只比较预算消耗与最终 KEEP 终值，无任何判定路径变化。

## 搜索自主性边界自查（合并裁决风险 1：逐条核对落地路径未绕过教师/学生/全量 gate）

| 通道 | 读数生产 | 判定/落地 | 自查结论 |
|---|---|---|---|
| pattern search | driver 跑 shrink-grid job，产读数表 | 赢家 = LANDABLE 行渲染给教师；落地唯一走教师 patch → 学生交付 → 全量 gate；driver 零 kernel 编辑（`_run_pattern_search` 只 reset/seed/spec/submit/parse，finally 恢复树） | ✅ 未绕过 |
| SH 淘汰 | sweep job 内决定"下一轮还测谁" | 终表与赢家规则与无 halving 同一；赢家仍走学生 SWEEP 交付 + 标准 gate | ✅ 未绕过 |
| MC 防线 | revote job 重测 top-2 | 赢家仍只能取自首批两变体，仍经标准全量 eval gate（`_mc_deliver` 后续路径不动） | ✅ 未绕过 |
| 复利池 bundle | `_run_pool_bundle` 一个 pricing job（STACK pass） | pays → 读数进块，落地教师走正常通道；null → combo-null 台账（教师侧先验）；**无任何 verdict/watermark 改写** | ✅ 未绕过 |
| runner-up 复活 | 池文件 + 靶子行（信息） | 复活 = 教师 milestone patch + 学生普通投递，读数经全量 gate（watermark/漂移/fresh-recheck 如常）才入账 | ✅ 未绕过（复活必重验，红线追加②） |
| EXPIRED / 耦合图 | 驱动记账与渲染 | 教师侧信息；不建价格、不判死活（EXPIRED 说的是"旧树价格无覆盖"，不是"lever 死了"）；边表 report-only | ✅ 未绕过 |
| STAGNATION 交互 | 池延后一轮加压 | WP10 块格式/override 通道一字不动；池空逐字节现状（WP10 集成测试全绿 + 单测） | ✅ 未绕过（追加⑤：池不机器改判） |

代理模型/地板/先验只排序不定价（追加①）在任务 7 与墓园同样成立：floor_progress 与墓园
读数均 report-only，永不进 gate。

## 验收对照

- ✅ pytest 全绿：2463 passed（基线 2437 + 26 新增）。
- ✅ lever_pool 快照测试（`test_serialize_parse_roundtrip_is_deterministic`：同输入同字节）。
- ✅ pattern search 网格单测（`test_pattern_search_contracts_on_a_known_surface`：已知响应
  面（峰 37）三轮收缩收敛到 ±1）。
- ✅ SH 回测赢家恢复率粘贴（上文，RESULT: PASS）。
- ✅ MC 防线近差用例（`test_mc_top2_gap_pct` + 带内/带外路径）。
- ✅ 池三路径（`test_trigger_three_paths`：plateau / age / 关；另有 <2 与空池）。
- ✅ STAGNATION 池空回归逐字节一致（`test_stagnation_block_byte_identity_…` +
  `test_wp10_search_depth.py` 集成级全绿）。
- ✅ runner-up 池复活三态（`test_revival_note_three_states` / `test_restart_target_lines_
  three_states`：有 #2 / 无 #2 / 池空）。
- ✅ dry-run：COMBO-POOL 注入样例、pattern search 读数表渲染、重启靶子行、EXPIRED 行、耦
  合图边表（`dry-run-wp16.md`）。

## 集成步接缝清单（交 WP23）

1. **screening.py 收编**：pattern search 的每-stage job 目前直调 `_sweep_job`（WP17 未在
   本基线）；WP17 合入后按契约 C9 把 `_run_pattern_search` 的 job 通道迁移到 screening
   原语（读数语义不变）。
2. **WP24 并行**：本包未触碰任何禁碰文件；driver.py / loop.toml 与 WP24 的 suite 最小删改
   若有残余冲突按 C18 归集成步。
3. **cli 转发**：无（C17 走 `_wp16_apply_cfg`，无 cli.py 增量，无 `m0_probe_rounds` 形状的
   断线风险）。
4. **combo-null 台账消费**：`_falsified_ledger_block` 现按 class 渲染，combo-null 行走通用
   路径已可读；如需专属排序（same-op combo 先行）归集成步微调。
5. **`halving` 上限 32**：若真机验证 32 点单 job 超预算，集成步在 A/B 矩阵后降
   `_SWEEP_MAX_VARIANTS_HALVING`（一次一翻，附证据与回退开关，红线追加③）。
