# CHANGES-WP17 — 筛选基建统一与 M0 保真（screening 原语 / 地板预排序 / 半实 ARCH_PROBE / 进程内变体 / 排程认证额度）

分支 `feat/wp17-screening-m0`（基于 kernel-ZXtune main @ `359696b`，即已发布 `ChaseZhang-ai/kernel-ZXtune` main；本地 clone 自 `/home/zcz/forgeKernelOptimize/zxtune-integration`，显式 `origin/main` 开出——clone 默认检出 integrated-v2，已按发包指令规避）。

| commit | 内容 |
|---|---|
| 本提交 | 任务 1–5：screening 原语 + 四通道迁移、地板预排序 + 台账、semi-real 两档 + 近差复测、进程内 bench worker、CANDIDATES 排程认证额度（机制 + SKILL/loop.toml + 测试） |
| （紧随） | `CHANGES-WP17.md` |

验收证据：`~/venvs/forgekernel/bin/python -m pytest tests/ -q` → **2468 passed, 0 failed**（基线 main = **2437 passed**；新增 31 个 WP17 测试，原有测试零删改语义——仅 `test_wp10_search_depth.py` 的 mc_prescreen 测试桩按新 API 补了一个 `clamp_pass_budget` 假函数，见 §3.4）。本机 GPU（RTX 3060 6GB）不代表目标硬件：验收以 pytest + fixtures + dry-run 为准（红线 11），board 端到端抽查注记见 §7。

dry-run 证据（pytest fixture 级 dry-run，输出原样摘录，三条均可复跑）：

1. **地板预排序日志**（`test_pre_rank_picks_the_floor_top_but_the_time_decides`）：
   `[loop] meta/distill: ARCH_PROBE probing the top-2 of 3 listed candidates (bracket 2, budget 3)`
   `[loop] meta/distill: ARCH_PROBE pre-ranked by the analytic floor (advisory — a sort suggestion, never a price or a verdict): arch_1 -> arch_0 -> arch_2`
   `[loop] meta/distill: ARCH_PROBE ledger -> <evidence>/arch_probe_ledger.jsonl (3 candidate row(s))`
2. **semi-real 两档 + 近差复测**（`test_near_diff_reprobe_one_job_redraws_the_pair`）：近差对触发一次复测（预算内 1 个空余 job、两配方同 job）→
   `[loop] meta/distill: ARCH_PROBE readings within the resolution floor (arch_0, arch_1) — one re-probe of the pair (1 spare job(s))`
   milestone head stamp 出现 `fidelity tiers (declared per recipe): arch_0=semi-real, arch_1=mock — …` 与 `re-probed once within the resolution floor (arch_0, arch_1); the re-draw replaces the first reading` 两行。
3. **排程认证记账**（`test_cert_scheduler_fires_on_the_schedule`）：
   `[loop] meta/wp17-cert: CANDIDATES head certification probe notes/cert.py -> PROBE FAIL (max_diff=unreported) (step 2, job ?) — the reserved round is spent`
   `[loop] meta/wp17-cert: workspace reset to the pre-certification head sha0; probe file(s) ['notes/cert.py'] re-materialised untracked (the -kbprobe branch keeps the commits)`

---

## 0. 总览与围栏声明

- **动过的文件**（全部在允许清单内）：`src/forgekernel/loop/screening.py`（新）、`src/forgekernel/loop/driver.py`、`skills/lever-pricing/lever_pricing_run.py`、`skills/lever-pricing/SKILL.md`、`skills/sweep-decision/sweep_run.py`、`skills/sweep-decision/SKILL.md`、`skills/arch-probe/SKILL.md`、`global_config/loop.toml`、`tests/test_screening.py`（新）、`tests/test_wp10_search_depth.py`（测试桩补一个 API 兼容函数）。**未碰**：timing.py、parse.py、四通道预算键（`max_probe_rounds`/`max_verdict_probe_rounds`/`m0_probe_rounds`/`m0_probe_candidates`）、`analytic_floor.py` 公式面、WP16 搜索策略、WP18 kb_cert_batch 消费逻辑、student_prompt.md、注入区（C10：WP17 不进注入区，全部新行为走 driver 日志与 milestone head stamp 既有通道）、WP24 禁碰清单（`leaderboard.py`/`card_template/`/`task.py`/`cli.py`/`docs/` 等零改动；driver.py 与 loop.toml 按 C2 追加纪律作业，残余冲突归集成步 C18）。
- **默认参数逐字节回归**：四通道在默认参数下的行为逐字节一致——既有 2437 个测试全数原样通过（含 `test_arch_probe.py` 14 条 stamp/预算/缺文件路径、`test_wp10_search_depth.py` 的 MC 预筛端到端、SWEEP 规格拒绝路径），另有 `test_stamp_default_bytes_unchanged` 以全等断言钉住 stamp 的前 WP17 字节。新机制（预排序/tier/复测/ledger/worker/排程）全部惰性激活：配方未声明新字段不触发排序与 tier 行、无空余预算不触发复测、默认 2×2 括号复测永不发生、worker 失败自动回退逐调用子进程、`cert_every_meta_steps` 默认 0=关。

## 1. 任务 1（P0）：统一 screening 原语（甲G14，契约 C9）

`src/forgekernel/loop/screening.py` 新建，为四通道公共基建的**唯一**实现，五件套：

1. **冻结 spec 校验**：`canonical_spec_hash`（canonical-JSON sha256）+ `sweep_spec_refused`（SWEEP 配方机械校验）。
2. **同 job 配对计时**：`lever_pricing_run.paired_order(variants)`（base 先行、变体按列序——SWEEP 与 MC 预筛逐行同构的 `order = ["base"] + …` 现在只有一处定义、两处真调用：`sweep_run.py` 与 mc_prescreen 内嵌源；driver 侧 screening.py 以注释锚定该纪律，注释与实现互指）。
3. **node-dirty 检测**：`node_dirty_reason`（PRICE-FAIL/ABLATE-FAIL/NODE-DITTY 三形态识别，单一规则）+ **GPU state 遥测消费**：`clock_drift`（WP1 `_sm_clock_drift` 原式迁入）与 `reading_dirty(gpu_state, ref_state)`（SM clock 带外 >2% 或温度差 >5°C → 该读数标 dirty 的判定原语；缺遥测一律 "unmeasured, never flagged"——WP1 原则）。
4. **预算钳制**：`clamp_bracket(want, listed, budget)`（ARCH_PROBE 括号钳制）；job 侧 pass 预算钳制统一为 `lever_pricing_run.clamp_pass_budget`（§3.5）。
5. **机器可读裁决**：`log_settled`（无 verdict 日志的归档重读恢复）+ `settled_job`（node-dirty / 无 verdict 无 traceback 各重买一次的策略）+ `pair_outcome(t0, t1, floor)`（ARCH_PROBE stamp 的 above/within-floor 决策核）。

### 四通道迁移对照表（函数级落点）

| 原手搓位置（driver.py） | 迁移后 | 语义保持方式 |
|---|---|---|
| `_canonical_recipe_hash`（体） | `screening.canonical_spec_hash`；driver 留同名委托 | 同一 sha256/canonical-JSON 参数，逐字节同值 |
| `_sweep_spec_refused`（体） | `screening.sweep_spec_refused`；driver 留委托 | 拒绝字符串逐字节同（`test_sweep_spec_refused_exact_bytes` 钉住） |
| `_node_dirty_reason`（体） | `screening.node_dirty_reason`；driver 留委托 | 三形态输出逐字节同 |
| `_TOOL_TRACEBACK`/`_NODE_DIRTY_MARKER`/`_ABLATE_FAIL_RE` 常量 | `screening.TOOL_TRACEBACK`/`NODE_DIRTY_MARKER`/`ABLATE_FAIL_RE`；driver 留别名 | 同值 |
| `_CLOCK_DRIFT_BAND` + `_sm_clock_drift`（体） | `screening.CLOCK_DRIFT_BAND` + `clock_drift`；driver 留委托 | 同带同式（WP1 判定消费点 `_confirm_suspicious_watermark` 不变） |
| `_sweep_log_settled`（体） | `screening.log_settled(word="SWEEP")` | 日志措辞/重读规则逐字节同 |
| `_mc_log_settled`（体，与上者重复） | 同一 `log_settled(word="MC")` | 同上 |
| `_sweep_job`（体） | `screening.settled_job(word="SWEEP", noun="recipe")` | 重买策略/日志措辞逐字节同 |
| `_mc_job`（体，与上者重复） | 同一 `settled_job(word="MC", noun="prescreen")` | 同上 |
| `_arch_probe_stamp` 两处 delta/floor 判定内联 | `screening.pair_outcome` | 同一算式同一 1.5% 缺省；stamp 渲染不动 |
| `_run_arch_probes` 的 `n = min(want, len(requests), budget)` | `screening.clamp_bracket` | 同一钳制 |
| job 侧：`sweep_run.py`/mc_prescreen（driver 内嵌源）各一份 `max(0, min(cap, left-margin))` | `lever_pricing_run.clamp_pass_budget`（两处调用点改一行） | 同式；`job_seconds_left=None` 时 cap 原样（hand-run 语义不变） |

未迁移（有意保留）：`parse_probe_verdict`/`parse_sweep_results`/`parse_mc_results`（parse.py 与通道专有协议行，属通道语义非公共基建）；`_lever_price_job`（杠杆定价的 PRICE/ABLATE 双 verdict 变体，与四通道重买策略同形但判词不同——其 dirty 检测已通过委托吃到统一规则）；sweep_run/mc_prescreen 的交错计时循环体（job 侧已共享 lpr 计时内核，本次再统一的是 pass 预算钳制与 bench 调用路由，见 §4）。

**WP15/WP16 消费不自建**：`screening.settled_job`/`log_settled`/`pair_outcome`/`floor_rank`/`reading_dirty` 即 pattern search（复用 sweep_run 交错计时内核）、复利池 bundle、runner-up 池复测、两级阶梯粗筛 nomination 的接入面；本包未实现任何搜索策略（围栏：策略归 WP16）。

## 2. 任务 2（P0）：ARCH_PROBE 地板预排序 + (family, arch) 台账（甲G15）

- **消费 WP9 单一实现，零第二公式**：配方新可选字段 `problem`（analytic_floor 的 problem schema：`axes`/`inputs`/`outputs`；`arch`/`family` 缺省自任务卡 `eval/leaderboard/problem.json` 经 `problem_from_card`——`_arch_probe_base_problem` 读的就是 `floor_from_op_dir` 读的同一文件）。`screening.recipe_floor(recipe, base_problem)` 合并视图后**直接调 `analytic_floor.floor_for`**；axes 按 key 深合并（tile 覆盖只写要改的轴）。配方未声明 `problem` → 不排序（默认字节不变）。
- **排序建议，永不价格（红线追加①）**：`screening.floor_rank`（stable：无 floor 的候选按文件序排在有 floor 者之后；空集 → 原序原样返回）。`_run_arch_probes` 仅当 `ranked != requests`（排序真实改变了次序）才打日志与在 stamp note 加 advisory 行，文案自述 "a sort suggestion, never a price or a verdict"。**判定权 untouched**：赢家/冻结/pin 一律来自时间读数——`test_pre_rank_picks_the_floor_top_but_the_time_decides` 专门构造"floor 最优者时间最慢"的反向用例，断言 pin 跟随时间赢家而非 floor 序。
- **预算内挑 top 而非文件序**：`probed = ranked[:n]`（n 仍为 `clamp_bracket` 产出；被切掉的候选数记入既有 cut note，排序致切时追加 "; probed top selected by the analytic-floor pre-rank (advisory)"）。
- **台账**：`<evidence>/arch_probe_ledger.jsonl`（`_evidence_dir` 既有通道；evidence 目录不可用时记日志跳过，永不影响探测结果）。schema（`screening.LEDGER_SCHEMA`，WP19 读数缓存 / WP16 墓园消费的三包接缝，集成步对齐）：`ts, op_family, arch, candidate, name, tier, floor_ms, bytes_min, flops, ms, verdict, job`。verdict ∈ measured / reprobed / no-reading / skipped-cut。默认写（键 `arch_probe_ledger=1`，kill switch 可关）；这是**记账新增件**而非 verdict 语义变化，stamp/log/判定字节不变（`test_stamp_default_bytes_unchanged` 钉住）。
- 配置：`[loop] arch_probe_floor_sort`（卡层，默认 true——惰性：无声明字段即无行为），经 `_wp17_apply_cfg` 解析（C17：卡层 free-form 键 + LoopContext None 哨兵，不经 cli）。

## 3. 任务 3（P1）：半实 ARCH_PROBE（乙A7）

- **两档解析**：配方可选字段 `fidelity: "mock" | "semi-real"`；`screening.recipe_fidelity`——缺省/拼错一律 `mock`（历史缺省，字节不变）。档位定义只进教师读集：`skills/arch-probe/SKILL.md` 新增 "Fidelity tiers" 节——semi-real 要求**主导数据轴流过真实尺寸数据**（memory 轴真实读写字节量、compute 轴真实 MMA issue 量），**仅计算逻辑可 no-op**；两档选择规则（数据移动/发射率差异 → semi-real；launch/occupancy 拓扑差异 → mock；不确定 → semi-real，轮数成本相同）。driver 无法验证数据尺寸（correctness 永不读——红线），故档位是教师自律声明，stamp 按声明如实标注。
- **近差强制复测**：`screening.near_diff_reprobe(readings, probed, floor, spare)`——决定性对差距 < resolvable_diff 且预算有空余 job（`m0_probe_rounds − len(probed)`，复测永不超买）时，返回该对（冻结 runner 本就接受多配方 argv，**一次复测 = 一个 job 载两配方**）；re-draw 替换首读数。默认 2×2 括号无空余 → 复测结构性不可达（字节不变）；bracket/预算放宽的 run（如 candidates=2, rounds=3）自动获得该防线。首读数与复测读数都入 ledger（verdict=reprobed）。
- **stamp 标注**：`_arch_probe_stamp_annotations`（tiers 行 + re-probe 行）只插在 stamp 闭行 `-->` 之前；声明缺失/复测未发生时两行都不存在（默认字节不变）。ARCH_PROBE 读数仍只定冻结、永不是 gate（`meta_milestone.md:175-185` 原则原文未动）；semi-real 仍只读时间。
- 配置：`[loop] arch_probe_near_diff_reprobe`（默认 true，同上惰性）。

## 4. 任务 4（P1）：进程内变体切换（甲G13）

- **机制**（全部在 `skills/lever-pricing/lever_pricing_run.py` + driver 内嵌 mc_prescreen 源，timing.py 逐字节未动）：新增长驻 **bench worker** 子进程——torch/CUDA context/卡 eval 模块 import 一次；每次计时调用经行协议 `RUN <id> <timeout>` → worker 内 `runpy.run_path("eval/bench.py", run_name="__main__")` 重执行卡 bench（模块命名空间每次全新——correctness、fresh inputs、配对窗每调用照跑，读数协议零改变）→ 捕获 stdout/stderr 回传 → `BENCH-DONE <id> rc=` 收尾。变体切换 = 写 kernel 文件（卡的 loader 每调用重新 exec kernel 文件——`load_entry` 本就是 by-path 逐次加载）。消灭的是每变体每轮的 torch import + CUDA context 冷启动（sweep job 预期 30-50% 提速）。
- **隔离与回退（读数语义一字不动）**：调用超时是**客户端**的——挂死调用按既有纪律 kill 进程组 → `sweep_gpu_orphans` → NODE-DIRTY 标记 → 通道既有 `settled_job` 重买一次；worker 干净死亡 = 崩溃调用（无读数，下一调用冷启动新 worker，等价于旧的逐调用子进程崩溃路径）；**completed 调用无 JSON = 不兼容信号** → 永久回退逐调用子进程并把该调用在子进程重读一次；`keep_cubin=True`（定价 cubin sha 路径）恒走子进程。
- **kill switch**：`FK_SCREENING_INPROC=0` → 恒走子进程（历史行为逐字节）。sweep/pricing/mc_prescreen 三处 `run_bench` 调用点零改动即继承（mc_prescreen 内嵌源经 `import lever_pricing_run` 自动获得）。
- **记账（红线 9）**：screening 读数的绝对水位随暖池略移（allocator 暖态），但筛选判据全部是同 job 相对读数（base 与变体同暖态），赢家判定分布不变；**gate（全价 eval job）进程模型一字未动**，筛选赢家仍必须过全量 gate。跨代绝对数比较不受影响（从不跨代比筛选读数）。
- **opt-in 遥测守卫**：`FK_SCREENING_CLOCK_GUARD=1`（默认关）时 `_bench_post` 消费 bench 自带 `gpu_state`——相对 pass 首读数 SM clock 偏差 >2% 或温度差 >5°C → 追加 NODE-DIRTY（该读数不产 verdict，pass 停止、driver 重买一次，诚实标注优于脏裁决）；job 侧 `_clock_guard_reason` 与 driver 侧 `screening.reading_dirty` 为同一规则的两侧实现（lpr 须保持无 harness 可导入，docstring 互指）。这是"node-dirty 消费现有 GPU state 遥测"的通道级接点；driver 侧原语本身已被 WP1 判定点（经 `_sm_clock_drift` 委托）消费。

## 5. 任务 5（P2）：CANDIDATES 排程认证额度（乙A12 后半）

- **排程器**：`_wp17_maybe_cert_candidates_head(ctx)`，挂在两处 `run_meta_step` 调用点的 `_apply_phase_close` 之后（M0 审计步与常规步）。到期判据读 meta 记忆长度（重启稳定，与 reopen_streak 同法）：`step % N == 0`。
- **一轮预算，无重试**：到期时一次教师 turn——读 recon `candidates.md` 队首未决条目、写一个认证 probe、以 `PROBE_REQUEST:` 收尾；driver 按既有 kb-probe 纪律执行（`-certprobe` 分支 commit/push → GPU job → verdict 日志 → reset 到 pre-head → probe 文件 untracked 重物化——`_detach_kb_probes` 增加了 `what` 参数缺省 `"meta/kb"`，既有调用字节不变）。教师写散文 → 预留轮次自动作废（lapse 记日志）；无 candidates.md → 先作废。预算 = 恰 1 个 probe 轮，永不连买。
- **与 WP18 接缝**：批量认证机制（`kb_cert_batch`）归 WP18；本包只管"何时预留、预留多少"（排程预算），verdict 只进 driver 日志，无任何 gate/prompts 消费（C10）。
- 配置：`[loop] cert_every_meta_steps`（默认 **0 = 关**，opt-in；历史行为即 `n<=0` 直接 return）。

## 6. loop.toml 与 C17

`global_config/loop.toml` 追加 `# --- WP17 ---` 注记块（沿 WP8 先例：本文件词汇表封闭，`[loop]` 键住卡层，本文件只记载键位与默认）。四个新键全部走 `_wp17_apply_cfg`（WP5 形态：LoopContext None 哨兵 + 卡层 free-form 读取，**不经 cli**——无 cli.py 转发义务，CHANGES 不留集成步转发欠条）。job 侧两个 env kill switch（`FK_SCREENING_INPROC`/`FK_SCREENING_CLOCK_GUARD`）同块记载。

## 7. M0 保真度 board 抽查注记（红线 11）

本包全部 M0 相关行为以 pytest + fixtures + dry-run 验收；**数值结论待 board 端到端抽查**（集成步 WP23 的 A/B 矩阵时机），届时建议核查三点：
1. **mock 排序保真度**：任取一次带 `ARCH_CANDIDATES: dual` 的真 run，对比 teacher 事前判断与 ARCH_PROBE 时间序（含 `problem` 字段声明前后各一次）——M0 押注质量的直接读数（乙A7 的动机量）。
2. **进程内 worker 的 A/B**：同一 SWEEP 配方 `FK_SCREENING_INPROC=1` vs `=0` 各跑一 job，核对 (a) job 墙钟差（30-50% 目标），(b) 赢家判定与 delta 序一致，(c) 绝对 device_median_ms 水位差记账（§4 暖池效应）。
3. **近差复测触发率**：真 run 中 `readings within the resolution floor … one re-probe` 出现频率与其预算占用（应只在放宽 bracket/预算的 run 出现）。

## 8. 红线自查表

| 红线 | 落实 |
|---|---|
| C9 screening 唯一实现 | screening.py 单文件；四通道只迁基建；四通道 verdict 语义逐字节回归钉住；WP15/WP16 接入面已备、无自建副本 |
| 排序不是价格（追加①） | `floor_rank` 只改探测次序；winner/冻结/pin 全由时间读数决定（反向用例钉住）；advisory 文案自述层级；floor 永不进 verdict/gate |
| ARCH_PROBE 只定冻结不是 gate | 读数通道原样；semi-real 仍只读时间；`meta_milestone.md` 未动 |
| node-dirty 不产裁决 | `reading_dirty`/clock guard 标 dirty 的读数一律不产 verdict（guard 默认关）；缺遥测 = 不标注（WP1 原则） |
| 复活必重验（追加②） | 复测 re-draw 是新鲜独立 draw；ledger 双 draw 均留痕；无任何跨代读数直接采信 |
| 一次一翻（追加⑤） | 本包零 gate 默认翻转；所有新机制惰性或默认关；`FK_SCREENING_INPROC` 默认开属 job 执行层（读数协议不变、有 kill switch、A/B 归 board 抽查——见 §7.2） |
| 统计回退纪律（追加⑥） | worker 任何退化（spawn 失败/死亡/无 JSON）回退逐调用子进程；evidence 目录缺失回退为日志；bad cfg 值回退默认 |
| 池与触发器不机器改判（追加⑦） | 预排序/复测/排程均为信息与预算通道，KEEP/REVERT/watermark/override 机器判定零改动 |
| C10 不进注入区 | 全部新输出走 driver 日志与 milestone head stamp 既有通道；无新 prompt 块 |

## 9. 观察记录（未立项，报总工程师备案）

29 号文档 N1 指出 `_run_arch_probes` 的 pin 分支不检查 delta 是否超 floor（within-floor 时 stamp 告诉教师"冻结跟随偏好"，pin 却已钉死裸胜者，二者矛盾）。该提案未进合并裁决 59 号 WP17 节，且修它即改四通道判定语义（违"语义一字不动"围栏），**本包未动 pin 行为**。合并裁决落地后若采纳 N1，改动点已隔离：`_run_arch_probes` 尾部 pin 块一处（`decisive` 判定处加 floor 比较），且 `screening.pair_outcome` 已提供所需的 floor 感知判定核，一行即可接上。

## 10. 测试清单（tests/test_screening.py，31 条）

- **原语五件**：`test_frozen_spec_hash_identity` / `test_sweep_spec_refused_exact_bytes`（spec）；`test_paired_order_base_first`（配对）；`test_node_dirty_reason_exact_bytes` / `test_reading_dirty_telemetry_bands`（node-dirty 双形态 + 遥测带外）；`test_budget_clamps`（双侧钳制）；`test_log_settled_recovery` / `test_settled_job_rebuy_policy` / `test_pair_outcome_floor_rule` / `test_sweep_and_mc_job_delegation_keep_their_words`（机器裁决与重买策略）。
- **迁移回归**：`test_stamp_default_bytes_unchanged`（stamp 前字节全等）+ 既有 2437 条全数原样通过。
- **任务 2**：`test_recipe_floor_consumes_analytic_floor` / `test_floor_rank_orders_and_stays_stable` / `test_pre_rank_picks_the_floor_top_but_the_time_decides`（排序断言 + "排序不产 verdict" 反向断言 + ledger 落盘）/ `test_pre_rank_off_switch_and_default_inertness` / `test_ledger_off_switch`。
- **任务 3**：`test_recipe_fidelity_parsing` / `test_near_diff_reprobe_schedule` / `test_near_diff_reprobe_one_job_redraws_the_pair`（近差触发 + 单 job 双配方 + tier 行）/ `test_near_diff_without_spare_budget_only_annotates_nothing_new`。
- **任务 4**：`test_inproc_worker_one_spawn_for_many_calls`（mock 层 torch import 次数断言：3 调用 1 次 spawn、0 次 sh）/ `test_inproc_kill_switch_restores_subprocess` / `test_inproc_worker_clean_death_reads_as_a_crashed_call` / `test_inproc_timeout_takes_the_dirty_path` / `test_inproc_completed_no_json_falls_back_to_subprocess` / `test_clock_guard_opt_in_marks_the_reading` / `test_run_bench_keep_cubin_stays_subprocess`。
- **任务 5 与配置**：`test_cert_scheduler_off_by_default` / `test_cert_scheduler_fires_on_the_schedule` / `test_cert_scheduler_lapses` / `test_wp17_apply_cfg_reads_the_card_layer`。
