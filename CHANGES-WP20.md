# CHANGES-WP20 — 红队与防线（测量保卫包）

分支 `feat/wp20-redteam-defense`（自 main `19e885b` 开出）。基线 `~/venvs/forgekernel/bin/python -m pytest tests/ -q` = **2332 passed** 全绿。开工门已验：WP24 已合入（`card_template/eval/leaderboard.py` 不存在、suite 零残留 grep 绿、常规评测驱动 = `benchkit.py` + cardkit 生成薄壳 `bench.py`；发包词 "bench.py" 即 `benchkit.py` 模板，生成卡工作区里两者并存）。

动机陈述（仅此一处，按包词口径）：公开的基准审计与事故报告一致表明，当奖励真实时作弊是系统性现象（有审计报出 ~14.5% 提交被标记作弊、150× 假加速的公开翻车）。一个假 best 压死后续所有真实增益。本包全部产出是测试资产与测侧防线，学生可见面零新增。

---

## 任务 1：红队作弊样例集 `tests/redteam/`（丙WP19-T1）

每类先写"预期防线"注记再写攻击；每 fixture 断言"防线抓到"或记录"已知无防线（TODO）"。资产边界：全部在 `tests/redteam/`（conftest.py 声明 C16），复用 `test_fresh_recheck` 的真实模板卡脚手架与 `test_loop_driver` 的 FakeBackend——红队跑的是生产 bench 代码。

### 覆盖矩阵（SOL 分类逐类）

| # | 攻击类（SOL 分类） | fixture | 预期防线 | 状态 |
|---|---|---|---|---|
| 1 | 缓存输出复用（指针/状态键） | `test_redteam_cache.py::test_attack1_pointer_keyed_cache_misses_in_a_fresh_process` | fresh_recheck **层1**（新进程：指针全换，停靠工件 key 不再命中） | ✅ 抓到 |
| 2 | 换 seed 内容寻址缓存 | `test_redteam_cache.py::test_attack2_content_keyed_cache_misses_the_shifted_draw` | fresh_recheck **层2**（child 换 seed 重绘，内容搬家） | ✅ 抓到（同内容命中/换 seed 失配双向验证） |
| 3 | try-except 回退慢实现/等价体 | `test_redteam_fallback.py::test_attack3a_*` | correctness（INCORRECT_NUMERICAL + 结构化 error/tolerance 行） | ✅ 抓到 |
| 3b | try-except 掩护死快路 | `test_redteam_fallback.py::test_attack3b_*` | **已知无防线（TODO）**：L0 fence 只扫 API、kb_lint 只扫 KB 面，无内核控制流静态扫描——记录暴露面，不假装有防线 | ⚠️ TODO 记录 |
| 4 | **stream 注入** | `test_redteam_stream.py::test_attack4_*` | **核查结论：可绕**（CUDA event 对 record 在计时 stream 上，侧 stream 工作与 event 无序）→ 补防线：WP20 stream 哨兵（任务 2） | ✅ 哨兵抓到 |
| 5 | 线程注入/JIT fork | `test_redteam_timer_proc.py::test_attack5_*` | 进程/线程计数基线对照（真实 /proc 实测；基线取首次调用后，JIT/线程池不误报）→ `anomaly` | ✅ 记账（**记账-only**：不自动定罪/击杀，见矩阵注记） |
| 6 | 计时器 monkey-patch | `test_redteam_timer_proc.py::test_attack6_*` | eval 文件完整性（harness_guard 既有）+ **计时函数身份哨兵**（新增，任务 2） | ✅ 抓到（mismatch→mode=timer fail 路径） |

注记：攻击 5 的"记账-only"是本包的明确姿态——census 异常进 eval JSON 但**零判定权**（fixture 断言 anomaly 时不翻 verdict）；自动定罪需要按 job 镜像约定的基线，属后续包工作。攻击 6 的裸 lambda 撕毁属性结构的重绑会让哨兵自身读不到 → 降级 `unknown`（保守不定罪；且窗口在裸 lambda 下根本无法运行，eval 自然失败）——fixture docstring 记录该形态。

---

## 任务 2：stream/timer 哨兵（丙WP19-T2，按任务 1 核查结论补防线）

`card_template/eval/benchkit.py` 新增 WP20 防线区（安装点 `run_bench` 在任何外来代码 import 之前；读取点 `_wp20_sentinel_blocks` 在**全部计时窗口关闭之后**——窗内协议一字未动，measure_paired 未触碰）：

- **stream 哨兵**：每个 timed group 结束后在窗外把同一调用重测两遍——协议同款 event 对（计时 stream 上）+ 全 device synchronize 下的墙钟排空。排空读数 > `STREAM_SPAN_RATIO(3.0)×event + STREAM_SPAN_SLACK_MS(0.5)` ⇒ timed-window 工作逃逸到侧 stream。violating → `fresh_recheck: fail (mode=stream)` → 走 WP9 既有 fail 路径（REVERT+suspicious），**零新判定逻辑**；真 recheck fail 已在结果上时不覆盖（recheck 是更强证据，stream 读数 rides beside, never under）。链回计时 stream（wait_stream）的诚实 kernel 两读数一致，不误伤。
- **timer 哨兵**：kernel import 前捕获 12 个时钟/窗口函数身份（perf_counter/time/Event/record/elapsed_time/synchronize + timing 的 6 个窗口例程），窗口后重解析比对 → eval JSON `timer_integrity: ok|mismatch|unknown`；mismatch → `fresh_recheck: fail (mode=timer)`。
- **proc census（攻击 5）**：pre-import / 首轮 correctness 后（note_first_calls_done——JIT、线程池的合法开销落在基线里）/ 窗口后三读；窗口后超基线 → `anomaly`。**accounting-only**，record 不因它 fail。
- **误报防护（硬要求）**：任何 env 失败 → `unknown`/None 降级，一律不定罪（WP9 fresh_recheck 姿态逐字沿用）。单测：env 失败、无 CUDA、基线缺失三条降级路径全断言。

## 任务 3：gate/judge 记账（丙WP19-T3）——零判定改动

- **WP24 手术遗留回归修复**（防线区同组装点，顺手修+回归测试钉住）：WP24 改名把 `leaderboard.py:2837` 的顶层补回丢了——`_run_task` 组装 eval JSON 时 `_correctness_json` 剥掉 precision 却无处补回 ⇒ 生成卡在 strict（WP24 后全任务默认）下 driver 读 `unreported` ⇒ `_precision_reading` 翻 correctness ⇒ 全 REVERT 死循环。现恢复顶层发射（`_run_task`，测试 `test_precision_block_rides_the_top_level_*`）。
- **precision 合同缺省审计行**：eval JSON `precision_audit: <contract> (default|problem|env)`——显性化每卡生效合同，零判定（`precision_audit_line`）。
- **gate_degraded**：precision `unavailable` 行占比 > `gate_degraded_ratio`（默认 0.5，`FK_GATE_DEGRADED_RATIO` 可覆写；loop.toml [loop] 同名键走 `_wp20_apply_cfg`）⇒ eval JSON 顶层 `gate_degraded: true`，record 字段同名。防"gate 静默失效"（乙 F 系核实 `precision_gate.py` gate_failure 只读 status=="fail"）。零判定：contract off 时不记账（无 gate 即无失明）。
- **corr_margin_ratio**：`parse.corr_margin_ratio` 从 correctness checks 解析 worst error/tolerance（原埋在 error 文本里的 max_diff/tol 结构化）；两个 REVERT 行（correctness FAIL、fresh_recheck FAIL）携带该字段。fixture 3a 断言来源行。
- **边缘带复检**：correctness FAIL 且 `0.9 < corr_margin_ratio < 1.1` 的**首犯**（每 run 一次，`_JUDGE_EDGE_RECHECKED` 按 status path 记账）自动买一次换 seed 复检——任务输入为 unseeded 绘制，新 job 即新 seed；复检走完全相同的协议（`_enforce_single_ref`/`_enforce_prep_contract`/`_precision_reading` 原样）；过 ⇒ 以新读数走原 KEEP 路径，row 记 `judge_edge_recheck: pass`；败/不可用 ⇒ 按原样 REVERT。**加一次测量再按原规则判，规则零改动**。loop.toml `judge_edge_recheck`（默认 true）为关闭开关；单测四条路径（首犯过/首犯败/第二次不买/关或带外不买）。
- **fresh_recheck 非确定豁免（形态区分）**：recheck fail 时附**同 seed 双跑一致性指纹**（`_nondeterminism_fingerprint`：judge 输入按组 seed 双绘双跑，max abs diff 为核自洽度）；双跑互不一致 ⇒ `form=nondeterministic`——**仍按 REVERT**（非确定 kernel 不可交付），record 以 `fresh_recheck_form` 区分两种 fail 形态，error 文本带 `form=`。单测断言两种形态的 row 字段与 suspicious 标记。

## 任务 4：杂项守门（P2）

- **milestone validator"全删标签"WARN**（乙N-B16）：`milestone.tag_wipe_warnings(baseline, text)`（~20 行）——基线带标签、现文本清零 ⇒ 一条 advisory WARN；**不拒不弹**（`validate_milestone(_WIPED) == []` 断言 accepted 裁定原样），driver 在 patch 流程 DAG warning 旁记日志（`_milestone_tagwipe_warnings`）。
- **板读数方差守门 → WP24 后映射**：原包的"板协议副读数"随 WP24 切除；幸存对应物 = 配对协议的**参照侧镜像对**（ref A-B-B-A 的两条 device 读数）。参照侧 spread > 2× kernel 侧 spread ⇒ `board_span_unstable: true` 进 record + 教师 workload 块旁一条 WARNING（report-only，零判定）。**字段名保留 `board_span_unstable`**——WP23 集成缝按此名 grep（语义已在注释中写明映射）。

## 配置

`global_config/loop.toml` 追加 `# --- WP20 ---` 注释块（键在卡层 [loop]，沿 WP12/WP8 封闭词表先例）：`gate_degraded_ratio=0.5`、`judge_edge_recheck=true`、`board_span_guard=true`。`_wp20_apply_cfg`（WP11 形态：显式设置胜出、坏值回默认、None=默认开）挂在 `run_two_layer` 入口。

## 触碰面与围栏自查

| 文件 | 改动 |
|---|---|
| `tests/redteam/`（新，5 文件） | 36 测试：6 类攻击 fixture + 记账/降级/零判定对照 |
| `card_template/eval/benchkit.py` | WP20 防线区（哨兵安装/读取、指纹、audit 行、gate_degraded、顶层 precision 发射修复）；`_task_side` 一行挂 span 读数 |
| `card_template/eval/timing.py` | **零改动**（stream 核对全部在窗边界外、benchkit 侧实现——窗内协议逐字节未动） |
| `src/forgekernel/loop/parse.py` | EvalResult 新字段（corr_margin_ratio/gate_degraded/precision_audit/stream_sentinel/timer_integrity/proc_census/judge_edge_recheck）+ `corr_margin_ratio()` 解析 |
| `src/forgekernel/inner/trajectory.py` | IterationRecord 5 字段（append-only；load_records 丢未知键，旧文件向后兼容） |
| `src/forgekernel/loop/driver.py` | `_judge_edge_recheck`/`_precision_degraded`/`_board_span_unstable`/`_wp20_apply_cfg`/`_milestone_tagwipe_warnings`；两条 REVERT 行与 KEEP 行的字段携带；fresh_recheck fail 的 form；board_span 渲染一行 |
| `src/forgekernel/milestone.py` | `tag_wipe_warnings`（advisory） |
| `global_config/loop.toml` | `# --- WP20 ---` 注释块 |

**未触碰**：measure_paired 镜像顺序与窗协议（timing.py 0 diff）、precision_gate.py 判定算式与常数（0 diff）、student_prompt.md、kb_lint、WP19 的 job 装配/缓存块、WP15 SPRT band 领地。

---

## 资产边界自查（契约 C16 逐条）

1. **作弊 fixture 全部进 tests/redteam/**：✅ `ls tests/redteam/` = conftest.py + 5 个 test 文件，无其他位置新增攻击代码（benchkit 侧只有防线，无攻击形态示例）。
2. **手法描述只存在于测试注释与 CHANGES**：✅ 各 fixture 的攻击注记全部在测试 docstring/注释与本文件；src/ 侧注释只描述防线与读数语义，无"怎么攻"内容。
3. **不进学生可读路径**：✅ 学生可见面（student_prompt.md、卡模板学生区、skills、KB 面未动；`git diff --stat` 可核）。KB 负知识条目（WP17 领地）零新增。
4. **sentinel env 探针不可被内核读取**：`FK_GATE_DEGRADED_RATIO` 由 driver/bench 自读，内核 fence 本就禁 `environ/getenv`（既有 KERNEL_FENCE/SEALED_FENCE/CUDA_KERNEL_FENCE 词表未动）。

## 验收对照（包词 §3）

1. pytest 全绿：**2368 passed**（= 基线 2332 + 新增 36，零回归）；红队每 fixture 一个"防线抓到/已知无防线（TODO）"断言 + 覆盖矩阵（上表，SOL 分类逐类）✅
2. stream/timer 哨兵：fixture 作弊 kernel 被抓（mode=stream / mode=timer → REVERT+suspicious，driver 消费单测）；env 失败降级 `unknown` 不定罪单测 ✅
3. gate_degraded / 边缘复检 / 非确定豁免 / board_span_unstable 各自单测；全部新记账行零判定改动对照测试（`test_zero_verdict_control_*`：关闭键时判定与基线一致、购买 job 数不变）✅
4. 本文件 ✅
