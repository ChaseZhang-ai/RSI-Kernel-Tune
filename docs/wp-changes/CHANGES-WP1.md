# CHANGES-WP1 — 测量统计升级

分支 `feat/wp1-measurement`（基线 main @ 5030503）。目标：让 <1.5% 的真实增益可被发现、确认并保留——判定分辨率不再是一个常数，而是这次运行自己测出来的量。

全量证据：`pytest tests/ -q` → **2128 passed**（本机 `~/venvs/forgekernel`，Python 3.14 + pytest 9.2；无 GPU，全部以 fixtures/stub-torch/dry-run 验收）。新增单测集中在 `tests/test_wp1_measurement.py`（44 例）。

## 任务 1 — 迭代记录保留原始读数与 spread

- `loop/parse.py`：`EvalResult` 新增 `device_readings_ms` / `ref_device_readings_ms`（镜像后的 per-window 原始读数）与 `gpu_state`、`clock_drift`（任务 4）；`_workloads()` 在原有 median/ratio 之外保留每 workload 的 `device_readings_ms`、`ref_device_readings_ms`、`host_enqueue_ms`，并计算 `device_spread_pct = (max−min)/min`（新纯函数 `readings_spread_pct`）。旧格式（无数组）产出与从前逐字节相同的紧凑块（有单测钉住）。
- `inner/trajectory.py`：`IterationRecord.gate_spread_pct`（run 级聚合，仅此一个新数值字段）；driver KEEP 行写入该值（gate 货币为 device 窗口时的原始读数 spread；e2e 窗口 bench 不上报原始样本，保持 None）。
- **digest 红线**：`build_digest` 的 `achieved["workloads"]` 投影为 median+ratio+spread，剥掉 `*readings*` 数组——digest 输出仍是 median+spread（这是 WP1 为满足"不输出原始数组"在 build_digest 里做的最小必要投影，未动 WP2 的输出结构；WP2 后续在 digest 顶部加的 `sol` 行与本改动正交）。

## 任务 2 — resolvable_diff 取代常数 1.5%（跨包契约 C1）

- `loop/parse.py` 新增（与 `detect_plateau` 同处，名字/模块/位置参数签名按契约钉死）：

  ```
  resolvable_diff(records, gate_window=..., *, floor=0.005, cap=0.015) -> float
  ```

  数据：最近 K=6 条 evaluated 记录（与 detect_plateau 同定义：gate_ms 非 None）的 in-job 配对 ratio（device 窗口读 `vs_ref_device`，e2e 读 `vs_ref`——同 job 配对，node 速度已抵消）。σ_roll = ratio 相对滚动最优（max）残差的 MAD×1.4826；返回 `clamp(3σ_roll, floor, cap)`。**样本 <4 条返回 cap=0.015（旧行为）**。
  - 规格笔误更正：`gate_window: int` 实为窗口名字符串（`"device"|"e2e"`），带默认值，位置调用不受影响。
- `detect_plateau` 改为 `margin = min(noise_floor, resolvable_diff(...))`：`noise_floor` 参数保留、语义不变（上限）；新增 keyword-only `floor`/`cap` 透传。数据不足时 resolvable_diff 返回 0.015 = 旧 margin，行为逐位一致（旧 fixture 原样通过，并有专门测试钉住）。
- driver 消费点（grep `0.015`/`noise_floor`/`1.5%` 逐处核对后）：
  - plateau 调用点：传 `floor=ctx.resolvable_floor, cap=ctx.resolvable_cap`；
  - 同阶段 gate 回归容差（`_GATE_REGRESSION_TOLERANCE`）：改为 `_ratio_noise()` = resolvable_diff，常量保留为上限兼回退；
  - patch 判定 margin（`_PATCH_IMPROVE_EPS`）：改为 `_patch_improve_eps()`，常量保留为上限兼回退；REVERT 日志打印实际用的 eps。
  - **未动**：lever pricing 的 0.5%（meta_shapes/lever_pricing_run，job 内交错计时语境）、`_GATE_REGRESSION_TOLERANCE_MS`（跨 job 绝对读数语境）、watermark_margin。
- `scripts/self_evolve.py`：try-import `forgekernel.loop.parse` 的 `RESOLVABLE_DIFF_CAP` 作为 `NOISE_FLOOR_MIN`，失败回退本地常量并注明对齐来源——单一事实源。导入的是**常量而非函数本身**：该脚本手里只有 confirmation ratio 列表、没有 records，`resolvable_diff(records, gw)` 无从调用；共享的事实就是上下界常量。
- 配置键：见下节。**注意全局 `global_config/loop.toml` 只接受 `[target]`/`[budget]`（`_unknown_keys` 严格拒绝其他表，已实测）**，因此 `resolvable_floor/cap/confirm_budget_per_iter` 的覆盖面是**卡片侧 `operators/<op>/loop.toml` 的 `[loop]` 表**（与 `watermark_margin`、`reanchor_band` 等既有键同一覆盖面），loop/cli.py 构造 LoopContext 时接线，默认值=历史界，不写即不变。

## 任务 3 — 疑似赢家确认带

- `_confirm_suspicious_watermark` 新增第三触发：改善幅度落在 `(resolvable_diff, watermark_margin)` 区间（默认 0.5%~3%）也触发确认。强触发（>margin 跳变、NCU 回归）保持原两 job 协议（confirm→tie-break）；band 触发只买 `confirm_budget_per_iter`（默认 1）个 job、无 tie-break——不一致时按既有语义落 median。确认 job 沿用既有预算豁免（不占迭代 wall-clock）。
- 防失控：band 触发每迭代最多 `confirm_budget_per_iter` 次（cli 接线 `max(1, …)`）。
- 回退：无 ratio 历史的 run 里 resolvable_diff 恒为 0.015，任何 sub-margin 改善都不进 band——旧 fixture（含 `test_watermark_within_noise_improvement_costs_nothing`）逐位不变。

## 任务 4 — 时钟/温度/功耗遥测 + clock_drift

- `card_template/eval/timing.py`：`sample_gpu_state` / `record_gpu_state` / `gpu_state_report`。读取次序 pynvml → `nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw --format=csv,noheader,nounits` → 字段置 None 并在 `note` 注明（对齐 LB_CLOCK_STATE 的上报精神）。采集时机：burn-in 后 + `_measure` 每个窗口后各一次（每 job 约 13 个样本），上报 per-field 均值；列表上限 64。**任何失败都不影响测量本身**（读取器抛异常=缺席）。上报位置：`paired_performance` 的 perf 块 `gpu_state` 键；`parse_eval` 同时接受 eval JSON 顶层（顶层优先）。
- `IterationRecord.gpu_state`（KEEP 与 correctness-REVERT 行都记）+ `clock_drift` 布尔。
- driver：`_CLOCK_DRIFT_BAND=0.02`、`_sm_clock_drift`（相对较低一侧时钟的偏差；任一侧无遥测=None，绝不做漂移断言）、`_best_keep_gpu_state`。当一次读数将夺 watermark 而 SM clock 与现 best 差 >2%：打 `clock_drift=True`，且该读数**不得单独定 watermark**——进入任务 3 的确认逻辑（band 级预算，若更强触发已 firing 则随强触发协议）；确认后落 median。
- `run_best_confirmation`：样本带 `gpu_state`（本任务暂不加额外逻辑，读数条件差异已由 clock_state/node band 报告，见 CONFIRM 既有 `off_keep_pct`/`nodes` 通道）。
- 真实读取逻辑全程 try/except 降级；本机无目标 GPU 全绿（stub-torch + 注入 query 单测覆盖均值/降级/非数值字段清洗）。

## 任务 5 — 普通 task 可选 board 协议窗口

- `task.toml` 新表 `[timing]`，键 `protocol = "device"（默认）| "board"`：`_KEYS["timing"]`、`_read_timing` 校验、`Task.timing`（`Timing` dataclass）、`dump_task_toml` 回写。未知值 TaskError。
- 渲染：`card_problem` 写 `timing_protocol`；board 卡的 `gpu_entry.sh` 锁定 SOL board 的 clock_lock preset（SM 1500 / mem 3996，`require_lock=1`——锁频是权限门控的 GPU 态，沿用模板 clock_note 的 permission 语义，退出时清除），`eval/bench.py` 换为设 `timing.SECONDARY_PROTOCOL = "board"` 的模板。**默认 "device" 渲染逐字节不变（快照断言钉住）。**
- 计时实现：`timing.py` 新增 `measure_board`（每次调用前 flush L2——`_flush_buffer` 2×L2 + persisting-L2 reset，单调用 event span，median；时钟**不在此设置**，锁是 entry script 的事）。board 读数为 **report-only**：`workloads_performance`/`paired_performance` 输出 `board` 块（workload 级 + geomean 聚合），**gate/best/plateau/任何 verdict 不读它**；board 读数失败只记 `error`，绝不让 bench 失败。
- 实现说明：规格写 "board 实现整体复用 leaderboard 模板"，但模板 `card_template/eval/leaderboard.py` 当时正被 WP2 大改（+91 行在制），且不在 WP1 允许清单内；board 协议所需的 flush+单调用实现很小，落在 WP1 自己的 `timing.py`（`measure_board` + 采样），不碰 leaderboard.py 模板。语义（flush + 单调用 median + 锁频权限）与规格一致。

## 新增配置键

| 键 | 位置 | 默认 | 语义 |
|---|---|---|---|
| `resolvable_floor` | 卡片 `loop.toml [loop]` | 0.005 | resolvable_diff 的下界：分辨率估计再紧也不低于 ~0.5% |
| `resolvable_cap` | 卡片 `loop.toml [loop]` | 0.015 | 上界，兼数据不足回退值（= 旧 1.5% 行为） |
| `confirm_budget_per_iter` | 卡片 `loop.toml [loop]` | 1 | band/漂移触发每迭代可买的确认 job 数 |
| `[timing] protocol` | `task.toml` | `"device"` | `"board"` 给卡片加 report-only 的 board 协议读数并锁频（权限门控） |

## 回退路径（红线 2）

所有统计改动的旧行为都是"数据不足"时的自动回退，不需要配置开关：

1. **任务 1/4 schema**：新字段全部带默认值；`load_records` 丢弃未知键 → 旧 status.jsonl 原样加载（有测试）。旧 bench 不产出数组/遥测 → spread=None、gpu_state={}、clock_drift=False，下游零分支差异。
2. **任务 2**：配对 ratio 样本 <4（年轻 run、无 in-job ref 的 bench、旧格式 record）→ resolvable_diff 返回 cap=0.015 → detect_plateau、gate 回归、patch 判定全部逐位等于旧常数行为。
3. **任务 3**：同上——无 ratio 历史时 band 永不触发，确认成本为零；默认配置下除"判定更灵敏"外无行为变化。
4. **任务 4**：无遥测的读数 `clock_drift=None 路径`（`_sm_clock_drift` 返回 None）→ 永不打标、永不触发；pynvml/nvidia-smi 均不可用 → 字段 None + note，测量照常。
5. **任务 5**：`protocol` 缺省即 "device"，渲染逐字节同旧卡（快照断言）；board 读数 report-only，gate 消费者一行未改。

## 红线核对

- correctness gate / precision gate / measure_paired 的 A-B-B-A 契约：未动一字。
- 防伪措辞（objective-only record、gate 锚定禁令）：未改写。
- SOL-ExecBench token：未进任何代码/测试/提交（board 时钟常数 1500/3996 来自 `leaderboard.SUITE_CLOCK` 公开常量）。

## 过程备注（集成步需要知道的）

1. **分支历史已重建为纯 WP1**：本分支最初检出时曾被并行会话写入（旧 WP2 提交 9e5d15b / 1b79d2e / 9416ae6 / 18c8db8 与旧 wp5 提交 a749131 混入其中，且任务 4 的主体代码被卷进 9416ae6）。该污染分支已更名为 `feat/wp1-measurement-superseded` 留档；本分支由 main 重新组织而来，只含 WP1 内容，提交序列为：
   - 任务 1（原始读数+spread）→ 任务 2（resolvable_diff，契约 C1）→ 任务 3（确认带）→ 任务 4（遥测+clock_drift，代码自 9416ae6 的 WP1 hunk 逐字节重建）→ 任务 4 record 字段 → 任务 5（board 协议窗口）→ CHANGES-WP1.md。旧的共享工作区污染事件不再适用于本分支；WP2 / WP5 的内容在其各自的重建分支（feat/wp2-telemetry / feat/wp5-governance）上。
2. WP1 提交均按显式路径 staging，未卷入其他会话的在制文件（ncu.py、leaderboard.py、meta/ 等）。
3. 规格与实现的偏差均已在各节注明：gate_window 类型注解（笔误）、self_evolve 导入常量而非函数（无 records 语境）、loop.toml 覆盖面在卡片侧（全局层严格校验）、任务 4 遥测随 perf 块上报（顶层同样接受）、任务 5 board 计时落在 timing.py（leaderboard.py 模板被 WP2 占用且不在允许清单）。
