# CHANGES-WP15 — 分辨率、确认与测量统计三代

分支 `feat/wp15-resolution-confirmation`，基线 `origin/main @ 19e885b`（WP24 去榜单化已合入的 main tip；发包指令记录的 main tip=合并后核验）。基线回归：**2332 passed**（`~/venvs/forgekernel/bin/python -m pytest tests/ -q`，262.94s）。交付时回归：全绿（见 §7）。

---

## 0. 就位与前置门（§0 核验记录）

1. 工作区按包头命令 clone 自 `/home/zcz/forgeKernelOptimize/zxtune-integration`，`git checkout -b feat/wp15-resolution-confirmation origin/main`，origin/main = `19e885b`。
2. **WP24 发包前置门（开工时核验）**：发包时 origin/main 仍是 `359696b`（WP24 未合入）→ 按 §0.4"停下等待"处理；随后 WP24 过门（独立审查 `95-审查报告-WP24.md` = PASS-WITH-NOTES，三条验证：`cardkit.py` 存在 ✓、`leaderboard.py` 不存在 ✓、`git grep -i "sol-execbench\|sol_board"` 仅剩 `test_deboard_regression.py` 自身禁词清单 2 处命中=回归测试本体 ✓）并合入本地 main @`19e885b`（未 push，push 归 WP23）。本包从该 tip 开出，timing.py/parse.py 锚点均为 WP24 后布局。
3. 必读已读：59 §三 WP15 节 + C8/C14/C17；28（甲G2/G3/G6/G7/G8/G10）、30（乙F1-F4/A1/A2/A10/A11/B3）、31（乙G8 设计强化+拒绝表）、42（丙逐行审计，全部 §1-§9）。

## 1. band 口径变化记账（跨代比较必读）

**本包改变 best 与分辨率带宽的数字语义**（30 号报告红线 2/红线 6 的记账义务）：

| 量 | 二代口径 | 三代（本包）口径 | 跨代比较规则 |
|---|---|---|---|
| `resolvable_diff` 返回值 | record 级比值对滚动 max 的残差 MAD×1.4826×3，clamp [0.005, 0.015] | **分层**：σ_within（job 内逐窗配对偏差 pool）与 σ_between（record 级比值 leave-one-out 残差 + 确认 job 比值注入）取 max 后 ×3，clamp 不变；drift 行剔除（两级回退） | 新旧数字**不可直接横比**：v2 去掉了 v1 的两类偏差（滚动 max 钉住残差 → 爬升段自我钝化；job 内窗噪声从未入估）。CHANGES §2 一致性矩阵给出同数据下的对照 |
| running best | 同 cluster 全部 KEEP 行 gate_ms 取 min（最幸运一抽） | min 不变，但 (a) drift 行剔除；(b) **每个新 best 落带即确认一次**，行上 gate 值=独立 job 读数的 median（median-of-medians）；(c) 行携带 `confirm_readings`/`best_standard_error`/`best_low_confidence` | 跨代比较终值时：三代 best 是确认中位数，二代是单 draw——三代同条件下**系统性略高**（去幸运偏差），这是特性 |
| plateau margin | `min(noise_floor, resolvable_diff)` | 同式（分母换 v2），窗口行同样剔除 drift | 与 watermark band 恒同源（G8.2 硬义务，测试钉死） |
| SPRT 停判带宽 | MAD×1.4826×3 clamp [floor, cap] | 停判用**未钳 cap** 的诚实带宽（floor 保留）；2 点持续性；族卡 3 窗停判地板；多 workload band pool 化 | 旧常数 SPRT_BAND_CAP 仍是文档/对漂口径；停判不再读它（§5 回测为证） |

**预算影响估算**（30 号红线义务）：A1 常规化使每个新 best 多买 1 个确认 job（`confirm_budget_per_iter` 共用，默认 1/iter 不变）。REVERT 主导的迭代序列不触发；爬升段每 KEEP +1 job。SPRT 本地默认开后损失方向早停抵回部分（回测：明确亏损省 3-60% 窗，§5）。

## 2. v1↔v2 一致性矩阵（验收 §1）

同一组 record 上同时跑 `_resolvable_diff_v1`（v1 算术原样保留，仅测试/对照消费）与 v2：

| 数据形态 | v1 | v2 | 判读 |
|---|---|---|---|
| 完全平坦（6×1.0） | floor 0.005 | floor 0.005 | 一致 |
| 数据不足（<4 样本/空） | cap 0.015 | cap 0.015 | **回退逐字节相等**（C8） |
| 安静噪声（σ≈0.1%） | floor | floor | 一致：分层不放宽安静 run |
| 5 平 + 1 个 5% 离群 | floor（MAD=0 塌缩） | floor | v1 对单一离群塌缩；v2 LOO 同样不屑一顾 |
| 单调爬升（1.00→1.05） | cap 0.015（自我钝化） | cap（趋势本宽） | v2 ≤ v1（测试断言方向性） |
| 已知 σ 逐窗配对样本（σ=0.4%） | 读不到（只看 record 级） | 3σ_cover ≥ 1.5σ_within，≤ cap | **v2 独有**：窗层覆盖测试（test_band_covers_three_sigma_of_known_window_noise） |
| raw 注入（确认 job 比值） | 无此参数 | 只 pool 不替换；clamp 纪律保持 | test_injected_confirmation_ratios / test_v2_clamp_discipline |

（测试文件 `tests/test_wp15_resolution_confirmation.py`；每行一个测试。）

**合成数据带宽覆盖率单测**（验收 §1 第 2 句）：`test_band_covers_three_sigma_of_known_window_noise` —— 已知 σ 的配对窗样本经 workload 块注入，断言 band ≥ 3×½σ（不瞎）且 ≤ cap（不撒谎）。

## 3. 任务 1：resolvable_diff v2（契约 C8）

- **签名/clamp/回退不变**：`resolvable_diff(records, gate_window, *, floor, cap)` + 新可选 `raw=None`（C8 注入点：`{"ratios": [...], "pairs": [...]}`，与 record 样本 **pool** 而非替换）。`<RESOLVABLE_MIN_SAMPLES` 回退 cap 逐字节不变。
- σ 分层：within = workload 块里 index 对齐的 `(ref_i/kernel_i)` 对本记录 own-center 的偏差（kernel 消掉，纯窗噪声；`_SprtController._candidate_pairs` 同族数据）；between = record 级比值 **leave-one-out** 残差（"除本条外其余读数的 robust 中心"——滚动 max 基线退役）+ 确认 memo 注入。合成=clamp(max(σ_w,σ_b)×3, floor, cap)。
- **drift 剔除两级回退**：`_drift_clean` 在 resolvable_diff/detect_plateau/`_best_ms` 三处生效——`clock_drift=True` 行在剩余干净行 ≥ 需要样本数时剔除；全 drift 历史照测（回退=现行为）。测试：`test_drift_rows_drop_from_best_plateau_and_sigma`、`test_plateau_excludes_drift_rows_with_fallback`。
- **detect_plateau 同步**（乙G8.2 硬义务）：margin 一处计算（同一 resolvable_diff），窗口行 drift 剔除，`raw` 透传；`test_plateau_and_watermark_band_move_together_g82` 钉死同源恒等式。
- **cluster 过滤补齐**（丙 §2 缺口 3）：确认带现在消费 `_on_cluster(...)` 行（driver.py `_confirm_suspicious_watermark`），与 detect_plateau 的窗口一致。
- **新函数名**（C8）：`combined_workload_test`（per-workload Stouffer 合成检验，z≥2 判 significant，单侧 p Abramowitz-Stegun；**判定主路零消费**——只进 plateau/audit 诊断块 advisory 行与 driver 触发文案）、`best_standard_error`（log 域 se：≥2 个确认 draw 用其 stdev/√n，否则 best 行 workload 块原始读数的 1.2533×MAD/√n；ci95 = best×exp(±1.96 se)）。
- **corr_margin_ratio 不在本包**（围栏：归 WP20）——`parse.py` REVERT 行未动。

## 4. 任务 2+3：确认常规化 + 遥测闭环（driver.py）

### 触发器全景（`_confirm_suspicious_watermark` 重写；返回 `(ev, info)`）

| # | 触发 | 预算 | 二代行为→三代 |
|---|---|---|---|
| 1 | 跳变 > max(watermark_margin, rd) | 两 job 协议（确认+仲裁）不变 | 不变 |
| 2 | [rd, margin) 带 | confirm_budget_per_iter | 不变 |
| 3 | **drift（平级 if）** | budget | `elif drifted`→平级：sub-rd+drift 漏洞（丙 §3/§6）关闭，预算不变 |
| 4 | NCU 回归 > max(1.05, 1+rd) | 两 job | 常数迁移（见 §6 对照表 #5） |
| 5 | **窗间 spread > 2×rd**（A11：`gate_spread_pct` 首个消费者） | budget | 新增：DVFS 双峰 draw 必确认 |
| 6 | **SPRT win 方向早停**（N-B14/C14） | budget | 新增：截断 draw 未过确认前不入 watermark/plateau/定价 |
| 7 | **卡上存在 loss-stopped workload**（回测族裁决） | budget | 新增：loss 方向可信、截断 workload 不得赦免整卡 → 族假赢结构性为 0 |
| 8 | **A1 近-best 常规**（乙A1+甲G8+丙） | budget | 新增：sub-rd 改进不再免确认——官方 8 条 <1.5% 胜利的漏斗闭合 |

- **协议**：band 触发买 `confirm_budget_per_iter` 个 job（默认 1，共用一个预算账户——C8"不造第二份预算"）；确认读数=同 job ratio+窗对偏差 **memo 回流** between-job/within 分量（`_memo_confirmation`/`_confirm_raw_memo`，上限 32 条，按 gate window 分桶）；watermark 取 **median-of-draws**；首确认失败保原读数（flaky queue 不否决 KEEP）不变；确认 draw 离散超 band → 行标 `best_low_confidence`（plateau/元/教师读数时携带告警）。
- **记账**：`IterationRecord` 新字段 `confirm_jobs`/`confirm_readings`/`best_low_confidence`/`best_standard_error`（验收 §3"确认预算记账进 IterationRecord"）。测试：`test_confirmation_accounting_rides_the_record`（含 round-trip）。
- **幸运抽样 fixture**（验收 §3）：`test_lucky_best_is_corrected_to_the_confirmation_median` —— 旧行为=幸运 1.976 直接当 bar（v1 一致性表为证），新行为=确认中位数 1.988 落账。
- **keep_disputed 回路**（42 §9）：`run_best_confirmation` 的争议以 **median-vs-keep 出带**为触发（单样本出带仍是日志注记），出带即买 max_extra 个追加 draw，claim=扩大集的 median；`best_confirmation.json` 增 `dispute_opened`。测试：`test_best_confirmation_without_a_lease_keeps_a_deviating_sample`（5 样本、dispute_opened）。
- **gate_spread_pct 逐窗化消费**：`gpu_state_report` 增 `window_readings`（最近 16 个有序样本）；digest 头部 `clock` 行（`clock_header`：最近窗 SM 时钟 vs job 均值差 %，report-only）——42 §6"末窗降频藏在均值里"闭环。
- **digest**：`achieved` 增 `best_ci95`/`best_standard_error`/`best_low_confidence`（客观统计行）；顶部增 `clock` 行。

## 5. 任务 4：SPRT 开闸 + 回测补 cell + win 方向早停

### timing.py（围栏内：SPRT band pool 化 + 逐窗遥测；`measure_paired` 一字不动）

- 停判带宽诚实化：`sprt_band` = MAD×1.4826×3，**floor 保留、cap 从停判中移除**（cap 钳制让置信区间在真噪声 >1.5% 时撒谎——回测族 cell 的 4% win-lost 即其代价）。
- 持续性：loss 停判须在候选 n 与 n-1 两个协议同时成立（同 sides 重算，零额外窗）。
- 族地板：`measure_windows_workloads` 置 `_SPRT_FAMILY_JOB`，控制器最小停判点 2→3 窗。
- band pool：同 job 各 workload 的 decided pairs pool（≤12）喂后续控制器。
- **win 方向**（N-B14）：`FK_SPRT_WIN_STOP`+`FK_SPRT_WIN_MARGIN` 门控，`sprt_win_stop_verdict`（ratio−band > 1+margin）；`paired_performance` 增 `sprt` 块（decided/direction/windows，带测量纪元 id——同进程多测量的 sweep/pricing job 不串块，`test_multi_workload_perf` 的闭集断言守住）。loss-only 的算术安全论证只覆盖 loss 方向；win 方向的安全=**C14 耦合**（driver 触发器 #6）。
- `measure_paired` 函数体逐字节未动（diff 可验）。

### 翻默认（硬前置=回测过门）

- `_wp11_apply_cfg`：**本地 delivery 默认 `sprt_early_stop=true`**；ssh delivery 维持默认关；config 显式值恒胜；**ssh 上 config 开但 submit.toml [env] 未声明 FK_SPRT_EARLY_STOP → 启动 WARN**（env 不跨线：驱动以为开了而 job 全采样）。测试：`test_wp15_sprt_default_is_delivery_dependent`。
- `global_config/loop.toml` 增 SPRT 注释块（默认、回退开关、ssh WARN、win stop、confirm_job_mode）。
- `sprt_win_stop` 默认**全路径关**（C14 耦合下才开）。

### 回测（`scripts/backtest_early_stop.py`，4000 paired trials/cell，完整输出=交付证据）

```
backtest_early_stop: 4000 paired trials per cell — full and early read the SAME windows;
DEVICE_REPS=5, stop band = UNCLAMPED (floor 0.005, cap 0.015 verdict-side only);
persistence=2; family floor=3 windows
GATE (iid loss (WP11 gate)): PASS (false-win total 0, worst win-lost 0.00% < 2.0%)
GATE (AR(1) autocorrelation rho=0.3 (WP15 cell 1)): PASS (false-win total 0, worst win-lost 0.00% < 2.0%)
GATE (AR(1) autocorrelation rho=0.6 (WP15 cell 1)): PASS (false-win total 0, worst win-lost 0.00% < 2.0%)
GATE (geomean family ...): PASS (false-win total 0, worst win-lost 0.03% < 2.0%)
GATE (win): PASS (no-fire on loss/null; won-early-lost-full <= 2%)
OVERALL: PASS
```

- 节省：clear loss x0.90 省 40%（iid）/39.9%（AR1 ρ=0.3）；loss x0.97 @1% 省 14.9%；族卡 loss 组合省 2.6-26.6%；整体 1,539,804/1,680,000 窗（8.3%）。
- **诚实记账**：族 battery 的 raw 口径（driver 无 veto 时）在 2% 噪声边界 cell 留有 1/60000 臂间分歧（`raw-false-win=1`，行内可见）——系统裁决（loss-stopped workload 否决卡级 KEEP，触发器 #7）下该方向**结构性为 0**（无停判→两臂全等；有停判→早臂不得判赢）。win 格：loss/null 不点火（算术单侧），x1.020 win-stopped 6.4% 且 won-early-lost-full 8/4000=0.2%——正是 C14 确认耦合的存在理由。
- 回测发现并修复的协议缺陷（cell 的价值）：capped band 停判在 2% 噪声下产生 4% 族 win-lost → 未钳带宽+持续性+族地板后 0.03%。

### win 方向早停（契约 C14）

停判读数在通过既有确认协议（触发器 #6/#7 + 水位 median-of-draws）前**永不**入 watermark/plateau/定价——行在确认返回后才落账，未确认读数物理上到不了这三个消费者。回测加 win 格（上）。红线条目对照：C14 耦合=触发器在 `_confirm_suspicious_watermark` 内部，非独立旁路。

## 6. 6 处硬编码常数迁移对照表（验收 §5）

| # | 常数 | 位置 | 旧值 | 迁移语义 | 消费点 |
|---|---|---|---|---|---|
| 1 | `NOISE_FLOOR_PCT` | `loop/meta_shapes.py:40` | 0.5% | `merged_floor(floors, floor=)` 可选参数=本 run `resolvable_diff×100`；**max(现值, resolvable)**——常数仍是安静 run 下界 | 杠杆定价合并表（driver 传入）；旧调用 None=原行为 |
| 2 | `NOISE_FLOOR_PCT` | `skills/lever-pricing/lever_pricing_run.py:156` | 0.5% | spec `noise_floor_pct` 键（driver 建 levers.json 时写入 resolvable band），`floor_pct()=max(常数, spec)`；旧驱动无键=原行为 | 三个消费点（noise_floor/ablation noise/stack noise） |
| 3 | `noise_floor` | `detect_plateau`（parse.py） | 0.015 上限 | **天花板语义不变**；live margin=v2 分层估计（§3）——与 watermark band 同源 | plateau 判定 |
| 4 | `watermark_margin` | driver.py LoopContext | 0.03 | 触发时 `margin=max(配置值, resolvable_diff)`——吵闹 run 的水位margin骑自己的带；配置值是下界 | `_confirm_suspicious_watermark` |
| 5 | `_NCU_REGRESSION_BAND` | driver.py | 1.05 | `_ncu_regression_band = 1 + max(0.05, rd)`——run 自己的分辨率看不见的 kernel-clock 移动不算回归信号 | NCU 回归触发器 + 确认清算 |
| 6 | `_GATE_REGRESSION_TOLERANCE` | driver.py:10061 | 0.015 | 消费点 `_ratio_noise` 透传 `raw`（确认 memo）→ live band=v2 分层估计；常数保留为 clamp 上限与 data-poor 回退（"旧 band，永不更松"） | `_gate_regression`（M0 gate 判定） |

## 7. 任务 5：gate-only 确认模式 + 失败反馈质量（甲G3/G6/G10）

- **driver 侧机制全量落地**：`confirm_job_mode`（LoopContext + [loop] 键，C17 词汇走 `_wp11_apply_cfg` 解析）——水位确认、run_best_confirmation 一律以 `ctx.confirm_job_mode` 提交。**submit 断言**：`test_confirm_job_mode_reaches_the_submit_payload`（mock backend 断言 mode="gate"）。
- **围栏注记（诚实交底）**：stock card_template 的 gpu_entry 把 `perf|all` 都映射到 `--mode all`——"gate"模式要真省 3-5×，eval 侧（benchkit.py/cardkit.py，**本包围栏外**）须提供 correctness 短路+门窗口径的模式。本包交付=机制+接缝+文档（loop.toml 注释块与 CHANGES 此节），eval 侧模式供给留给集成步/后续包；把 `confirm_job_mode` 设成 entry 不认识的 mode 会让确认 job 失败→确认回退单读数（安全，无意义）。
- **数值 FAIL top-k**（甲G6）：`_correctness_fail_detail`（parse.py）——eval JSON 的 `checks` 行按归一化误差排序取 top-3（`CORRECTNESS_FAIL_TOP_K`）注入错误反馈（"worst checks: g0/y: error 0.5 vs tol 0.01 — worst; ..."）；`worst_elements` 键为**预留 seam**（元素级 {output,group,index,value,expected,abs_diff}，bench 侧产出在围栏外，此处 verbatim 透传）。测试 2 件。
- **fresh-recheck 共享父输入重生成**（甲G10）：实现在 `eval/benchkit.py`（围栏外）——**未实施**，见 §9。

## 8. 任务 6：解析地板收尾（丙WP15-T5+乙F5）

- **bwd 保守系数**：`analytic_floor.py` `_family_flops` 对名字含 `bwd|backward|wgrad|dgrad|grad` 的 attention/conv 问题按 **×2.0**（`_BWD_FACTOR`，recompute+反传链的保守下界——真链 ~2.5×，地板仍只低估不高估）计 flops，assumption 行记账。测试：`test_backward_attention_charges_the_conservative_chain_factor`（fwd 名字不受影响）。
- **围栏注记**：任务 6 要求改 floor 公式，公式唯一居所 `analytic_floor.py` 不在"只许动"清单、也不在"不许碰"清单——按任务清单实施（最小手术：一个分支+系数），此处明示交由审查裁决。
- **`backtest_floors.py --from-run <op_dir>`**：读 `best_confirmation.json`（settled median，含 per-workload medians）+ `eval/problem/problem.json`（workload axes），按 axes 匹配回填 `best_ms`+`best_provenance`；**门先跑在填过的数据上，ratio<1.0 则不写回**（回填不能洗白公式 bug）；过门才写回 manifest。`floor_backtest.json` note 同步更新。selftest 保持 PASS。

## 9. 明确未实施项（围栏外交底）

| 项 | 原因 | 状态 |
|---|---|---|
| gate-only 模式的 eval 侧供给（benchkit/cardkit） | 围栏外 | driver 机制+seam+文档已交付（§7） |
| fresh-recheck 共享父输入重生成（甲G10） | 实现在 benchkit.py，围栏外 | 未实施；CHANGES 记账 |
| `worst_elements` 元素级产出 | 同上（benchkit） | parse 侧消费就绪，键已预留 |
| Stouffer 的**判定主路**消费 | C8：判定主路不动 | 只作触发文案与教师 advisory（红线追加④的回退纪律由既有 resolvable 回退测试承担） |

## 10. 范围围栏自查

改动文件：`src/forgekernel/loop/parse.py`、`src/forgekernel/loop/driver.py`、`src/forgekernel/loop/meta_shapes.py`、`src/forgekernel/loop/analytic_floor.py`、`src/forgekernel/inner/trajectory.py`、`src/forgekernel/card_template/eval/timing.py`、`skills/lever-pricing/lever_pricing_run.py`、`scripts/backtest_early_stop.py`、`scripts/backtest_floors.py`、`scripts/floor_backtest.json`、`global_config/loop.toml`、`tests/`（WP1/WP11/multi_workload/analytic_floor 断言更新 + 新 `tests/test_wp15_resolution_confirmation.py`）。

- 清单内：parse.py（resolvable v2+新函数+drift 过滤+top-k 反馈）、driver.py（确认常规化/drift 独立触发/常数迁移/disputed 回路/digest 行/SPRT 配置）、trajectory.py（客观统计行+记账字段+时钟头）、timing.py（仅 SPRT 机制+逐窗遥测；**measure_paired 逐字节未动**）、两脚本+manifest+loop.toml ✓。
- **清单外交底**（两处，任务清单原文要求但清单未列）：`meta_shapes.py`（常数 #1 迁移的宿主）、`lever_pricing_run.py`（常数 #2 的宿主+spec 键）、`analytic_floor.py`（任务 6 唯一居所）。三者均不在"不许碰"清单，且无其他 Wave-3 包认领；请审查裁决。
- **未碰**：precision_gate ✓、screening.py ✓、lever_pool.py ✓、注入区（Stouffer advisory 只进 plateau/audit 块既有 diagnose_extra 组装位置，学生 prompt 零改动）✓、per-shape 字段（WP14）✓、student_prompt.md ✓。

## 11. 回归与验收核对

- 基线 2332 → 交付全绿（§0）；WP15 新测试 25 件（一致性矩阵/覆盖率/幸运 fixture/记账/新统计/drift 剔除/G8.2 同步/top-k/seam 断言/win/loss-stopped 耦合/mode 断言/交付依赖回测 PASS 断言）。
- **顺带修复一处存量测试卫生缺陷**：`test_wp11_apply_cfg_reads_loop_toml`/`test_wp15_sprt_default_is_delivery_dependent` 原用 `monkeypatch.delenv` 清理 `_wp11_apply_cfg` 写入 `os.environ` 的 `FK_SPRT_EARLY_STOP`——`monkeypatch.delenv` 的 teardown-undo 会**恢复生产代码设置的值**，泄漏给后续测试；本包给 `paired_performance` 加 `sprt` 块后，该泄漏让 `test_multi_workload_perf` 的 workload 闭集断言在特定顺序下失败（环境变量开着 → FakeGpu fixture 的 loss 组合点火）。清理改为 `os.environ.pop`（不经 monkeypatch 账本），并加 conftest autouse fixture 兜底弹 `FK_SPRT_*`（本地翻转是生产行为：每个 FakeBackend loop 测试都会置位 env）。
- **顺带修复 sprt 块的延迟上报缺陷**（真 bug，非测试问题）：`workloads_performance` 在**全部 workload 测完后**才逐块调 `paired_performance`——模块级 `_last_sprt_decision` 里只剩最后一个 workload 的决策，且纪元检查恒等通过，导致末位 workload 的 SPRT 决策错骑到**所有** workload 的 eval 块上。修法：`measure_windows_workloads` 在每个 measure_paired 返回时把**本 workload 的决策就地挂到它的 paired dict**（`_sprt_decision`，私有键，paired_performance pop 消费、永不进 JSON），单测量 bench 保留"全局+纪元相等"回退路径。
- 回测输出粘贴 §5；真实 record 回放的新旧带宽对比表：**无真实 run 存档可用**（仓库不存绝对时间，42 §8 同款限制）——以一致性矩阵（§2）+ 合成覆盖率替代，此为诚实交底。
- dry-run 断言：`best_ci95` 行（test_digest_carries_best_ci95）、低置信标注（test_band_confirmation_disagreement…）、gate-only submit 断言（mock backend 载荷差异，test_confirm_job_mode_reaches_the_submit_payload）✓。
- CHANGES：band 口径变化记账 §1 + 6 常数对照表 §6 + SPRT 翻默认的回测依据引用 §5 ✓。
