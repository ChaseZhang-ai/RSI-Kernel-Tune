# 通电与收获协议（WP21 / 甲 G25）

> 执行归 **WP23**（集成与收获）：本文档是协议与固定动作清单；本包（WP21）只落文档
> 与 `scripts/energize_ab.py` 参数化骨架，**跑批与翻默认都不发生在本包**。
>
> 背景：甲 G25（二代休眠机制验证债）——`PK_BUILD_CACHE=off`、`SPRT=off`、
> MC-DELIVERY 通道开但需教师写 `multi_candidate` 才激活、stagnation/multi_candidate
> 的 cli 转发行曾缺（dcfdcb5 已接）、floor 门在 floor_backtest `best_ms` 填充前
> inert、survey 通道 opt-in 默认关、NCU flags 的 board A/B 从未跑、JustGoal A/B
> 协议写了从未跑。一代红线 8（board 真测仅端到端抽查）欠下的债，三代以本协议偿还。

## 0. 总则

1. **一次一翻**（one default at a time）：每轮只翻一个默认；翻前有基线 A、翻后有
   实验 B，同任务集同机器同预算；证据落盘后才允许翻下一个。
2. **每个默认必须有回退开关**，回退开关必须与默认位分离（env / cli / loop.toml 键），
   回退动作是一行命令，不允许"改代码回退"。
3. **证据先于翻转**：A/B 对比用例 + 机制行为用例（机制确实触发了）两类证据齐备，
   才进入默认翻转提议；翻转提议写在 CHANGES，由 WP23 集成步执行。
4. **本机 3060 只验链路不产性能结论**（红线 11）：A/B 的"度量"在 3060 上只回答
   "机制是否触发、数据流是否完整、墙钟构成是否改善"；任何以百分比收益为结论的
   跑批必须在目标机器（sm90/sm100 板卡）上重做。
5. **GPU 独占**：所有真跑以 `~/tools/bin/with-lock gpu 1800 -- <命令>` 包裹。
6. **JustGoal 对照制度化**（见 §3）：机制上线前后同 job A/B + 换基座重测，是每机制
   的固定验收动作，不是可选项。

## 1. 验收 A/B 矩阵

| # | 项 | 度量方案 | 证据要求 | 回退开关 | 翻默认顺序 |
|---|---|---|---|---|---|
| A1 | **普通任务集执行**（对照组基线） | 固定 mini 任务集（`tasks/wp21mini` + 1 个正式小卡）×2 job；记录：status.jsonl 行数、eval/NCU job 完成率、墙钟构成表五桶、digest 产出 | run 归档 + status.jsonl + 两份 run.log；基线不入库存档（run 目录即证据） | —（基线态无开关） | 先跑 A1 作为一切 A/B 的 A 侧 |
| A2 | **NCU flags A/B**（WP9 遗留 `ncu_flag_ab` 方向确认） | `scripts/ncu_flag_ab.py`（WP9 已落，3060 无 CUDA torch 时未跑——现已有 ml venv CUDA torch 可跑）：同内核新旧 regime flags（`--cache-control none --clock-control none`）各采一次，比对 SOL/L2/DRAM 读数偏移方向 | 两次 NCU 报表 + 偏移表落盘；结论只写"方向确认"，不写性能结论；跨代 NCU 数字不可混读须在报表头注明 | NCU flags 生成点全部带开关（WP9 原生），关即回旧 regime | 任何时候（纯测量无默认翻转；只为跨代读数口径立证） |
| A3 | **SPRT 默认**（`sprt_early_stop`，默认关） | A1 任务集 ×2：off vs on；度量：每迭eval job 时长分布（早停生效的窗口占比）、verdict 一致性（KEEP/REVERT 与 A 完全一致才算过） | 两轮 status.jsonl 的逐迭代 eval_job_s 对比 + verdict 对照表；**SPRT 只提前输永不造 KEEP**须逐行核对（loop/driver 的 SPRT 段） | `--sprt-early-stop` cli / loop.toml `[loop] sprt_early_stop`（默认 false 即回退） | WP15/WP19 合入后第 3 位（在 build cache 之后、floor 之前） |
| A4 | **build cache 默认**（`PK_BUILD_CACHE`，默认关） | 乙线折中先行：①真跑构建日志验证"同内核跨 job 重复构建"假设（丙 T5 标注推断）；②key 覆盖用例：改头文件必 miss / 改注释必命中；③然后 A1 任务集 ×2 度量墙钟构成表 eval_job_s 桶 | ①构建日志摘录（同 sha 两次 nvcc 调用）；②key 覆盖单测输出；③两轮 wall-clock 构成表；manifest 逐产物 sha 复核日志 | `PK_BUILD_CACHE=0`（env，一行回退）；board 卡与密封 ref 不启用（范围护栏） | **第 1 位**（乙线主张本地默认开；证据齐后进 CHANGES 建议，翻转归 WP23） |
| A5 | **floor_stop / floor_margin**（解析地板门，默认关） | 前置：`scripts/backtest_floors.py --from-run <归档>` 把 `scripts/floor_backtest.json` 的 25×null 回填（WP21 冒烟已验 --from-run 通路）；回填后 backtest_early_stop.py 回测 floor_stop 假命中率；A1 任务集 ×2：off vs on 度量"误提前停=0、plateau 判定漂移=0" | floor_backtest.json 回填 diff + 回测早停报告 + 两轮 run.log 的 floor 行 | `[target] floor_margin` / `[loop] floor_stop`（task 层键，默认关） | 第 4 位（依赖 A5 前置回填完成 + WP15 resolvable v2 合入） |
| A6 | **survey 通道**（`teacher_prior_survey`，默认关） | A1 任务集 ×2：off vs on；度量：survey 产物落盘（priors/ 路径）、教师诊断 turn 数变化、SOL 头行是否被教师引用（诊断文本引用率）；**机械存在检查**（甲 F18：义务句空转=fail） | 两轮 run.log + priors 产物 + 诊断 turn 计数表；survey 未做而无机械拦截 = 本项 fail | `[loop] teacher_prior_survey = false`（card 层 loop.toml） | 第 5 位 |
| A7 | **MC-DELIVERY 激活**（通道开，需教师写 `multi_candidate`） | A1 任务集 ×2：普通 vs 教师里程碑含 `multi_candidate:` 块；度量：variant_timings 落行、合并 job 预筛行（smoke 判定表）、赢家独过全门（gate 行只在赢家） | 两轮 status.jsonl（variant_timings 字段）+ 合并 job 日志的 MC 段；**注意 variant_timings 基线无读者（WP21 审计基线 #7）——本项证据同时是 WP16 池的供料验证** | 教师不写 `multi_candidate` 块即天然关闭；`[loop] multi_candidate_max` cli 转发（dcfdcb5 接缝） | 第 6 位（依赖 WP16 池合入后联测） |
| A8 | **stagnation cli 转发复核**（WP10 遗留，dcfdcb5 已修） | 静态+动态双证：静态=cli.py 转发行 grep（`stagnation_k`/`multi_candidate_max`）；动态=cli 传非默认值跑一轮，driver 侧生效值比对（run.log 的 STAGNATION 触发阈值行） | grep 输出 + 一轮动态 run.log（阈值与 cli 值一致） | cli 默认值即回退（不传参） | 第 0 位（纯复核，无翻转；WP23 接缝检查固定项） |

**翻默认顺序汇总**：A8 复核（无翻转）→ A2 flags 立证（无翻转）→ **A4 build cache** →
A3 SPRT → A5 floor → A6 survey → A7 MC。理由：A4 有独立回退 env 且省墙钟最直接；
A3/A5/A6 依赖三代 WP15/WP19 合入后的判定面；A7 依赖 WP16。每步翻完跑 A1 任务集
回归一遍（防交互回归），全绿才进下一项。

## 2. 度量与证据的落盘约定

- 所有 A/B 的 run 归档保留在 `.artifacts/`（不打捞进 git），CHANGES 引用 run 目录名。
- 墙钟构成表（run.log 末尾 `[wp11] wall-clock composition`）是过程提速类翻转
  （A3/A4）的主度量：五桶（agent_turn/eval_job/ncu_job/teacher/queue_wait）逐轮对比。
- verdict 一致性是正确性类机制（A3 SPRT/A5 floor）的主度量：**任何 verdict 漂移 =
  该轮 fail，回退并记录**。
- 3060 上的所有数字只作"链路/行为"证据引用；写进 CHANGES 时必须带
  "3060 链路证据，非性能结论"标注（红线 11）。

## 3. JustGoal 对照制度化（升级为验收动作）

每机制上线前后，固定执行（缺一即该机制不许进 CHANGES 翻默认建议）：

1. **同 job A/B**：同一 run 归档、同一 kernel 树，机制 off/on 各跑一次合并 job；
   对比 EvalResult 全字段（两窗读数、fresh_recheck、gpu_state、gate_spread_pct）。
   `scripts/energize_ab.py --mode same-job` 封装。
2. **换基座重测**（上游"换基座须全机制重测"债务）：机制 on 态在第二块卡/第二架构
   （部署层自决）至少端到端一轮，证据=该轮 run 归档 + status.jsonl。3060 上允许
   以 mini 任务代正式卡，只验机制触发不验收益。
3. **机制行为用例**：pytest 级触发用例（每个机制一条：SPRT 早停边界、build cache
   key 命中/失配、floor 回填门、MC 预筛、stagnation 阈值转发）——已有者在 CHANGES
   引用测试名，缺者由归属包补。
4. 三条证据（同 job A/B、换基座、行为用例）以清单形式进 CHANGES 的机制条目，
   WP23 按清单验收。

## 4. 与 WP21 审计的接缝

- `tests/test_dataflow_audit.py` 是本协议的 CI 侧哨兵：任何 A/B 项要消费的字段
  （`gate_spread_pct`、`clock_drift`、`ref_host_enqueue_ms`、`variant_timings`、
  `build_digest`）目前在基线清单——翻转对应机制前先确认其读者已随 WP15/WP19/WP16
  合入（审计测试会自动要求删基线条目）。
- 冒烟发现的机制静默不触发项（CHANGES-WP21.md 冒烟一节 defect 清单）按归属包分发，
  修复后由本协议对应项的 A/B 复验。
