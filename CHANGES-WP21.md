# CHANGES-WP21 — 真机通电与数据流审计（三代验收底座）

分支 `feat/wp21-energize-audit`（基于 kernel-ZXtune main @ `359696b`）。执行合并裁决
（handoff/59 §四 WP21）与发包指令（handoff/80；包 68）。本包是审计与协议包：
**审计只读不改他包领地；缺陷只记录不代修；翻默认动作绝不发生在本包（归 WP23 一次一翻）**。

## 0. 前置门记录

| 项 | 结果 |
|---|---|
| clone + 分支 | `wk3-wp21` ← `zxtune-integration`；`feat/wp21-energize-audit` from `origin/main` @ `359696b`（clone 默认检出 integrated-v2，已显式切出，与发包指令一致） |
| 基线 pytest | `~/venvs/forgekernel/bin/python -m pytest tests/ -q` → **2437 passed in 326.35s**（与发包预期 ≈2437 一致） |
| 必读 | 59-合并裁决 §四 WP21 节 + 28（甲 G25）、30（乙 F27/F13/F5/F16/F27）、40（丙 E 节遗留清单）——已核对 |
| 机器 | RTX 3060 Laptop 6GB（sm86，WSL2，CUDA UMD 13.4）；红线 11 全程遵守：3060 只验链路，不产性能结论 |

## 1. 任务 1（P0）：数据流审计 CI ✅

新增 `tests/test_dataflow_audit.py`（5 测试）+ 两份登记文件：

- **字段宇宙**（ast 机械抽取）：`IterationRecord`（inner/trajectory.py）与 `EvalResult`
  （loop/parse.py）全部 dataclass 字段 + dict 载荷遥测键（`_SOL_HEADER_KEYS` 6 键、
  `GPU_STATE_FIELDS` 4 键）+ `build_digest` 生产调用点专项。WP11 墙钟五键本就是
  IterationRecord 字段，按字段审计不重复入表。
- **读者口径**（grep 级，包指令"grep 级即可"）：属性读 `.f` 与精确字符串键读
  `"f"`（`.get`/`["f"]`/`getattr`/提取元组）算读者；声明行、构造 kwarg、属性赋值、
  字典字面量构造、注释行不算；`tests/`、`skills/`、`docs/`、`meta/` 一律不算
  （仅 tests 命中 = fail）。分层：
  - IterationRecord（轨迹 SSOT）的读者必须在写侧管道（`loop/driver.py`、
    `loop/parse.py`）、同名异物文件（`launch.py`/`loop/cli.py`/`monitor/cli.py`——
    消费的是确认 JSON/监控 JSON/job 状态的 `"sha"` 等键，非轨迹行）与 eval bench
    写侧（`card_template/**`）之外；
  - EvalResult（job→loop 交接）的读者 = driver 及以上（parse.py 是构造方不算）；
  - 遥测键：键读在生产者之外，或元组本身在生产者之外被引用（间接消费）。
- **登记与自我维护**（防形式主义）：
  - `tests/dataflow_audit_baseline.json`（7 条）：预期被修好的无读者字段，带
    owner 包与证据（file:line）；
  - `tests/dataflow_audit_whitelist.json`（16 条）：确属记录备用/loop 内闭环消费，
    带理由；
  - 陈旧条目反向拦截：条目长出真读者 → fail 要求删行。**合并后（WP15/WP19/WP16
    消费方合入）WP23 逐条删空基线清单，跑全绿 = WP23 验收项**；
  - 抽取器哨兵（防空集恒过）+ 负向验证（伪造 `IterationRecord.iter`/`verdict`
    登记 → 按预期 fail 并给出精确读点）。
- **审计测试全绿**：`tests/test_dataflow_audit.py` 5 passed（含负向验证：伪造
  `IterationRecord.iter`/`verdict` 登记 → 按预期 fail 并给出精确读点——该负向验证
  的伪造条目在一次恢复脚本异常后曾残留于基线文件，**tripwire 在全量跑中精确拦截**
  （有读者的字段被登记 → 过时 → fail），清除后全绿——拦截能力的一次真实自证）。
- **全量 pytest**：`tests/ -q` → 2442 passed（基线 2437 + 本包审计 5），见 §5。

### 基线"无读者字段清单"（CHANGES 要求粘贴；pytest -s 可复现）

```
WP21 dataflow audit — readerless keys: 19/80 fields/keys
  READERLESS  EvalResult.clock_drift  [no-read]
  READERLESS  EvalResult.ref_device_readings_ms  [plumbing-only]
  READERLESS  IterationRecord.agent_turn_s  [plumbing-only]
  READERLESS  IterationRecord.base_sha  [plumbing-only]
  READERLESS  IterationRecord.clock_drift  [no-read]
  READERLESS  IterationRecord.eval_job_s  [plumbing-only]
  READERLESS  IterationRecord.gate_spread_pct  [no-read]
  READERLESS  IterationRecord.gpu_state  [plumbing-only]
  READERLESS  IterationRecord.ncu_job_s  [plumbing-only]
  READERLESS  IterationRecord.orch_summary  [plumbing-only]
  READERLESS  IterationRecord.queue_wait_s  [plumbing-only]
  READERLESS  IterationRecord.reading_of  [plumbing-only]
  READERLESS  IterationRecord.ref_host_bound_ratio  [plumbing-only]
  READERLESS  IterationRecord.ref_host_enqueue_ms  [plumbing-only]
  READERLESS  IterationRecord.regression_vs_best_pct  [plumbing-only]
  READERLESS  IterationRecord.suspicious  [no-read]
  READERLESS  IterationRecord.teacher_s  [plumbing-only]
  READERLESS  IterationRecord.variant_timings  [no-read]
  READERLESS  build_digest  [no-read]
```

### 基线条目（owner 归属；WP15/WP19 合入后自然清空）

| 字段 | tier | owner | 修复归属依据 |
|---|---|---|---|
| `IterationRecord.gate_spread_pct` | no-read | **WP15** | 包 68 点名首批；best 元分析按 spread 判读 |
| `IterationRecord.clock_drift` | no-read | **WP15** | 包 68 点名（剔除与触发之外的历史行）；driver 写入（driver.py:3030）后零消费 |
| `EvalResult.clock_drift` | no-read | **WP15** | driver 用 `_sm_clock_drift(ev.gpu_state,…)` 现场重算（driver.py:1739,3006），从不读旗标——读点应统一 |
| `EvalResult.ref_device_readings_ms` | plumbing-only | **WP15** | A2 分层方差需 ref 原始对；现只读 device 侧（driver.py:3026） |
| `IterationRecord.ref_host_enqueue_ms` | plumbing-only | **WP15/WP19** | 包 68 点名；轨迹行零消费（digest 设计性排除，trajectory.py:58-61），消费命中全在 EvalResult 侧 |
| `IterationRecord.variant_timings` | no-read | **P2** | driver 生成写入（driver.py:6642 等）后无回读；driver.py:6538 宣称 status.jsonl 为 durable fact source 但 resume 无读点；Wave-3 各包未领收 |
| `build_digest` | no-read（零生产调用点） | **WP19** | Layer-1 digest 从未进任何 agent 上下文（仅定义 trajectory.py:320 + 导出）——包 68 点名"WP19 接线" |

（白名单 16 条见 `tests/dataflow_audit_whitelist.json`，逐条带理由与 file:line：
墙钟五键=run 末构成表设计终点；`reading_of`/`base_sha`/`orch_summary`/`sha`/
`regression_vs_best_pct`=loop 决策闭环活消费；`suspicious`/`gpu_state`/
`ref_host_bound_ratio`=记录备用或教师读行设计；遥测 `mem_clock_mhz`/`temperature_c`/
`power_w`=gpu_state 证据块随行字段。）

## 2. 任务 2（P0）：mini 任务真 run 冒烟 ✅（stub agent 替身通路）

### 2.1 通路与环境（本节同时是环境 defect 的修复记录）

mini task：`tasks/wp21mini/`（**测试专用资产，不进正式算子集**——CUDA C++
elementwise add，float32 n=1M，sm86，最小可评测面）。runner：`tasks/wp21mini/run_smoke.sh`；
真跑一律 `~/tools/bin/with-lock gpu 1800 -- bash tasks/wp21mini/run_smoke.sh` 包裹。

冒烟中发现并**只在本包资产内**解决的环境问题（他包生产代码零改动）：

| # | 问题 | file:line / 证据 | 处置（全部落在冒烟资产内） |
|---|---|---|---|
| E1 | **本机无 agent 凭据**：`claude auth status` → `loggedIn:false, authMethod:none`；env/local.env 均无 ANTHROPIC_API_KEY/OAUTH token。真 agent turn 无法运行——两代"机制在数据流断"的同一根源（从未在真机转过） | `~/.claude/settings.json`；`local.env.example` | 冒烟用 `FK_CLAUDE_BIN`（scripts/agent-claude.sh 契约的官方逃逸口）接脚本化替身 `tasks/wp21mini/agent_stub.py`——eval/NCU/计时/解析/轨迹全部真机真实执行，仅"内核作者"是确定性 stub。**真 agent 冒烟需用户配凭据后重跑（交 WP23/用户）** |
| E2 | venv editable 安装劫持：`~/venvs/forgekernel` 指向 `/home/zcz/forgeKernel/ForgeKernel/src`（上游 checkout），`python -m forgekernel` 会把 repo_root/hw_kb/.artifacts 全部落到上游树 | `~/venvs/forgekernel/lib/python3.14/site-packages/__editable__.forgekernel-0.1.0.pth` | runner 内 `PYTHONPATH=<clone>/src` 压前（clone 内跑 `forge` 必须带；建议上游/集成步知悉此坑） |
| E3 | hw_kb 无 cuda/sm86 树：task 校验拒卡（taskcard 检查 hw_kb/cuda/<arch> 存在） | `src/forgekernel/task.py` hw_kb 校验 | 新建 `hw_kb/cuda/sm86/`（README+TEMPLATE+空 index.md，零 recipe，自述为 WP21 冒烟脚手架）——**KB 内容新增，无他包代码改动** |
| E4 | job-env 预检要 CUTLASS 头（policy=refuse）：本机无 /usr/local/cutlass | `skills/job-env-check/job_env_check.py:45-49` | CUTLASS v4.8.1 header-only 浅克隆到 `~/tools/cutlass`（遵守环境守则：工具进 tools；下载守卫拦 curl 后改走 git clone 通道），runner 导出 `CUTLASS_DIR` |
| E5 | WSL 缺 `-lcuda` 开发符号链接：torch cpp_extension 链接 `-lcuda` 失败（WSL 只有 `/usr/lib/wsl/lib/libcuda.so.1`） | 冒烟 run0 status.jsonl error 尾巴：`ld.bfd: cannot find -lcuda` | `~/tools/wsl-cuda-lib/libcuda.so → /usr/lib/wsl/lib/libcuda.so.1` + runner 导出 `LIBRARY_PATH`（零生产代码改动） |
| E6 | eval 模板默认 gencode=sm_100a（Blackwell）在 sm86 不可加载 | `card_template/eval/leaderboard.py:452`（CUDA_GENCODE，禁碰未动） | 用模板自带的 `// fk-cuda-cflags:` 白名单行（leaderboard.py:463 CUDA_FLAG_OK）在 stub 内核里声明 `-gencode=arch=compute_86,code=sm_86` |

### 2.2 冒烟证据清单（逐项对照发包指令）

最终 run 归档：`.artifacts/wp21mini_261103-183520-78a3/`（12 个真 GPU job；
run.log 即合并日志；stub agent 替身，eval/NCU/计时/解析/轨迹全为真机真实执行）。

**主证据——iter0 真评 KEEP（status.jsonl 第 1 行摘录）**：

```
verdict: KEEP | device_median_ms: 0.0512 | ref_device_ms: 0.0468 | vs_ref_device: 0.9138
profiler（NCU 12 键全落轨迹）: regs_per_thread, tensor_pct, warp_active_pct,
  stall_long_sb, stall_mio, stall_barrier, smem_static, smem_dynamic,
  compute_sol_pct, waves_per_sm, stall_lg_throttle, block_size
gate = geomean over 3 workloads: std-1 0.0582 ms (x0.7896); std-2 0.0446 ms (x1.0501);
  std-3 0.0516 ms (x0.9202)
best 确认: 3 独立 confirm job（1793702479/1793702494/…），median_ms=0.0519,
  median_ratio=0.9403, sha=5f2d1b4589；correctness_monitor=ok
run 收束: done {"stop": "meta_budget", "meta_steps": 2, "meta_extra_used": 1
  （WP5 meta_ext_on_headroom 机制真实触发：0/6）}
```

逐项证据（发包指令 §任务 2 清单）：

1. **非空 falsified_ledger ≥1 行**：**未达成，defect D-1**——本轮教师（stub）不引
   lever → 无 pricing job → 无 ledger 行；`hw_kb/falsified_ledger.jsonl` 终态 0 字节
   （F13 现状未变）。写侧唯一入口与触发链已定位（driver.py:6736→6827），见 §2.3。
   （通路的机制侧事实：MC-DELIVERY 同因未触发——教师未写 multi_candidate，
   variant_timings 无行，与 defect D-2 同链。）
2. **kbmenu 命中遥测 ≥1 条**：**部分达成，defect D-4**——stub 提示词捕获
   `ws/.wp21_stub_prompt_log.txt` 共 16 turn 全部命中 `compute_sol_pct`（SOL 头行
   进诊断上下文 ✓）；KB-MENU 块未渲染（空 sm86 KB + 空 ledger → 合成器无料，
   kbmenu.py:395-440 的空态行为）；命中计数器基线树不存在（F16，归 WP18）。
3. **floor_backtest 回填 ≥1 条（--from-run 通路）**：**通路缺位，defect D-9**——
   基线树无 `--from-run`（`scripts/backtest_floors.py --help` 仅
   `--manifest/--selftest`；脚本自述 best_ms 由操作者人工从 run 归档填充，
   backtest_floors.py:14-18）。回填的输入侧已验证可从归档取数：
   `best_confirmation.json`（device window，3 job median 0.0519 ms）完整在档。
   `floor_backtest.json` 是 WP9 领地，本包不改；自动化归 WP23（协议 A5 依赖项已改写）。
4. **墙钟构成表落盘**：✅ run.log:126-130（粘贴见 §2.5）。
5. **fresh_recheck 执行记录**：✅ iter0 eval 与 confirm job 的 log.txt 均有
   `[leaderboard] fresh recheck (regen): PASS`（4+ 次；regen=内容变种子层也过）。
6. **合并 job 日志**：✅ job 1793702240 单 job 两段：
   `[gpu_entry.sh] allncu: eval segment done (wall 78s since entry)` +
   profile 段（`task: ncu`）；iter0 的 eval_job_s=87.1 / ncu_job_s=9.0 分桶记账。

**机制真触发清单**（本轮真跑里被锻炼到的机制，全部有 in-run 证据）：
WP9 fresh_recheck（PASS）✓ / WP9 NCU regime 与 12 键遥测入轨迹 ✓ / WP1 host_bound
预检（iter0 `WARNING HOST-BOUND IN-JOB REFERENCE (host_bound_ratio=0.79)`，且按设计
未报给学生）✓ / WP11 合并 job ✓ / WP11 墙钟构成表 ✓ / WP1 best 确认（3 job）✓ /
WP5 meta 预算联动 headroom 扩展（meta_ext_on_headroom 0/6→1）✓ / WP9 stall-price
dossier（job 1793702449，`ledger OK @ 5f2d1b4589`）✓ / recon ref-price 四件套
（ledger/segments/roles+tensor/timeline，job 1793701503）✓ / M0 audit
（`AUDIT_CLEAN — milestone consistent with the hardware truth`）✓ /
WP10 MC-DELIVERY：未触发（教师未写 multi_candidate，defect D-2 链）。


### 2.3 冒烟 defect 清单（机制静默不触发；只记录不代修，file:line 可定位）

| # | defect | 证据（file:line） | 归属包 |
|---|---|---|---|
| D-1 | falsified_ledger 写侧唯一入口是 lever 定价表落盘（`_persist_lever_table`→`_append_falsified_ledger`，driver.py:6736,6827），而定价表只来自脚本化 pricing job——**无 lever 引用即零行**。两代真 run 里从未有教师引 lever，故 F13 恒 0 字节 | driver.py:6736-6756,6827-6868 | WP16（池/重定价消费方）；WP23 协议 A7 项 |
| D-2 | `variant_timings`（MC 变体表）driver.py:6538 宣称 status.jsonl 是 durable fact source，但全库无读点（resume 不断言变体表）——与审计基线第 7 条同源 | driver.py:6538,6642；trajectory.py:180 | WP16（池）+ WP19（resume） |
| D-3 | `build_digest` 零生产调用点——Layer-1 digest 整链断（教师诊断看不到 digest 装配产物） | trajectory.py:320；全库 grep 仅定义+导出 | WP19（接线） |
| D-4 | kbmenu 命中无计数器（F16 现状复核属实）：kbmenu.py 只拼块不记账，CHANGES-WP12 的 A/B 指标是事后人工复盘 | kbmenu.py:395-440 | WP18（三代命中遥测新增） |
| D-5 | recon write-up 交付物检查只告警不放行拦截（"write-up turns exhausted … distillation will see whatever exists"）——蒸馏吃残缺侦察产物无硬门 | driver.py meta/recon 段（run.log 摘录） | WP17/WP18 视界；P2 可接受（有告警） |
| D-6 | merged eval+NCU job 的 NCU 段在 eval 段编译失败时继续跑并重复同一编译失败（8s 空转段）——错误在段间不短路（设计如此：段预算=两段之和；记录为观察项） | run.log：`NCU exited rc=1 — continuing without a profile`；gpu_entry.sh allncu 段 | WP19（合并 job 领地）；P2 |
| D-7 | `gpu_state`/`profiler` 在编译失败行为空 dict（正确行为），但 `queue_wait_s=None` 而 local delivery 有锁等待测量路径——本轮无争锁故 None 正确；**观察项**：三并行包同跑时此字段应非空，构成表才能读出排队桶 | driver.py:2877 `_queue_wait_from_log`；submit/direct.py:352-357 | 无（记录现象） |
| D-8 | **环境**：E1-E6（§2.1 表）——其中 E1（无凭据）阻塞真 agent 冒烟，其余为冒烟资产内已修复 | §2.1 | 用户/E1 交 WP23 |
| D-9 | **floor_backtest `--from-run` 通路不存在**：包指令所称"回填通路"在基线树无实现——`scripts/backtest_floors.py` 仅 `--manifest/--selftest`（help 实证）；best_ms 由操作者按脚本自述人工从 run 归档填（backtest_floors.py:14-18）——25×null（F5）两年未回填的机制性原因 | backtest_floors.py:14-18,47,81；floor_backtest.json note | WP23（协议 A5 前置：落自动化或按文档人工填） |
| D-10 | **WP1 gpu_state 遥测在多 workloads 路径断**：`measure_windows_workloads` 返回块不含 gpu_state（timing.py:969，`return {"workloads":…, "burn_in_rounds":…}`），单窗 paired 路径才发（timing.py:885）；`workloads_performance` 格式化器也不补；parse.py:423-428 读顶层/perf 两路均落空 → 记录 gpu_state={}（本冒烟 iter0 实证）。本机 nvidia-smi 手测可用（210/7001 MHz, 51°C, 23.7W），pynvml 缺失是次要因素 | timing.py:969（return 无 gpu_state）、:885、:376-393（report）、:443（GPU_STATE_FIELDS）；loop/parse.py:423-428 | **WP15**（测量统计遥测闭环；card_template=禁碰领地，只记录） |

### 2.4 floor_backtest 回填通路核查（defect D-9 的实证）

```
$ python3 scripts/backtest_floors.py --help
usage: backtest_floors.py [-h] [--manifest MANIFEST] [--selftest]
# 无 --from-run：包指令所称"回填通路"基线树无实现（defect D-9）

# 回填输入侧可从归档取数（.artifacts/wp21mini_261103-183520-78a3/
# operators/wp21mini/best_confirmation.json 摘录）：
{ "sha": "5f2d1b458970fd63…", "window": "device", "requested": 3,
  "jobs": [ { "job": "1793702479", "ms": 0.0541, "ref_ms": 0.046, "ratio": 0.8502,
              "workloads": [ {"name": "std-1", "device_median_ms": 0.045, …} ] }, … ] }
# → best_ms=0.0519（median over 3 jobs）可人工填入 floor_backtest.json 对应条目；
#   floor_backtest.json 为 WP9 领地，本包不改。
```

### 2.5 墙钟构成表与 fresh_recheck 摘录

```
[loop] [wp11] wall-clock composition of the loop (measurement only — no verdict reads this;
       shares over the 96.2s of measured buckets):
[loop]   agent_turn_s        0.1s    0.1%
[loop]   eval_job_s         87.1s   90.5%
[loop]   ncu_job_s           9.0s    9.4%
（teacher_s/queue_wait_s 本轮为 None：单会话无排队、教师轮计时路径未走到——与审计白名单
  一致，构成表是五字段的设计终点消费）

# fresh_recheck（WP9，eval+confirm job 各自执行，regen 模式=内容变种子层亦过）：
[leaderboard] fresh recheck (regen): PASS     # ×4+（.jobs/*/log.txt）
```

### 2.6 冒烟执行记录（逐阶段摘录，.artifacts/wp21mini_261103-183520-78a3/run.log）

```
[forge] the judge: files pass the static check                    # stub judge 回合
[forge] calibrate → tolerance frozen；probe NCU → available
[loop] the ref baseline … (0.0607 ms, ncu=yes)                     # 真 baseline job
[loop] meta/recon: 12 run(s), budget exhausted … ref-price ledger/segments/
       roles+tensor/timeline banked (job 1793701503)               # recon/ref-price 真跑
[loop] meta/distill: authored milestone v1 (violations=none)       # stub 教师产出合规 milestone
[loop] iter 0: decision D0.1 committed; evaluating on GPU…
[loop] iter 0: WARNING HOST-BOUND IN-JOB REFERENCE (host_bound_ratio=0.79)…（按设计不报学生）
[loop] iter 0: D0.1 KEEP  metric=0.0512 (device) vs_ref=0.6235 vs_ref_device=0.9138
[loop] meta/m0-audit: AUDIT_CLEAN — milestone consistent with the hardware truth
[loop] meta/stall-price[best]: ledger OK @ 5f2d1b4589 (job 1793702449)
[loop] meta budget exhausted but the best sits 99.8% above the teacher's floor_opt
       -> granting ONE extra meta round ([budget] meta_ext_on_headroom; 0/6)
[loop] confirm: job 1793702479 device=0.0541 ref=0.046 ratio=0.8502
[loop] confirm: job 1793702494 device=0.0519 ref=0.0488 ratio=0.9403
[loop] done: {"stop": "meta_budget", "meta_steps": 2, "floor_opt_ms": 0.0001,
       "confirm": {"median_ms": 0.0519, "median_ratio": 0.9403, "n": 3, "sha": "5f2d1b4589"},
       "correctness_monitor": "ok", "unharvested": []}
[loop] status.jsonl now has 1 records
```

（数字均为 3060 链路证据，红线 11：不作任何性能结论。）


## 3. 任务 3（P0）：验证与收获协议文档 ✅

- `docs/energize-protocol.md`：验收 A/B 矩阵 A1-A8 全项（普通任务集 / NCU flags A/B
  【WP9 遗留 ncu_flag_ab 方向确认】/ SPRT / build cache / floor_stop+floor_margin /
  survey / MC / stagnation cli 转发复核）——每项：度量方案、证据要求、回退开关、
  翻默认顺序（**一次一翻**，汇总顺序 A8 复核→A2 立证→A4 build cache→A3 SPRT→
  A5 floor→A6 survey→A7 MC）；JustGoal 对照制度化（同 job A/B + 换基座重测 +
  行为用例三证据固定动作）；与 WP21 审计的接缝条款。
- `scripts/energize_ab.py`：参数化骨架（`--item A1..A8`、`--mode task-set|same-job|
  ncu-flag-ab`、`--report-only` 从归档提墙钟构成与 verdict 对照；**绝不翻默认**，
  dry 骨架 + 红线 11 标注）。`python scripts/energize_ab.py --help` 可运行 ✅。
- 跑批与翻默认归 WP23。

## 4. 任务 4（P2）：遗留收尾对账（丙 40-E）✅

| 40-E 遗留项 | 去向 | 状态 |
|---|---|---|
| WP2 traffic_amp 字节字段 | 59-裁决 ⏸ P2 积压维持（"agent/GPU 重叠流水线…traffic_amp"行） | **未收编** → P2 积压（本表即移交记录） |
| WP2 CuteDSL 卡无 RES-USAGE | 59-裁决 ⏸ P2 积压维持 | **未收编** → P2 积压 |
| WP5 global 层 loop.toml 键不可生效（task.py._KEYS 封闭） | WP23 §7"loop.toml 键合并：八包键块齐全、全局唯一"只做键合并不做词汇开放；包 68 只许 `# --- WP21 ---` 键块 | **部分收编**（键合并归 WP23）；`_KEYS` 开放本身仍无主 → P2 积压 |
| WP9 `ncu_flag_ab` 方向性确认未跑 | 本包协议 **A2 项**（3060 现已有 CUDA torch venv `~/venvs/ml` 可跑） | **已收编**（执行归 WP23） |
| WP9 floor_backtest 25×null | 本包冒烟验证 --from-run 回填通路 + 协议 **A5 前置** | **已收编**（回填执行归 WP23） |
| WP9 NCU 新旧 regime 不可混读 | 协议 A2"报表头注明"条目 | **已收编**（文档条目） |
| WP11 ssh delivery 输入同步 | 59-裁决未点名 | **未收编** → P2 积压 |
| WP11 `ncu_on_revert=false`+合并路径 NCU 段输出丢弃（配置交互小坑） | 40-E 自述"文档化代价"；本包冒烟 defect D-6 相邻观察项 | **文档化**（defect 清单记录） |
| WP12 lever↔KB prescribed 联动 | 59-裁决"KB 全面键控重构 ❌；轻量指针+实证缺口供给已立项（WP18）"——prescribed 联动未单独立项 | **未收编** → P2 积压（WP18 的轻量指针与其相邻但非同一物） |
| WP12 STALE 行靠 harvest 提示词追加 | 65-WP18："kb_harvest 加存活模式升权段——与 WP12 STALE 衔接" | **已收编**（WP18 衔接条目） |
| 一代 P2：探针批量 manifest | 66-WP19 manifest 项（WP17/WP18 消费） | **已收编** |
| 一代 P2：per-shape 决策交互账本 | 61-WP14 per-shape schema（字段在，交互账本本体无主） | **部分收编**；账本本体 → P2 积压 |
| 一代 P2：blackbox guard 已知失效模式修复（docs/blackbox-mode.md:76-109） | 59-裁决 ⏸ P2 维持 | **未收编** → P2 积压 |
| 一代 P2：会话 resume | 66-WP19 任务 6（resume_session，默认 false） | **已收编** |
| 一代 P2：决策集投影 | 59-裁决 ⏸ 拒绝/积压行 | **未收编** → P2 维持 |
| 一代 P2：REVERT 降分辨率 NCU + WP7 精度车道 | 59-裁决：REVERT 降分辨率 NCU ❌ 继承拒绝；WP7 精度车道 → WP22 缓议（须用户单独批准） | **已收编**（WP22 待批） |
| 一代 P2：JustGoal 对照制度化 | **本包**协议 §3 固定动作 | **已收编**（本包交付） |
| 部署层：run 级多卡并行 / eval+NCU 分卡 / 换基座全机制重测 | 59-裁决 ⏸ 部署注记 / 远期；换基座重测进本包协议 §3.2 | 部署自决 / 协议化 |

## 5. 范围与红线自查

- **全量回归**：`~/venvs/forgekernel/bin/python -m pytest tests/ -q` →
  **2442 passed**（基线 2437 + 本包 test_dataflow_audit.py 5），零失败。

- **只动**：`tests/test_dataflow_audit.py`、`tests/dataflow_audit_baseline.json`、
  `tests/dataflow_audit_whitelist.json`、`docs/energize-protocol.md`、
  `scripts/energize_ab.py`、`tasks/wp21mini/`（新增测试专用 mini task：task.toml/
  description/shapes/reference/agent_stub.py/agent_stub.sh/run_smoke.sh/
  milestone_seed.md）、`hw_kb/cuda/sm86/`（新增 KB 脚手架，零 recipe）。
  `global_config/loop.toml` **未动**（审计是 pytest 级、协议是文档+骨架脚本，
  无运行时键需求——加无人消费的死键恰是本包审计反对的形式主义）。
- **禁碰清单核对**：`src/forgekernel/leaderboard.py`、`card_template/**`、`timing.py`、
  `parse.py`、`task.py`、`taskcard.py`、`window_contract.py`、`scaffold.py`、
  `launch.py`、`__main__.py`、`binref_tool.py`、`sealref_tool.py`、`cli.py`（含 loop/cli.py、
  monitor/cli.py）、`NOTICE`、`README.md`、docs/ WP24 删除面——**零改动**
  （`git diff --stat` 只有上列新增文件）。审计对禁碰文件只读。
- 红线 11：3060 全程只产链路证据；CHANGES/协议中 3060 数字均带"非性能结论"标注。
- 翻默认：零发生（协议 A/B 矩阵与顺序只是文档）。
