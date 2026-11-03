# CHANGES-WP11 — 过程提速包：job 冷启动税

分支 `feat/wp11-throughput`（基线 kernel-ZXtune main @ c1971f1 = ForgeKernel @5030503 + WP1–WP6 集成）。
目标：零行为变化地砍掉重复支付的固定成本——eval+NCU 两次冷启动、每 job 全量传输输入张量、固定采样无早停、eval 失败零细节、墙钟不可见。

**验收基线：`pytest tests/ -q` → 2222 passed（merge-base），交付时 2257 passed（+35 例全绿，零回归）。** 新增单测集中在 `tests/test_wp11_throughput.py`（35 例，六任务各一组）+ `scripts/backtest_early_stop.py`（统计门）+ 本机 3060 真跑冒烟（任务 1，日志证据见文末）。

**零行为变化证据（硬验收 §4.2）**：`test_switches_off_matches_baseline_records` —— merge_eval_ncu=false + build_cache 不启用（默认）+ sprt_early_stop=false 时，同一脚本化场景（一 KEEP 一 REVERT）跑主循环，对 merge-base 语义手写的逐字段 golden 断言（verdict/读数/mfu/profiler/kernels/error 前缀/metric_before/telemetry 缺省全部钉死），且合并路径（开关开、卡支持）产出**除 error 的 log-tail 后缀外逐字段相同**的记录——合并是传输层变更，不是语义变更。此外 `test_loop_driver.py` 等全部既有用例（2222 例）原样通过即为逐字节回归。任务 5 的失败路径 error 附加 tail 与任务 4 的五个遥测字段是规格内的**记录性增量**（无开关、与 WP1/WP2 字段追加同一先例：load_records 丢弃旧文件未知键），golden 测试对前者按前缀断言、对后者显式剥离。

## 配置键与词汇表（契约 C7）

沿用 **WP5 先例：`LoopContext` 默认区 + driver 内 `_wp11_apply_cfg(ctx)` 在 run_two_layer 入口解析**（`_wp5_cfg_sections` 同一读取通道：卡片 `loop.toml` `[loop]` 覆盖 `global_config/loop.toml`，读取失败即用内建默认；显式赋值的字段——测试/调用方——永不被覆盖）。未动 `task.py` 的封闭词汇表，未动 cli.py：

- `[loop] merge_eval_ncu = true`（默认 **true**，kill switch 写 false 回两 job 路径）
- `[loop] sprt_early_stop = false`（默认关；置 true 时 driver 把 `FK_SPRT_EARLY_STOP=1` 写进本进程环境，本地 delivery 的 job 继承之；ssh delivery 环境不过 ssh，需在 submit.toml `[env]` 显式放同名变量——CHANGES 注记，机内 3060 场景为本地）
- **build cache**：规格原文写 `[submit] build_cache = false`；本包落成 **submit.toml `[env] FK_BUILD_CACHE = "1"`**（默认不写 = 关）。原因：submit.toml 的表词汇（[repo]/[exec]/[job]/[env]）由 `submit/config.py` 定义，不在本包围栏内；`[env]` 是 runner 与 job 双侧都能看到的既有通道（`_job_env` 全量下发），语义与规格一致——per-submission 配置、默认关。注释写明 **board 卡与密封 ref 树不启用**（`card_template/submit.toml` 与模板块内均有注记）。

## 任务 1（P0）— eval+NCU 合并单 job

- **范围**：只合并主迭代 eval 路径（`run_inner_iteration` 的 eval submit + `_ncu_profiler` 调用点）；`_confirm_suspicious_watermark` 的确认/tie-break job、board/blackbox 专用路径、probe 循环（契约 C4）一律不动。
- **新 gpu_entry 任务 `allncu`**（模板 `_WP11_ALLNCU_CASE`，经 format kwarg 注入，避免 shell 花括号转义地狱）：先跑 `run_with_timeout python3 eval/bench.py --mode all`（与 `all` 任务逐字节相同的计时协议），随后同 job 以独立进程跑 NCU 段——无参时与 `ncu` 任务同一条命令（同一 metrics 过滤、同一 flags——**WP9 领地未碰**，`--launch-count 24` 字面量保留供 taskcard 替换）；带参时跑调用方命令（`ncu_bounded bash -c "$*"`——orch_census 变体，镜像 cmd 任务）。mode 串：`allncu`，census 变体 `allncu:python3 notes/orch_census_run.py ...`（direct.py `_entry_argv` 按 cmd: 同款 shlex 拆分）。
- **卡支持检测**：driver 读 workspace 的 `gpu_entry.sh` 找 `^\s*allncu\)` 标记——旧卡自动回退两 job 路径（`_card_supports_merged_job`，读不到 = 不合并）。
- **超时纪律（规格红线）**：合并 job 预算 = **两段之和，绝不沿用单段 timeout 装两段活**。gpu_entry 内 eval 段仍受 `TASK_TIMEOUT` 全额约束、NCU 段仍受 `NCU_TIMEOUT` 约束（ncu_bounded）；runner 侧 `_deadline_for(mode)` 对合并 mode 取 `cfg.timeout + RUN_HEADROOM_S + NCU_TIMEOUT`（NCU_TIMEOUT 从 job env 读，缺省 600 = gpu_entry 自身默认，坏值回退默认）。选择"按现有超时键放大（两段预算求和）"，未新建 `[loop] merged_job_timeout_s`——driver 无法改写 job env，按段求和是围栏内唯一能真正落实的形态，CHANGES 特此说明。
- **解析**：合并日志整体喂给两套既有解析器——`parse_eval`（eval JSON 是全 log 唯一带 correctness/performance 键的对象）与 `_profiler_from_log`（从 `_ncu_profiler` 原样抽取的折叠体：key metrics + kernel 表 + dominant-kernel resusage + census 字段 + orch summary + SASS precision；纯解析无 submit 无 retry，`_ncu_profiler` 改为调用同一函数，行为不变）。NCU 段输出协议、census 行剥离、RES-USAGE 行全按现有协议拼装。
- **失败语义**：合并 job 的 NCU 段在 **PASSING kernel 上空手而归**（旧路径的 retry 条件）→ **自动回退旧两 job 路径一次**（`_ncu_profiler` 原函数保留、内部 empty-retry 纪律原样）；correctness-FAIL 的空 NCU 不回退（旧纪律：kernel 坏了 NCU 空手是合法读数，不浪费 GPU）。`ncu_on_revert=false` + correctness FAIL 时：合并 job 天然已跑 NCU 段，driver **丢弃**其输出使记录与旧路径逐字段一致（该段 GPU 时间是此配置形状下合并的文档化代价；规格"NCU 仍每迭代必跑、ncu_on_revert 语义不变"逐条保持——KEEP/REVERT 都跑、correctness 联动参数平移为回退条件）。
- **纪律不变清单（逐条自查）**：NCU 每迭代必跑（KEEP/REVERT 都跑，合并≠跳过）☑；`ncu_on_revert` 语义不变 ☑；NCU correctness 联动（retry 只在 passing kernel）平移为回退条件 ☑；REVERT 迭代同样合并 ☑；probe 循环与 SWEEP 未碰 ☑；NCU flags 未碰（WP9 领地）☑；job 超时 = 两段之和 ☑；其他 eval 调用点未动 ☑。
- **单测**：合并模式 submit 次数 2→1 且解析结果与两 job 等价（`test_merged_job_runs_one_submit_and_parses_both_segments`）；census 变体 mode 拼装（driver + direct 两侧）；NCU 段失败回退一次（`test_merged_ncu_failure_on_a_passing_kernel_falls_back_once`）；kill switch/旧卡两 job 路径（`test_merge_off_or_old_card_runs_the_two_job_path`）；ncu_on_revert=false 时合并记录无 NCU 且无回退（`test_merged_revert_with_ncu_on_revert_false_keeps_the_record_ncu_free`）；deadline 求和（`test_merged_deadline_is_the_sum_of_the_segment_budgets`）。
- **3060 真跑冒烟**（规格要求，非 mock）：见文末证据。

## 任务 2 — 内容寻址构建缓存（默认关）

- **位置**：job 侧（规格"job 脚本先查缓存"）。gpu_entry.sh 模板新增 `fk_build_cache_*` 函数块（`_WP11_CACHE_BLOCK`，经 format kwarg 注入）：`FK_BUILD_CACHE=1` 时，key = **sha256(job 树 payload hash（runner 经 `FK_TREE_SHA256` 下发，direct.py `_job_env`）+ nvcc 版本行 + torch 版本)**，缓存目录 `~/.cache/forgekernel-build/`（`FK_BUILD_CACHE_DIR` 可覆盖）——run workspace 之外、wipe 区之外。编译 flags 在源码树的 `fk-cuda-cflags` 行内，已含于树 payload hash，天然入 key。
- **语义红线论证**：`direct.py` 头注释守的 "never inherits build products" 防的是 **stale**——fresh tree 防的是上一棵树的产物混进这一棵。key 在源码内容上：同 sha ⟺ 同源码同 flags，命中产物即本源码的产物，**陈旧产物不可能命中**；且命中后仍做产物自检（manifest.json 记录每个 .so/.o/.a 的 sha256，逐个复核，失配即 miss→全量重建）；toolchain 升级（nvcc/torch 变版）→ key 变 → 天然 miss。缓存永不成为正确性单点：任何一步失败（key 算不出、目录建不出、verify 失败）都回退普通全量构建（`FK_KERNEL_BUILD_DIR` 不设置 = 既有 /tmp 路径）。
- **回填**：miss → 构建导向 `staging.$$`；任务 rc=0（EXIT trap 里 `$?`）→ promote（原子 rename 进 key 位；他 job 先到 = 同 key 同内容，弃 staging）；rc≠0 / promote 失败 → 弃 staging。EXIT trap 同时兜底 `lb_clocks_reset`（cache 开启时本 trap 替换时钟 trap，锁频是必须每出口清理的 GPU 态——原语义不变；cache 关闭时模板行为逐字节不变，有 `test_build_cache_off_by_default_and_silent` 钉住）。
- **cutedsl 模式**：`fk_build_cache_prewarm`——cache 开启时任务前做一次有界（300s）best-effort `import <kernel 模块>`，把 DSL 自己的 JIT file cache（用户级受管位置，不在 workspace——本包不改动其位置）在计时进程前焐热；失败只打一行提示继续。
- **踩过的坑（已修并有测试）**：模板在 `set -euo pipefail` 下运行，key 探测管道里 nvcc/torch 缺失会使 pipeline 非零直接杀 job——探测逐一 `|| true` 卫兵化（`test_build_cache_shell_*` 全部在 `set -euo pipefail` 下跑）。
- **单测**：key 稳定性（同内容同 key→HIT、改树 sha→新 miss）、miss→promote→hit→**篡改产物→verify 拒绝** 三路径（bash 子进程 harness）、默认关闭零输出零副作用、渲染模板 bash -n 通过且含两块。

## 任务 3 — payload 输入张量内容寻址同步（direct 后端 = 本地同机）

- **位置**：`submit/direct.py`（本地 delivery）。每个 job：先 `sync_inputs_to_store`——扫 op 树中 `inputs/` 路径段下 ≥1 MiB 的提交文件（规格的 `tasks/<op>/inputs/` 与卡片实际的 `leaderboard/inputs/…` 两个布局都以路径段覆盖），逐文件 sha256 写进 `~/.cache/forgekernel-inputs/<sha256>`（`FK_INPUTS_STORE` 可覆盖）；已存在且 hash 相符的条目跳过（每 run 首个 job 付费、后续 job 自愈——store 缺文件时从本 job 的树补齐）。随后 **git archive 层排除**（`:(glob,exclude)**/inputs/**`——strip 后置仍要付读账，排除是真正的省）。
- **装载**：解包后 `restore_inputs_from_store` 按 manifest 逐文件从 store 装载（**copy**，与现有 inputs 装载方式一致——树里是普通文件；装载前逐文件复核 sha256，字节等同或抛错）。任何不一致（store 缺失/hash 失配/写不进）→ **回退现行全量传输路径**（重打全量 archive 重解包），日志写 `inputs sync failed ... — re-delivering the full archive`。缓存永不成为正确性单点，树逐字节等同。
- **范围**：ssh delivery 保持全量 payload（跨机同步留 P2，CHANGES 注记）；`FK_INPUTS_SYNC=0`（runner env 或 job env）关闭；无 inputs 目录的旧 task 零影响（manifest 为空 = 原路径）。
- **单测**：manifest 生成/小文件与非 inputs 文件不进 manifest/幂等；archive 排除 + restore 与全量解包逐字节等同；hash 失配拒绝（`test_restore_refuses_a_hash_mismatch`）；禁用开关与无 inputs 树；端到端 runner 级回退（注入 restore 失败 → job 仍 Succeeded 且日志含回退行——真实 hash 失配会被 runner 的自愈补齐，故在 restore 缝隙注入，docstring 写明）。

## 任务 4 — 墙钟构成遥测（先测量后优化）

- `inner/trajectory.py::IterationRecord` 追加五个可选字段：`agent_turn_s` / `eval_job_s` / `ncu_job_s` / `teacher_s` / `queue_wait_s`（全 None 缺省；与 WP9/WP10 字段追加共存，load_records 丢弃旧文件未知键）。**纯测量**：无任何判定/预算/停止读它们；run 末构成表是唯一消费者。
- driver 填数边界：agent 轮 = `run_inner_iteration` 内全部学生 agent turn 累计；eval/NCU job = submit→terminal 墙钟（合并路径的 NCU 段先记解析墙钟，回退时记回退 job 墙钟；不要求 NCU 的迭代为 None）；teacher 轮 = run_meta_step 的 diagnose/patch turn（`_wp11_teacher_timed`），累计到下一条 appended 记录上（`_wp11_wallclock_fields` 读断式）；queue wait = runner 在 GPU 锁上的等待（direct.py `gpu_lock` 现在 yield 等待秒数并在日志写 `[direct] GPU <g> acquired after <n>s wait`，30s 轮询粒度；driver `_queue_wait_from_log` 解析；ssh 不报 → None）。
- run 结束打印构成表：`_wp11_wallclock_summary` 在 run_two_layer 全部 9 个 stop 出口打印（纯 stdout/`_log`，占比按五桶实测合计；没量到就不打印）。
- 单测：字段缺省/读写回环、teacher 桶读断语义、queue-wait 正则、汇总表输出与空运行静默、真实时钟下 KEEP 行三桶非 None。

## 任务 5（P0）— eval 失败错误带宽

- `loop/parse.py`：`job_log_tail`（尾 N 行再截尾 M 字符）与 `_with_log_tail`；`EVAL_TAIL_LINES=150` / `EVAL_TAIL_CHARS=8000` **复制自 driver 的 `_VERDICT_PROBE_TAIL_LINES/_CHARS`**（probe 通道现成协议；两处副本由注释与 `test_tail_constants_match_the_probe_protocol` 钉在一起）。
- `parse_eval` 三个失败面全部附 tail：`no eval JSON block`（job 死在 verdict 前，log 是唯一证据）、数值失败（per-tensor detail 之后附 log 上下文）、COMPILE_ERROR（其 nvcc/ptxas 错误尾就在 log 尾部；JSON 内的 `why[-6000:]` 保留在前）。tail 带 `--- job log tail (last 150 lines, oldest first) ---` 标记行，空 log 不附。log 来自 student 自己树的 job，无 reference 泄露面；进 record 属客观事实。
- 单测：三失败面含 tail、通过路径 error 逐字不变、tail 有界、常数对照钉。

## 任务 6 — SPRT 序贯早停（opt-in 默认关，契约 C5）

- **开关链**：`[loop] sprt_early_stop` → driver `_wp11_apply_cfg` 导出 `FK_SPRT_EARLY_STOP=1` → timing.py 模块旗标 `SPRT_EARLY_STOP`（env 读取）→ `measure_paired(..., early_stop=...)`（显式参数可覆盖模块旗标）。
- **实现（timing.py 自含，不 import forgekernel）**：`_SprtController` 在 **device 窗循环边界**（`time_device` 新增可选 `on_window(index, ms) -> bool`，缺省 None = 循环逐字节不变）做"是否采下一窗"的判定。判定点只在 **第二个 ref 半段的窗边界**（A-B-B-A 顺序下第一个完整的候选 n-窗配对集在此出现）；判定数据是**候选 n-窗协议自己的 index 对齐配对**（两镜像半段 index<n 的 ref/kernel 对），gate ratio 用 **bench 上报的同一估计量**（镜像半段 device median 均值之比，即 vs_ref_device 的算法）。
- **带宽**：候选配对的 MAD×1.4826×3，clamp 到 [0.005, 0.015]——与 driver 侧 `resolvable_diff` 同族统计同系数。**两处公式并列漂移对照**（先例：`_LEVER_NOISE_DEFAULT_PCT == lever_pricing_run.NOISE_FLOOR_PCT`）：
  - `timing.py::sprt_stop_verdict`：`band = clamp(3*1.4826*MAD(candidate pairs), SPRT_BAND_FLOOR, SPRT_BAND_CAP)`
  - `loop/parse.py::resolvable_diff`：`min(cap, max(floor, 3.0 * 1.4826 * mad))`，`RESOLVABLE_DIFF_FLOOR == SPRT_BAND_FLOOR == 0.005`，`RESOLVABLE_DIFF_CAP == SPRT_BAND_CAP == 0.015`
- **硬约束逐条**：窗内 A-B-B-A 协议与 `measure_paired` 一字未动（burn-in、半段顺序、窗内部全原样；只有**窗数**可少）☑；最低样本 ≥4 配对读数且 ≥2 完整窗（`SPRT_MIN_PAIRED=4`/`SPRT_MIN_WINDOWS=2`，先于下限的一律继续采满）☑；决定后**所有半段补齐到决定窗数、所有 side 截齐到同一 n**——上报读数是一个干净的 n-窗协议，判定数据 = 判定所依据的数据。
- **假赢方向的零容忍是算术结构，不是调参**：规则只朝**亏方向**停（ratio + band < 1 = 区间顶仍在平价之下），赢/骑墙永远采满；且停判用的 ratio 与门判定用的 ratio 是**同一个估计量、同一份(截齐后的)数据**。于是：没停 ⟹ 与全采样逐窗相同 ⟹ 判定逐位相同；停了 ⟹ ratio < 1 − band ≤ 1 − 0.005，任何 ≥0 的 margin 都判 REVERT。**"早停 KEEP 而全采样 REVERT" 在算术上不可能发生**。早停只可能把一个全采样会 KEEP 的边界读数提前收成 REVERT（更保守、方向安全；该方向由回测钳在 <2%）——这正是"早停只可能让 σ 估计变宽 → 判定更保守，不制造假赢"的可验证形态。
- **离线回测** `scripts/backtest_early_stop.py`（构造分布；本脚本 torch-free，控制器逐字复制并注明漂移对照）：每格 (真值 ratio × 窗噪声) 4000 次**成对**试验——早停与全采样读**同一窗序列**（成对比较，差异只可能来自截断）；门判定 margin 用 run 级带（建模 driver 的 resolvable_diff：整格一个 margin，非逐测量拟合）。**验收门输出（粘贴）**：

```
backtest_early_stop: 4000 paired trials per (scenario, noise) — full and early read the SAME windows; DEVICE_REPS=5, band=[0.005, 0.015], rule: LOSS-ONLY, verdict-estimator ratio
scenario              noise | full KEEP% early KEEP%   win-lost% false-win | early windows saved%
clear loss  x0.90      0.3% |       0.0%        0.0%     0.00%         0 |                 60.0%
clear loss  x0.90      1.0% |       0.0%        0.0%     0.00%         0 |                 60.0%
clear loss  x0.90      2.0% |       0.0%        0.0%     0.00%         0 |                 60.0%
loss        x0.97      0.3% |       0.0%        0.0%     0.00%         0 |                 60.0%
loss        x0.97      1.0% |       0.0%        0.0%     0.00%         0 |                 59.7%
loss        x0.97      2.0% |       0.0%        0.0%     0.00%         0 |                 55.1%
marginal    x0.995     0.3% |       0.0%        0.0%     0.00%         0 |                13.1%
marginal    x0.995     1.0% |       0.0%        0.0%     0.00%         0 |                 8.4%
marginal    x0.995     2.0% |       3.5%        3.4%     0.10%         0 |                18.7%
null        x1.000     0.3% |       0.0%        0.0%     0.00%         0 |                 0.2%
null        x1.000     1.0% |       0.2%        0.2%     0.00%         0 |                 2.5%
null        x1.000     2.0% |       7.5%        7.3%     0.17%         0 |                11.8%
marginal    x1.006     0.3% |       0.0%        0.0%     0.00%         0 |                 0.0%
marginal    x1.006     1.0% |       5.1%        5.1%     0.00%         0 |                 0.4%
marginal    x1.006     2.0% |      21.0%       20.7%     0.25%         0 |                 5.7%
real win    x1.020     0.3% |     100.0%      100.0%     0.00%         0 |                 0.0%
real win    x1.020     1.0% |      82.8%       82.8%     0.00%         0 |                 0.0%
real win    x1.020     2.0% |      68.4%       68.4%     0.03%         0 |                 0.5%
big win     x1.080     0.3% |     100.0%      100.0%     0.00%         0 |                 0.0%
big win     x1.080     1.0% |     100.0%      100.0%     0.00%         0 |                 0.0%
big win     x1.080     2.0% |     100.0%      100.0%     0.00%         0 |                 0.0%

TOTAL false-win (early KEEP / full REVERT): 0 (gate: 0); worst win-lost rate: 0.25% (gate: <2%); windows taken: 1347132/1680000 (19.8% saved)
GATE: PASS
```

  假赢 0 例（结构保证的实证）；赢丢失最差 0.25% < 2%；明确亏迭代省 55–60% device 窗，整体 19.8%。
- 单测：控制器只在合法边界对明确亏停（win/骑墙不停、n<2 不判）、stop-verdict 单元（四方向+样本下限）、**开关关=逐字节现状回归**（`early_stop=False` 与无参 measure_paired 同读数同顺序）、明确亏早停改变读数且协议顺序不变、赢/平价采满、模块旗标 env 联动、回测门通过。

## 任务 4 顺带（规格附言）— stall 价格缓存落盘

- `_STALL_PRICE_LOGS` 纯内存 dict 在 driver 重启时清空，重启的 run 重复买价。现读写穿到 `<workspace>/notes/wp11-stall-price/`（`index.json` + `<key>.log`，key=(status_path, ref, cmd) sha1；LRU 上限 8 条）。**从不进 job payload**：目录自带 `.gitignore`（`*`），git archive 只运提交文件、`git add notes/` 也跳过 gitignored 树。读写全部 best-effort：notes 不可写时退化为原内存缓存（job 再买一次，与改动前完全一致）。`_run_stall_pricing` 的复用判定（只收整读数）原样不变。

## 范围围栏自查

只动了：`driver.py`（主 eval 路径合并、墙钟字段填充、stall 价格缓存落盘）、`submit/direct.py`（合并 mode 拆分、deadline、输入同步、FK_TREE_SHA256、锁等待上报）、card 模板 job 脚本生成（`leaderboard.py` 模板两块 + `taskcard.write_card` 传递 kwarg）、`card_template/eval/timing.py`（**仅**窗边界早停判定 + on_window 透传；协议一字未动）、`inner/trajectory.py`（5 个可选字段）、`loop/parse.py`（log-tail）、`global_config/loop.toml` 未改（两键默认走 LoopContext 默认区——按 WP1/WP5 先例，[loop] 键的实际覆盖面是卡片侧 loop.toml；global_config 的封闭词汇表未扩，WP1 已注记 [target]/[budget] 之外会被 `_unknown_keys` 拒绝）、`scripts/backtest_early_stop.py`（新）。**未碰**：measure_paired 镜像顺序与窗口内部协议、precision_gate.py、注入区、probe 循环（契约 C4）、NCU flags（WP9 领地）、student_prompt.md。`cli.py` 零改动（配置走 `_wp11_apply_cfg`，与 `_wp5_apply_cfg` 同模式）。

## 预期节省估算（标注：落地后用任务 4 遥测实测校准）

| 机制 | 依据 | 估算 |
|---|---|---|
| eval+NCU 合并 | 每迭代省一次完整解包+构建+往返；lever_pricing_run.py 自认 build 占 job clock ~40%，nvcc+CUTLASS ~35–40s | 每迭代省一次冷启动 ≈ 数十秒～分钟级；两位数百分比墙钟（总览裁决同款） |
| 输入张量同步 | attention 类 GB 级输入每 job 一次 archive+extract；省读/写各一遍 | 大输入任务每 job 省一次 GB 级 I/O；小输入任务 ≈0 |
| SPRT 早停 | 回测：明确亏迭代 device 窗省 55–60%（device 窗约占计时段大头）；win 迭代不省 | 亏迭代（多数）计时段两位数百分比；整体采样 ~20%（回测口径） |
| 构建缓存 | 命中即免 nvcc+CUTLASS ~35–40s/job | 默认关；开启后热树近似免构建 |
| eval 失败带宽 / 墙钟遥测 | 不省墙钟，省**人的迭代**（失败诊断不再盲猜；加速优化的第零步） | 不可计，先测量 |

## 3060 真跑冒烟证据（任务 1 验收项）

本机 RTX 3060 Laptop GPU（WSL，sm_86，torch 2.14.0+cu130，job python = `~/venvs/ml`）：手工小卡（`/tmp/wp11-smoke/operators/smoke`）携带**新模板渲染的 gpu_entry.sh**（`bash -n` 通过、含 `allncu` 任务），eval/bench.py 用卡片真实的 `timing.py::measure_paired`（bf16 rmsnorm kernel vs torch 参考，burn-in 1s），eval/ncu_run.py 是真 NCU 目标（profiler start/stop 圈 4 次发射）。经**真实 runner**（`python -m forgekernel.submit smoke submit --mode allncu`，全程 `with-lock gpu` 持锁）提交，job **1793674836 → Succeeded**：

- **合并单 job**：`[direct] local GPU 0 branch=main mode=allncu`——一次 submit 一次解包一次构建；
- **计时协议原样**：eval 段 `=== JSON ===` 携带镜像 device readings（kernel `[1.9193, 2.3625]` / ref `[2.0221, 2.0374]`）、host_bound_ratio、`vs_ref_device=0.8452`、gpu_state 13 样本（SM 1963 MHz / 64.2°C / 97.4W）——`paired_performance` 全部键原样；
- **两段分隔按协议**：`[gpu_entry.sh] allncu: eval segment done — starting the NCU segment` 之后 NCU 段独立进程启动，**真实采到 CSV**（`==PROF== Connected to process`、Kernel Name 表头 + `vectorized_elementwise_kernel`(rmsnorm 逐元素核) 的 regs/smem/waves 等指标行，`ncu_filter_metrics` 在 sm_86 上正常过滤）；
- **收尾协议**：`[gpu_entry.sh] task=allncu finished`，runner 判 Succeeded——driver 侧 `parse_eval` 与 `_profiler_from_log` 对这份合并 log 各取所需（eval JSON 是全 log 唯一 correctness/performance 对象；CSV 表头前的 bench 输出被 NCU 解析器按既有规则忽略）；
- **失败语义旁证**（第一次冒烟，bench 核验 NaN → rc=1）：`set -e` 使 job 在 NCU 段前终止，log 以 `[direct] gpu_entry allncu exited rc=1` 收尾——driver 将读得 correctness FAIL + 空 NCU，不回退不重试（与两 job 路径的 FAIL 语义一致，即"correctness 坏的 kernel 无 NCU 是合法读数"）。
