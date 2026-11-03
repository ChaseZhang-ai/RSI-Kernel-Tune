# CHANGES-WP19 — 过程提速三代

分支 `feat/wp19-throughput-v3`，基线 = origin/main @ `19e885b`（WP24 后布局；merge-base 逐项核实过）。
本包是纯提速包：gate 判定、测量语义、NCU 每迭代必跑、防造假纪律零改动；一切新机制默认关或默认不变，
「零行为变化」以 `tests/test_wp19_throughput.py::test_wp19_switches_off_matches_baseline_records`
的逐字段黄金（在基线 @19e885b 上采集，采集方法见该文件 docstring）加全部 2332 条基线测试原样通过为准。

**就位核查**：cardkit.py 存在 / leaderboard.py 不存在 / suite 零残留 grep 绿（开工前验证，见提交历史首条）。
**基线**：`pytest tests/ -q` = **2332 passed in 262.72s**（开工时实测）；终态 **2405 passed in 278.90s**
（2332 基线全绿 + 73 条 WP19 测试；1 条 WP8 carry 测试的**上界集合**随七件扩件合法放宽，见任务 7）。

---

## 任务 1（P0）：构建缓存 key 窄化（乙B1+丙T5，契约 C11）

**实现**：`cardkit.py` 的 `_WP11_CACHE_BLOCK` 组入 `fk_build_cache_key_narrowed`（`FK_BUILD_CACHE_KEY_NARROW=1`
时生效，默认关）。key = sha256(kernel .cu + 递归引号 `#include` 闭包（逐文件**去注释后**内容哈希——注释不改
codegen，改注释必须命中）+ `fk-cuda-cflags` 行（先于去注释提取）+ 构建脚本（eval/benchkit.py、eval/bench.py、
problem.json 的**逐字节**哈希））再与 nvcc/torch 版本合并。C11 机械枚举与回退纪律：
- 任何异常（不可解析 include / include 逃出树 / 缺 builder 文件 / 非 .cu kernel / 任何 IO 麻烦）→ `sys.exit(1)`
  → **回退整树 key**（只丢命中，永不假命中）；
- 命中侧的逐产物 sha 复核（`fk_build_cache_brain verify`）**原样保留**——陈旧产物仍被拒；
- board 卡与密封 ref 不启用：与 WP11 相同的 opt-in 语义（FK_BUILD_CACHE 本就按 job env）。

**前置验证（丙T5 推断 → 真跑实证）**：本机 3060 真跑（with-lock gpu 全程包裹，锁记录见 §冒烟），
渲染 cardkit 模板成真 gpu_entry 卡树，经真实 `forgekernel.submit` 在 GPU 0 上跑了多个真 job。
整树 key 的 miss-by-构造不再是推断：

| 真 job | 树状态 | FK_BUILD_CACHE_KEY_NARROW | 构建缓存 key（job log 原行） |
|---|---|---|---|
| B | commit「builders」@ tree sha X | 1 | `MISS be0bb653397ed4ebf96ab043ee154be5c1facd263479ea105dc269b2d35d2e7e` |
| B' | 新 commit（仅 notes/probe_a.py 变，payload sha 变） | 1 | `MISS be0bb653397ed4ebf96ab043ee154be5c1facd263479ea105dc269b2d35d2e7e` —— **同 key，跨 commit 命中成立** |
| C | 同 B' 的树 | 未设（旧路径） | `MISS 57c51b9ca3f8c664cef37adca8aa9578be6b80bd5d316dc5cc87c7041639b1c0` —— **key 随 commit 移动，旧语义每迭代必 miss 实证** |

真跑还覆盖了回退路径：无 builder 文件的树上 narrow=1 静默回退整树 key（日志里 key 形态与旧路径一致）——
C11 的"异常=全量构建"在真机上走通。**本地默认开作为建议记录在案（key 覆盖用例 + 真跑证据齐全），
翻转归 WP23 收获步一次一翻**。

**附（甲G24/乙B7）inputs manifest 降本**：`submit/direct.py` 源侧 sha 以 (path,size,mtime_ns) memo
（`<store>/source-manifest.json`，损坏/不可写→全量重算）；store 复核改抽样全量哈希（每
`INPUTS_VERIFY_SAMPLE_STRIDE=4` 个候选一个）+ size 校验；restore 的逐字节哈希复核与回退全量 archive
**原样保留**（安全网不动，抽样只决定是否重拷）。测试：memo 命中（第二次同步 0 次源读）、mtime 变更 miss、
抽样捕获损坏（重拷）、manifest 损坏回退全量。

**附（乙B6）失败迭代快速失败**：job 侧 `cuda_compile_preflight`（FK_CUDA_PRECHECK=1，默认关）——
经 benchkit **自己的** builder（同 flags、同构建目录，成功时计时跑零成本复用）短超时预编译；构建失败 = 既有的
"无 eval JSON + 日志 tail 带编译错误" verdict，只提前不改语义不新增 verdict 类别；预检超时（rc=124）与
非 CUDA 树（rc=3 静默跳过）都走原路径。仅 perf|all|cardref|allncu 四个计时任务跑预检（probe 的 cmd 任务
不跑）。driver 侧 `_ast_preflight_warn`：python kernel 提交后 ast.parse，SyntaxError 打一条 WARNING
（**永不入 verdict/gate**），.cu 走 job 侧预检。

**测试**：改注释必命中/改头文件必 miss/跨树同 key、默认关=整树 key 逐字节、三路回退、缓存 roundtrip
跨树 HIT、篡改产物仍拒（C11 硬验收 4 条全真跑 bash）；B7 三条；B6 预检 off/失败/成功/超时/非计时任务
+ driver WARN 不入记录。

## 任务 2（P0）：probe 批量 manifest（乙B2+丙T3+甲F11）

**实现**：`parse.py` 增 `PROBE-MANIFEST:` 标记（显式单脚本 PROBE_REQUEST 优先； student 轮不新增提交面——
student 探针循环遇 manifest=True 轮仅回"not available to the student"反馈）。driver 侧
`_read_probe_manifest`（≤4 行、去注释空行、缺失/超限→带反馈拒绝不烧轮）、`_probe_batch_mode`（一条
`cmd:` job：逐脚本 begin/end 标记 + `python3 <script>`，**脚本间内联 `lb_clocks_reset` 等价物**（cmd 任务的
子 bash 不继承 gpu_entry 函数，重述 -rgc/-rmc 复位）+ `FK_PROBE_BATCH_SETTLE_S`（默认 3s）settles；
DNF 结构性隔离：`;` 顺序 + 子 bash 无 set -e）、`_split_probe_batch_log`（按标记切逐脚本日志片；任一
begin 缺失 → None）。接缝：meta 步探针分支（轮预算语义不变——一个 manifest = 一轮 = 一 job）与 KB 认证
循环（逐脚本 verdict，全 PASS 才 finalize）；**job 级失败回退逐脚本旧路径**（旧路径提取为
`_run_meta_verdict_probe` 原样保留，回退即旧路径本身）。kill switch `[loop] probe_manifest`（默认 true：
无人生成该标记 ⇒ 默认态零行为变化，回归钉 `test_t2_default_state_leaves_existing_probes_alone`）。

**真跑证据**（with-lock 下真 job log 原文）：`=== fk-probe-batch begin: notes/probe_a.py === /
PROBE PASS max_diff=0.001 / ... end ... rc=0 / === begin: notes/probe_b.py === / ... end ... rc=7` ——
probe_b DNF（exit 7）后批次继续、probe_a 的 PASS 独立成立：**单脚本 DNF 不连坐在真 job 上走通**。

**测试**：标记解析/优先级、batch 模式标记与隔离结构、日志切分与缺标记回退、manifest 边界（>4 拒绝/缺失拒绝）、
meta 4→1 断言、逐脚本独立 verdict、DNF 不连坐、整批失败回退 3 job、config 关、student 面零新增、
KB 单 job 全 PASS finalize + 失败反馈 + 回退。

## 任务 3（P1）：空闲 GPU 后台测量车道（乙B4，契约 C15）

**实现**：`BackgroundLane`（driver.py）+ `submit_gpu_start/finish`（cli.py——非阻塞提交/收集对；
fuse 的 `submit_gpu` **逐字节不动**，gate 路径不变）。学生探针循环即测量窗口：窗口开（到期任务非阻塞提交，
`submit_gpu_start`；后端无该对→窗口不动，同步路径永不入窗）→ agent turn → 窗口收（`submit_gpu_finish`
收集并交付）。C15 三硬约束**结构性**成立并有逐条测试：
1. **gate 永不消费后台读数**——车道读数只达任务自己的 `deliver`（confirm → `ctx.lane_confirm_samples`，
   即 run-stop 后 claim 的 seed 池，WP15 的 `run_best_confirmation(seed_samples=...)` 参数；测试断言车道读数
   前后 gate 字段逐字节不变）；
2. **cooldown guard**（默认 60s，按最后 GPU 释放时间）+ **温度 guard**（复用 eval JSON 已有的 gpu_state
   temperature_c 遥测，≥83°C 推迟启动不丢弃——不新增探测不新增 job）；
3. **默认关**——`[loop] background_lane` 默认 false；off 时无 lane 对象、无窗口、记录逐字段同基线。
`gpu_lock` 语义不变：车道 job 就是普通提交（direct 的锁）。中断回退：窗口未关（agent turn 异常）→ 下一窗口
`abandon` 陈旧 inflight（读数永不消费，车道自愈可用）；车道任何失败 drop 不重试不阻塞迭代。
v1 只允许 confirm/candidates 两类（其余 kind `ValueError`）；candidates 的真消费点随 WP18 落地（其 deliver
即挂点）。

## 任务 4（P1）：师生上下文有界化（甲G20+乙B5.1+N-B12+丙T13，契约 C13）

**学生侧**：`build_digest`（gen1 死代码）接线为 `_write_student_digest` —— 对既有记录的**确定性投影**
（零新增行类型：sol/target/achieved/flags 摘要行 + 逐决策行，尾部 `student_tail_lines`（默认 50）截窗）
落 `student/status_digest.jsonl`。`{STATUS_JSONL}`/`{TRAJECTORY_PATH}` 改绑**仅学生渲染**（kill switch
`[loop] student_digest_binding` 默认 false = 逐字节现状且不写文件；共享 bindings 字典复制不改——教师等
raw status.jsonl 读者零影响）。两态对比测试 + golden 钉住。

**教师侧**：NCU 视图行界压缩（`[loop] teacher_context_bound` 默认 false）：best 迭代整行保留（best-only）
+ 历史 tail `teacher_ncu_tail_lines`（默认 8）行——**按块裁不按字节裁**（行列表，无任何字节切），本包
未引入任何新字节截断（4k 事故纪律）。
**每加一块配一块削减清单**：学生 digest = 投影+截窗（削减 raw 全量读取）；教师 NCU 压缩 = 只减不增；
run-level 墙钟桶 = stdout 行（无记录字段）。**未做**（记录在案）：stall/ref ledger 超龄段折叠摘要——
既有 [:9000]/[:6000] 为 HEAD 字节切片（基线现状），折叠需动 stall-price 解析器（skills/ 围栏外），
本包只保证不新增字节切；bounce 轮不重发已读块——与 resume 耦合（Task 5 的合并弹回已天然只发新违规块），
随 resume 会话语义在 WP23 一并评估。

## 任务 5（P1）：教师诊断合并校验弹回（乙N-B11，契约 C12）

**实现**：诊断门合并为**一趟**——reachable/INFEASIBLE 首弹/INFEASIBLE-mention/locus-segment/
reopen-budget/reopen-price/R2-structural-price/phase-verdict 各自检查、各自旗标（一次弹回后接受/拒绝语义
逐门原样），违规块收进 `violations` 列表，**一次 re-prompt 报全部违规**（旧行为：每门一回合 teacher turn；
大批次实测 695 回合底数）。patch-shaping 门对 INFEASIBLE 停止请求照旧不适用（`cause != "INFEASIBLE"` 门控），
停止请求旗标用尽照旧 break 走后置拒绝；ENUM 门在循环外原样。弹回文本 = 原块重组（`"".join(violations)`），
**零新增注入内容**（文本逐句测试）。`meta_format_retry` 计费在 run_two_layer 层，未触碰。

## 任务 6（P1）：会话 resume + 结构化输出（乙B5.2/B5.3+丙T1/T2+甲G26）

**前置 CLI 能力验证（已运行，记录如下）**：`scripts/verify_claude_resume.sh`（机器 claude 2.1.316）：
```
CAPABILITY cli-present: YES
2.1.316 (Claude Code)
CAPABILITY flag--session-id: YES
CAPABILITY flag--resume: YES
CAPABILITY flag--print: YES
VERDICT: FLAGS-OK (live round-trip skipped; set FK_RESUME_VERIFY_LIVE=1 to spend it)
```
（旗标验证零成本通过；真两轮回程 `FK_RESUME_VERIFY_LIVE=1` 时按需花费，验证脚本即源码。）

**实现**：`agent-claude.sh` 增 `FK_AGENT_SESSION_ID/MODE`（create→`--session-id`，resume→`--resume`；
**默认不传旗标逐字节现状**）。`cli.py`：`resume_session`（`[loop]` 键经 `_wp19_apply_cfg`，默认 false）+
`agent_session_scope`；driver 在**同一 meta 条目**（diagnose+patch 共 `meta<N>` 作用域）与迭代
（`iter<N>`）设置作用域——首 turn create、后续 turn resume（按 attempt 现算 env：失败重试天然续会话）、
新作用域新会话。**失败重试改"续会话+错误 tail 注入"**：retry prompt 追加上一 attempt 的失败 tail
（仅 resume 开时）。**学生断网四层不弱化**：`agent_policy.install`（policy/settings 重写、sandbox、
tripwire、notice）与 skills mirror **每 turn 照旧**——resume 的会话重读的是每 turn 新重写的 settings
（测试钉住 install/notice 每轮调用 + notice 前缀）。**未做（记录在案）**：关键协议行 schema 约束——
改输出格式是全局协议变更（解析面全动），其收益（降 unparseable 整轮重跑）已由任务 5 的合并弹回部分兑现；
建议 WP23 按 CLI 能力再评估。

## 任务 7（P1）：meta 采集合并单 job + 换代读数级复用（丙T4/T11+乙B8）

**换代读数级复用（对账三态，全部落地）**：
- **carry 扩件**（`scripts/self_evolve.py`）：四件 → 七件——`probes/`（父代 notes/*.py 探针脚本）、
  `tools.md`（stall/timeline 工具 seed，明示子代 harness 自行 overlay、永不因此跳测量）、
  `ref_census.md`（父代 best 树的 SASS/NCU census，**机器头标注 reference_kernel_sha256**（best kernel
  内容 sha256）+ parent_best_sha/window/median_ms/ratios + 五段 census 正文，缺失段落记 absent 不猜）。
- **对账**（driver，`_ensure_ref_pricing` 挂点）：子代 reference = 父代 best 逐字拷贝 ⇒ census 头 sha 与
  子代 ref 树 kernel 内容 sha 逐字一致才可引用（失配 → 不引用，直接全套）。引用 = **一条 confirmation job
  对账**（mode `ref`，ref 树计时）：实测与 carry 读数差 ≤ `_REF_RECONCILE_BAND=0.03` → **采纳**（五件
  ref 侧 evidence 落盘 + marker 静默，census 全套不再跑；采纳要求 role+pm_timeline 双段齐全——不齐全视为
  不可采纳回全套）；失配 → **回退全套实测**；无 carry/sha 失配/任何异常 → **现状**（逐字节基线路径）。
  读数复用是加速不是省测量：confirmation job 就是那次重测。三态纯函数
  `reconcile_carried_ref_reading` + 解析/采纳/确认全部单测（采纳=恰好 1 个 ref job、refute=回退不采纳、
  无 carry=不买任何 job、sha 失配不引用）。

**meta 采集合并单 job（merge_meta_jobs）——实现状态如实记录**：盘点后确认**本项的合并主体已由既有
WP11 census 机制承载**：`_run_stall_pricing` 的一个 job 同产 stall ledger + segment account + dossier +
role（含 tensor 项）+ PM timeline（其 docstring 明言 "one job yields the stall ledger, segment account,
dossier, role table WITH tensor and the phase timeline"），且带 (status_path, ref, cmd) 键的内存+落盘
缓存与"只存完整读数"纪律。本包未引入名为 `merge_meta_jobs` 的新键：唯一还在单跑的 meta job 是
`_run_timeline`（notes/timeline_probe.py，独立工具），把它并进 census 命令需要改 census 运行器的命令面
（skills/ 围栏外）并把 timeline 的 banked 复用/stale overlay/dirty 重跑语义移植进合并段——按围栏与
"所有失败路径回退旧行为"纪律，本包不做，**留给 WP23 评估**（timeline 的 banked 复用本身已把实测开销
摊薄：大批次 1672/1887 命中 bank）。群卡侧读合并照任务书缓议（P2）。

## 任务 8（P2）：小项捆

- **乙B9 墙钟桶补全 + 常数校准建议**：run-level 桶（`wp11_teacher_recon_s/distill_s/lever_s` ——
  recon/distill/lever-design teacher turn 经 `_wp11_teacher_timed(bucket=...)`（iteration 结构外的 turn
  不入记录字段）；`wp11_job_pricing_s/timeline_s/confirm_s` —— 三类 job 的 submit+poll span 经
  `_wp11_job_timed`）；run 末汇总表加行 + **常数校准建议输出**（如 "one confirmation job ≈ Ns —
  confirm_budget_per_iter/meta_hours_per_round may want re-deriving" —— 建议打印，绝不自动改配置）。
  全部 stdout-only，无新记录字段、无 gate 输入（测试钉住 `teacher_s` 不受桶污染）。
- **乙N-B15**：floor_stop 出口补打与其它出口相同的墙钟构成表。
- **丙T6**：cutedsl prewarm 与构建缓存解耦——`fk_build_cache_prewarm` 去掉 FK_BUILD_CACHE 门（纯性能
  无关正确性的 warm-up，默认也做）；problem.json 路径修正为 eval/problem/（根兼容）。
- **乙B10**：ssh delivery lzma preset 6→3（tar 包上最慢档无收益、压缩 CPU 每 job 分钟级；对宿主透明，
  仍 `tar xJ`）。模块内一行 + 断言。
- **未做（记录在案）**：机器级价格/读数缓存（甲G18）——教师侧跨 run 缓存，触及价格表与 ARCH_PROBE 消费面
  （WP17/WP18 领地接口），缓议；skills 镜像 mtime 增量、submit `_forge` 进程内调用——落点
  `agent_policy.py`/`skills/__init__.py`（围栏外）与进程隔离语义变化，缓议；ssh lzma 已做（见上）。

---

## 冒烟（验收 §3.6）——本机 3060 真跑，with-lock 全程

`scripts/smoke_wp19.sh`（入册可复跑）：渲染 cardkit 模板成真卡树 → 真实 `forgekernel.submit` 在 GPU 0
跑真 job。**with-lock 记录**：`[with-lock] 已获得锁 gpu，开始执行: bash scripts/smoke_wp19.sh` /
`[with-lock] 命令结束（exit 0），锁 gpu 已释放`（另含逐 job 的 300–600s with-lock 包裹）。
真 job 证据（log 原文已粘贴于任务 1/2 节）：
1. **manifest 批量**：双脚本批次，probe_a `PROBE PASS max_diff=0.001`（rc=0）、probe_b rc=7 DNF，
   批次继续逐脚本 verdict —— DNF 不连坐真跑成立；
2. **窄化 key**：narrow=1 跨 commit 同 key `be0bb653…`（B/B' 行）、narrow=0 同树不同 commit key 移动
   `57c51b9c…`（C 行）、builder 缺失树静默回退 —— T5 假设由推断变真跑实证；
3. **CLI 能力**：claude 2.1.316 旗标 FLAGS-OK（任务 6 节）。

**WP11 墙钟构成表前后对比（如实标注）**：本机 venv 的 torch 为 CPU 版（2.15.0+cpu）且 repo 内唯一任务
rmsnorm 面向 sm100（本机 GPU sm86）——**端到端 loop 真跑在本机不可运行**，构成表的 A/B 需要可运行任务，
以真实测量在 WP23 通电步完成。本包交付的机制级真测量如下（真秒、真锁，`_wp11_wallclock_summary` 原样输出，
"前后"= 纯 WP11 桶 vs WP19 增桶）：
```
[loop] [wp11] wall-clock composition of the loop (measurement only — no verdict reads this; shares over the 1.3s of measured buckets):
[loop]   eval_job_s          1.3s  100.0%
[loop] [wp19] run-level wall-clock buckets (stdout only — no record reads these):
[loop]   job_pricing             1.2s
[loop]   job_confirm             1.3s
[loop]   calibration: one confirmation job ≈ 1.3s — confirm_budget_per_iter and meta_hours_per_round may want re-deriving from this table (suggestion only; the flip belongs to the integration step)
[loop] [wp19] measured wall clock including the run-level buckets: 3.7s
```
（job_pricing = 真实 lzma preset-3 压缩 42.4 MiB 的 span；job_confirm = with-lock 下一个真 cmd job 的
submit+poll 全程 1.26s——即单条确认 job 的真实固定开销，供任务 1/7 的节省估算校准。）

## 预期节省估算（全部标注：以墙钟遥测校准——上表的真秒 + 43 号报告口径，数值结论归 WP23）

| 项 | 估算依据 | 预期节省 |
|---|---|---|
| 任务 1 窄化 key | T5：构建占 lever-pricing job 时钟 ~40%（CHANGES-WP11:119），35–40s/次（CUTLASS），同内核每迭代 2–3 job 重复 | 每 run 数十 job × 35–40s 归零（默认关，翻转后生效） |
| 任务 2 manifest | T3：probe job 固定开销 0.5–2min，meta 步 1–4 个探针 job | 每 meta 步 (k−1)×固定开销；本机实测单 cmd job 固定开销 ≈1.3s（上表） |
| 任务 3 车道 | F20：agent turn 分钟级 × 每迭代 4–15 turn 全空闲 | confirm/CANDIDATES 墙钟从串行追加变隐藏（默认关） |
| 任务 4 有界化 | F21/G20：提示词无界增长、iter 80–120 时 20–40k token 死历史 | token/时长两位数百分比（迭代越深越大） |
| 任务 5 合并弹回 | 695 回合/批次实测底数，每门一回合 → 一回合 | 每多门违规步省 (k−1) 个 teacher turn |
| 任务 6 resume | T1：CLI 冷启动秒级 + 21–45KB 提示词重放 | 每 turn 冷启动+重放；teacher 长回合与截断重试受益最大 |
| 任务 7 对账 | T11：每代 ref 定价 1 job（census 1–3min）→ 1 条轻确认 job | 每代省一个 census job（本机确认 job 实测 ≈1.3s 固定开销 + ref 计时） |
| 任务 8 | B9/B15（可观测性）、T6（JIT 预热不再依赖缓存开关）、B10（ssh GB 级输入压缩 CPU 分钟级） | 遥测闭环 + 部署层 |

## 零行为变化清单（自查）

默认/未使用态逐项核对，硬验收 = golden（在基线 @19e885b 采集）+ 2332 条基线测试原样通过 + 2405 全绿：

1. **构建缓存窄化**：`FK_BUILD_CACHE_KEY_NARROW` 默认未设 → key 与旧路径逐字节一致（真跑+单测两证）；
   任何回退路径只换 key 来源，miss/失配/异常=全量构建不变；逐产物复核原样。
2. **inputs manifest**：memo 损坏/缺失=全量重算；抽样复核不改变"损坏永不入树"（restore 逐字节哈希 + 回退
   全量 archive 原样）；FK_INPUTS_SYNC=0 行为不变。
3. **B6 预检**：`FK_CUDA_PRECHECK` 默认关=无预检输出（真跑验证 off 态 job 形状不变）；driver WARN 不入记录。
4. **manifest**：`probe_manifest` 默认 true 但无人生成 PROBE-MANIFEST ⇒ 现有输入零变化（回归钉）；学生提交面
   零新增；回退=旧路径本身。
5. **车道**：`background_lane` 默认 false=无 lane 对象/无窗口（记录逐字段断言）；gate 字段无车道痕迹。
6. **digest**：`student_digest_binding` 默认 false=不写文件+绑定逐字节现状；教师渲染不受影响。
7. **教师 NCU 压缩**：`teacher_context_bound` 默认 false=全历史渲染（monkeypatch 计数断言）。
8. **合并弹回**：单违规回合的 re-prompt 文本 = 原块原文（逐句断言）；旗标/停止通道/ENUM 门语义原样；
   `meta_format_retry` 计费代码未触碰。
9. **resume**：`resume_session` 默认 false=不传旗标+retry prompt 无注入（逐字节断言）；四层断网与 mirror
   每 turn 重装未被弱化。
10. **对账**：无 carry=不买任何 job；sha 失配=不引用；refute=全套；采纳只走 confirmation job（一条），
    census 状态机 marker 语义不变。
11. **任务 8**：墙钟桶/建议=stdout-only（`teacher_s` 不受污染断言）；prewarm 解耦是 job 侧行为（任务书
    "默认也做"明令）；lzma preset 为传输透明变更。
12. **WP11/WP15/WP20 领地**：timing.py **零 diff**（审查项 B 的 FAIL 条件）；measure_paired、precision_gate、
    注入区（C10）、probe/SWEEP 分支语义（manifest 只加分支）、SPRT 领地、student_prompt.md 文案（digest 绑定
    是渲染层路径替换）全部未触碰。新键全部走 `_wp19_apply_cfg`（C17）：probe_manifest、
    student_digest_binding、student_tail_lines、teacher_context_bound、teacher_ncu_tail_lines、
    background_lane、resume_session（经 ctx.backend 设定）。

**围栏外改动清单（如实）**：`tests/test_wp8_prior_supply.py` 的 carry 四件上界集合放宽为七件（任务 7 的
合法扩展，测试自身未改断言语义）；`tests/conftest.py` 未动；`tests/test_wp11_throughput.py` 的
`Ctx` 桩补 6 个新桶字段默认 0.0（其测试语义未变）。
