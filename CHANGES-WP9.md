# CHANGES-WP9 — 测量锚定校准（NCU 协议对齐 + 解析地板 + 接线五修 + 新鲜输入复检）

基线：`zxtune-integration` main（= ForgeKernel @5030503 + WP1–WP6 集成，2222 tests 全绿）。
工作区 `wk2-wp9`，分支 `feat/wp9-measurement-anchor`。完成后 `pytest tests/ -q`：**2276 passed**（+54）。

---

## 任务 1（P0）：NCU 协议对齐

### 改动
全部 4 个 ncu 命令生成点加 `--cache-control none --clock-control none`（grep `profile-from-start` 找齐，逐处改）：

| 生成点 | 消费者 |
|---|---|
| `src/forgekernel/sealref.py`（ref_tree_gpu_entry 的 ncu 任务） | 密封源参考 recon 树 |
| `src/forgekernel/binref.py`（同函数的 ncu + binref 两处） | 密封二进制参考 recon 树 |
| `src/forgekernel/leaderboard.py::GPU_ENTRY_TEMPLATE`（ncu 任务） | 每次迭代的 NCU job + 全部 board 卡 |

`taskcard.write_card` 对生成文本的 `--launch-count 24` 后处理不受影响（该字面量保留，测试钉住）。

### 锁频协同（核查结论）
gpu_entry.sh 的时钟状态在 TASK case **之前**建立：先清残留锁（`-rgc/-rmc`），按卡需要锁频（SOL 卡锁 SM 1500/mem 3996 并读回校验，`LB_CLOCK_STATE` 导出），EXIT trap 清理。计时任务在该时钟状态下运行；`--clock-control none` 让 NCU 不再把板卡锁改回 base——**两者协同后，NCU 读数与计时读数出自同一时钟状态**。FlashInfer-Bench 卡（不锁频）下 `none` 使 NCU 保持在自由时钟，与该卡计时的自由时钟同样一致。旧默认 `--clock-control base` 会把 SOL 锁频卡的 NCU 读数搬到 base 频率，SOL 分母变形。

### ⚠️ 指标语义变化（总览红线 6 记账）
**历史记录与新记录的 NCU regime 不同**：
- 旧记录（≤WP8 集成）：flush-L2 + base 频率（NCU 默认 `--cache-control all --clock-control base`）。
- 新记录（本包起）：warm-L2 + 板卡实际时钟状态（`none`/`none`）。

方向性后果：`lts__t_sector_hit_rate.pct`（l2_hit_rate_pct）新记录显著高于旧记录——attention 类 K/V 跨 CTA 复用发生在 L2，flush 版系统性读低（teacher 拿着假低读数会去追已存在的复用）；`compute_sol_pct`/SOL 头在 SOL 锁频卡上分母从 base 变为板卡锁频。**跨代比较（如 Table 1 自进化链 vs 新一代）必须注明采集 regime**；归档/摘要工具今后读混合 regime 的记录时应当作两种测量条件。

### 论证：这是指标语义变化，不是 gate 协议变化
gate 读数（target/plateau/verdict）全部来自 **timing**（measure_paired A-B-B-A，timing.py 一字未动），不来自 NCU。NCU 是诊断输入（教师侧 evidence）。本包不触碰任何 gate 算式与常数；INFEASIBLE 门、precision gate、correctness 协议零改动。因此本改动不触红线 1/5，属红线 6 的记账事项。

### 解析侧健壮性（核查）
`inner/ncu.py` 的解析按**指标名**读 CSV（`KEY_METRICS` 名字表 + `_to_number` 值解析），对值域无任何 regime 假设（不钳位、不按 base 归一）；两个 flag 不改变 CSV 列名与格式 → 解析对 flush 与 warm 两 regime 均健壮，零改动。`skills/ncu-trace/SKILL.md` 已加 regime 注记（历史/新记录不可混读）。

### A/B 验证（scripts/ncu_flag_ab.py）
脚本对一张卡的 `eval/ncu_run.py` 两套 flag 各 profile `--repeats` 遍，比对 l2_hit_rate_pct / compute_sol_pct / time，方向性判定 + 方差守卫（warm 方差 > 2× flush 方差 ⇒ 建议保留 `--clock-control none`、回退 `--cache-control none` 并记账）。dry-run 模式供无 GPU 环境冒烟（有测试）。

**本机（RTX 3060 Laptop 6GB）无法跑**，原因：venv 的 torch 为 CPU-only 构建（2.15.0+cpu，`torch.cuda.is_available()=False`），`eval/ncu_run.py` 无法启动 GPU kernel；且仓库不含已生成的 operator 卡片（脚本输入）。ncu 2026.2.1 在位，board 侧直接：
`python scripts/ncu_flag_ab.py operators/<op> --run --launch-count 6`。

---

## 任务 2（P0）：解析地板求解器

### 模块：`src/forgekernel/loop/analytic_floor.py`（C7 单一实现，消费方直接导入）
纯函数 `floor_for(problem) -> {"bytes_min", "flops", "floor_ms", "floor_ms_band", "family", "assumptions"}`。

- **bytes** = Σ(输入+输出张量字节数)，每个操作数只计一次（tile 重读只增不减，故为可证下界）；dtype 尺寸表覆盖 fp64/fp32/tf32/fp16/bf16/fp8(e4m3,e5m2)/fp4/int8-64/bool；**未知 dtype 或不可解析 shape 的张量计 0 并写进 assumptions**（跳过项只会缩小下界，总 bytes 保持可证）；变长/稀疏 shape 退化为纯带宽 roofline。
- **flops 公式表**（只做可证的部分，保守取向）：

| 族 | flops | bytes 下界 | 来源 |
|---|---|---|---|
| gemm | `2mnk`（每个 MAC 一次） | A+B+C 各一次 | GEMM 定义 |
| attention | `2·BH·L²·d`（causal，mask 半边）或 `4·BH·L²·d`（非 causal）；**causality 未声明时取半（可证方向）**；投影/softmax 不计 | Q/K/V/O 各一次（KV 重读保守下界 = 1×） | score+PV 两个矩阵乘 |
| conv | `2·N·H_out·W_out·C_out·C_in·k_h·k_w(·k_d)`（im2col MAC 计数） | in+W+out 各一次 | im2col 形式化 |
| norm / elementwise | None（字节流主导，不计 flops 保持可证） | x(+w)+y | — |
| reduction | 元素数（树形下界 n−1≈n，单输入可数时） | in+out | 树形归约 |
| 未知族 | **None（比算错好）** | 照算（通用 roofline 仍成立） | — |

- **峰值表**（`_PEAKS`，dense/无稀疏化的厂商 datasheet 数字 = paper-peak 约定；真实 kernel 不可能超过，故 floor 可证）：

| arch | 带宽 | FP16 TC dense | FP8 | 来源 |
|---|---|---|---|---|
| sm90（H100 SXM） | 3350 GB/s | 989.4 TFLOPS | 1978.9 | NVIDIA H100 datasheet |
| sm100（B200） | 8000 GB/s | 2250 TFLOPS | 4500 | NVIDIA Blackwell datasheet |
| sm80（A100 SXM） | 2039 GB/s | 312 | 624 | NVIDIA A100 datasheet |
| consumer（RTX 3060） | 360 GB/s | 25.3 TFLOPS（fp16-acc） | — | GeForce Ampere（仅方向参考，不进回测） |

flops 类按**操作数字节最大的 dtype 类**选（fp8 GEMM 的 fp32 scale、attention bwd 的 fp32 lse 不得把峰值拽到 fp32——测试钉住）；未知 arch → bytes/flops 照算、floor_ms=None 并写明。
- `floor_ms = max(bytes/带宽, flops/峰值)`；`floor_ms_band = (可证地板, 可实现地板)`，上界用 `_ACHIEVABLE`（带宽 0.80 / 算力 0.80，STREAM/cuBLAS 同硅实测档）——band 上界以内 = "贴地板"。
- `problem_from_card`：从卡的 `eval/leaderboard/problem.json`（definition inputs/outputs + workload axes）构造 problem；shape 表达式支持整数/轴名/算术（SOL 的 shapes.py 语法子集，AST 白名单求值）。

### 回测：`scripts/backtest_floors.py` + `scripts/floor_backtest.json`
- manifest 覆盖 README 的 24 个已发布 kernel + 论文后 DSV4.1 kernel，共 25 行；problem 定义逐个从 kernel.py 冻结契约转录（contract 里没写 campaign 轴的条目如实标 `axes: null`，**不编造**）。
- **数据诚实声明**：仓库对 24 个 kernel 只记录**相对加速比**，无任何绝对毫秒——`best_ms` 全部为 null，ratio 列如实显示 no-data；**门（任一 ratio < 1.0 ⇒ exit 1）对每个带实测成绩的条目自动生效**，board/论文归档的绝对成绩填入 `best_ms` 即触发全表 gate。本机 3060 不跑真数（无可比成绩，任务书同款约束）；门自身的算术由 `--selftest` 证明（synthetic 条目 1.5×/1.1× 过，故意 <1 的条目被 exit 1 抓住——pytest 钉住）。
- 回测输出（本仓库可算部分）：

```
kernel                                               arch     floor_ms   best_ms   ratio  band
----------------------------------------------------------------------------------------------------
attention/attn_fwd_gqa16/h100/gen0                   sm90       0.5556         — no-data  fill best_ms from the run archive
attention/attn_bwd_mha/h100                          sm90       0.5556         — no-data  fill best_ms from the run archive
attention/attn_bwd_mha/b200                          sm100      0.2443         — no-data  fill best_ms from the run archive
attention/gqa_paged_decode/h100                      sm90       2.4073         — no-data  fill best_ms from the run archive
gemm/gemm_fp16/h100                                  sm90       1.1113         — no-data  fill best_ms from the run archive
gemm/fp8_gemm/h100                                   sm90       0.0695         — no-data  fill best_ms from the run archive
gemm/fp4_grouped_gemm/b200                           sm100     13.7439         — no-data  fill best_ms from the run archive
conv/conv3d_fwd/h100                                 sm90       0.0224         — no-data  fill best_ms from the run archive
（…全部 25 行；3 个 DSA 条目 + dsv41 契约未记轴 → "contract on file does not pin the axes"）
0 ratio(s) computed, 0 below 1.0
```

附注：attention flops 公式计价 fwd score+AV 链；bwd kernel 实际功 ≈2.5×，floor 刻意低估（比值偏高但方向永不 <1，manifest note 写明）。H100 若干成绩在 SM 1590 MHz 锁频下测得（README），paper-peak 在 boost 下定义——只会放宽健康带。

### 三处消费（全部 advisory/opt-in，教师侧；INFEASIBLE 机器门零改动）
1. **NCU ground truth 小节尾两行**（driver.py `_floor_teacher_line`/`_mfu_teacher_line`，渲染进 `diagnose_extra` 的 ncu_view 尾）：`floor=<x.xx ms> (bytes=<n> flops=<n|n/a>, at-floor band ≤ …, family=…)` 与 `MFU=<x>% (headroom <y>% — time if MFU reached 100%)`。floor 从卡的 problem.json 惰性读出并缓存（`ctx.floor_ms`）；MFU 取 sitting best KEEP 的 mfu。**学生可见 digest 零改动**（`build_digest` 无这两行——测试双侧断言）。
2. **diagnose_prompt.md §1c 对账段**（5 行，C6 限额内）：analytic floor 是 §1c 的机器分母，teacher 的 `floor_opt` 在其上构造（unhideable 项做加法，不替换）；差异超 ×1.5/×0.8 须在 evidence 解释，未解释的差距降 confidence，BLOCKED 压在不一致 floor 上视为未证明；`floor=n/a` 时回落手估（原路径）。
3. **opt-in 配置**：
   - `[target] floor_margin = 0.30`：无实测参考目标的 run，target = floor×(1+margin)（`_resolve_target` 尾部小改；实测 ref target 恒优先）。task.py：`_KEYS["target"] += {"floor_margin"}` + `_read_target` 解析 + `Target.floor_margin` 字段（C7 同步扩词汇表）。
   - `[loop] floor_stop`（默认 false）+ `[loop] floor_stop_margin`（默认 0.10）：best ≤ floor×(1+margin) **且** 结构通道已探（本基线的机器信号 = WP5 的 verified-BLOCKED 诊断 `ctx.wp5_reach_blocked`；STAGNATION 是 WP10，本基线无此通道——CHANGES 如实记录）⇒ 允许以 stop="floor" 关闭（`_floor_stop_check`，可单测）。band 在而结构未探 ⇒ 不停并记日志。

### C7 配置词汇表选择（写明）
- task.toml：扩 `_KEYS["target"]`（如上）。
- loop.toml 新键走 **WP1 先例：LoopContext 默认区（driver.py `# --- WP9 ---` 键块）+ cli.py 的 LoopContext(...) 三行 cfg 读取**。global_config/loop.toml **未追加**键块：其顶层表词汇表同样封闭（`_unknown_keys` 拒绝未知表——实测 `[loop]` 表直接 TaskError，test_launch 抓住），历史上它只携带 [target]/[budget]，运行级 loop.toml 由 cli 直读不过该校验。
- **围栏声明**：cli.py 的 3 行（`target_floor_margin`/`floor_stop`/`floor_stop_margin` 的 cfg 读取）超出围栏字面清单，是 LoopContext 默认区路线的必需接线（WP1 键块同形、同位置）；除此之外 cli.py 零改动。

---

## 任务 3：接线五修（各一个单测）

- **C1**：`taskcard.timeline_probe_auto(task)`——description/name/shape 命中 latency/decode/single-call（下划线连字符视作分隔，`mla_paged_decode` 命中）⇒ `write_loop_toml`/`write_board_loop_toml` 写 `timeline_probe = "auto"`；其余卡片**一个字节都不多写**（历史默认 off 不变）；ncu=False 仍写显式 "off"。driver 的 auto 语义原样（milestone 声明 `bottleneck: latency` 才开）。延迟型 kernel 的时间轴证据不再整段缺失。
- **C2**：`IterationRecord.ref_host_enqueue_ms: float | None`（trajectory.py），KEEP 构造处从 `ev` 填入（REVERT 行与其他 ref_* 字段同规则不携带）；digest 不展示。parse.py 的解析本就存在（WP2 schema）。
- **C3**：MFU→headroom 行（见任务 2 消费 1，与 floor 行同处，教师侧，digest 无行）。
- **C4**：`ctx.candidate` 接线——`_run_arch_probes` 的 decisive winner（freeze 本就跟随它，distill_blackbox_prompt 自己的规则）写入 `ctx.candidate = Path(win).stem` 并记日志；无探针 run 保持 None（`{CANDIDATE}` 渲染历史 fallback 文本，行为不变）。核查记录：全库此前无人赋值，变量语义未死（distill prompt :284 消费），故接线而非删除。
- **C5**：`_reanchor_memo(ctx)` 访问器——进程内存 memo 为空时**首次咨询**读 `<op_dir>/reanchor.jsonl` 重建（confirmed_gap 且 best_ms/median_ratio/equivalence_ratio 齐全的最后一条；note 从记录数字紧凑重建，决策内容等价）；`paired_residual_memo` 同源重建。三处 consult 点（false-win replay、phantom-gap、confirmed-gap）换线。**跨机器不旅行**：记录 cluster ≠ 当前 active cluster 的记录不重建（paired_memo 自己的规则，重启后仍成立——现有 test_loop_driver 用例钉住）。同 gap 不再重复买 ref job（测试断言 0 个 GPU job）。

---

## 任务 4：timing 后新鲜输入复检（作弊保险）

### 核查结论（先核查后动手）
对 `card_template/eval/leaderboard.py`（生成物）逐行确认：
1. **共用实例**：三条路径全部 `verdict, sets = _correctness(...)` → `measure(problem, timed_sets/sets, ...)`（单 workload `timed_sets = sets` 显式别名；group/task 路径同样把 judge 建的 sets 原组传进 timing）——correctness 与 timing 操作同一批张量对象。
2. **timing 后无复检**：单 workload / `_run_group` / `_run_task` 三条路径的 timing 完成后直接 `_emit(result)` 返回；期间仅有的检查是 thread-injection（timing 后线程计数）与 timer/clock 警告——都不是对输出的新鲜复检。
3. 现有防线全在测侧/静态：`kernel_fence`（文件扫描）、`_lazy_output`（DPS 迟发输出）、prep snapshot（`mutated_saved`，防输入变异）、`host_bound_ratio`（host 假象）、thread injection——**agent 侧无 timing 后复检**。
4. `measure_paired`（timing.py）A-B-B-A 与窗口协议零改动；复检只在全部窗口结束后的 `_emit` 前运行（契约 C5）。

### 两层设计（防误报为硬要求）
- **层 1（默认，mode=process）**：全部窗口结束后，父 bench spawn **全新进程**跑 `eval/bench.py --mode correctness --fresh-recheck`——新解释器、新 CUDA context、全新分配：指针全变、模块状态清零。子进程跑卡片自己的 correctness 协议，输出一行 `FRESH-RECHECK-VERDICT {...}`，不递归、不 timing、不 emit JSON。
- **层 2（有输入生成器的卡，mode=regen）**：同卡片、换 seed 重画同 shape/dtype 输入再跑 correctness。SOL：`sol_correctness` 读 `FK_FRESH_SEED`（父进程层 2 设 SOL_SEED+7919）；task 套件的生成器组在复检子进程内加 seed shift（`_recheck_seed_shift`，env 只在子进程存在，父进程恒 0）；FIB 协议本就 unseeded，层 1 已含新内容（mode 如实记 regen）。两层用任务自己的生成代码，**任务不变量按构造不破坏**。
- 语义判定（防误报）：复检子进程首调即"新键"——若 kernel 的 miss 路径诚实重算则照常通过（诚实 kernel 全层通过，测试钉住）；被抓的是"correctness 调用算好**侧缓冲**、其后按指针/内容键返回缓存、缓冲外输入返回错值"的形态——进程内 eval 永不踩它的 miss 路径（correctness=计算调用、timing=同一批张量的命中），复检进程必踩。文件落盘的侧缓冲跨进程存活，正是"指针全变、模块状态清零"要击破的承载。
- 结果 `fresh_recheck: {mode, pass: true|false|null, layers: [...]}` 进 **eval JSON 顶层**；子进程失败/超时 ⇒ pass=null（如实记 no reading，绝不冤判）；`--impl ref` 自测不复检；correctness FAIL 不进 timing 也就无复检。
- fail ⇒ driver 硬 REVERT（error 载明 fresh_recheck FAIL 与层细节）+ **record 标 `suspicious=true`**（trajectory 新列，objective 轨迹纪律：学生自产工作的客观测量，非教师注入；digest 不加行）。`parse.py` 仅动此处（EvalResult.fresh_recheck 解析）。
- eval 代码自包含：`_fresh_child` 只以子进程方式调本卡的 bench.py，PYTHONPATH 自补 workspace+eval，不从 loop 包 import（验证：card fixture 无 forgekernel 依赖，测试里子进程在纯卡片目录跑通）。

### fixture（验收 4）
`tests/test_fresh_recheck.py`：
- 层 1：parked-buffer 作弊 kernel（correctness 落盘侧缓冲、缓冲外零返回）在复检进程 FAIL；honest kernel 全层 PASS。
- 层 2：内容键变体——缓冲覆盖的内容复检通过、regen shift 后必 FAIL（对照卡证明抓它的是内容移动）。
- 子进程协议：恰一行 verdict、无嵌套、无 JSON。
- parse：顶层 fresh_recheck 进 EvalResult。
- driver：fail ⇒ REVERT + `suspicious=True` + error 载因（scripted backend，0 额外 job）。

---

## 测试与验证
- 基线 2222 → **2276 passed, 0 failed**（`~/venvs/forgekernel/bin/python -m pytest tests/ -q`）。
- 新增：`test_ncu_flags.py`(6)、`test_analytic_floor.py`(25)、`test_wp9_floor_wiring.py`(9)、`test_wp9_wiring_fixes.py`(9)、`test_fresh_recheck.py`(5)。
- 验收 3 端到端证据：`test_wp9_floor_wiring.py::test_floor_and_mfu_lines_render_teacher_side` + `::test_student_digest_carries_neither_line`（诊断提示词 NCU 段尾有 floor 与 headroom 行、学生 digest 无——两断言都在）；`test_wp9_wiring_fixes.py::test_reanchor_memo_rebuilds_…` + `::test_rebuilt_memo_prevents_the_rebuy`（重启重建、不再买 job）。
- 验收 1 的逐族单测：gemm/attention/norm/conv/reduction/退化路径（未知 dtype、不可解析表达式、未知 arch、n/a 渲染）全在 `test_analytic_floor.py`。

## 文件清单（C7 新增 + 改动）
新增：`src/forgekernel/loop/analytic_floor.py`、`scripts/backtest_floors.py`、`scripts/floor_backtest.json`、`scripts/ncu_flag_ab.py`、5 个测试文件。
改动：`sealref.py`、`binref.py`、`leaderboard.py`（NCU 模板 flags + BENCH/child 相关仅本包条目）、`taskcard.py`（C1）、`inner/trajectory.py`（ref_host_enqueue_ms + suspicious 两列）、`inner/ncu.py`（零改动，核查记录）、`loop/driver.py`（NCU 段尾两行 + floor 缓存 + `_floor_stop_check` + `_resolve_target` floor_margin + C4/C5 + fresh_recheck REVERT + LoopContext WP9 键块）、`loop/parse.py`（仅 fresh_recheck 解析）、`loop/cli.py`（3 行 LoopContext 接线，围栏声明如上）、`task.py`（`_KEYS["target"]` + `_read_target` + Target.floor_margin）、`meta/diagnose_prompt.md`（仅 §1c 对账段）、`skills/ncu-trace/SKILL.md`（regime 注记一段）、`card_template/eval/leaderboard.py`（fresh_recheck 机制 + 子进程模式 + 三处 hook + SOL seed 覆盖）。
**未触碰**：`card_template/eval/timing.py`（契约 C5）、driver 注入区其他小节（C3）、`precision_gate.py`、`student_prompt.md`、`kb_lint.py`、`meta/distill*`/`recon` 提示词（WP8 领地）、`global_config/loop.toml`（表词汇表封闭，见上）。

## 遗留给 board
1. `ncu_flag_ab.py` 方向性确认（3060 无 CUDA torch，无法在本机跑）；若 warm 方差恶化超 2×，按脚本结论回退 `--cache-control none` 并在此记账。
2. `floor_backtest.json` 填入 board/论文归档的绝对 best_ms（24+1 条），全表 ratio ≥1.0 门自动生效；健康带 [1.0, ~2.5]，超带条目复核公式或标注（attention bwd 的比值预期偏高，公式刻意低估）。
