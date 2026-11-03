# CHANGES-WP10 — 结构搜索深化（架构逃逸触发器 + flags 决策轴 + MC-DELIVERY）

分支 `feat/wp10-search-depth`（基于 kernel-ZXtune main @ `c1971f1`，WP1–WP6 集成树）。

| commit | 内容 |
|---|---|
| 本提交 | 任务 1–3：STAGNATION 触发器、flags 决策轴、MC-DELIVERY（机制 + 提示词供给 + 测试） |
| （紧随） | `CHANGES-WP10.md` |

验收证据：`~/venvs/forgekernel/bin/python -m pytest tests/ -q` → **2242 passed, 0 failed**（分支顶端；基线 main = 2222 passed，新增 22 个 WP10 测试，零回归）。本机 GPU（RTX 3060 6GB）不代表目标硬件，验收以 pytest + fixtures + dry-run 为准。dry-run 证据：`test_stagnation_injection_is_minimal_block_or_nothing`（注入区出现 STAGNATION 块且含 override 句、组装 diff 恰好等于块本身）与 `test_mc_prescreen_script_runs_end_to_end_against_a_stubbed_timing_kernel`（driver 生成的预筛脚本真实执行：冒烟→交错计时→MC-WINNER 行→树还原）。后者在验收过程中抓到并修掉一个真 bug（脚本常量作用域缺失，见 §3 末）。

---

## 1. 任务 1：架构逃逸触发器（STAGNATION）

**问题**：参数级局部最优没有机器判据（matvec 案例：启发式卡 2.09ms，须换骨架才到 1.95ms）。WP5 的停滞治理只读 `floor_opt`（目标侧估计），`detect_plateau` 只负责把内层循环交给 meta——没有任何机制告诉 teacher"最近 K 次参数级尝试在测量分辨率内全部读平、而真差距仍在"。

**聚合逻辑**（`driver._stagnation_attempts` / `_stagnation_report`，全部新代码在 `# --- WP10 ---` 标记段）：

- 取**参数级尝试序列**，含两类：
  1. **内层交付**（inner）：每条被评估的 decision 行（KEEP，或 correctness=True 的性能 REVERT——被判定并输掉的尝试，增益记 0）。排除账本行：gate 行、REOPEN/REOPEN-REVERT、`part k of n` 伴随组步骤（联合判定，不单独计）、`reading_of` 复用读（不是新尝试）。SWEEP 赢家与 lever bundle 落地本就以普通 decision 行落地，天然在内（SWEEP 行由 milestone 侧 `sweep:` 配方可识别，但识别只影响将来的标注、不影响计数）。
  2. **additive meta 补丁的闭环结果**：KEEP_PATCH（增益 = best_before→best_after）与 REVERT_PATCH（增益 0）。structural/split 补丁**不计入**——它们就是本触发器要逼出的动作。
- 每次尝试的**实际增益**：主源 = WP4 已合入的 per-decision 账本字段 `regression_vs_best_pct`（KEPT 但慢于当时的 best，增益取负）；**备源**（账本未写的行：带内 KEEP、新 best、字段存在前的行）= 该行自身 `metric_before` 对 gate 读数的算术（`(metric_before − ms)/metric_before`）。CHANGES 注明：主源 WP4 账本，备源迭代记录等价算术。
- meta 结果的**时钟对齐**（无共享时钟的尽力合并）：meta outcome 发生在其内层段结束之后，取 `metric_before == 该记录 best_before` 的最新内层行 iter +0.5；无匹配则放最新端（outcome 在段后才判定，放晚侧是诚实方向）。
- **判定**：最近 `stagnation_k` 条尝试全部 `|gain| < resolvable_diff`（C1：WP1 的 `parse.resolvable_diff` 直接导入，floor/cap 取 ctx 配置）**且**当前 `gap_pct > resolvable_diff`（best 对 `_resolve_target` 的差距——还有可测的真差距才谈逃逸）。任一不满足、尝试不足 K、best/target 缺失、或 `stagnation_k = 0` → 不触发。
- **注入**：`run_meta_step` 组装区，紧随 WP4 REMOVE_CANDIDATES（契约 C3 渲染顺序：…→ WP4 REMOVE_CANDIDATES → **WP10 STAGNATION** → WP12 KB-MENU）。块 ≤10 行（默认 K=3 时 7 行）：逐条列出尝试与增益、band、gap、约束句（cause_class 须为结构类 R1 带价 / R2-structural 带价 / ARCH_PROBE_REQUEST / survey 请求；坚持参数级须写 `stagnation_override: <理由>` 并在 confidence 说明为何本次不同）、历史 override 次数。
- **override 通道**（防逼出假结构动作）：诊断文本中的 `stagnation_override:` 行仅在**本轮注入了块**时被认（`_parse_stagnation_override`）；认下后**约束解除靠文本本身**——触发器只能加压、不机器改判 cause_class、不跳过教师诊断，因此无机器门可"解除"，override 的意义在于**记账**：`[stagnation_override]` 标记追加进该步的 `patch_direction`，随 MetaRecord 落 meta_status.jsonl（重启存活），后续 `reflect_summary` 与下一个 STAGNATION 块的 "Past stagnation overrides: N" 可见。不新增 MetaRecord 字段（schema 零改动）。
- **红线的落实**：块只进教师读集（diagnose_extra），学生不可见；不碰 `_parse_cause_class`、不加 bounce。

**单测三路径**：`test_stagnation_triggers_on_k_flat_attempts_with_a_real_gap`（触发）、`test_stagnation_silent_when_an_attempt_clears_the_band` / `..._without_a_real_gap` / `..._needs_k_attempts_and_honors_the_off_switch`（不触发三情形）、`test_stagnation_block_injected_into_the_diagnose_prompt_and_override_recorded`（override 记账 + 下轮计数）。

### 逃逸触发器与 floor 判据的关系

floor_stop 是 **WP9** 的 opt-in 门（解析地板给出 `floor_opt` 的机器分母 → INFEASIBLE/目标判定的硬门）；本触发器是**行为判据**（最近 K 次参数级尝试的实测增益序列 → 诊断提示词加压）。两者**独立、集成时互不依赖**：本包不导入 WP9 的 `analytic_floor.py`，不读它的配置键与 digest 字段；WP9 不消费 STAGNATION。可同时成立而不冲突——floor 判据回答"这个架构还能不能到"，行为判据回答"参数级行走还在不在付钱"；一个 BLOCKED 且行为停滞的 run 会同时看到两个机制各自的输出，互不引用对方的代码、配置或数据。

## 2. 任务 2：flags 决策轴（纯供给，零学生改动、零机制改动）

白名单已存在于 `card_template/eval/leaderboard.py:428`（`CUDA_FLAG_OK`）与 `src/forgekernel/leaderboard.py` 镜像注释；学生本就写 `// fk-cuda-cflags:` 行（`meta/cuda/student_prompt.md:50`）。本任务只把 flags 当**可定价决策轴**教给 teacher：

- `meta/diagnose_prompt.md` Hard rules 加 **"Flags are a lever axis"** 段（8 行）：flags lever = 对 `// fk-cuda-cflags:` 行的文本 edit 对（现有 lever/SWEEP 基建承载，无新请求类型）；列白名单族 `--maxrregcount=N` / `--ftz` / `--fmad` / `--prec-div` / `--prec-sqrt` / `--extra-device-vectorization` / `-O[0-3]` / `-lineinfo`；`--use_fast_math` 允许列出但注明 **numerics 由 precision gate 仲裁、风险自担**。
- **联动提示**（读数顺序段 1b 加一句）：`regs_per_thread ≥ 0.9×` 架构寄存器档位（sm100：128/168/255）时**先定价 `--maxrregcount` 档位扫描**再考虑结构判读——spill 悬崖停在档位下沿的 kernel 是 flags 决策而非结构决策。
- `skills/sweep-decision/SKILL.md` 加 **"Flags are sweepable parameters (WP10)"** 配方示例：`maxrregcount ∈ {128, 168, 224, 255}` 梯度（编辑 cflags 行的 exact-text 对）+ 判读（spill 清零且超 floor = 赢家；spill 清零但读平 = 占用受限，先看 `waves_per_sm` 档位跃迁再重扫）。
- **单测**（`test_wp10_search_depth.py`）：加载卡侧 `eval/leaderboard.py`（torch 缺席时用 stand-in，同 cuda-student 测试的装载方式），断言 WP10 教学的全部示例 flag 行被 `cuda_kernel_flags` 接受（含梯度全集）、非白名单（`-I/-l/-L/-Xlinker`、伪拼写 `--use-fast-math=true`）被拒、`--use_fast_math` 可用且 diagnose 文案含 precision gate 仲裁句。**card_template 零改动**。

## 3. 任务 3：MC-DELIVERY（学生结构 fan-out，预筛 + 冒烟）

**语义链**（全部 `# --- WP10 ---` 标记段）：

1. **授权**：teacher 在 milestone 某决策行写 `multi_candidate: 2|3`（`_milestone_multi_candidate` 读 milestone 文件本体，与 sweep recipe 同通道同形状）；driver 侧钳到 `[loop] multi_candidate_max`（默认 3；0 = 通道关闭）。gate 行/REOPEN 行永不 fan-out。`meta/patch_prompt.md` 加授权句（6 行）：授权字段、变体命名（kernel 文件同目录同 stem：`<stem>_v0/_v1/_v2.<ext>`）、**变体间须是结构差异（循环组织/stage 划分/角色划分），不只是参数值——参数差异走 SWEEP**、字段不携带代码（L0 不变）。`meta/student_prompt.md` 追加**一句**（C6 允许的唯一一句）：授权决策的多变体交付说明。`multi_candidate` 字段只在 patch_prompt 声明，**meta_milestone.md（POLICY）一字未动**；实测带该字段的 milestone 过 `validate_milestone` 无违规。
2. **变体交付**：学生一次标准 commit 交付多个**自己写的**同接口实现（记录块照旧）；无 L0 泄露面——变体是学生树内文件，driver 从不向学生树写任何实现文本。
3. **预筛**（`_mc_deliver`，授权且 ≥2 个变体文件存在才走）：driver 生成 `notes/mc_prescreen.py`（源码常量 `_MC_PRESCREEN_SRC` 在 driver 内，随 `_mc_seed_toolkit` 落盘并 commit，连同 vendored `lever_pricing_run`——**与 sweep_run 同一复用手法：只复用其交错计时内核/节点卫生/噪声读法，SWEEP 分支语义零改动，契约 C4**）+ 提交时写入 spec（`notes/__mc_<did>.json`：kernel_file、变体表、rounds=2、代表 workload scope），一个 job（`cmd:python3 notes/mc_prescreen.py …`，`-i<iter>mc` 分支，结算重买语义与 sweep 相同）。job 内：**每变体先一次 correctness 冒烟**（单 workload 快速调用）——冒烟不过记 DNF、不进计时；幸存者与 base 交错计时；`MC/MC-WINNER/MC-OK/MC-FAIL` 行进日志；树原样还原。
4. **赢家交付**：时延最优且冒烟通过者由 driver **应用**（写入 kernel 文件、全部变体文件移出交付树：非赢家归档 `notes/variants/<iter>/`，赢家内容已在 kernel 文件）、一次 driver 记账 commit，随后走**标准 eval job 全量 gate**（correctness + precision + measure_paired 原样，NCU 只测赢家——天然成立，NCU 跑在赢家树上）；其余路径不碰。赢家文件不可读等意外按无赢家处理。
5. **失败语义**：变体级 DNF 不影响其余；**全部失败 → 普通 REVERT**（不买 eval，workspace 复位，REVERT 行 error 注明 multi-candidate，表照记）；预筛 job 本身失败（BackendError）→ 不阻塞交付，按原树普通 eval 继续（同 pricing"永不阻塞一步"纪律）。
6. **记录**：`IterationRecord.variant_timings: list | None = None`（`inner/trajectory.py` 仅追加此一个可选字段，与 WP9/WP11 的字段追加互不冲突；load 侧 `_FIELDS` 白名单自动容忍旧文件）。行 schema：`{variant, smoke, ms, ratio, delta_vs_base_pct, winner}`——纯客观测量，学生自产工作的客观记录（红线 3 明文不算注入）。

**防造假核查表（CHANGES §要求，逐条）**：

| # | 项 | 落实 |
|---|---|---|
| ① | 变体是学生自写——无 L0 泄露面 | driver 全链路只读不写变体内容；授权句明文"字段不携带代码"；L0/断网/kb_lint 未触碰 |
| ② | 交付仍是 1 commit | 赢家是**唯一交付实现**，标准 commit 交付；赢家应用 commit 是 driver 记账类（与 sweep/probe 的 seeding commit 同类），不是第二次学生交付；"每迭代一次交付"纪律不变 |
| ③ | 赢家过全量 gate | 赢家树走原样 eval 路径（`parse_eval(be.submit_gpu(branch,"all"))` + precision + watermark + NCU），无任何 gate 稀释；未选中的变体永不进 eval |
| ④ | teacher 只见客观时延表与冒烟结果 | 变体表只进 IterationRecord（机器 SSOT）；教师读集无新增注入 |
| ⑤ | 预筛 job 计入本迭代预算 | 每授权迭代恰 1 个预筛 job，变体数 ≤ `multi_candidate_max`；语义上它先于该迭代的 eval，是授权决策的交付仲裁成本，不是新增 job 维度（无 probe 轮次消耗，学生无感） |

**单测**：`test_mc_delivery_full_chain_winner_applied_and_banked`（2 变体：授权 → 预筛 job（mock backend）→ 赢家写入 kernel、变体归档、record 记表 → 标准 eval/NCU 恰好一对）；`test_mc_delivery_dnf_variants_revert_without_an_eval`（DNF 路径：无 eval、复位、表照记）；`test_mc_delivery_without_authorization_is_byte_identical_to_the_old_path`（无授权回归：job 序列恰为 `["all","ncu"]`、`variant_timings is None`）；`test_mc_delivery_authorized_but_no_variant_files_is_the_old_path`；`test_mc_prescreen_script_runs_end_to_end_against_a_stubbed_timing_kernel`（生成脚本真实执行，抓到真 bug：脚本引用的预算常量只定义在 driver 侧、脚本作用域缺失——已修进脚本本体；这类错误 mock 测试测不到，dry-run 能）。

## 4. 配置与词汇表选择（契约 C7）

新增两个 LoopContext 键（`# --- WP10 ---` 块，driver.py 数据类默认区）：

| 键 | 默认 | 语义 |
|---|---|---|
| `stagnation_k` | `3` | 触发 STAGNATION 注入所需的连续平坦参数级尝试数；0 = 关闭 |
| `multi_candidate_max` | `3` | `multi_candidate:` 授权的绝对钳制上限；0 = MC 通道关闭 |

**选择与理由**：按 WP1 先例走 **LoopContext 默认区**（C7 明文选项），**不动** `global_config/loop.toml`、**不动** `task.py`。原因（WP5 CHANGES 已实测同一结论）：global 层 loop.toml 过 `task.py._unknown_keys` 的**闭合词表**校验，`_KEYS` 无 `[loop]` 表——加入即 `TaskError`（每个 `load_task` 都炸），而本包围栏不许改 `task.py`；WP1 的测试注释亦明文"the global layer's loop.toml takes [target]/[budget] only"。键的生效覆盖面 = 卡层 `operators/<op>/loop.toml` 的 `[loop]` 段（`_load_loop_config` 自由格式读取）。**cli.py 的 `LoopContext(...)` 转发行按 WP6 `m0_probe_rounds` 的先例留待集成步添加**（本包围栏不含 cli.py；WP6 分支同样只交付默认值 + 注释，集成提交 `c1971f1` 补入转发行）——两个键的字段注释里已写明该契约，集成步照 `m0_probe_rounds` 形状各补一行 `int(cfg.get("loop", {}).get("<key>", <default>))` 即可生效；未补前默认值照常治理（stagnation_k=3 默认开启触发器，multi_candidate_max=3 默认钳制）。

## 5. 契约与围栏遵守清单

- **C3**：注入区只加 STAGNATION 一个小节（≤10 行），位置在 WP4 REMOVE_CANDIDATES 之后、留给 WP12 KB-MENU 的邻居位之前；`# --- WP10 ---` 块注释；无其他注入小节。
- **C4**：probe 循环区本包只**新增** MC-DELIVERY 预筛路径；WP4 SWEEP 分支语义（`_run_sweep_request`/`sweep_run.py`/SWEEP 行协议）与 WP6 `_run_arch_probes` **一字未动**（git diff 可证）；预筛结果（`variant_timings`）独立字段，不与 SWEEP 表/ARCH_PROBE 读数共享 schema。
- **C5**：`card_template/eval/timing.py` 未触碰；预筛不进入 measure_paired 协议（它是赢家选择，赢家仍过全量 paired gate）。
- **C6**：`diagnose_prompt.md` 只加两段一句（flags 族段 + STAGNATION 消费段 + 1b 联动一句，合计 17 行 < 30）；`student_prompt.md` 只追加一句；`patch_prompt.md` 加授权句 ≤6 行；POLICY（meta_milestone.md）未动。
- **围栏**：改动文件 = `driver.py`、`inner/trajectory.py`（仅 variant_timings 字段）、`meta/diagnose_prompt.md`、`meta/patch_prompt.md`、`meta/student_prompt.md`（一句）、`skills/sweep-decision/SKILL.md`、`tests/test_wp10_search_depth.py`（新增）。未触碰：`parse.py`（resolvable_diff 直接导入）、`precision_gate.py`、`kb_lint.py`、`task.py`、`cli.py`、`global_config/loop.toml`（理由见 §4）、card_template 全部。
- **每迭代一次交付**：赢家才是交付；变体表是客观记录。
- **触发器只加压**：无机器改判、无诊断跳过；override 有通道且记账可见。
