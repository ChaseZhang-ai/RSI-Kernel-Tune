# CHANGES — WP14 形状分域决策通道（两阶段）

分支 `feat/wp14-shape-regime`（基线：kernel-ZXtune main `359696b`，从 `origin/main` 开出；clone 默认检出 integrated-v2，已按发包指令显式避开）。
基线回归 **2437 passed**（5:42）；交付后 **2457 passed**（新增 20 个用例，全绿，零回归）。全部教师侧信息与机器校验，零模型调用、零 gate 协议改动。

**围栏注记（交集成步裁决）**：本包改动文件全部在"只许动"清单内，另有 **`src/forgekernel/milestone.py`** —— 既不在允许清单、也不在禁碰清单。落点理由：任务书"milestone 决策行协议加 shape-keyed lever 值域声明"与"谓词解析/覆盖性校验"需要一个协议解析落点，milestone.py 是该协议的单一住所（模块章程即"milestone format validation"，SYNC/GATHER/PHASE 机器词汇的先例所在；driver 早已 import 其 `_decision_blocks`/`_section_body`）。**纯追加**：五个新函数名（`parse_regime_predicates` / `parse_regime_expr` / `regime_expr_matches` / `regime_coverage` / `parse_per_shape_declarations`），现有函数一字未动；`validate_milestone` 对带 regime/per_shape 行的 milestone 判定结果与基线逐项一致（`test_authoring_milestone_with_regime_lines_still_conformant` 钉住），全 137 个 milestone 既有测试全绿。若集成步判定越界，四个纯函数可机械平移至 driver.py（组合点 `_wp14_declaration_violations` 本就在 driver），行为不变。

禁碰清单零触碰（`timing.py`/`parse.py`/`precision_gate`/`screening.py`/`lever_pool.py`/`kb_lint`/`leaderboard.py`/`card_template/`/`task.py`/`taskcard.py`/`window_contract.py`/`scaffold.py`/`launch.py`/`__main__.py`/`binref_tool.py`/`sealref_tool.py`/`cli.py`/`NOTICE`/`README.md`/`docs/` 全部未动；`skills/` 未动）。

## 任务 1（P0，阶段一）— per-shape 账本 + size-dispatch 窄版（甲G1+乙A5）

**`inner/trajectory.py`（契约 C8 单一定义）**：`IterationRecord` 增两字段（追加式，`load_records` 对旧文件天然兼容）：

- `per_shape_ratios: dict` — 本行 eval 的逐 workload vs-ref 比值（`vs_ref_device` 回落 `vs_ref`，与定价表同币种），`{workload: ratio}`。单输入 bench = `{}`（顶层字段已是那个 shape 的读数）。
- `per_shape_best: dict` — 截至（含）本行的各 workload 历史最优快照 `{workload: {"ms": 门窗口径 ms, "iter": 创造行}}`：继承最近携带行、仅 KEEP 且超过在位者才推进（REVERT/失败行只记 ratios、账本原样继承）。

账本算术的唯一实现在 trajectory.py：写侧 `per_shape_ledger(records, workloads, iter_no, *, improve, gate_window)`，读侧 `per_shape_best_readings(records, gate_window)`（从 KEEP 行自己的 `workloads` 块重算——pre-WP14 行同样入账，digest 永不信任可以重算的快照）。两定义一致性由 `test_ledger_roundtrip_and_recompute_agree` 钉住。`build_digest` 顶层增 `per_shape_best` 键（geomean 无法示意的账面）。

**driver 落账**：KEEP 行与 correctness-FAIL / fresh-recheck-FAIL 两类 REVERT 行（唯一携带 `ev.workloads` 的三个记录点）按上式落账，`# --- WP14 ---` 注释标注。

**决策行协议（milestone.py 新函数）**：`per_shape: <lever>[<workload>]=<值域>`（逗号分隔、括号平衡切分，`raster[prefill]={row,col}` 一对一解析；不可解析对以 key="" 回报供 driver 点名）。`shape_split` 判类从此有接收器。

**SWEEP 逐 shape 报读数列（现有 per-workload 读数平移）**：`per_shape:` 声明的决策，`_run_sweep_request` 改走 `_run_sweep_request_per_shape`——每 workload 一个 scoped sweep job（复用定价通道既有 per-workload job 纪律；scoping 用工具自身的赋值路径：`lever_pricing_run.main` 解析 `--workload` 后写 `lpr._SCOPE_WANT["name"]`，驱动前导只做同一赋值，`_sweep_scope_cmd`），rounds 钳到定价通道的 `META_ROUNDS`（多 workload 预算纪律同源）。回显表逐 shape 列（每列各自 job 内同 job 比值/Δ/赢家），**不跨 job 选机械赢家**（跨 job 比值不是同一测量）；组合是教师 `per_shape:` 声明的职责，落地仍是学生普通交付走全量 gate。未声明的决策单 job 行为逐字节不变。

**落地形态不变**：分派 = 学生写的运行时 shape 查表分派代码（同一 kernel 内），普通决策交付、全量 gate 判定（`test_dispatch_delivery_walks_the_standard_gate_path` 钉住；patch_prompt 交付段同文）。**不做 per-shape kernel split/composite**（上游非目标维持排除，本包零实现面）。

## 任务 2（P0，阶段二）— M0 级 per-regime 拓扑词汇（丙WP14）

**谓词与覆盖性（milestone.py）**：`regime: <axis-predicate>` 行（决策块内归属该决策，per-decision duty 同 SYNC/GATHER/PHASE）。受限语法即纯度红线：仅 `<axis> (<|<=|==|>=|>|!=) <数字>` 的 and-连接——数据内容/计时读数/运行时状态在语法上无法成句（`test_regime_expr_purity_whitelist`）；语义纯度由 `regime_coverage` 持有：谓词的轴必须是卡片声明的形状轴（`best_ms` 这类计时名被点名拒绝）。

**覆盖性论证失败行为定义（验收点名项）**：机器把教师的互斥+覆盖论证钉在卡片真实 workload 上——每个 workload 恰好命中一个谓词（多命中=互斥违规、零命中=覆盖违规），违例文本带 `覆盖失败 = 冻结无效`；执行点两处：蒸馏 authoring 循环（`run_distill_step` 违例入 violations → 弹回重写，冻结无效=不过冻结）与 patch 循环（只对 patch **新增**声明持义，baseline 匹配豁免——gate-line hygiene 同款 grandfathering，`test_wp14_declaration_violations` 全覆盖）。卡片无 workload 轴可校验时，谓词存在本身即违例（不可验证的冻结=无效冻结，保守）。

**per-regime 定价**：诊断注入区 lever 价格块后增 regime 标签行（`_wp14_regime_label_line`：workload→regime 映射 + "沿 regime 分裂的 lever 是 per_shape 候选"）；`diagnose_prompt.md:245-246` shape_split 语义升格（指向 per_shape 声明与 per-shape 账本）+ 新增 Shape regimes 消费条（合计 3 行 ≤30）。

**per-regime NCU 证据**：`_wp14_regime_evidence_block`——里程碑带 regime 谓词时，每个 regime 取代表 workload 复用 stall/SASS pricing 通道的 sanctioned `--workload` scope（= `FK_LB_WORKLOAD_SCOPE` 点名），每 meta 步新购 ≤ `regime_ncu_max`（默认 2，0=关；C17 走 LoopContext 默认区 + `_wp11_apply_cfg` 同款卡片层解析，**不经 cli**），(tree, workload) 备忘防重购，超出预算的 regime 留给后续 meta 步补买。渲染在注入区**最后**（C10：per-regime 证据行 ≤6 行，KB-MENU 之后）。**仅 meta 步发生**：eval 路径零改动。

**零 regime 回归 = 逐字节一致**：无 regime/per_shape 行的 run，三个渲染块全返 `""`、零新购、`_arch_probe_stamp` 与基线逐字节相等（`test_regime_label_and_evidence_zero_regime_byte_identical`、`test_arch_stamp_zero_regime_byte_identical_and_regime_note_line`、既有 ARCH_PROBE 全套测试钉住）。

**ARCH_CANDIDATES per-regime 行**：值 token 可带 `regime=<谓词>` 注记（`ARCH_CANDIDATES: dual regime=m >= 1024`；全角冒号兼容）；`_parse_arch_candidates` 新函数解析，注记经 `_arch_probe_stamp(regime_note=...)` 进里程碑头注一行（probe 判定规则不变，time-only 不变）；裸写法逐字节不变。

**meta_milestone.md 只加词汇段**（PHASE 条目与 "What the machine check rejects" 之间，22 行）：regime/per_shape 语法、纯度规则、互斥+覆盖义务、覆盖失败=冻结无效、落地形态=L0 不变。**:391-400 红线原文一字不动**（md5 与基线一致：`e12b7ef06c60dd0b619a0baf490083f4`），全文件对基线纯追加（diff 零删除行）。`meta/distill_prompt.md`（M0 撰写段 +7 行 ≤8：per-regime 拓扑合同义务 + ARCH_CANDIDATES per-regime 行写法）、`meta/recon_prompt.md`（+4 行 ≤8：候选行 regime 注记 + 代表 workload 外的 regime 不得冻结）。

## 任务 3（P1，可选延伸）— MC-DISPATCH（乙A14）

MC 预筛按 workload 计时：`per_shape:` 授权的决策，`_mc_deliver` 改走 `_mc_deliver_per_shape`——每 workload 一个预筛 job（toolkit 自带的 `spec["workload"]` scope，授权/预算/纪律全沿任务 1：`_sweep_per_shape_workloads` 同一授权函数）；**全部 workload 同一赢家 → 既有机械赢家路径**（`_mc_apply_winner`，普通交付全量 gate）；**逐 shape 赢家分歧 → 无机械赢家**，普通 no-winner REVERT（不烧 eval），逐 workload 读数带 workload 标签入 `variant_timings`（学生自产客观记录豁免类），REVERT 文本指明"组合是决策的 per_shape 声明（教师据此 patch；学生交付分派代码）"。教师仅在结构分叉比 lever 值分叉更彻底时经 per_shape 声明选用——driver 从不合并分叉代码（L0）。单 job 路径逐字节不变（`test_wp10_search_depth` 24 例全绿）。

## 注入区纪律（契约 C10）

三块均在 `run_meta_step` 挂点、C10 序位：workload readings 后 per-shape 行（≤6 行）；lever 价格块后 regime 标签行；KB-MENU 后 per-regime 证据行（≤6 行，注入区最后）。每块源码区 `# --- WP14 ---` 标注；全空 run 三块皆空串（逐字节一致见上）。

## 知识注入 A/B 注记（红线 9）

- **A/B 口径**：本包不改任何测量量、gate 数字与 best 语义；新增信息面全部为教师侧注入（per-shape 账本/regime 标签/per-regime 证据）+ 机器校验弹回。A/B 对比 = 同 run 开关 `regime_ncu_max`（0=关，三块消失、零新购）与里程碑是否携带 regime/per_shape 行；WP21 通电包跑双 regime fixture（rmsnorm 类）真 run 时按此取 A/B。
- **学生可见面**：零新增教师 curated 信息流——`per_shape_ratios`/`per_shape_best`/`variant_timings.workload` 全部是学生自己 eval job 的客观读数（红线 2 第②类豁免，student_prompt.md 客观轨迹段一句注明，diff 恰 1 句）；分派代码永远是学生实现（patch_prompt 交付段 + diagnose 消费条同文钉死）。
- **换基座重测**：不适用（无知识条目注入；KB 零改动）。

## 验收对照（61 包 §3）

1. pytest 全绿（2457；新增 20 例：账本落账×4、谓词解析/覆盖报错×3、声明解析×1、driver 校验×2、per-shape/regime 渲染×3、per-regime NCU 选择×2、零回归逐字节×3、SWEEP 逐 shape×3、ARCH_CANDIDATES×2、MC-DISPATCH×1、per_shape 谓词键空 shape 集×1、分派 gate 路径 fixture×1——部分用例覆盖多项）。
2. dry-run（双 regime fixture，prefill m=8192 / decode m=64）：蒸馏提示词出现 per-regime 合同与 per_shape 值域 ✓；SWEEP-OK 表逐 shape 列（`| variant | prefill ratio/Δ/winner | decode ratio/Δ/winner |`）✓；诊断注入区 regime 标签行 + per-shape 账本行 ✓；覆盖性机器校验对 rmsnorm fixture 返回 `[]` ✓。
3. `meta_milestone.md:391-400` grep 一字不差（md5 实测同基线）；`student_prompt.md` diff = 1 句。
4. 本文件即 CHANGES-WP14.md：每任务一段 + A/B 注记 + 覆盖性论证失败行为定义（见任务 2 段内粗体项）。

## 测试环境

`~/venvs/forgekernel/bin/python -m pytest tests/ -q`；动手前基线 2437 全绿已记录。
