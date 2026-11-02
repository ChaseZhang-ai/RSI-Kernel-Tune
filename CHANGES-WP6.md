# CHANGES-WP6 — M0 冻结前实测 + R2 结构通道

分支 `feat/wp6-m0-structural`（自 main @ 5030503 开出，本地保留，未合并未推送）。
三个任务各一个 commit：`ef87bfd`（任务 1 ARCH_PROBE）、`c854d90`（任务 2 R2-structural）、`938db91`（任务 3 制度文档），外加一个 house 不变量修正 commit（SKILL.md/README 行、cuda overlay 同节；见"测试证据"）。
基线与验收：干净 main 全量 `pytest tests/ -q` = **2054 passed**（见文末"测试证据"）。

工作方式说明（并行共存）：本包开发期间，共享检出 `/home/zcz/forgeKernelOptimize/ForgeKernel` 的分支被并行的 WP1 会话切走（`feat/wp1-measurement`，工作区带未提交改动）。为不干扰它，本包在独立 worktree `/home/zcz/forgeKernelOptimize/wk-wp6` 中检出本分支完成全部工作；分支本体在该共享仓库的 `.git` 中，集成步可直接合并。

---

## 任务 1：ARCH_PROBE — M0 候选架构冻结前实测

### 协议

**触发**：teacher（meta agent）在蒸馏产出 v1 时，若某关键资源拓扑轴（cluster 尺寸 / tile 方案 / warp 角色划分 / smem 拓扑）存在 ≥2 个未被观测证伪的自洽读法，必须给出 **top-2 候选**，各附一个骨架配方；无分歧写 `ARCH_CANDIDATES: single`（零 job）。source 模式下 ORACLE 源码在 call site 定出值即为证据，不适用本通道。

**milestone 机器行**（title 行之下）：

```
ARCH_CANDIDATES: dual
ARCH_PROBE_REQUEST: notes/arch_0.json   # 首位 = teacher 首选
ARCH_PROBE_REQUEST: notes/arch_1.json
```

**配方 schema**（`notes/arch_<n>.json`，teacher 写）：

```json
{"name": "<label>", "rationale": "<一句话依据>",
 "language": "cuda" | "python",
 "rounds": <计时重复次数，默认 20>,
 "source": "<cuda: 定义 void arch_probe_run() 的 CUDA C++（设置该候选的 grid/block/cluster/smem/warp 角色骨架并跑一遍主循环形状）；python: 定义 def arch_probe_run(): 的片段>"}
```

骨架 kernel 是**拓扑 mock，非实现体**：最小、哑数据、正确性无关。所有权对齐 verdict probe：**teacher 写、driver 冻结执行**，学生线零接触（见下"执行点"的复位语义）——L0 边界不放松。

**执行点**：`run_two_layer` 蒸馏入口，`run_distill_step` 之后、`distilled.milestone_violations` 检查**通过后**（v1 将冻结才探；malformed 直接 `distill_malformed` 不买 job）、内层循环开始（冻结生效）前。driver.py:11843 附近（`_run_arch_probes(ctx)`，紧随 violations 早退块）。

**预算**：`m0_probe_rounds`（driver.py:452，默认参数区 `# --- WP6 ---` 块，默认 2 = 每候选 1 job；0 = 通道关）。两候选需两轮，预算 <2 时整体跳过；候选 >2 时只探前 2；recipe 文件缺失、runner 缺失、job 抛 BackendError 均跳过并记录。**每个 skip 都在 milestone 头部留痕**，文案：`M0 probes skipped (budget)` / `(only one candidate recipe)` / `(recipe file missing)` / `(runner missing)`。

**执行语义**（`_run_arch_probes`，driver.py:11560）：
1. workspace 提交一个 `meta-probe: ARCH_PROBE skeleton candidates` 快照（milestone + 配方 + 种子 runner —— job 只带已提交文件），push 到 `<branch_prefix>-archprobe` 分支；
2. 每候选一个 job：`cmd:python3 notes/arch_probe_run.py notes/arch_<n>.json`；runner 从 `skills/arch-probe/arch_probe_run.py` 种子进 `notes/`（`_seed_arch_probe_runner`，driver.py:11506）。`skills/arch-probe/` 按 house 规范携带 SKILL.md 并入 skills/README 表（`tests/test_skills.py` 强制每个 skill 目录有 SKILL.md、README 有行）——SKILL.md 会像其他 skill 一样随 mirror 进 agent 工作区（job-env-check / lever-pricing 同为先例），其首行即声明"teacher-side，学生不用"；骨架配方本体仍是 teacher 专属（probe 工作区复位，永不进学生树）；
3. runner（driver 冻结）解析配方 → `language=cuda` 走 torch cpp_extension in-job 编译（与 CUDA 卡学生内核同一 nvcc+CUTLASS 工具链），`language=python` exec；一次预热 + CUDA event 计时重复；每候选打印一行 `ARCH_PROBE <stem> ms=<均值>`（编译/启动失败打 `status=FAIL`，即"无读数"；**只读时间，正确性永不读**——verdict probe 的 "wrong results are allowed" 哲学）；
4. `finally` 中 reset 回 probe 前 head（teacher 骨架材料不进学生线；real backend 上 reset --hard 会把 probe commit 加入的文件从工作树移除），milestone 用执行前快照文本**重新写入**并打头部戳；
5. 判定（`_arch_probe_stamp`，driver.py:11522）：两读数齐全且相对差 `(max-min)/min > resolvable_diff(records, gate_window)` → **赢家候选锁定为 v1**（头部戳 `winner <path> (+Δ% faster than …)`，`the M0 freeze follows the WINNER candidate's topology reading`）；差值在分辨带内或一方无读数 → **teacher 首选冻结**、读数附于头部戳。`resolvable_diff` 走 C1 垫片（driver.py:11439-11441；WP1 未合入时回退 0.015；WP1 合入后 data-poor run 按其自身 cap 回退，同为 0.015）。
6. 头部戳为 HTML comment（紧随 `# ` title），§9 校验不受影响；`ARCH_PROBE_REQUEST`/`ARCH_CANDIDATES` 行随即被消费删除（`_strip_arch_probe_lines`）。

**M0 audit 联动**：`meta/m0_audit_prompt.md` 审计程序新增 3c —— 关键拓扑裁决无实测证据（无 ARCH_PROBE 头部戳、无同源实测）时，**必须列为审计结论首要风险项**（不阻塞，但显式记录）。

### 代码落点（driver.py，全部为本包新区域）

| 内容 | 位置 |
|---|---|
| `m0_probe_rounds` 默认参数（`# --- WP6 ---` 块，仅追加） | :438-452 |
| C1 垫片（`resolvable_diff` try/except → 0.015） | :11439-11441 |
| `ARCH_PROBE_REQUEST`/`ARCH_CANDIDATES`/verdict 行解析、头部打戳、runner 种子 | :11450-11560 |
| `_run_arch_probes`（提交/执行/判定/复位全流程） | :11560-11635 |
| 调用点（`run_two_layer` 蒸馏入口，violations 门后） | :11843 |

未触碰：1748-1796 `_ncu_profiler`（WP2）、2157-2214 probe 分支（WP4）、373-393 与 6217-6222（WP5）、`parse.py`（WP1）、`student_prompt.md`、11601-11700 meta 预算段（WP5，现分支行号有漂移，按内容规避）。
**遗留（集成步一行）**：`[loop] m0_probe_rounds` 的 loop.toml 接线需在 `cli.py` 的 `LoopContext(...)` 构造处加一行（`m0_probe_rounds=int(cfg.get("loop", {}).get("m0_probe_rounds", 2))`）——cli.py 不在本包围栏内，未动；接线前默认 2 对所有 launch 生效。

---

## 任务 2：R2-structural — 教师侧 ORACLE 外结构提案（附价格）

### 提示词契约（meta/diagnose_prompt.md，仅本包拥有的两段）

- R2 定义段（原 :138）新增子类：**R2-structural — a skeleton NOT present in the ORACLE**；仅接受**本 run 内产生的实测价格** `reopen_price: <delta> via probe:<path>#<n>`（按路径引用的 ablation JSON 亦算 probe）；lever/KB/推断一律不算价（红线 3：不放宽为推测/KB 推断）；无价机器拒绝（弹回一次、再犯拒绝）；落地走普通 milestone patch（structural reopen 或 additive rung 扩展，按现有事务类），标准 gate 兜底。
- 输出契约段新增提案格式行（插在 `patch_kind:` 与 `reopen_price:` 之间，两行原文未改）：

```
r2_structural:     <write the literal tag `R2-structural` ONLY when the proposed skeleton is NOT present in ORACLE …; otherwise leave empty>
```

### 机器检查逻辑

**标记解析**（driver.py:3340-3348）：`_R2_STRUCTURAL_LINE = r"r2_structural\s*[:：]\s*(\S+)"`（容忍全角冒号，对齐 `_CAUSE_ENUM`/`_REOPEN_PRICE_LINE` 房规），值 == 字面量 `r2-structural`（大小写不敏感）。

**严格验证**（`_verify_r2_structural_price`，driver.py:3362）：
1. 无 `reopen_price` 行 → 拒（错误行给出唯一可接受形式）；
2. 引用无 `probe:/lever:/KB:/ablation:` 源 → 拒；
3. 源非 `probe:` → 拒：`an R2-structural prices only a probe measured in this run; <kind>:<name> is a lever/KB-class source and does not price an ORACLE-external skeleton`；
4. 复用 `_verify_reopen_price` 的 probe 分支（lever_table 传空）：本步 probe_readings 必须含该 path（有 #n 再匹配 contract），且读数非 n/a/unreported——**"本 run 内产生"由 in-memory 读数证明**；
5. `check_disk=True`（仅弹回门，见下）：证据文件必须存在于本 run 的 `notes/`（workspace 及 workspace/notes 双拼）或 `evidence/`（`ctx.evidence_dir`，recon 目录兜底）——错误行：`the cited evidence file <path> is not in this run's notes/ or evidence/ — an R2-structural's price needs its probe evidence on disk`。

**弹回一次（诊断循环内，driver.py:6708-6722，R1 价格门之后）**：诊断带 tag（**绑定 `r2_structural:` 标记本身，不绑定 cause 类**——教师把提案误归入非 R2 类（如 G 类）时同样先收到一次"如何补价"的错误行，而不是在 patch 校验点被从未获知路线地硬拒；R1 类误标仍由其前的 R1 价格门先弹回，属正常兜底次序）且严格验证不过 → `r2_price_demanded` 置位、注入 `_R2_STRUCTURAL_PRICE_BLOCK`（driver.py:3393，含 probe 路线与"lever/KB/推断不算价"红字）+ driver check 结果行 → re-prompt。注意弹回不设 `_reopen_gate_on` 前置——红线 3 无例外（无测量通道的 run 中该提案只能改投他类）。

**再犯拒绝（patch 校验点，driver.py:6953-6969，reopen-capped 检查之后、既有 structural/split 价格检查之前）**：诊断带 tag（无论落地 patch_class——structural 或 additive 一视同仁，红线 3 绑提案不绑事务类）且严格验证（`check_disk=False`）不过 → milestone 回滚到 patch 前快照、`_record(patch_class="unpriced-reopen", outcome="UNPRICED_REOPEN", ...)`、返回 unpriced-reopen——与既有 UNPRICED_REOPEN 模式完全同形。`check_disk=False` 的原因：诊断循环结束时的 probe 复位已把证据文件归档出工作树，此时 in-run 读数即 provenance（弹回门已在文件可查时把过关）。弹回后被教师坚持的"有读数无盘上文件"提案按读数放行——证据实质在（读数 + `-metaprobe` 分支归档），符合"缺价弹回一次再犯拒绝"的本义（再犯仍缺**价**才拒）。

**回归不变量**：不带 tag 的普通 R2/R1/G 类行为逐字节不变（严格检查以 tag 存在为前提；R1 三源门、G1-structural 补价门、reopen 预算/封顶全部原样）；学生 prompt 零改动。单测 `test_untagged_r2_with_a_structural_patch_keeps_the_old_rule` 锁定。

未触碰：373-393 reopen 参数区（WP5）、6217-6222 封顶读数区（WP5）。

---

## 任务 3：制度层

- `meta_milestone.md` §3 M0 条目末尾新增段：**冻结是要价签的动作**——冻结未实测的关键拓扑推断前必须有 ARCH_PROBE 或同源实测读数；重开是纠偏（价签的 structural 事务）；probe 读数永远不是 gate；无实测的冻结由 M0 audit 记为首要风险项。**:391-400"图外增益无法产生也无法收割"段原文未动**（git diff 仅含 §3 一段新增）。
- `meta/recon_prompt.md` 候选表节新增段：纯推导幸存的候选是赌注不是读数——关键拓扑轴上无法区分的 fork 诚实保活，交给蒸馏步 ARCH_PROBE 冻结前实测；不许把 back-solve 值写成"读出来的"。

---

## 提示词改动 diff 摘要

| 文件 | 改动 |
|---|---|
| `meta/distill_prompt.md` | "M0 freezes at its finish" 段后新增 ARCH_PROBE 协议块（触发条件、配方 schema、机器行格式、single 免探、判定与冻结规则、L0 不放松说明） |
| `meta/distill_blackbox_prompt.md` | 同上，且绑定 blackbox 语境（recon 给的是带置信度的推断、live fork 正是风险所在；赢家同时裁决 live fork，伪证条款在纸面兑现）；写明"不因配方放松 L0 与 do-not-claim-to-have-read-the-source" |
| `meta/cuda/distill_prompt.md` / `meta/cuda/distill_blackbox_prompt.md` | 对应 CUDA 变体同步（`cudaLaunchKernelEx`、in-job nvcc+CUTLASS 工具链措辞） |
| `meta/verdict_probe_brief.md` | 末尾新增 "ARCH_PROBE (the distill step's pre-freeze cousin of this channel)" 协议文档：schema、冻结执行、时间哲学、所有权、预算 skip 留痕、never-a-gate |
| `meta/m0_audit_prompt.md` | 审计程序新增 3c（无实测且无 ARCH_PROBE 的关键拓扑冻结 = 结论首要风险项，不阻塞） |
| `meta/diagnose_prompt.md` | R2 定义段子类 + 输出契约 `r2_structural:` 行（读数顺序段、价格块消费段、Hard rules 全部未动） |
| `meta_milestone.md` / `meta/recon_prompt.md` | 任务 3 两段（见上） |
| `meta/student_prompt.md` 及学生线 | **零改动** |

## 测试证据

- 新增 24 例：`tests/test_arch_probe.py`（14）+ `tests/test_r2_structural.py`（10）。覆盖 brief 验收清单：蒸馏输出含 2 候选 → **2 个 probe job 提交、赢家锁定**；单候选（`ARCH_CANDIDATES: single`）→ **零 job**；预算 1（<2）→ **跳过并记录 "M0 probes skipped (budget)"**；垫片回退态（WP1 未合入，`resolvable_diff([], "e2e") == 0.015`）行为正确；R2-structural：带合法本 run probe 价格通过并按普通 patch 落地、无价弹回一次再拒绝（UNPRICED_REOPEN、milestone 回滚）、**标记带在误归类（G2+additive）下同样先弹回**、lever/KB 价被拒、未运行 probe 被拒、证据文件弹回门、additive 落地仍需价、普通 R2+KB 价回归不变。
- House 不变量回归：`tests/test_skills.py`（每个 skill 目录须有 SKILL.md + README 行 → 补 `skills/arch-probe/SKILL.md` 与 README 行）与 `tests/test_cuda_mode.py`（cuda overlay 须与共享 prompt 同节 → 补 `meta/cuda/verdict_probe_brief.md` 的 ARCH_PROBE 节）。
- 分支全量：`pytest tests/ -q` = **2078 passed**（干净 main 基线 2054 + 本包 24；两次全量运行期间该机器负载有波动，均无失败）。

## 红线自查

1. M0 冻结语义未放松：ARCH_PROBE 只发生在冻结前（`run_two_layer` 蒸馏入口、violations 门后）；冻结后契约与 "M0 fixes the architecture" 措辞原样保留（student_prompt.md 零改动）。
2. ARCH_PROBE / R2-structural 只读时间、永不 gate：判定戳仅进 milestone 头部备注与 teacher 决策；落地全走标准 correctness + precision gate。
3. R2-structural 接受条件 = 本 run 内 probe 实测价（lever/KB/推断机器拒绝）；ARCH_PROBE 判定阈值 = 本 run 分辨率（C1 垫片），不放宽。
4. 学生 prompt 与学生线纪律零改动；teacher 骨架材料 commit→job→reset，不进学生树（skill 镜像按 house 规范覆盖 `skills/arch-probe/SKILL.md`，其中明确 teacher-side；配方本体只随 probe commit 走，job 后复位）。
5. 未引入任何 token/凭据；`local.env` 未触碰。
