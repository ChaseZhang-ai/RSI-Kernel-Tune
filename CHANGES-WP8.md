# CHANGES-WP8 — 先验供给线（SOTA survey → ARCH_PROBE）

分支 `feat/wp8-prior-supply`（自集成 main @ c1971f1 开出，独立工作区 `/home/zcz/forgeKernelOptimize/wk2-wp8`，未合并未推送）。
基线与验收：干净 main 全量 `pytest tests/ -q` = **2222 passed**；本包完成后全量 = **2252 passed**（含复审接缝补齐的 5 项新测试）（见文末"测试证据"）。

**前置门偏差记录（重要）**：`/home/zcz/forgeKernelOptimize/kernel-ZXtune` 尚不存在；集成仓库以 `zxtune-integration` 的形式就位（main = `c1971f1 kernel-ZXtune: ForgeKernel @5030503 + WP1-WP6 integrated`，`handoff/02-集成报告.md` 已落盘——总览 §四前置门的实质条件满足，只差目录名）。姊妹 Wave-2 包（wk2-wp9、wk2-wp10）同一批拿到任务，其工作区 origin 均指向 `zxtune-integration`，本包沿用同一先例 clone 开工，未触碰旧 ForgeKernel 仓库。若总工程师期望 kernel-ZXtune 由集成步另行落位，本分支 `git remote set-url` 即可，工作不受影响。

教师侧工具权限核查（任务 1 的配置检查项）：`agent_policy.py` 中教师 `offline()` 恒为 False（联网放行，blackbox 仅 `deny_urls` 拒密封 reference 的上游）、`allow_write("teacher")` 含整个 run 仓库（`notes/` 可写）——**无配置缺口，零代码改动**。models.toml 无工具白名单概念（agent 以 `claude -p --dangerously-skip-permissions` 运行，权限由策略 settings 控制），同样无缺口。

---

## 任务 1：教师调研 skill `skills/sota-survey/`

`skills/sota-survey/SKILL.md`（新；已按 house 规范登记 `skills/README.md` 表，`tests/test_skills.py` 强制）。要点：

- **时机**：recon（blackbox）/蒸馏（source）阶段、M0 冻结之前；产出 `notes/prior_art.md`。
- **目标源**：FA3/FA4、FlashInfer、cuDNN-frontend 文档、DeepGEMM、CUTLASS examples、ThunderKittens、Liger——全宽松 license，读与描述无法律障碍。
- **产出格式（强约束）**：每条 = 决策（骨架假设 / config 起始值）+ 预期作用轴 + 机制转写（自己的话）+ license/provenance + **验证方式**（落到现有请求格式：`ARCH_PROBE: notes/arch_<n>.json`（skills/arch-probe 协议）/ `SWEEP:` 决策字段（skills/sweep-decision 协议，≤8 变体）/ `PROBE: notes/probe_<x>.py`）。骨架假设 2–3 个、每个须与本代 reference 架构不同。
- **四条硬边界**写死在 SKILL 内：① deny_urls 照旧（blackbox 禁查密封 reference 自己的上游仓库与镜像、raw/api、包管理下载）；② 任何代码体不得进 milestone/KB/prior_art；③ survey 结论未经本 run 实测不得影响任何 gate 数字或 M0 冻结（只能作 `[ESTIMATE]` 候选——借来的权威不作数）；④ survey-lane run 分 lane 标注（指向 `docs/prior-supply.md`）。

## 任务 2：注入点（task.toml 键 + 教师提示词 + driver 注入）

**键与 schema**（`src/forgekernel/task.py`）：

- `_KEYS["agent.role"]` 增加 `prior_survey`（bool）、`prior_art`（str）——封闭词汇表同步扩展，`[teacher]` 误拼表 / 未知键照旧被 `_unknown_keys` 拒绝（负例测试在案）。
- `AgentSpec` 增 `prior_survey: bool = False`、`prior_art: str = ""`（默认关——不配置则一切行为与先前逐字节一致）。
- `_read_agent` 校验：bool / str 类型错报清晰问题；**学生侧拼写被拒**（`prior_survey is a teacher-side key; the student has no prior lane`）；global models.toml `[agent.teacher]` 同样可设（`merge_config` 天然生效，全局 opt-in 实验可用，单测在案）。
- `load_task` 内 `_inside(d, teacher.prior_art, ...)`：声明卷宗必须相对 task 目录且存在（绝对路径拒、缺文件拒）。

**教师提示词**（`meta/recon_prompt.md`、`meta/distill_prompt.md`、`meta/distill_blackbox_prompt.md` + 三个 `meta/cuda/` overlay——见文末围栏注记）：各加一段 `## Prior inputs`（≤20 行，三段语义一致、措辞按模式适配：distill/source = "before the ORACLE"，blackbox = "before the recon report"，recon = "before you design the probes"），两端用 HTML 注释标记 `<!-- PRIOR-INPUT:BEGIN/END (WP8) -->` 包裹，段内含：

- 读 `PRIOR_ART`（声明卷宗 / notes/prior_art.md）/ `CARRY` / `PRIORS`（repo `priors/index.md` 指针，教师自行检索 top-k）；
- 分歧候选骨架 → M0 `ARCH_CANDIDATES` + `ARCH_PROBE_REQUEST` 配方（供 ARCH_PROBE 实测）；config 表 → 对应决策 `sweep:` 起始值（"starting values marked tunable"，L0 :339 的合法通道）；
- **carry 纪律句逐字注入**："for hypothesis reuse and reconciliation only — re-verify every line against this run's own measurements; never a gate anchor, never a price source"；
- "全部输入为 (none) 则整段跳过"的教师侧指令 + 借来的权威不作数 / 禁实现体红线句。

**driver 注入**（`src/forgekernel/loop/driver.py`，新 `# --- WP8: prior supply ---` 区 + `_bindings` 三 token）：

- LoopContext 新字段 `teacher_prior_survey / teacher_prior_art / carry_dir / m0_probe_candidates`（默认区 `# --- WP8 ---` 块，全部默认关/空）。
- `{PRIOR_ART}` = 声明卷宗（路径解析序：绝对 → workspace 相对 → run 快照 `operators_dir().parent/task/` 相对，即 "[agent.teacher] prior_art 由 task.dir 拼" 的运行时等价物）+ 本 run survey 产物（`notes/prior_art.md`）+ survey 义务句（opt-in 且无产物时）；`{CARRY}` = carry 四件套渲染（各自截断，总纪律句在前）；`{PRIORS_INDEX}` = 仓库 priors/index.md 路径。
- 应用点 = 蒸馏入口 `run_distill_step` 与 `_recon_prompt`（其全部调用方统一生效）：`_apply_prior_input(render, ctx)`——lane 开着时原样放行（token 已填充），关着时 `_PRIOR_INPUT_RE` 剥除整段（含前后换行，**剥后与 pre-WP8 模板逐字节一致**——六个提示词文件的字节一致性有专项验证，两个控制组测试照此更新）。
- **学生侧零改动**：student_prompt.md 一字未动（测试 pin 其无任何 prior token）；注入只经 `_bindings`（学生模板无该 token，渲染不受影响）。

**cli 接线**（`src/forgekernel/loop/cli.py`，LoopContext(...) 内 `# --- WP8 ---` 注释块，WP1/WP6 同址先例）：四键从卡片层 loop.toml 读（`teacher_prior_survey`/`teacher_prior_art`/`carry_dir`/`m0_probe_candidates`）；carry_dir 缺省时快照优先（`_default_carry_dir`：run 快照 → `<repo>/tasks/<op>/carry` 回退）。task.toml 拼写（`[agent.teacher]`）在 AgentSpec 上随快照携带，`forge run` 经卡片发射进真实链（见文末"接缝补齐"——本包内闭环，不留集成步）。

**`docs/prior-art/`**（新）：README（"只写决策不写实现体"的格式说明）+ `template-op.md` 空模板。

## 任务 3：仓库内先验库 `priors/`

- `scripts/build_priors.py`（新）：扫描 `kernels/` 全部 25 个已发布 kernel，产出每 kernel 的架构决策图摘要骨架（六节：Persistent scheduling / Warp-role split / Pipeline depth / Data residency / Epilogue / Shape domain），元数据取自 kernels/README.md 成绩表；`--write` 落盘、`--force` 刷新骨架但**永不覆盖 `Status: filled` 文件**、`--repo` 供测试/hermetic 使用；默认 dry-run。骨架只含元数据 + 章节桩，无实现体。
- 已落盘 25 份：**3 份高质量样例（filled）**——`priors/gemm/fp8_gemm_h100.md`、`priors/attention/attn_fwd_gqa16_h100_gen2.md`、`priors/conv/conv3d_fwd_h100.md`（各 ~18 行，决策+数值散文，花括号密度抽查为 0，见测试证据）。**偏差说明**：任务文本要求 GEMM/attention/norm 各一，`kernels/` 无已发布 norm kernel——第三份按同标准取 conv 族（且其 im2col-TMA 事实正与任务 4 第 3 行互证），不虚构不存在的 kernel。
- `priors/index.md`：按算子族 + **按特征**（"要决定什么"）双检索，与 hw_kb index 的 symptom-first 组织同风格；distill/recon 提示词先验段已含其指针。

## 任务 4：`hw_kb/CANDIDATES.md` 增补 6 行

按现有 schema 追加（现有 7 行一字未动；`Status` 全 TODO；kb_lint 全树 0 findings）：

1. **CTA raster/swizzle order**（三树，High）——五报告/总览实锤零条目；public 栈 5–15%；认证闭环走 WP2 `l2_hit_rate_pct`（字段已在 `inner/ncu.py:89`，raster 变更 → L2 命中率位移可直接观测，不靠裸计时）。
2. **TMA store reduce**（三树，Medium）——epilogue 归约直接进 global（split-K 部分和/softmax 分母），免原子、免第二 kernel；现条目只覆盖 store 拼写不覆盖 reducing store。
3. **im2col-mode TMA 跨树转写**（`cuda/sm100`，Medium）——cutedsl 侧已认证（发布的 conv3d_fwd 即该供给形态、对 cuDNN 仅 +7.6%，kernels/README.md），cuda 树零条目；CUDA 拼写还摆脱 DSL 版本依赖。
4. **DSMEM/cluster 直接交换**（cutedsl/sm90+sm100，Medium）——现有条目只有 multicast（扇出）方向；交换方向（CTA-pair halo/部分和替代全局原子）零条目，附 probe 形态。
5. **`cp.async.bulk.prefetch`**（三树，Medium）——现条目全是带 mbarrier 的 smem 目标；L2-only fire-and-forget 预取是 decode 型 attention 下一 tile 的延迟掩护，附 stall-ledger probe 形态。
6. **Load-modifier census**（三树，Low）——.nc/.cg/.cs 与 L1 evict 提示按 load 族建账；streaming 算子 L1 污染控制从 folklore 变可认证配方，probe 便宜（每热 load 一次 modifier A/B）。

## 任务 5：ARCH_PROBE 括号扩容 + 跨代携带

**括号扩容**（`_run_arch_probes` / `_arch_probe_stamp`，driver.py）：

- 新键 `m0_probe_candidates`（LoopContext 默认区 WP8 块，默认 2 = WP6 行为；cli.py 卡片层接线；global_config/loop.toml 词汇表封闭，按 WP1 先例键走卡片层，**该文件只追加注释块**说明键位——`[loop]` 表若写入会过不了 `_unknown_keys`，CHANGES 特此写明选择）。
- 语义：`n = min(want, len(requests), rounds)`（want≥2 下限，<2 预算整体跳过的 WP6 语义不变）；`n < len(requests)` 时打 "probing the top-n of N listed candidates (bracket …, budget …)" 日志并在 milestone 头部戳记 skip 行（"…further listed candidate(s) beyond the probed top-n skipped"）；`n == len(requests) 且 want>2` 打 "bracket widened"。消费 `requests[:n]` 替换全部 `[:2]` 切片。
- 判定规则不变、对前二名应用（最快 vs 次快，分辨率带判定；带内冻结 teacher 首选）；`_arch_probe_stamp` 恰好两候选且无截断时**逐字节保持 WP6 渲染**（既有测试零改动通过），>2 时每读数一行（最快在前）+ 同一 winner/floor 措辞。

**跨代携带**（`scripts/self_evolve.py::next_generation`，签名增 `run=None`）：

- 换代时落盘 `tasks/<op>-g<k>/carry/` 四件套：`milestone.md`（父代 final：决策+合同+理由注记，头部自带 ARCH_PROBE 戳）、`levers.json`（价格表）、`arch_probe.md`（从 milestone 头部抽取的 ARCH_PROBE 读数）、`falsified.md`（`hw_kb/falsified_ledger.jsonl` 中 `op_family == 本 op` 的行）；外加一页 `README.md` 标注四件与纪律句。best-effort：父代没产出的工件记 "absent"，不猜。
- 消费点 = distill/recon 提示词先验输入段（任务 2 同段），**不进诊断注入区**（契约 C3）；段内纪律句（不作 gate 锚、不作价格来源）driver 侧逐字注入，测试 pin。
- gate 锚定禁令不破（meta_milestone.md:290-296 禁的是把过去 trajectory 当 gate 锚；carry 是教师侧输入，且新版 reference 树 = 父代 best 树，旧读数对它精确有效——但契约仍要求逐行复核）。

## 任务 6：ops 纪律文档 `docs/prior-supply.md`

一页纸三条：① **lane 分报**——任何先验 lane 开着的 run 是 survey-lane，报告分 lane 标注、永不与 goal-only/source 基线混列（FA4/DeepGEMM 是公开代码，"读过机制描述"与"自发写出"不可判别，lane 标注就是诚实答案；carry 是强度最高的 lane，仍只是假设复用）；② **ref 换血**——自进化链起点选最强**合法**参考（expert-seed 是合法赛道、单独 lane）；"市面最强"以本机配对计时为准（不信 README 跑分）；每次换代重检市场，更强合法参考可换血（换血记 lane）；供给线终点仍在教师，学生永不看等价实现；③ **license/provenance**——GPL/AGPL 排除、每条 entry 记 license+版本；进入 KB 走正常认证，provenance 落 `docs/kb-provenance/`。

---

## 围栏注记（超出字面清单的三处，均为功能必需，请集成步复核）

1. **`meta/cuda/{recon,distill,distill_blackbox}_prompt.md`**：CUDA 模式经 `lang_mode.prompt_path` 读 overlay 全文（非增量），只改 base 三个文件 = cuda 模式静默无此功能。三份 overlay 各加同一先验段（与 base 措辞一致）。
2. **`cli.py`**：LoopContext(...) 构造内键接线 + 注释，及其 carry_dir 默认表达式的小 helper `_default_carry_dir`（快照优先；放在表达式旁边的唯一理由是可读性，不碰 cli 其余任何逻辑）——任务文本"新键按 WP1 先例走 LoopContext 默认区"的必然落点（WP1/WP6 键同址接线）。
3. **既有测试更新**（三处，均系 WP8 被授权改动的契约所系）：`test_loop_driver.py` 两个 recon/distill 控制组测试改比"剥除先验段后的模板渲染"（= pre-WP8 字节）；`test_self_evolve.py` fixture 适配 `next_generation(…, run=)` 签名；`test_kb_candidates.py` 队列行数 7→13（任务 4 的合法增量）。

## 契约遵守

- C3：carry/先验只进 distill/recon 提示词段，未碰 driver 注入区其他小节、未碰 `meta/diagnose_prompt.md`。
- C6：三个 base 提示词只加先验输入段（其余一字未动），关闭 lane 时渲染输出与 pre-WP8 逐字节一致（六文件 strip 字节一致性验证 + 两个控制组测试）。
- C7：新文件清单与总览 §五一致（skills/sota-survey/、priors/、scripts/build_priors.py、docs/prior-art/、docs/prior-supply.md）；task.toml/loop.toml 键均同步扩词汇表或走 LoopContext 默认区。
- 红线：学生侧零改动（student_prompt.md 未动、agent_policy.py 学生限制未动、kb_lint/timing/parse/precision_gate/diagnose_prompt 未动）；借来的权威不作数（先验段与 SKILL 四边界、CARRY 纪律句、docs/prior-supply.md 三处成文）；禁实现体（prior_art 模板、priors/ 26 文件、CANDIDATES.md 新 6 行均无代码体，测试机械检查）。

## 知识注入 A/B 度量方案（总览红线 5，逐任务注记）

| 任务 | 上线前后同 job A/B 度量方案 | 基座模型更换后 |
|---|---|---|
| survey skill + prior_art 注入（任务 1/2） | 同一 task.toml 同一 seed，跑三对 run：goal-only 基线 vs `prior_survey=true`（survey lane 单独标注）；读数 = M0 首冻结达成迭代数、前 3 迭代 ARCH_CANDIDATES 分歧数、到 target 的墙钟与 GPU job 数。预期信号：M0 一次冻结率上升（返工蒸馏轮减少）；负信号哨兵：[ESTIMATE] 候选未经理化直接进 gate 的次数（应为 0，kb_lint/审计可查） | 重测（JustGoal 消融显示机制收益随基座漂移） |
| priors/ few-shot（任务 3） | 同上三对，仅开 `PRIORS` 面（survey 关），读数 = 教师在 milestone 内引用 prior 条目的决策其后续 KEEP 率 vs 未引用决策的 KEEP 率（同 run 内配对） | 重测 |
| carry/（任务 5） | 同链 g→g+1 两代，carry 开 vs 删 carry 目录对照；读数 = g+1 代的死路重试数（falsified_ledger 重复行）、lever 重复定价 job 数、到 target 的总墙钟 | 重测 |
| 括号扩容（任务 5） | `m0_probe_candidates` 2 vs 3：M0 首段迭代收益（首批 KEEP 的 ≥resolvable_diff 增益）、代价 = +1 probe job；按 WP6 的"决策读数 vs job 价"口径合并评判 | 与基座弱相关，可延后 |
| CANDIDATES 增补/prior-supply 文档（任务 4/6） | 非行为改动（无 agent 读该文件的 run 行为不变），无需 A/B；queue-hit 命中率随 harvest 记录自然积累 | 不适用 |

## 测试证据

- 基线：干净 main @ c1971f1，`pytest tests/ -q` → **2222 passed**（306.91s）。
- 完成后全量：**2252 passed**（新增 `tests/test_wp8_prior_supply.py` 30 项 = 25 项主体验收【键 schema 正/负例、prompt 两态与字节还原、bracket 扩容四态、carry 四件套/缺席记录/无 run 不写、build_priors hermetic 写与 filled 保护、priors 无代码体、CANDIDATES.md kb_lint 专项扫描】+ 5 项接缝补齐【卡片发射正/反态、board 卡发射、foo-g2 代形 carry 指向快照、cli 默认快照优先/回退】）。
- kb_lint：`lint_kb_dir(hw_kb)` = 0 findings（含 CANDIDATES.md 新 6 行）；`lint_exam_surfaces`（meta/ + skills/ + 模板 + 循环内提示词字符串常量）= 0 findings。
- dry-run 证据（验收 2）：fixture task 带 `[agent.teacher] prior_survey=true` + `prior-art/op.md` 过 `load_task`；蒸馏/recon 提示词组装注入段全文见下方附录（PRIOR_ART 卷宗内容、CARRY=(none)、PRIORS 指针、纪律句齐备）；关掉键后整段消失（"section present: False, marker present: False"）；负例 `[teacher]` 误拼表被拒且报 "unknown table [teacher]"（单测 pin 同文案）。
- 花括号密度抽查（验收 3）：`grep -c '[{}]' priors/*.md` 全 0（25 文件）；`python3 scripts/build_priors.py --help` 可运行。
- strip 字节一致性：六个提示词文件 `_PRIOR_INPUT_RE.sub("", template) == 集成 main 同名文件`（逐字节 True，x6）。

### 附录：dry-run 注入全文（distill，source 模式）

```
<!-- PRIOR-INPUT:BEGIN (WP8) -->

## Prior inputs (teacher-only; every claim is a hypothesis until THIS run measures it)

This run's prior inputs are listed below; "(none)" names an input that does not exist. **If every input reads
"(none)", skip this section entirely** and work exactly as the rest of this brief says. Read them **before the
ORACLE**: skeleton hypotheses that diverge from the source's readings become M0 `ARCH_CANDIDATES` lines (each
carrying an `ARCH_PROBE_REQUEST: notes/arch_<n>.json` recipe, so THIS run measures them before the freeze); a
prior config table becomes `sweep:` starting values ("starting values marked tunable") on the matching decision.

- `PRIOR_ART` = ### Dossier (declared prior_art): `<ws>/prior-art/op.md`
  ### 1. persistent cluster-pair skeleton
  - Decision: cluster 2x1x1, A multicast
  ### Survey duty
  this run opted into a survey (`prior_survey = true`) and `notes/prior_art.md` does not exist yet: ...
- `CARRY` (the parent generation's closing record) = (none)
- `PRIORS` (the repo prior library; read the top-k entries matching this operator's family) = <repo>/priors/index.md

`CARRY` is **for hypothesis reuse and reconciliation only — re-verify every line against this run's own
measurements; it is never a gate anchor and never a price source** (a parent price was read on the parent's tree;
on this run's reference tree it is a hypothesis until re-priced). Mechanism descriptions in your own words only:
**no implementation body** enters the milestone, the KB or any notes file, and **no unmeasured prior claim moves
a gate number, the M0 freeze or any stopping judgment** — unmeasured claims stay `[ESTIMATE]` candidates.

<!-- PRIOR-INPUT:END (WP8) -->
```

### 接缝补齐（复审 P0/P1，已在本包完成——不存在"集成步替我接线"，本包交付即终态）

**任务级 opt-in 在真实启动链上断线**（P0）：`load_task` 校验并挂上 AgentSpec 的两键，但 `write_loop_toml`/`write_board_loop_toml` 不发射，而 loop cli 只读卡片 loop.toml 的 `[loop]`——`forge run` 真实路径上先验线永远暗着，全局 models.toml opt-in 同样死在链上。补齐：

1. **发射**（`src/forgekernel/taskcard.py`，`write_loop_toml` 与 `write_board_loop_toml` 各加一块 `# --- WP8 ---`）：
   - `task.teacher.prior_survey` 为 True → 发射 `teacher_prior_survey = true`；
   - `task.teacher.prior_art` 非空 → 发射 `teacher_prior_art = "<快照绝对路径>"`，用现成 `_snapshot_path(run, task, task.dir / task.teacher.prior_art)` 解析（`run.task_dir / 相对路径`；driver 的 `_resolve_prior_art_path` 对绝对路径直接使用，链路即通）；
   - `(task.dir / "carry").is_dir()` → 发射 `carry_dir = str(run.task_dir / "carry")`（`Run.task_dir` 即快照；快照是整个 task 目录的 copytree，`_SNAPSHOT_IGNORE` 只排 `.git/__pycache__/*.pyc/.DS_Store`，carry 一定随行）。**无键 task 产出的文件与现状逐字节一致**（三行只在对应条件成立时出现，发射点在固定键序区内，测试 pin 正/反两态）。
2. **cli 默认快照优先**（`src/forgekernel/loop/cli.py`，仅 carry_dir 默认表达式 + 其 helper `_default_carry_dir`）：卡片键 → run 快照 `operators_dir().parent / "task" / "carry"`（存在才用）→ 回退 `<repo>/tasks/<op>/carry`。resume / `--harvest-only` / 手动裸跑 loop 无卡片写入也正确。
3. **carry 默认的稳健性强化**（上一轮复审判为 P1 故障修复，经本轮复核更正为防御性改进）：`task.name` 来自目录名（`load_task`：`name = name or d.name`），换代目录改名 `<base>-g<k>` 后 `run_loop` 传 `--op` 随之指向新代——因此修改前的 repo 回退默认 `<repo>/tasks/<args.op>/carry` 在标准自进化链上**本就指向正确的 carry 目录**，不存在"永指 gen0"的故障（该错误根因出自上一轮复审提示词，本轮复核纠正）。本次快照优先默认 + 卡片发射的真实价值是对 resume / 换机 / 仓库树搬迁等场景的强化：快照 carry 随 `runs/<id>/` 走、不依赖 repo 树的 `tasks/` 布局仍在原地，卡片发射让显式路径总能压过任何默认推断——双保险，任一在位都指对本代 carry。
4. **测试**（`tests/test_wp8_prior_supply.py` 新增 5 项，共 30）：带两键 fixture → 产物含三键且 `teacher_prior_art` 指向快照内存在文件；foo-g2 代形（目录改名、carry 在 task 内）→ `carry_dir` 指向 `exists()` 的快照目录、绝不落 `tasks/<name>/carry`，且发射文本走 cli 同款 `cfg.get("loop",{})` 读取三键可读回；board 卡同发射；无键 task（含 board）三键全无；`_default_carry_dir` 快照优先/回退两态（monkeypatch `operators_dir`）。

### 顺手清理（P3）

- `scripts/build_priors.py`：删死常量 `_KERNEL_FILES`，`_kernels()` docstring 改准（现仓库唯一 .cu kernel 目录同时有 kernel.py，kernel.py 即标志，无实际漏）。
- 骨架标题/元数据修复：复审只点了标题 fallback 落 GPU 代号（dsv41 骨架标题曾是 "# prior: b200"），排查发现更深一层——`_rows()` 以 README 相对路径为键而查找用 `kernels/` 前缀路径，**元数据从未匹配上**，且目录列正则漏了 markdown 链接的 `[`（`` [`dir`](dir/) ``），`_rows()` 恒为空。三处一并修：键对齐 + 正则补 `\[?` + 标题 fallback 改算子目录段（`kernels/<family>/<op>/<gpu>` 第二段，永不落到 "b200"）。`--write --force` 刷新全部 22 份自动骨架（现全部带 README 算子名与 Published 行；3 份 `Status: filled` 未动，脚本保护在案）。

### 集成缝（已清——无遗留）

原"WP13 待办两行"已由上节接缝补齐取代：卡片层拼写与发射、cli 快照优先默认、无键字节一致，全部在本包内交付并有测试。集成步只剩常规检查（分支合并 + 全量回归），无 WP8 专属接线要做。
