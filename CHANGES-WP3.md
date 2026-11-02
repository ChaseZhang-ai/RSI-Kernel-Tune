# CHANGES-WP3 — 知识复利：证伪杠杆台账、probe 归档、KB 待认证队列

分支 `feat/wp3-knowledge`（基线 main @ 5030503），每任务一个 commit。行号基于本分支合并 WP3 后的
`src/forgekernel/loop/driver.py`（与 WP1–WP6 并行改动会挪动行号，以函数名为准）。

## 测试基线与证据

- 环境：仓库自带 pyproject（`pythonpath=["src"]`），用 `~/venvs/forgekernel/bin/python -m pytest tests/ -q`。
- 基线（改动前，同环境）：**1966 passed, 15 skipped, 2 failed**；2 个失败是
  `test_correctness_monitor.py` / `test_launch.py` 各一例懒加载 `import torch`
  （`ModuleNotFoundError: No module named 'torch'`，`card_template/eval/leaderboard.py` 触发），
  纯环境问题，与本包无关（本包不需要 GPU）。全套最终结果见文末。

---

## 任务 1：跨 run 证伪杠杆台账（falsified-lever ledger）

**Schema**（`hw_kb/falsified_ledger.jsonl`，一行一记录，键序固定）：

```json
{"op_family": "<ctx.op>", "arch": "<ctx.kb_arch>", "lever": "<lever>",
 "class": "harmful|coupled-negative|faded|null|noop",
 "delta_pct": <add 腿读数，landed 行取 -remove；无读数为 null>,
 "condition": "<一句话成立条件（含 noise floor / 决策来源 / re-price span）>",
 "run": "<workspace 目录名 = run id>", "date": "<UTC ISO 日期>"}
```

**写入点**（同 run 幂等：`(run, lever)` 去重；evidence 目录写法未动，ledger 是成功持久化后的旁路）：

- `driver.py:4952` `_persist_lever_table` —— 函数内仅加一行旁路调用（:4975
  `_append_falsified_ledger(ctx, table, best_ref=..., job=..., reprice=...)`，位于原
  `else: _bank_evidence(ctx, "lever prices")` 分支内）。
- ledger 实现（`_persist_lever_table` 之后新加的独立小节，共 4 个函数）：
  - `driver.py:4999` `_falsified_ledger_path` —— `Path(ctx.kb_dir).parent / "falsified_ledger.jsonl"`
    （生产即 `<repo>/hw_kb/falsified_ledger.jsonl`；位于两棵语言树之上，不在任何学生读集内）。
  - `driver.py:5015` `_falsified_ledger_row` —— 行级判定：定价工具自身 class 阶梯
    （harmful / null / noop）直取；`reprice=True` 的表按 staleness 判词覆盖
    （faded；reversed→harmful，翻转写入 condition；对照"剔除本次已入库表之后"的最新历史读数，
    历史 job 为字符串，比较走 `str()`）；`coupled` 且 add<0 → coupled-negative
    （即 diagnose_prompt 阅读规则"负 add 但 all-on 缺不得"）。非负读数（independent/coupled/
    dependent/shape_split/broken/unstacked/unmeasured）不入账。
  - `driver.py:5066` `_append_falsified_ledger` —— 读旧账建 `(run, lever)` 去重集、逐行判定、
    追加；任何 I/O/解析异常只记日志吞掉（定价不得感知它的存在）。
- 空文件 `hw_kb/falsified_ledger.jsonl`（0 字节）已入 git（知识工件，与 .artifacts 相区分）。

**消费点**（meta 诊断注入区，该区域归本包）：

- `driver.py:6468`（`run_meta_step` 内、`diagnose_extra` 初建处）：
  `diagnose_extra += _falsified_ledger_block(ctx)`。
- `driver.py:5112` `_falsified_ledger_block` —— 按 `arch == ctx.kb_arch` 过滤；
  排序 same-op → generic → 其他 op（组内保持文件序）；最多 20 行
  （`_LEDGER_MAX_LINES = 20`）；行格式 `LEDGER <lever> <class> Δ<delta>% <condition>`
  （delta 为空打 `Δn/a%`）；文件缺失/为空 → 整节省略。小节自述定位：
  **PRIOR INFORMATION，只喂 teacher 的设计/定价决策，不进任何 gate/验收线；
  当前 best 上的新价格表永远压过台账（a price is a reading on one tree）**。

**测试**：`tests/test_falsified_ledger.py`（11 例）——死杠杆入账与绿表不入账、
(run, lever) 去重、coupled-negative、reprice 的 faded/reversed 判词与 span 文本、
landed 行按 -remove 记账、损坏行不炸且不重置去重、注入块有/无台账两态、arch 过滤 +
同 op 优先 + 20 行上限、以及两例经 `run_meta_step` 的端到端 prompt 断言
（有台账 → prompt 含 `LEDGER …` 与 PRIOR INFORMATION；无台账 → 无该节）。

## 任务 2：probe / 定价配方归档

**归档落点**（先定位后动手：kb harvest 的执行函数是 `run_kb_harvest`）：

- `driver.py:3723` `_archive_probe_recipes`（新函数）—— 把本次 run `notes/` 下的
  `*.py` / `*.json` 配方（各通道 PROBE_REQUEST 脚本、`levers.json`、ablation JSON spec）
  复制到 run 的 `<evidence>/probes/`，镜像 notes 相对路径；**排除驱动播种的工具箱**
  （`RECON_SEEDED_MODULES` / `PRICING_SEEDED_MODULES` 模块名 + `_ORCH_CENSUS_RUNNER`，
  每次 run 重新播种、无需保留）；其余后缀（如 `.md` 草稿）不动。选择规则是文件式的、
  与通道无关（学生 probe 战 / meta verdict probe / 认证 probe 重物化的文件都在 notes/ 下），
  driver 重启也不丢。best-effort：任何异常只记日志，永不上抛。
- `driver.py:3801` `run_kb_harvest`（:3770 起）—— 在 axis 账单之后、
  `if not debt and not axis_debt: return []` **之前**调用（有账无账都归档，归档失败降级为空表）；
  归档非空时向 teacher 注入第三段 prompt 小节（:3814 起）：绝对路径仅用于本轮对照，
  条目内一律写 run 相对形式 `evidence/probes/<file>`（绝对路径/ run id 会被 kb_lint 当叙事拦下）。

**约定面**：

- `hw_kb/cutedsl/TEMPLATE.md` 与 `hw_kb/cuda/TEMPLATE.md`（Sources 节）新增可选字段：
  `- Probe archive: evidence/probes/<file> …`，缺失时必须显式写
  `Probe archive: probe not archived`（"行必填、值可以说没有"）。
- `meta/kb_harvest_prompt.md` 与 `meta/cuda/kb_harvest_prompt.md`：
  - Inputs 节声明第三段"probe recipe archive"小节（归档先于收割、notes/ 会被下次认证覆盖）；
  - "What you deliver" 第 1 条新增硬性要求：本轮写/改的每个条目必须带 `Probe archive:` 行
    （引归档副本 `evidence/probes/<file>`，或显式 `probe not archived`；漏行是唯一禁止项）。
- `docs/kb-provenance/README.md`："What writes here" 之下新增
  `### The probe recipe archive (run-side convention)` 小节（归档约定、run id 只进本目录不进条目）；
  顶部"本 release 不含历史 probe"的原文未动。

**测试**：`tests/test_probe_archive.py`（9 例）——归档清单与目录布局（含子目录镜像、
工具箱排除、非配方后缀排除）、无 notes/ 或无 evidence 通道返回空、同 run 重复归档幂等覆盖、
经 `run_kb_harvest` 的端到端（归档文件 + prompt 含 `evidence/probes/hard.py`）、
无账单也归档且不产生 teacher 轮、两树 TEMPLATE 字段、两树 prompt 渲染冒烟、
provenance README 约定 + 原文保留、`hw_kb` 全树 lint 仍干净。

## 任务 3：KB 待认证候选队列

- `hw_kb/CANDIDATES.md`（新文件，进 git）——表列
  `Candidate entry | Tree(s) | Priority | Basis | Status`，初始 7 行（全部 TODO）：
  1. `tcgen05_blockscaled_mma_sf_tmem` 的 fp4(e2m1)/CTA-pair 认证 → `cuda/sm100`（High；对照已认证的
     `cutedsl/sm100` 同名条目转写 + 重认证，成本减半；cuda 树 index 现标 "fp4 / pair uncertified"）
  2. `tma_multicast`（cluster TMA multicast operand load）→ `cutedsl/sm100`、`cuda/sm100`（High；
     sm90 只有 draft（GPU probe pending），sm100 两树只有 non-multicast pair-load 条目）
  3. Grid 级 split-K / stream-K 工作分解 → 三树（High；现仅 warp 级
     `cutedsl/sm90/mma_sync_subtile_broadcast_splitk.md` 一条；DeepGEMM/CUTLASS stream-K 看家本领）
  4. Wave quantization / 尾波（persistent 调度 + tile 高度对 132/148 SM 整除性）→ 三树（Medium）
  5. L2 access policy window / persisting set-aside → `cuda/sm100`（Medium；现有条目仅
     inline-PTX `.L2::cache_hint`；cutedsl 是否有 API 待查后再入队）
  6. PDL launch 侧 → `cutedsl/sm100`、`cutedsl/sm90`（Medium；wait 侧已有、sm90 index 行是 TODO、
     cuda 树 `pdl_griddepcontrol_launch.md` 可作转写种子）
  7. 2:4 结构化稀疏 MMA → 按需、任一树（Low；全树 grep 零命中）
- 队列纪律（写入文件头并由单测钉死）：**排队 ≠ 认证**；行内只有"知识缺在哪、为何值得 probe"，
  不含任何机制结论、不带任何 ✅/PROBE PASS 认证标记；进树仍走现有认证流程。
- 收割触发器（对齐 trigger one/two 的写法，两语言树各一份）：
  `meta/kb_harvest_prompt.md` / `meta/cuda/kb_harvest_prompt.md` 头部新增
  **"> Trigger three (queue-hit)"**：本轮落地/认证的机制命中 `{KB_DIR}/../CANDIDATES.md` 某行时，
  把该行 Status 改 `DONE` 并在同一行链到新条目；仅在本轮真的写出了条目时才翻行
  （`NO_RECIPE` 轮不翻）；队列不得外泄机制结论——认证读数落在 KB 条目里。

**测试**：`tests/test_kb_candidates.py`（6 例）——表头五列与 7 行初始 TODO、树名合法性、
basis 引用的既有条目路径逐一存在、无认证标记 + `hw_kb` 全树 lint 干净、文件头自述契约、
两树 prompt 的 trigger three 文本 + 渲染冒烟。

## 红线对照

1. 认证标准一字未降：CANDIDATES 行全部 TODO、无机制结论（测试钉死）；ledger/归档/队列均不参与任何 gate。
2. ledger 只进 teacher 的诊断注入（prior information），无任何 gate 读它（测试钉死两态与措辞）。
3. lever pricing 执行与计价行为未动：`_persist_lever_table` 仅在成功持久化后追加旁路；
   相关旧测试（test_price_staleness / test_loop_driver / test_lever_pricing_run / test_meta_shapes）全绿。
4. SOL-ExecBench token 只在 `ForgeKernel/local.env`，未进任何代码/测试/提交。

## 全套测试结果（提交后最终运行）

`~/venvs/forgekernel/bin/python -m pytest tests/ -q`：
**2081 passed, 0 failed, 0 skipped（全绿）**。

说明：基线记录为 1966 passed / 15 skipped / 2 failed，2 个失败均为环境性懒加载
`import torch`（`card_template/eval/leaderboard.py` 触发，本包无关）。最终运行前
`~/venvs/forgekernel` 里被并行会话装入了 `torch 2.15.0+cpu`，于是那 2 个失败转绿、
15 个 torch 条件 skip 全部实际执行——收集数从 ~1983 涨到 2081（+26 例为本包新增的
三个测试文件，其余 ~72 例为条件用例解禁），且无一失败。与本包改动相关的子集
（test_falsified_ledger / test_probe_archive / test_kb_candidates = 27 例）单独跑亦全绿。

## 与并行包的边界确认

仅改/增：`hw_kb/falsified_ledger.jsonl`、`hw_kb/CANDIDATES.md`、`hw_kb/{cutedsl,cuda}/TEMPLATE.md`、
`docs/kb-provenance/README.md`、`meta/{,cuda/}kb_harvest_prompt.md`、`driver.py` 的
`_persist_lever_table`+新增 ledger 小节（4952–5155 一带）与 `run_meta_step` 注入区一行（:6468）、
kb harvest 执行函数 `run_kb_harvest`（:3770）+ 新函数 `_archive_probe_recipes`（:3723）、`tests/`。
未触碰：`inner/ncu.py`、`loop/parse.py`、`IterationRecord` schema、pricing 核心执行区 4557–4801、
probe 分支 2157–2214、`run_two_layer` 的 meta/预算段与蒸馏入口、`timing.py`、`diagnose_prompt.md`。

**工作树说明**：共享 checkout `/home/zcz/forgeKernelOptimize/ForgeKernel` 中途被并行会话（WP1 等）
占用并提交，本包自 `git worktree add ../wp3-knowledge feat/wp3-knowledge` 起全部工作在独立工作树
`/home/zcz/forgeKernelOptimize/wp3-knowledge` 进行；分支 `feat/wp3-knowledge` 停在本地仓库，
未合并 main、未 push 远端。
