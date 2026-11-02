# CHANGES-WP5 — 预算与停止治理

分支 `feat/wp5-governance`（基于 main @ 5030503），每任务独立 commit：

| commit | 任务 |
|---|---|
| `b741c75` | 任务 1：格式失误不计 meta 预算 |
| `b5c34cc` | 任务 2：meta 预算与剩余墙钟联动 |
| `5b0ab1d` | 任务 3：headroom 进停止准则 + 1 轮预算扩展 |
| `c4d4c48` | 任务 4：BLOCKED 时解除 reopen 封顶 |
| `0c11ade` | 任务 5：`ncu_on_revert` 配置项 |

验收证据：`pytest tests/ -q` → **2076 passed, 0 failed**（分支顶端；共享 venv 中途被并行会话装了 torch，基线时 2 个 torch 环境失败现已转绿；driver 相关 4 个测试文件在装 torch 前后均全绿）。本机 GPU 不代表目标硬件，验收以 pytest + 逻辑覆盖为准。

---

## 1. 新配置键表

所有键的读取入口：`driver._wp5_apply_cfg(ctx)`（`run_two_layer` 入口调用一次）。**卡层优先**：`operators/<op>/loop.toml`（所有 `[loop]` 键的既有 home，自由格式）覆盖 `global_config/loop.toml`（底层）。显式赋值（测试/调用方）永不被 toml 覆盖。任何读失败 / 值非法 / 键缺失 → 回退默认（= 红线：回退现状）。

| 键 | 段 | 默认 | 语义 | 回退 |
|---|---|---|---|---|
| `meta_format_retry` | `[loop]` | `3` | 同一 meta 条目内连续 FORMAT-only 反弹的前 N-1 次免费重试（不计 meta_steps）；第 N 次恢复计费并停止重试 | `1`（=每次反弹都计费，现状）；`<=0` 按 1 |
| `reopen_blocked_uncap` | `[loop]` | `1` | 每个 streak regime（一次 KEEP_PATCH 归零为新 regime）内，BLOCKED 解封结构性 reopen 的次数上限 | `0` = 关闭解封（现状） |
| `ncu_on_revert` | `[loop]` | `True` | `False` 时 correctness-FAIL（规则上必 REVERT）的迭代跳过 NCU job；KEEP/新 best 永不跳 | `True` = 现状（每次提交迭代都测 NCU） |
| `meta_hours_per_round` | `[budget]` | `1.5` | 一轮 meta（诊断+补丁+其内层段）的计划墙钟成本；入口计算 `effective_max_meta` 时用 | `<=0` = 不上调（cap = max_meta，现状） |
| `meta_ext_cap` | `[budget]` | `6` | 有效上限的绝对封顶；与任务 3 的 headroom 扩展共用记账（`meta_extra_used ≤ meta_ext_cap`） | `0` = 禁止一切上调/扩展 |
| `meta_ext_on_headroom` | `[budget]` | `True` | meta 预算耗尽且 `measured_headroom > 20%` 时追加 **1** 轮 meta（每 run 至多一次） | `False` = 不追加；floor_opt 缺失绝不追加 |

**配置层位说明（重要）**：新键放卡层 `operators/<op>/loop.toml`。`global_config/loop.toml` 被 `forgekernel/task.py` 按**闭合词表**校验（`_KEYS`，实测：加入 `[loop]` 表或未知 `[budget]` 键 → TaskError），本包围栏不许改 `task.py`，故 global 层键今日不可生效。`_wp5_cfg_sections` 已实现 global 层读取——集成步若扩展 `task.py._KEYS`，键上移 global 层即自动生效，无需改 driver。

## 2. 错误分类清单（任务 1）

判定落点：`driver._substantive_violations()` + `_WP5_FORMAT_VIOLATION_RES`（消费 `_patch_landing_violations` 与 `_validate_milestone_file`→`milestone.validate_milestone` 返回的**消息串**）。

**FORMAT（免费反弹，前 N-1 次）**——机械性 schema 失败，re-prompt 可修：
1. `missing title header (...)`（缺标题）
2. `missing required section: <sec>`（缺节）
3. `sections out of order (...)`（节序错）
4. `free-optimization section present (...)`（出现禁用节）
5. `§3 has no parseable decisions (...)`（§3 解析失败）
6. `§3 <D id> missing five-tuple keys: [...]`（决策块缺必需键）
7. `§3 [SYNC|GATHER|PHASE: <val>] is outside the machine vocabulary (...)`（**非法枚举值**）
8. `no decision id (D<x>.<y>) found`（解析失败）
9. `no [MEASURED]/[ESTIMATE] evidence marker found (...)`（全局缺 schema 元素）
10. `milestone file unreadable (...)`（文件级读取失败）
11. `The patch never reached the file: ...`（补丁回合没碰文件——报告了但没写）
12. `` `changed_entries` declares <id>, but the milestone file does not contain it ``（报告与文件不符）

**SUBSTANTIVE（立即计费，现状流程）**：悬空依赖（`depends on undefined decision`）、缺 Mechanism-verification 行、[SYNC]/[GATHER]/[PHASE] 判定无 [VERIFY:] 门线、prefetched 无 issue order、ping-pong 无 phase contract / 无数值 blocking count、行级性能数字无 provenance、operand-source 违规、gate hygiene、downgrade schedule。**未知消息一律按 substantive（fail-closed 到旧计费）**。混合列表（含任一 substantive）按 substantive 计。

语义对齐：M0 audit 路径的 malformed patch（driver 原 :11549-11560）不计数、无补丁续跑——本包把同款处理带进 plateau 路径。计费恢复（第 N 次反弹）后：预算尽 → `meta_budget`；尚有预算 → 按该分支形状**无补丁续跑内层段**（M0 audit 形状），绝不第四次反弹。

## 3. headroom 计算与数据来源（任务 3）

- `measured_headroom = (last_best_ms − floor_opt_ms) / last_best_ms`；`last_best_ms` = 停止判定时 gate window 下的 best（`_best_ms(_on_cluster(records, ctx), ctx.gate_window)`，与现状同源）。
- `floor_opt_ms` **来源唯一**：meta 诊断输出里 teacher 写的 `reachability:` 行（diagnose_prompt §1c 本就强制要求），用机器门自己的解析器复用解析（`_parse_reachability` / `_reach_quantity`，即 `_infeasible_refusal` :3120-3126 用的那套），**不新造来源**。由 `run_two_layer` 在每次 `run_meta_step` 返回后经 `_wp5_note_reachability` 线程到 `ctx.wp5_reach_floor_ms/target_ms`（floor/target 粘滞保留 teacher 最后写下的值；`wp5_reach_blocked` 则要求**当前**诊断验证，见 §4）。
- 扩展规则：meta 预算耗尽 且 headroom > 20%（`_WP5_HEADROOM_EXT_MARGIN`，BLOCKED margin 的镜像语义）→ 追加恰 1 轮（`meta_cap = meta_steps + 1`，每 run 一次）；追加轮消耗与普通轮相同、经 `meta_extra_used = max(0, meta_steps − max_meta)` 进共享账本（≤ `meta_ext_cap`，与任务 2 上调共用）；追加轮后无论结果按现有流程收尾。
- 追加的三重保险：floor 无法解析 → 不追加；剩余墙钟 < `meta_hours_per_round` → 不追加（不为凑轮数硬塞）；共享账本用尽 → 不追加。
- 辅助旁证：WP2 的 `compute_sol_pct`/`memory_sol_pct` 从最新携带它们的 KEEP 行读取（`getattr`，**字段缺失 → 不出现，绝不阻塞**），两值均 <70 时在 stop 摘要写 `sol_corroborates_headroom: true`。
- `meta_budget` stop 记录新增字段：`floor_opt_ms` / `measured_headroom` / `meta_extra_used`（+上述 SOL 旁证，若有）——主门与三条 retry 路径的 meta_budget 返回统一携带。

## 4. 解封状态机（文字版，任务 4）

状态：`streak`（`_fruitless_reopen_streak`，重启安全——读 meta records）、`blocked`（当前诊断验证的 BLOCKED：`-> BLOCKED` 且 `floor_opt > max(行内 target, _resolve_target(ctx)) × 1.20`，逐诊断刷新，无行/未验证即清除）、`used`（本 regime 已借次数，**内存态**——driver 重启会重新获得 1 次借支，cap 本身不受影响）。

每次 `run_meta_step` 的封顶判定（原 :6217-6222）：
1. `streak == 0` → `used ← 0`（KEEP_PATCH 落地即新 regime；真实流程中判定紧邻下一 meta 步，零点必被观察到）。
2. 若 `0 < reopen_max_fruitless ≤ streak` 且 `blocked` 且 `used < reopen_blocked_uncap` → **解封**：`used += 1`，`streak ← 0`，log 记录借支；`reopens_capped` 为假 → 封顶提示块 / R1 bounce / R1 拒绝 / landed-patch 撤回全部不再触发。
3. 价格门照常：无 `reopen_price` → 先 UNPRICED_REOPEN 弹回一次、再拒绝（现状语义不变）。
4. 借来的 reopen 若 fruitless 关闭 → 下一判定 raw streak = 3 ≥ cap，`used` 已达上限 → **恢复封顶**（旧账不抹）；若 KEEP_PATCH → streak 归零、账本重置、一切回默认语义。
5. 非 BLOCKED / `reopen_blocked_uncap = 0` → 行为与现状完全一致。

## 5. 与 WP4 / WP6 的接缝

- **WP4（lever pricing / `_confirm_suspicious_watermark` / KEEP 段）**：零接触。任务 4 的落点只在封顶计算块（原 :6217-6222）——它在价格门**之前**、只影响"封顶是否生效"；`reopen_requires_price`/`_verify_reopen_price`/定价表一概未动（解封后的 reopen 仍须附价）。`loop.toml` 我未加任何 lever 相关键；driver 默认参数区我的追加块以 `# --- WP5 ---` 界定，不与 WP4 的追加相交。
- **WP6（蒸馏入口 11450-11530）**：零接触。任务 2/3 的入口计算在 `run_two_layer` 签名之后、蒸馏入口之后；`_needs_authoring`/`run_distill_step` 一概未动。stop 摘要新增字段只在 `meta_budget` 返回上，蒸馏路径的 `no_ref_baseline`/`distill_malformed` 返回不变。
- **WP2（`inner/ncu.py`、SOL 读数）**：只读旁证（`getattr(rec, "compute_sol_pct", None)` 模式），WP2 合入后字段即自动出现在 stop 摘要；缺失时降级为只用 floor_opt，无阻塞。任务 5 的开关在 `_ncu_profiler` 调用点（:2387 一带）外侧包一层判断，未进入 `inner/ncu.py`。
- **共享热点**：driver 默认参数区（`kb_tripwire_rounds` 之后）与 `_ctx` fixture 的追加均以注释界定为 WP5 块；`global_config/loop.toml`、`meta/diagnose_prompt.md`、`timing.py` 未动。

## 6. 集成注意事项

⚠️ **杂散 commit**：`feat/wp1-measurement` 上有一个 `a749131`（"wp5/task1: format-only patch bounces..."）——并行会话轮换共享 checkout 时，我的 task-1 提交误落在其分支顶端，且**混入了 WP1 当时的在途编辑**（resolvable_diff 等）。集成的正确版本是 `feat/wp5-governance` 上的 `b741c75`（纯净、369 行）；请在集成分支合并时**丢弃 wp1 分支上的 a749131**（例如 wp1 rebase 到其 4f40c1b，或集成时 cherry-pick 选择跳过）。

其余：未改 correctness/precision gate、plateau 判定、`target.ratio` 锚定；`loop.toml` 的 `max_meta` 语义不变（下限）；first_green 强制 M0 audit 的预算跳过仍读 `max_meta`（围栏外，语义保守）。SOL-ExecBench token 未出现在任何代码/测试/提交中。
