# CHANGES-WP24 — 去榜单化与框架纠偏（Wave 3.0 前置包）

分支 `feat/wp24-deboard`（自 origin/main `359696b` 开出）。契约 C18 + 合并裁决 59 §三 WP24 节 + 终审 80 §九/§十。
总量：72 文件，+3576 / −9064（`git diff --stat 359696b..HEAD`）。

## 0. 前置就绪门（记录）

- 工作区：`git clone zxtune-integration wk3-wp24 && git checkout -b feat/wp24-deboard origin/main`（clone 默认检出 integrated-v2，已显式从 origin/main 开分支）✓
- 基线：`~/venvs/forgekernel/bin/python -m pytest tests/ -q` → **2437 passed in 293.97s**（与发包指令预期 ≈2437 一致）✓
- 必读：59 §三 WP24 + C18；80 §九/§十 ✓

## 1. 与发包指令的两处工程解释（实现裁决记录）

1. **评测模板"整文件删除"→"git mv + 剥皮"**：`card_template/eval/leaderboard.py`（3020 行）并非纯 board 件——它是**普通任务的判题/计时引擎**（`taskcard.write_card` 把它复制进每张普通任务卡，生成的 `eval/bench.py` 调它的 `run_leaderboard_bench`；`card_problem` 也写 `eval/leaderboard/problem.json`）。整删即断普通任务路径，与本包"普通任务 dry-run 走通"验收门矛盾。处置：`git mv` 为 **`benchkit.py`**，删除全部 FIB/SOL 协议代码（两套件判据/正确性协议/输入生成 FIB 分支/CUPTI span 协议计时/`sol_score`/`board_block`/上游计时器/`check_only`/`_run_group` 等），保留 forgekernel 任务套件、timing.py 窗口门、WP9 fresh_recheck、精度车道、密封参考加载、CUDA 构建、resusage、fence、`FK_LB_WORKLOAD_SCOPE`（Wave-3 锚点名，未改）。文件名 "leaderboard" 零残留达标；文件名本身即"整文件删除"的对象——模块名不复存在。
2. **冷 L2 轴随板删除**：合并裁决 59 明文"板卡双 gate / 冷 L2 轴作废……冷 L2 如需作为通用严格测量轴，归 WP22 评审时重议"。故保留的是**锁频机制**（LB_CLOCK_STATE / gpu_entry 锁频纪律，目标改 `[protocol]` 配置）与 timing.py 的 `measure_paired`/窗口协议（逐字节未动，diff 验证见 §5）；引擎侧的 cold-L2 单调用 CUPTI span 协议（board 门读数）随板删除。

## 2. 任务 1（P0）：删除榜单面向机器

### 文件级删除（git rm）
- `docs/leaderboards.md`、`docs/upstream-readme.md`、`docs/upstream-readme_zh.md`、`docs/assets/{figure1-self-evolution,figure2-architecture}.png`（仅被 upstream-readme 引用）
- `tests/test_leaderboard.py`（762 行，board 卡单元测试）、`tests/test_leaderboard_runtime.py`（963 行，FIB/SOL 运行时测试；任务套件运行时由 test_task_suite/multi_workload/fresh_recheck/precision 系列覆盖，通用件救入新 `test_benchkit_runtime.py`）、`tests/test_leaderboard_cuda_student.py`（board CUDA 导出面；CUDA fence/构建/lineinfo 用例移入 test_cuda_mode（本地 card_lb fixture）与 test_binref）

### 函数/字段级删除（改名后的两模块 + 调用方）
- **cardkit.py（原 leaderboard.py，1746→530 行）**：`SUITE_FIB/SUITE_SOL/SUITES/UPSTREAM/DATASETS/FIB_UNSUPPORTED`、`_LFS_POINTER`、`SOL_BOARD_API/SOL_BOARD_CLOCK/SUITE_CLOCK`（榜单钉死常数 SM 1500/3996 及出处注释）、`Problem` dataclass、数据集读取器（`_pick_workload(s)/_group_entry/_check_entries/workload_rows/_blob_target/_collect_blobs/fib_problem/_attach_fib_baseline/sol_problem/_NO_BOARD/_git_head/list_workloads`）、board 拉取（`_http_json/fetch_sol_board/load_sol_board/sol_score/time_for_score/attach_sol_board/sol_board_row`）、卡片渲染（`problem_json/META_WORKLOADS/meta_workload_names/_AXIS_OPS/_tensor_elements/describe_workloads/_signature/_spec_line/_var_note/render_kernel_stub(_cuda)/render_window_toml/render_loop_toml_example/_geomean/_group_board_example/render_readme_section/write_card/blob_bytes`）、导出面（`_EXPORT_LANG/infer_language/_CPP_FN/cuda_signature_findings/signature_findings/export_solution/_loop_language/main`（`forge <op> leaderboard show|export` CLI）、`SEALED_KINDS/BLOB_WARN_BYTES`（无残余消费者）
- **benchkit.py（原 card_template/eval/leaderboard.py，3020→1748 行）**：FIB/SOL 常数与 config（`SUITE_*/SUITES/UPSTREAM/FIB_*/SOL_ROUNDS/SOL_SEED/SOL_WARMUP/SOL_ITERS/SOL_TOLERANCE_DEFAULTS/SOL_POOL_ALIGNMENT/SOL_MAX_SHIFT_BYTES/BASIS_CUPTI/BASIS_COLD_EVENTS/TIMERS/_recheck_seed`）、判据（`fib_config/fib_error_stats/fib_judge/sol_tolerance/sol_sanity/sol_error_stats`）、正确性协议（`fib_correctness/_lazy_output`（随后按需恢复给 task_correctness）/`sol_correctness`）、时钟与协议计时（`_flush_buffer/_reset_persisting_l2/_Shifted/_kernel_events/_profile/_select_sequence/time_profiler_spans/_cupti_*/time_cupti_spans/time_cold_events/_cupti_python/_upstream_timer/choose_timer/PROFILER_NOTE/_time_once/_protocol/measure/UNSTABLE_PAIR_RATIO/unstable_sets`）、board JSON（`sol_score/board_block/performance_block`）、`check_views/judge_check_only`（correctness-only workloads）、`_run_group/_ref_alone`（board 分支）、gen_inputs 的 FIB 分支与 custom-inputs 分支（SOL-only）、normalize_outputs 的 dict 分支、fresh_recheck 的 SOL 双子分支、`check_only/sol_board` 在 workload_views/apply_workload_scope/group_verdict/precision_result 中的残余
- **task.py**：`MODE_LEADERBOARD/LEADERBOARD_PROBLEM/Leaderboard/BOARD_SUITES/_read_leaderboard/_board_shapes/Shape.board/TIMING_PROTOCOL_*/Timing/Task.timing/_read_timing`；`_KEYS` 词汇表收缩（删 `timing`、`leaderboard` 表）
- **taskcard.py**：`BOARD_DIR/TIMING_PROTOCOL_BOARD/BOARD_CLOCK_SM_MHZ/BOARD_CLOCK_MEM_MHZ/BOARD_BENCH_TEMPLATE/install_board_trees/write_board_loop_toml/_toml_value`；card_problem 的 `suite/timing_protocol` 键
- **scaffold.py**：`forge new --leaderboard` 参数组全部分支 + `_leaderboard_problem/_leaderboard_rows`（`--binary-ref/--sealed-ref` 随 `--leaderboard` 分支不可达，一并删除）
- **launch.py**：`BOARD_TASK_TOML/new_board_task/cmd_init --leaderboard 分支/deny_paths 的 leaderboard 块/_drive_board/describe 的 board 分支/cmd_export 的 board 导出分支/MODE_LEADERBOARD`
- **__main__.py**：`forge <op> leaderboard show|export` 派发与帮助行
- **timing.py（card_template/eval，仅 board 分支，红线 1）**：WP1 副读数窗四处——`paired_performance` 的 board 块、`measure_windows_workloads` 的 `SECONDARY_PROTOCOL=="board"` 挂点、`workloads_performance` 的 board 聚合、文件尾 WP1 段（`SECONDARY_PROTOCOL/BOARD_WARMUP/BOARD_ITERS/_flush_buffer/measure_board`）。**`measure_paired` 及窗口协议逐字节未动**（见 §5 diff 验证）
- **window_contract.py**：`LEADERBOARD_CALL` → `BENCH_CALL = "run_bench"`；board 协议注释删除；problem.json 路径改 `eval/problem/`
- **路径更名（卡片数据布局）**：`eval/leaderboard/` → **`eval/problem/`**；`eval/leaderboard/problem.json` → **`eval/problem/problem.json`**（cardkit.PROBLEM_DIR/PROBLEM_FILE、taskcard、precision_gate.PROBLEM_FILE、meta_shapes、analytic_floor、goal_ref、cli._card_kernel_file、driver._sass_precision/_analytic_floor_ms、window_contract 同步）；eval JSON 性能块键 `leaderboard` → **`bench`**、`leaderboard_timer` → **`bench_timer`**；日志前缀 `[leaderboard]` → `[bench]`
- **锁频拒绝 marker 去品牌（两端同步）**：`CLOCK_LOCK_REFUSED_MARKER` = `"[gpu_entry.sh] REFUSED: this card times under a configured clock lock"`（cardkit GPU_ENTRY_TEMPLATE 与 submit/jobs.py 同步；同步性有回归测试钉住）

### 评测路径零网络外发
全树唯一 http 客户端 `fetch_sol_board/_http_json` 已删；`urllib.request`/`urlopen` 零命中（仅存 `urllib.parse.unquote` 的本地路径解码，非网络 I/O）。agent CLI 的 LLM 后端与教师调研 skill 保留（与榜单无关）。

## 3. 任务 2（P0）：通用基建保留并更名

- `git mv src/forgekernel/leaderboard.py src/forgekernel/cardkit.py`；六处调用方更新为 cardkit（`launch.py/taskcard.py/__main__.py`）或删除（scaffold/board 导出），binref/sealref 工具的卡片侧 import 改 `benchkit`
- 保留并去品牌：卡片组装模板（`BENCH_TEMPLATE/NCU_TEMPLATE/GPU_ENTRY_TEMPLATE/_WP11_ALLNCU_CASE/_WP11_CACHE_BLOCK` 即 `fk_build_cache_*`）、NCU 指标模板与 arch 自适应过滤、`is_cuda_kernel_file/cuda_kernel_flags/_cpp_param`、CUDA 构建惯例与 fence、resusage、job 装配/submit 管线、sealref/binref 通用黑盒密封
- **锁频目标配置化**：task.toml 新增 `[protocol]`（`sm_clock_mhz/mem_clock_mhz`，正整数，缺省 = 不锁频，行为与改造前 unlocked 路径一致）。task.py `_read_protocol`/`Protocol` dataclass、`dump_task_toml` 表序列更新；taskcard.write_card 由 `task.protocol` 渲染 gpu_entry（锁频时 `require_lock=1`：锁不住即拒绝除 help 外全部任务——拒绝语义逐字节保留，仅文案去品牌）；`render_gpu_entry` 改为显式参数驱动（clock/mem_clock/require_lock/clock_note），`SUITE_CLOCK`/`SOL_BOARD_CLOCK` 删除
- 全局 grep 验收：src/、card_template/（=src 子树）、skills/、scripts/、tests/ 中 "leaderboard" **零残留**（见 §5 粘贴）；回归测试 `test_deboard_regression.py` 固化

## 4. 任务 3（P0）：文档、定位与许可合规

- 删除 `docs/leaderboards.md`、`docs/upstream-readme.md/_zh.md`、`docs/assets/` 两图；`docs/design/v1-launch.md` 顶部加"历史设计存档（WP24 注记）"，正文未改
- `README.md` 重定位："纯粹的 AI 优化算子框架"（五层机制能力优先 + 任务供给 + 零榜单关联声明 + `[protocol]` 快速开始）；**OpenBMB/ForgeKernel 上游致谢与 Apache-2.0 声明原样保留**；WP1–12 改进表保留（WP1 行加"board 协议副读数窗已随 WP24 切除"备注）；删对已删文档的引用；`kernels/` 原样保留并在文档索引注明"先验/教材资产，出处如实保留"
- **NOTICE 处置（逐文件法律自查，结论）**：
  | 文件 | 改编源 | WP24 后 |
  |---|---|---|
  | card_template/eval/leaderboard.py（旧，NOTICE 声明载体） | fib utils/default evaluator + sol io/correctness/timing/eval_driver | **改编面全部删除**（判据/协议/计时/FIB 生成器/CUPTI collector/ShiftingBuffer/sol_score） |
  | card_template/eval/benchkit.py（新） | sol core/bench/io.py 的 `_rand_tensor/_cast_to_fp4x2/_generate_heuristic_tensor`（含 `_is_*` 名字谓词） | **残余保留**（服务任务套件的精度压力采样与 recon 工具探针输入）→ NOTICE 缩减保留，逐文件映射列出 |
  | card_template/eval/timing.py | 无（本仓原创窗口引擎；WP1 board 副读数窗已删） | 无残余 |
  | card_template/eval/precision.py | 无（WP7 原创；K_RATIO 校准数据出处为事实陈述非代码派生，注释已改写） | 无残余 |
  | card_template/eval/closed_ref.py | solution JSON 格式惯例（格式非代码） | 无 NOTICE 残余（注释去品牌） |
  | binref.py / binref_tool.py | sol solution 格式的构建惯例（compile_options/顶层源/CUTLASS 路径） | 保守起见纳入缩减 NOTICE 映射 |
  | loop/analytic_floor.py | sol shapes.py 表达式文法 | 保守起见纳入缩减 NOTICE 映射 |
  - **结论**：有残余 → **保留缩减版 NOTICE**（仅列保留文件及其来源 + SOL-ExecBench NOTICE 原文；FlashInfer-Bench 段随其改编代码一并移除）。"拿不准从保守"落实为：binref/analytic_floor 的惯例级派生虽非原 NOTICE 范围，一并列入映射。

## 5. 任务 4（P0）：回归与零残留门

### 全量回归
- 前：**2437 passed**（基线 `359696b`）；后：**2332 passed in 215.78s**（全绿）。净差 105 = 删除的 board/suite 测试 − 新增/改造测试（新 `test_deboard_regression.py` 10 项 + `test_benchkit_runtime.py` 9 项 + binref/sealref 重写为普通任务 fixture）。
- **`measure_paired` 红线验证**：`git diff 359696b..HEAD -- src/forgekernel/card_template/eval/timing.py` 仅含 §2 所列四处 board 分支删除；measure_paired/measure_windows/burn-in/镜像顺序函数体零 diff（timing.py 净 −93 行全部在列位置）。
- `precision.py` 判定算式与常数（K_RATIO/FLOOR_ULPS/FLOOR_REL/stress profiles）零 diff，仅 docstring 出处措辞改写。
- `driver.py` loop 语义面最小触碰：仅注释/文案/路径常量与 `_too_few_workloads` 的谓词改名（`is_leaderboard_card`→`is_generated_card`，语义同：有 problem.json 即生成卡）；无逻辑重构。

### 零残留 grep（粘贴）
```
$ git grep -i -l -E "leaderboard" -- src skills scripts tests
（无输出）
$ git grep -i -l -E "sol-execbench|flashinfer-bench|sol_board|fetch_sol_board|SOL_BOARD" -- src skills scripts tests
scripts/floor_backtest.json
tests/test_deboard_regression.py
```
两处均为已文档化豁免：`floor_backtest.json` 是 kernels/ 语料的诚实出处 fixture（WP9 回测清单，范围围栏保留件）；`test_deboard_regression.py` 是验收门自身（它引用被禁模式作为断言）。kernels/、CHANGES 存档、docs/ 在范围围栏之外（kernels 原样保留；docs/ 三删一注记已执行）。

### test_deboard_regression.py（10 项）
① 双 grep 零残留（含豁免白名单）；② cardkit/consumer import smoke + benchkit 布局断言（`eval/problem/problem.json`、`run_bench`、无 `run_leaderboard_bench`）；③ 锁频拒绝 marker 两端同步；④ `[protocol]` 卡钉锁渲染 + 无 `[protocol]` 卡 unlocked 行为不变；⑤ 普通任务 `forge new --dry-run` → write_card → write_loop_toml 全程无 board 路径（eval/problem 布局、benchkit 引擎、precision strict 缺省、sealref ref_mode）；⑥ `--from-kernels` 供给闭环 dry-run（含 .cu 拒绝 → P2 积压）；⑦ CLI 无 board 面。

### 改造的锁频拒绝安全用例（必须保留项）
`test_clock_lock_refusal.py` 4 用例原样保留（拒读行首匹配、refused 重提不回传、连续拒绝 ClockLockRefusedError、parse_eval 拒绝命名），仅 fixture 日志文案去品牌并与新 marker 同步；`test_binref.py::test_recon_tree_refuses_every_task_where_the_clock_lock_does_not_hold` 以 `[protocol]` 普通任务 fixture 复现 build/binref/ncu/cmd 全拒绝。

## 6. 任务 5（P1）：任务供给闭环

`forge init <op> --from-kernels --axes '{"k": v}' [--arch h100|b200] [--shape-name campaign]`：
- 从 `kernels/<family>/<op>/<arch>/kernel.py` 冻结合同脚手架普通任务：`task.toml`（language=cutedsl、arch=h100→sm90/b200→sm100、`[reference] run=reference/run.py:run` + `blackbox = true`）；`reference/run.py` = 已发布 kernel 原文 + `run = <entry>` 别名（AST 取最后一个顶层非下划线 def）；`description.md` = kernel docstring（冻结合同）原文 + 签名待填段；`shapes.json` = 单 campaign 形状，axes 由 `--axes` 显式给定（工具只脚手架，不猜合同）
- 黑盒参考 = 既有 sealref 机制：`forge run` 时 `taskcard.seal_reference` 自动把 reference/ 密封进卡片与 recon 树（key 在 gitignored submit.local.toml），教师与学生均不可读源——去榜单后框架自带任务语料，成果 = 真超车
- `.cu` 已发布 kernel 明确拒绝并提示 P2 积压（binref 编译参考路由未接）；`--axes` 缺失/非法 → rc=2
- 验收样例：`test_deboard_regression.py::test_from_kernels_scaffolds_a_task_from_a_published_kernel`（rms_h64 @ sm90：task.toml/别名/描述断言 + `load_task` 过 + `forge new --dry-run` 计划输出 "mode blackbox" + .cu 拒绝 rc=2）

## 7. 范围围栏遵守与已知残余（记录）

- 未动：driver.py loop 语义面（只删不重构）、`measure_paired`、`precision.py` 算式、kb_lint、`meta/`、`hw_kb/`、`kernels/`、`inner/trajectory.py`、历史 CHANGES、`local.env.example`
- 已知残余（grep 门范围之外，记录在案）：`docs/blackbox-mode.md` 等保留文档与 `meta/` 提示词中仍有 "leaderboard" 字样（docs/meta 不在验收 grep 面内；文档语义指"卡片"历史称谓，不影响代码）；`FK_LB_WORKLOAD_SCOPE`/`LB_*` 环境变量名保留（Wave-3 锚点与遥测协议名，"LB" 缩写不匹配零残留模式）
- 遗留待 Wave-3/P2：binref 编译参考路由未接入普通任务（.cu kernel 参考的 P2 积压）；冷 L2 轴若需恢复按 59 裁决归 WP22 重议

## 8. 交付

- 分支 `feat/wp24-deboard` tip 带 `CHANGES-WP24.md`；已合入本地 `main`（zxtune-integration），合入后在 main 上全量 pytest 绿（见 §9 记录）；未 push 任何远端（push 归 WP23 集成步统一发布）
