# `forge board` — SOL-ExecBench 榜单客户端

用途：把本地做好的 `submission.py` 交给榜单、轮询/回读分数、设为公开、上传赛道压缩包，并把
结果导回 campaign 的台账。**它只做网页界面能做的同样操作，并在客户端遵守榜单限速**；令牌只从
环境变量 `SOL_EXECBENCH_TOKEN` 读，任何输出、日志、异常里都不会出现令牌。

实现：`src/forgekernel/ext/board.py`（命令）＋ `src/forgekernel/board/{client,policy,validate,importer}.py`。
只用标准库（`urllib`），不引入 `requests`。

> 工作区提示：本分支的基线提交 `b2c8ae8` 里 `forgekernel/ext/` 的自动发现（第 0 步扩展点）尚未落地，
> 所以当前在 worktree 里用 `python -m forgekernel.ext.board …`（或 `PYTHONPATH=src` 后同样）
> 运行；集成第 0 步后 `forge board …` 直接可用。两种入口是同一个 `main()`。

---

## 1. 命令

| 命令 | 做什么 |
|---|---|
| `forge board whoami` | `GET /api/me`：账号、账号组、提交策略（并发上限、结果延迟）、今日用量 |
| `forge board submit <题目名> <submission.py> [--wait\|--no-wait] [--timeout 分钟]` | 单题提交。默认在当天前 5 次的“即时”额度内等结果；第 6 次起只提示预计公布时间、不等（`--wait` 可强制等） |
| `forge board status [id]` | 不带 id：列出自己最近的提交（排队/运行/待公布/完成/失败/失格）；带 id：打印该提交摘要 |
| `forge board result <id> [--json]` | 回读结果：正确性（AC/WA/DQ）、SOL 分数、逐 workload 的延迟/基线/加速/S、是否失格、评测栈版本 |
| `forge board publish <id> [--yes]` | 设为公开（默认要确认；`--yes` 跳过） |
| `forge board submit-collection <赛道> <zip> [--yes]` | 上传赛道压缩包（默认要确认）。赛道用 `l1` / `l2` / `quant` / `flashinfer-bench` 或 id |
| `forge board import <campaign> [--since 7d]` | 把自己所有已完成的单题提交导进 `<campaign>/ledger.jsonl`，并按题目名配上本地分数算比值 |

通用参数：`--json`（机器可读输出）、`--log PATH`（审计日志，默认 `.artifacts/board/log.jsonl`）、
`--api-base URL`（默认官方前缀）、`--http-timeout 秒`。`submit` / `submit-collection` / `publish` 支持
`--dry-run`：只打印将发送的请求（方法、URL、表单字段、文件名/大小/sha256），不发送。

退出码：`0` 成功；`1` 一般错误/取消；`2` 自检未通过或参数不合法；`3` 被限速/策略拒绝/等待超时；
`4` 网络或服务端错误；`5` 认证失败（401/403）。

### 例子

```bash
export SOL_EXECBENCH_TOKEN='…'                    # 见第 4 节
forge board whoami
forge board submit 069_rms_norm out/069_rms_norm/submission.py --wait --timeout 15
forge board status
forge board result 12345 --json
forge board publish 12345 --yes
forge board submit-collection l1 out/l1-<UTC时间戳>.zip --yes
forge board import .artifacts/campaign --since 7d
```

---

## 2. 接口核实表（唯一来源：榜单前端脚本）

前端 bundle：`https://research.nvidia.com/benchmarks/sol-execbench/assets/index-3d17caa7.js`
（2026-11-24 拉取，2,476,394 字节）。下面的“来源”列指向该 bundle 里的调用点；minify 后的函数名
每次构建都会变，所以引用的是可辨别的代码片段，不是函数名。

接口前缀：bundle 里 `const ut="/benchmarks/sol-execbench"`，所有调用都是 `${ut}/api/…`，
即 `https://research.nvidia.com/benchmarks/sol-execbench/api`。认证头：
`function ra(){const e=HCe(); return e?{Authorization:\`Bearer ${e}\`}:{}}`。

| 用途 | 方法 / 路径 | 请求 | 响应（已核实字段） | 来源（bundle 片段） |
|---|---|---|---|---|
| 当前账号 | `GET /me` | 认证头 | `{data:{authenticated, user:{id, email, username, display_name, account_group, groups, submission_access:{limits:{max_concurrent_submissions}, usage:{active_submissions, kernel_submissions_today}, result_delay:{max_delay_minutes, collection_delay_hours}}}}}` | 登录校验 `fetch(\`${ut}/api/me\`,{headers:{Authorization:\`Bearer ${f}\`}})`；个人页读 `t.submission_access.limits/usage/result_delay` |
| 题目列表 | `GET /kernels` | 无 | `{data:{kernels:[{id, name, title, gpu_types, tags}]}}` | `async function sE(){const e=await fetch(\`${ut}/api/kernels\`)…return (await e.json()).data}`；`T.kernels` |
| 题目详情 | `GET /kernels/{id}` | 无 | `{data:{id, name, title, gpu_types, tags, workloads:[{axes, baseline_latency_ms, sol_ms}], inputs, outputs, baseline, scoring_baseline, latency_unit}}` | 题目页 `QCe(e)`；表格渲染 `te.axes/te.baseline_latency_ms/te.sol_ms` |
| 赛道列表 | `GET /collections` | 无 | `{data:{collections:[{id, name, title, gpu_types, kernel_count, submission_count, kernels?}]}}` | `async function lE()`；首页 `w.collections` |
| 赛道详情 | `GET /collections/{id}` | 无 | `{data:{…, kernels:[{id,name,title,…}]}}` | `async function JCe(e)` |
| **单题上传** | `POST /submissions/upload` | multipart：`file`, **`kernel_id`**, `gpu_type`, `submission_mode` | `{data:{…}}`（提交 id 的键**未知**：前端忽略返回值；代码兼容 `submission_id` / `id`） | `a4e`：`a.append("file",e),a.append("kernel_id",String(t)),a.append("gpu_type",n),a.append("submission_mode",r)` |
| 提交列表 | `GET /submissions?…` | 认证头；query：`kernel_id, user_id, status, collection_id, collection_submission_id, gpu_type, evaluation_stack_version, username, submitted_from, submitted_to, submitted_from_utc, submitted_before_utc, offset, limit` | `{data:{submissions:[…], total}}` | `dN(e)` 的 `URLSearchParams` 逐项 `t.set(...)` |
| 提交详情 | `GET /submissions/{id}` | 认证头 | `{data:{id, status, is_correct, is_disqualified, disqualification_source, disqualification_reason, review_decision, review_summary, review_reasons, sol_score, latency_ms, fast_1_count, fast_1_total, avg_speedup, gpu_type, username, user_id, kernel_id, kernel_name, kernel_title, collection_id, collection_title, collection_submission_id, type, submission_mode, original_filename, submitted_at, started_at, finished_at, worker_id, evaluation_stack_version, result_available_at, error_log, run_log:{stdout,stderr}, aggregate_sol_score, kernel_submissions?, benchmark_results:[{workload_axes, spec, latency_ms, baseline_latency_ms, speedup_factor, sol_score}]}}` | `Lg(e)`；详情页字段列表与 “Benchmark Results” 表 |
| 下载提交文件 | `GET /submissions/{id}/download` | 认证头 | 二进制；文件名在 `Content-Disposition` | `zee(e)` 读 `t.headers.get("content-disposition")` |
| **设为公开** | `POST /submissions/{id}/make-public` | 认证头（无 body） | `{data:{…}}` | `o4e(e)` |
| **赛道上传** | `POST /collections/{id}/submit` | multipart：`file`, `gpu_type`, `submission_mode`（**没有** `kernel_id`） | `{data:{submission_id}}` | `i4e(e,t,n,r)`；调用处 `if(xe.submission_id){…}` |
| 队列状态 | `GET /queue/status` | 无 | `{data:{queue_length, queues:{…}}}` | `Uee()`；Workers 页 |
| 逐题榜 | `GET /leaderboard/kernel/{id}/{gpu}?evaluation_stack_version=v1.1` | 无 | `{data:{kernel_name, kernel_title, kernel_id, gpu_type, latency_unit, sol_entry, reference_entries, rankings}}` | `t4e(e,t,n)` |
| 赛道榜 | `GET /leaderboard/collection/{id}/{gpu}?evaluation_stack_version=v1.1` | 无 | `{data:{collection_name, collection_title, collection_id, sol_entry, reference_entries, rankings}}` | `Bee(e,t,n)` |
| 榜单索引 | `GET /leaderboards` | 无 | `{data:{kernel_leaderboards, collection_leaderboards}}` | `e4e()` |
| SSO 换令牌 | `POST /auth/token` | `{code}` | `{token}` | 登录回调 `pKe()`；**CLI 不用它**（见第 4 节） |

枚举（同样从 bundle 读出）：

* `gpu_type`：`DM=["A100","H100","B200"]`；提交页默认选 `gpu_types[0]`，评测机是 **B200**。
* `submission_mode`：上传控件只有两个取值 `private`（“Your result is visible only to you.”）与
  `release`（“Your result is visible to everyone.”）；显示时 `release` → “Public”。
* 提交状态：`["QUEUED","PENDING","RUNNING","EVALUATING","PENDING_RESULT","COMPLETED","FAILED","SYSTEM_ERROR"]`；
  前五个是“在途”（bundle 的 `fN` 数组，用于自动轮询）。`PENDING_RESULT` 表示结果已算出但按策略未公布。
* 上传大小：上传控件对**所有**文件（含压缩包）都按 `size>10*1024*1024` 拒绝（“File size exceeds 10MB limit.”）。
  指南另外给出 50 MB 压缩包 / 100 MB 解压后的上限，自检一并执行。
* 赛道 id/名称（榜单快照实测）：`1=l1 "L1 - Single Operations"`、`2=l2 "L2 - Fused Operations"`、
  `3=quant "Quantization"`、`4=flashinfer-bench "FlashInfer-Bench"`。

**未核实 / 未知**（代码里做成兼容或可配，集成联调时优先确认）：

1. 单题上传响应里提交 id 的字段名（前端忽略返回值；客户端两种都认）。
2. 其它必填字段：bundle 只发 `file/kernel_id/gpu_type/submission_mode`，是否还有服务端必填项未知；
   错误时服务端返回 `{"detail": …}`（代码把它带进错误信息）。
3. 错误体结构：只核实了 `detail`（以及兜底 `message` / `error`）；429 是否带 `Retry-After` 未知（代码按有/无都处理）。
4. `evaluation_stack_version` 的合法值（前端固定 `v1.1`）；榜单将来加版本时可能变化。
5. 分页上限、`limit` 的默认值与上限（客户端固定每页 100 并翻页）。
6. `/api/v1/...` 前缀下还有 profile/通知等端点，与提交无关，未使用。

---

## 3. 限速（客户端强制执行，不绕过）

来源：官方《Submission Guide》的 “Submission Policy” 一节（`reports/board-snapshot-2026-11-24/
blog_submission-guide.json`），以及 `GET /api/me` 的 `submission_access`。两者不一致时以 `/api/me`
为准，指南数字用于提交前的预估。

* **单题**：当天（UTC）前 5 次即时；第 6 次起结果延迟公布 10、20、30… 分钟，上限 60 分钟；
  11 次起恒 60 分钟。**在途（排队+运行+待公布）最多 5 个**；没有每日总量上限。
* **赛道**：结果固定约 6 小时；**每个赛道每 UTC 日最多 2 次**；校验失败的压缩包不计次。
* 客户端行为：`submit` 前查 `/api/me`，`active_submissions >= max_concurrent_submissions` 直接拒绝（退出码 3）；
  `submit-collection` 前按 `collection_id` 数当天（UTC）的父提交，达到 2 次就拒绝并打印下一次可提交时间
  （UTC 零点）。
* **等待以服务端 `result_available_at` 为准**：轮询到 `COMPLETED/FAILED/SYSTEM_ERROR` 后，如果
  `result_available_at` 还没到，继续等到那个时刻（`--timeout` 之内）；指南的 10/20/60 分钟只用于
  提交前的预估提示。
* 收到 HTTP 429 不自动重试：打印服务端 `Retry-After`（若有）与建议，退出码 3。

---

## 4. 令牌怎么给

1. 浏览器登录 `https://research.nvidia.com/benchmarks/sol-execbench`。
2. DevTools → Application（应用）→ Local Storage → 该站点 → 复制 `solbench_token` 的值（前端就是把它
   放进 `Authorization: Bearer`）。
3. `export SOL_EXECBENCH_TOKEN='<粘贴>'`（放进 shell 配置或从密码管理器注入；**不要**写进
   `local.env`、源码、命令行参数或报告）。

没有令牌时所有命令打印同一段提示并以 1 退出。令牌只从 `SOL_EXECBENCH_TOKEN` 读取；客户端对
令牌做了遮蔽（`redact`）：错误信息、`--json` 输出、审计日志里都不会出现，服务器返回的
`detail` 里如果混进令牌也会被替换成 `***`。

### 没有令牌时的等价手工步骤

单题：打开题目页 → 选择 Device（B200）与 Submission Mode（Private/Public）→ 上传
`submission.py` → 在 `/status` 页看状态与 `result_available_at` → 详情页 “Make Public”。
赛道：打开赛道页（`/collection/<id>`）→ 选 Device/Mode → 上传压缩包 → 6 小时后在提交详情页看
`Aggregate SOL Score`。`forge board import --from-saved` 这类离线替代**没有实现**（网页另存的 JSON
由 campaign 的 `sync --from-saved` 处理，见其任务文件）。

---

## 5. 上传前自检（`forgekernel/board/validate.py`）

单题（`validate_kernel_file`）：`.py` 扩展名；≤ 10 MB；UTF-8；能 `ast.parse`；顶层有 `def run(...)`
（`async def run` 报错——评测器同步调用；`run = …` 赋值给警告）；非空。

赛道（`validate_collection_archive`）：压缩包 ≤ 10 MB（上传控件上限）且 ≤ 50 MB（档案上限）；
解压后 ≤ 100 MB；每个顶层目录必须含且只含一份 `submission.py`；给了赛道题目名列表时目录名必须
严格匹配（不匹配报错）；缺题默认只警告（该题记 0 分），`--require-all` 时升级为错误；拒绝符号
链接/硬链接、绝对路径、`..` 穿越、NUL 字节、重复成员；根目录的多余文件按指南忽略（并给警告）。
只读归档的成员表，不解压、不执行任何东西。campaign 的 `pack` 可以直接调用这两个纯函数。

---

## 6. 审计日志与安全

* 每次**真实**调用在 `.artifacts/board/log.jsonl` 追加一行：时间、命令、题目/赛道、submission id、
  gpu_type、mode、kernel 数、成功/失败与错误摘要。失败也会记一行（`ok:false` + `error`）。
  `--dry-run` 不写日志（没有真实调用）。
* 日志与错误信息都过 `redact`，令牌被替换为 `***`。
* 客户端的生产传输层只允许 HTTPS、主机名允许列表（默认 `research.nvidia.com`，可用
  `SOL_EXECBENCH_ALLOWED_HOSTS` 显式追加镜像域名）、拒绝重定向、解析出的地址是环回/私网/链路本地
  就拒绝连接，并限制响应体大小。它不做任何写操作以外的自动化。

---

## 7. campaign 集成（软依赖）

* `forge board import <campaign>` 写 `<campaign>/ledger.jsonl`，每行
  `{"ts", "kind":"board", "source":"board", "problem": <kernel_name>, "submission_id", "kernel_id",
  "status", "gpu_type", "sol_score", "local_s"?, "ratio"?, "is_correct", "is_disqualified",
  "latency_ms", "fast_1_count", "fast_1_total", "avg_speedup", "submitted_at", "finished_at",
  "evaluation_stack_version"}`。
  `local_s` 取台账里同题目名已有本地分数的最好值（字段 `local_s` / `mean_s` / `est_s` / 非 board 行的 `s`）；
  `ratio = sol_score / local_s`。按 `submission_id` 去重，可重复运行。
* 若 `forgekernel.campaign.sync` 暴露 `record_board_readback(campaign_dir, records)`，`import` 改为调用它，
  campaign 自己落账（board 不写文件）。接口约定写在 `reports/board-report.md` 的集成备注里。
* 只导入 `status == "COMPLETED"` 的单题提交；赛道父提交与其子提交跳过（赛道分由 campaign 从子提交算）。

---

## 8. 风险与未覆盖项

* **接口是非官方的**：上表来自前端 bundle 的当前版本。页面改版（新的 bundle hash）后字段可能变化；
  代码对缺失字段宽容（读不到就显示 `-`，不崩溃），但新的必填字段会导致上传被服务端拒绝——
  第一次联调时先跑 `whoami` 与一次 `--dry-run`，再交真实提交。
* **未知字段**见第 2 节末尾；联调时最需要确认的是单题上传返回的 id 字段名与错误体结构。
* **没有实机验证**：本任务只允许离线实现。所有命令的真实调用（`whoami`、单题提交、赛道上传、
  `publish`）都需要用户提供 `SOL_EXECBENCH_TOKEN` 并在场做第一次调用；测试全部用注入的假
  transport 覆盖到调用边界。
* **未实现**：`--from-saved`（离线导入网页另存的 JSON，归 campaign）；`/api/submissions/{id}/download`
  的 CLI 入口（客户端函数有，命令未暴露）；赛道提交的“子提交逐题回读”聚合展示（`result <collection id>`
  会打印 `kernel_submissions` 表，但不逐题下载）。
* **不做的**：不改写 `submission.py`；不上传没通过自检的文件；不绕过任何限速；不替用户选题。
