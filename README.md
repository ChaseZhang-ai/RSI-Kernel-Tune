# kernel-ZXtune

kernel-ZXtune 是基于 [ForgeKernel](https://github.com/OpenBMB/ForgeKernel) 的 GPU 算子优化增强版：在上游自进化 kernel 代理框架之上，集成了六条并行工作线（WP1–WP6）的改进——测量统计升级、每轮遥测扩容、知识复利、搜索机制升级、预算与停止治理、以及 M0 冻结前实测与 R2 结构通道。

## 上游致谢与合规

- 本仓库基于上游 [ForgeKernel](https://github.com/OpenBMB/ForgeKernel)（fork 自 `https://github.com/OpenBMB/ForgeKernel.git`，基线 commit `5030503`）修改而成。
- 上游代码以 **Apache License 2.0** 授权；本仓库遵循同一许可，[LICENSE](LICENSE) 与 [NOTICE](NOTICE) 原样保留。
- 上游的 README 见 [docs/upstream-readme.md](docs/upstream-readme.md)（英文）与 [docs/upstream-readme_zh.md](docs/upstream-readme_zh.md)（中文）。

## 本仓库的改进（WP1–WP6）

| 包 | 改进摘要 |
|---|---|
| WP1 测量统计升级 | 判定分辨率从 1.5% 常数改为本次运行自测的 `resolvable_diff`（可配 `resolvable_floor/cap`）；迭代记录保留原始读数与 spread；疑似赢家确认带与 clock 漂移旗标；eval JSON 顶层 `gpu_state` 遥测；board 协议计时窗口 |
| WP2 遥测扩容 | NCU 指标 17→25 项（SOL/L2 命中/尾波/合并度）；build 资源遥测 `resusage`（寄存器/共享内存）；digest 顶部 SOL roofline 头行与诊断读数顺序 |
| WP3 知识复利 | 跨 run 的证伪杠杆台账（`hw_kb/falsified_ledger.jsonl`）作为设计/定价先验；probe 归档；KB 待认证队列（`hw_kb/CANDIDATES.md`） |
| WP4 搜索机制升级 | `LANDABLE` 组合一次落地；`SWEEP_REQUEST` 教师参数扫描（学生 sweep 禁令不变）；慢决策记账与 `REMOVE_CANDIDATES` 诊断小节 |
| WP5 预算与停止治理 | 格式失误不计 meta 预算（`meta_format_retry`）；meta 预算与剩余墙钟联动（`meta_hours_per_round`）；headroom 扩展与 BLOCKED 解封（`meta_ext_*` / `reopen_blocked_uncap`）；`ncu_on_revert` 开关 |
| WP6 M0 冻结前实测 + R2 结构通道 | `ARCH_PROBE` 拓扑候选实测（`m0_probe_rounds`，冻结跟随赢家）；`R2-structural` 骨架必须附实测价格（未定价机器拒绝）；配套制度文档 |

各包的详细交付文档集中在 [docs/wp-changes/](docs/wp-changes/)（CHANGES-WP1.md … CHANGES-WP6.md，与各实现分支 tip 逐字一致）。

## 快速开始

上游的完整文档（安装、任务卡片、运行循环、leaderboard）在移动后的上游 README 中：

- 英文：[docs/upstream-readme.md](docs/upstream-readme.md)
- 中文：[docs/upstream-readme_zh.md](docs/upstream-readme_zh.md)

环境要求：

- **Python ≥ 3.11**；
- 测试无需安装即可运行：`python3 -m pytest tests/ -q`（`pyproject.toml` 已配 `pythonpath=["src"]`）；
- GPU job（评测、NCU、ref 采集）需按上游 docs 的说明配置提交后端与算子任务卡（见 `docs/design/` 与上游 README 的 Quick Start 一节）。

## 仓库名与包名

本仓库名为 **kernel-ZXtune**，但 Python 包名仍是 `forgekernel`、CLI 命令仍是 `forge`——包名与导入路径是上游身份和全部测试的依赖，本仓库未作改动；仓库名仅是发布身份。

## 文档索引

| 路径 | 内容 |
|---|---|
| [docs/upstream-readme.md](docs/upstream-readme.md) / [docs/upstream-readme_zh.md](docs/upstream-readme_zh.md) | 上游 ForgeKernel README（原根 README 移动而来） |
| [docs/wp-changes/](docs/wp-changes/) | WP1–WP6 六份交付文档 |
| [docs/design/](docs/design/) | 上游设计文档（launch、milestone 蒸馏等） |
| [meta/](meta/) / [skills/](skills/) | 循环提示词与技能包 |
