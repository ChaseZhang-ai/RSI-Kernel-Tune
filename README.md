# kernel-ZXtune

**纯粹的 AI 优化算子框架**：给定一个算子（描述 + 参考实现 + 判据容差 + 输入），系统以五层机制迭代产出更快的 kernel——

1. **教师-学生决策图蒸馏**：教师从参考实现的黑盒行为（运行、计时、剖析）蒸馏出架构决策图（milestone），学生逐行实现；
2. **诚实测量**：配对窗口计时（`measure_paired` 镜像顺序）、精度门（默认 STRICT）、新鲜输入复检（fresh_recheck）、可声明并验证的测量时钟（task.toml `[protocol]`）、锁频拒绝安全（锁不住即拒绝计时，机器故障不计入判罚）；
3. **认证知识复利**：跨 run 的知识库（KB）供给与生命周期管理、证伪杠杆台账、确定性 KB-MENU 注入；
4. **定价搜索治理**：杠杆定价、预算与停止治理、结构天花板通道（解析地板、结构逃逸）；
5. **结构天花板通道**：NCU 证据驱动的瓶颈定位与硬件极限对齐。

任务供给 = 用户自供（`tasks/<op>/` 普通任务：教师判题 + 自供 inputs）；`forge init --from-kernels <op>` 可从 `kernels/` 已发布算子的冻结合同脚手架任务，并以既有 sealref 机制把已发布 kernel 密封为黑盒参考——框架自带的任务语料，自进化有题可做，超越已发布 kernel 即真超车。

**与一切外部榜单/评测组织零关联**：评测路径零网络外发（agent CLI 的 LLM 后端与教师调研通道保留，均与榜单无关）。2026-11-03 用户裁决（WP24）：原上游继承的榜单面向机器（双榜单 suite 供给线、board API 拉取、board 评测模板、榜单钉死时钟、榜单 docs）已一次性切除，详见 `CHANGES-WP24.md`。

## 上游致谢与合规

- 本仓库基于上游 [ForgeKernel](https://github.com/OpenBMB/ForgeKernel)（fork 自 `https://github.com/OpenBMB/ForgeKernel.git`，基线 commit `5030503`）修改而成。
- 上游代码以 **Apache License 2.0** 授权；本仓库遵循同一许可，[LICENSE](LICENSE) 保留，[NOTICE](NOTICE) 为 WP24 缩减版（逐文件自查记录见 `CHANGES-WP24.md`）。

## 本仓库的改进（WP1–WP6 一代 / WP8–WP12 二代）

| 包 | 改进摘要 |
|---|---|
| WP1 测量统计升级 | 判定分辨率从 1.5% 常数改为本次运行自测的 `resolvable_diff`（可配 `resolvable_floor/cap`）；迭代记录保留原始读数与 spread；疑似赢家确认带与 clock 漂移旗标；eval JSON 顶层 `gpu_state` 遥测（其 board 协议副读数窗已随 WP24 切除） |
| WP2 遥测扩容 | NCU 指标 17→25 项（SOL/L2 命中/尾波/合并度）；build 资源遥测 `resusage`（寄存器/共享内存）；digest 顶部 roofline 头行与诊断读数顺序 |
| WP3 知识复利 | 跨 run 的证伪杠杆台账（`hw_kb/falsified_ledger.jsonl`）作为设计/定价先验；probe 归档；KB 待认证队列（`hw_kb/CANDIDATES.md`） |
| WP4 搜索机制升级 | `LANDABLE` 组合一次落地；`SWEEP_REQUEST` 教师参数扫描（学生 sweep 禁令不变）；慢决策记账与 `REMOVE_CANDIDATES` 诊断小节 |
| WP5 预算与停止治理 | 格式失误不计 meta 预算（`meta_format_retry`）；meta 预算与剩余墙钟联动（`meta_hours_per_round`）；headroom 扩展与 BLOCKED 解封（`meta_ext_*` / `reopen_blocked_uncap`）；`ncu_on_revert` 开关 |
| WP6 M0 冻结前实测 + R2 结构通道 | `ARCH_PROBE` 拓扑候选实测（`m0_probe_rounds`，冻结跟随赢家）；`R2-structural` 骨架必须附实测价格（未定价机器拒绝）；配套制度文档 |
| WP8 先验供给线 | 教师侧先验供给三通道（`[agent.teacher] prior_survey` 公开 SOTA 调研 / `prior_art` 离线档案 / `carry/` 跨代携带），全部默认关闭（关闭即逐字节同前）；`priors/` 先验库（25 算子架构决策图）+ `scripts/build_priors.py` 骨架生成；`skills/sota-survey/` 与 `docs/prior-art/` 模板 |
| WP9 测量锚定校准 | NCU regime 对齐 flags（`--cache-control none --clock-control none`，全部生成点）；解析地板 `analytic_floor.py`（bytes/FLOPs 机器分母，`floor_stop`/`[target] floor_margin` 缺省关）；timing 后新鲜输入复检 `fresh_recheck`（不再现即硬 REVERT 并标 `suspicious`）；reanchor memo 落盘重启安全 |
| WP10 结构搜索深化 | `STAGNATION` 架构逃逸触发器（连续 K 次 `stagnation_k` 带内尝试即注入结构路线约束，`stagnation_override` 减压阀，只加压不改判）；`multi_candidate:` MC-DELIVERY 多候选单 job 预筛（赢家独过全门）；compile-flags 决策轴（白名单内合法杠杆） |
| WP11 过程提速包 | eval+NCU 合并单 job（省一次冷启动，段预算=两段之和绝不沿用单段；`merge_eval_ncu` 缺省开、false 回两 job）；内容寻址 build cache（`FK_BUILD_CACHE`，默认关）；SPRT 窗边界早停（只提前输、永不造 KEEP，`sprt_early_stop` 默认关）；墙钟构成遥测五字段（纯测量，无门读它） |
| WP12 知识注入与 KB 生命周期 | 确定性 KB-MENU 注入（`kbmenu.py` 唯一合成器：KB-HINTS/LEVER-MENU/CALIBRATION，检索辅助、无未测数字）；KB 条目陈旧度与死路跨 run 传播；`skills/probe-recipes/` 预置判定探针配方库（占用曲线/相位消融/有效带宽三元组） |

各包的详细交付文档：一代集中在 [docs/wp-changes/](docs/wp-changes/)（CHANGES-WP1.md … CHANGES-WP6.md），二代在仓库根（CHANGES-WP8.md … CHANGES-WP12.md），均与各实现分支 tip 逐字一致。

## 快速开始

- **Python ≥ 3.11**；
- 测试无需安装即可运行：`python3 -m pytest tests/ -q`（`pyproject.toml` 已配 `pythonpath=["src"]`）；
- 一个任务的最小闭环：

```
forge init myop                     # tasks/myop/ 脚手架：填 description.md / shapes.json / reference/
forge new tasks/myop --dry-run      # 计划预览（判题、校准、loop 配置）
forge new tasks/myop                # 建运行目录
forge run .artifacts/myop_<id>      # 判题 -> 生成卡片 -> 校准 -> 蒸馏 -> 学生迭代 -> 报告
forge export .artifacts/myop_<id>   # 导出确认的最优 kernel
```

- `forge init myop --from-kernels --axes '{"seqs": 64}'`：从 `kernels/` 已发布算子的冻结合同脚手架任务（已发布 kernel 经 sealref 密封为黑盒参考）；
- GPU job（评测、NCU、参考采集）按 `docs/design/` 与 `submit.toml`（`local.env`）配置提交后端；
- 测量时钟可声明（`task.toml [protocol]`）：声明的锁由 gpu_entry 验证，锁不住即拒绝计时。

## 仓库名与包名

本仓库名为 **kernel-ZXtune**，但 Python 包名仍是 `forgekernel`、CLI 命令仍是 `forge`——包名与导入路径是上游身份和全部测试的依赖，本仓库未作改动；仓库名仅是发布身份。

## 文档索引

| 路径 | 内容 |
|---|---|
| [docs/wp-changes/](docs/wp-changes/) | WP1–WP6 六份交付文档（二代的 CHANGES-WP8…WP12.md 在仓库根） |
| [docs/design/](docs/design/) | 设计文档（v1-launch 为历史存档：榜单章节已过时，见文首注记） |
| [meta/](meta/) / [skills/](skills/) | 循环提示词与技能包 |
| [kernels/](kernels/) | 26 条已发布 kernel（先验/教材资产；其 README 出处记录如实保留——诚实出处 ≠ 榜单关联） |
