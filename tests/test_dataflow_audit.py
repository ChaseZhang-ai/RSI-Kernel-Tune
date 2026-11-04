"""WP21 数据流审计 CI：每个 IterationRecord / EvalResult 字段与遥测字段必须有读者。

背景（乙 F27 / 甲 G25，59-三代总览与裁决-合并版 §四 WP21）：两代机器大量"机制在、
数据流断"——字段写进轨迹却无人消费（ledger 0 字节、floor 回测 25×null、多个字段
零读者）。本审计把这件事变成 pytest 可拦截项：对 ``inner/trajectory.py`` 的
IterationRecord、``loop/parse.py`` 的 EvalResult 的字段清单，机械生成"读者 grep"
断言。

读者口径（grep 级，机械可判定）
------------------------------
* 只扫 ``src/**/*.py``；``tests/``、``skills/``、``docs/``、``meta/`` 一律不算——
  仅 tests 命中 = fail，提示词文本/文档 prose 不是代码消费者。
* 算作读者的命中：
  - 属性读 ``.field``（排除属性写 ``.field = x``）；
  - 精确字符串键读 ``"field"``——``.get("f")`` / ``["f"]`` / ``getattr(x, "f")`` /
    提取用元组常量；排除字典字面量构造 ``"f": ...``（写侧）。
* 不算读者的命中：dataclass 声明行、构造 kwarg ``field=...``、属性/就地赋值写、
  注释行（# 起始或行尾 # 后）、字典字面量构造。
* 分层（这是本审计的核心判别）：
  - IterationRecord（轨迹 SSOT，受众是 digest/收割/跨 run 层）——读者必须落在
    ``loop/driver.py``、``loop/parse.py``（写侧管道）与 ``card_template/**``
    （eval bench 写侧）之外；
  - EvalResult（job→loop 交接，受众就是 driver）——除构造方 ``loop/parse.py``
    与 bench 写侧 ``card_template/**`` 外，driver 读点算数；
  - 遥测键（``_SOL_HEADER_KEYS`` / ``GPU_STATE_FIELDS`` 两个 dict 载荷元组，ast
    抽取；``_WP11_TELEMETRY_FIELDS`` 五键本就是 IterationRecord 字段，按字段审计）
    ——键名在元组定义文件之外有键读，或元组本身在生产者之外被引用（间接消费）。
* ``build_digest`` 单独审"生产调用点"（ast Call 节点，定义文件 docstring/导出不算）
  ——Layer-1 digest 是轨迹的唯一消化出口，零调用点=整条链断。

清单与自我维护（防形式主义）
----------------------------
* 无读者字段必须登记在 ``tests/dataflow_audit_baseline.json``（预期被修好，带
  owner 包与证据）或 ``tests/dataflow_audit_whitelist.json``（确属记录备用，带
  理由）；两者皆需注明依据。
* **陈旧条目反向拦截**：登记条目一旦长出真读者，测试立即 fail 要求删行——
  WP15/WP19 的消费方合并后，WP23 逐条删空基线清单，"合并后跑全绿"即验收。

已知局限（grep 级的边界，见 68-WP21 包指令"grep 级即可"）
----------------------------------------------------------
* 名字级扫描无法区分接收者类型：跨类同名（如 ``.error``/``.sha`` 在别的对象上）
  由管道层排除集压制；残余撞名风险由白名单的人工理由兜底。
* 元组间接消费只识别本文件定义的三个遥测元组；新增遥测元组需扩展 ``TUPLES``。
* 直接跑本文件可打印当前无读者清单：``python tests/test_dataflow_audit.py``。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
SRC = ROOT / "src"

TRAJECTORY = SRC / "peregrine_kernel" / "inner" / "trajectory.py"
PARSE = SRC / "peregrine_kernel" / "loop" / "parse.py"
TIMING = SRC / "peregrine_kernel" / "card_template" / "eval" / "timing.py"
DRIVER = SRC / "peregrine_kernel" / "loop" / "driver.py"

# IterationRecord 的写侧管道与同名异物文件：命中不计读者。launch.py / loop/cli.py /
# monitor/cli.py 消费的是 best_confirmation.json / monitor / job-state 等 JSON 导出件
# 的 "sha" 键与同名 CLI 参数/局部变量，与轨迹行同形不同源（loop/cli.py:634 job 状态、
# :1797/:1814 确认与监控 JSON、:3184 配对确认；monitor/cli.py:33 确认 JSON、:72 args），
# 不是 IterationRecord 行的读者。
IR_PLUMBING = ("src/peregrine_kernel/loop/driver.py", "src/peregrine_kernel/loop/parse.py",
               "src/peregrine_kernel/launch.py", "src/peregrine_kernel/loop/cli.py",
               "src/peregrine_kernel/monitor/cli.py")
IR_BENCH = "src/peregrine_kernel/card_template/"
# EvalResult：构造方与 bench 写侧不算读者（driver 是它的设计受众）。
EV_PLUMBING = ("src/peregrine_kernel/loop/parse.py",)
EV_BENCH = "src/peregrine_kernel/card_template/"

BASELINE_PATH = TESTS_DIR / "dataflow_audit_baseline.json"
WHITELIST_PATH = TESTS_DIR / "dataflow_audit_whitelist.json"

# 遥测元组常量：(定义文件, 元组名, 键名空间标签, 生产者文件/目录前缀)。
# 只收 dict 载荷键（profiler/gpu_state 内部键，字段级审计够不到的）；_WP11_TELEMETRY_FIELDS
# 的五个键本就是 IterationRecord 字段（白名单按字段登记），不重复入表。
TUPLES = (
    (TRAJECTORY, "_SOL_HEADER_KEYS", "SOL", "src/peregrine_kernel/inner/ncu.py"),
    (TIMING, "GPU_STATE_FIELDS", "GPU_STATE", "src/peregrine_kernel/card_template/"),
)

DIGEST_FN = "build_digest"


# ---------------------------------------------------------------- 机械抽取 ----

def _dataclass_fields(path: Path, cls: str) -> list[str]:
    """类体 AnnAssign 名单（即 dataclass 字段清单），按源码顺序。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [s.target.id for s in node.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
    raise AssertionError(f"{cls} not found in {path}")


def _tuple_const(path: Path, name: str) -> list[str]:
    """模块级 ``NAME = ("a", "b", ...)`` 的元素清单。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets) \
                    and isinstance(node.value, ast.Tuple):
                return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    raise AssertionError(f"tuple const {name} not found in {path}")


def _tuple_referenced(path: Path, name: str, producer: str) -> bool:
    """元组名在生产者之外被引用（定义文件的消费函数如 trajectory.sol_header 算数；
    生产者文件/目录内部的引用是写侧自用，不算消费）。"""
    producer_path = (ROOT / producer).resolve()
    for py in sorted(SRC.rglob("*.py")):
        resolved = py.resolve()
        if producer_path.is_dir():
            if resolved.is_relative_to(producer_path):
                continue
        elif resolved == producer_path:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        def_lineno = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets):
                def_lineno = node.lineno
        if any(isinstance(n, ast.Name) and n.id == name and n.lineno != def_lineno
               for n in ast.walk(tree)):
            return True
    return False


def _production_callers(fn: str) -> list[str]:
    """src 下对 fn 的真实调用点（ast Call 节点；定义文件内的 def 与 docstring 自然
    不构成 Call）。返回 "rel:line" 列表。"""
    callers: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        rel = py.relative_to(ROOT).as_posix()
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            named = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None)
            if named == fn:
                callers.append(f"{rel}:{node.lineno}")
    return callers


# ---------------------------------------------------------------- 读者分类 ----

def _src_lines() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for py in sorted(SRC.rglob("*.py")):
        rel = py.relative_to(ROOT).as_posix()
        out[rel] = py.read_text(encoding="utf-8").splitlines()
    return out


def classify(name: str, files: dict[str, list[str]]) -> tuple[list, list]:
    """全 src 扫描 name 的命中，分 (reads, writes)。

    reads/writes 元素为 (rel_path, lineno, tag)；tag ∈ attr|key / decl|kwarg|dictlit。
    """
    reads: list = []
    writes: list = []
    attr_re = re.compile(r"\." + re.escape(name) + r"\b")
    kwarg_re = re.compile(r"(?<![\w.])" + re.escape(name) + r"\s*=(?!=)")
    key_re = re.compile(r"[\"']" + re.escape(name) + r"[\"']")

    for rel, lines in files.items():
        for lineno, raw in enumerate(lines, 1):
            if re.match(rf"\s*{re.escape(name)}\s*:", raw):
                writes.append((rel, lineno, "decl"))
                continue
            code = raw.split("#", 1)[0] if not raw.lstrip().startswith("#") else ""
            if not code.strip():
                continue
            for m in attr_re.finditer(code):
                tail = code[m.end():].lstrip()
                if tail.startswith("=") and not tail.startswith("=="):
                    continue  # 属性写
                reads.append((rel, lineno, "attr"))
            for m in kwarg_re.finditer(code):
                writes.append((rel, lineno, "kwarg"))
            for m in key_re.finditer(code):
                tail = code[m.end():].lstrip()
                if tail.startswith(":"):
                    writes.append((rel, lineno, "dictlit"))
                else:
                    reads.append((rel, lineno, "key"))
    return reads, writes


def _consumption(reads: list, plumbing: tuple[str, ...], bench: str) -> list:
    """按类分层取消费层读点。"""
    return [r for r in reads
            if r[0] not in plumbing and not r[0].startswith(bench)]


def audit() -> dict[str, dict]:
    """全字段审计表：key → {status, tier, hits, note}。

    status: "read"（消费层有读者）| "readerless"。
    key 形如 "IterationRecord.f" / "EvalResult.f" / "telemetry:TAG.key" / "build_digest"。
    """
    files = _src_lines()
    table: dict[str, dict] = {}

    for cls, path, plumbing, bench in (
        ("IterationRecord", TRAJECTORY, IR_PLUMBING, IR_BENCH),
        ("EvalResult", PARSE, EV_PLUMBING, EV_BENCH),
    ):
        for f in _dataclass_fields(path, cls):
            reads, _ = classify(f, files)
            cons = _consumption(reads, plumbing, bench)
            table[f"{cls}.{f}"] = {
                "status": "read" if cons else "readerless",
                "tier": "consumption" if cons else (
                    "plumbing-only" if reads else "no-read"),
                "hits": [f"{r[0]}:{r[1]}({r[2]})" for r in cons[:5]],
            }

    for path, tup, tag, producer in TUPLES:
        keys = _tuple_const(path, tup)
        indirect = _tuple_referenced(path, tup, producer)
        def_rel = path.relative_to(ROOT).as_posix()
        for k in keys:
            reads, _ = classify(k, files)
            direct = [r for r in reads
                      if r[0] != def_rel and not r[0].startswith(producer)]
            ok = bool(direct) or indirect
            table[f"telemetry:{tag}.{k}"] = {
                "status": "read" if ok else "readerless",
                "tier": "consumption" if ok else "no-read",
                "hits": ([f"{r[0]}:{r[1]}({r[2]})" for r in direct[:5]]
                         or ([f"indirect:{tup}@{def_rel}"] if indirect else [])),
            }

    callers = [c for c in _production_callers(DIGEST_FN)
               if not c.startswith("peregrine_kernel/inner/trajectory.py")]
    table[DIGEST_FN] = {
        "status": "read" if callers else "readerless",
        "tier": "consumption" if callers else "no-read",
        "hits": callers[:5],
    }
    return table


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def readerless_keys(table: dict[str, dict]) -> list[str]:
    return sorted(k for k, v in table.items() if v["status"] == "readerless")


def report_text(table: dict[str, dict]) -> str:
    """人读报告：当前无读者清单 + 登记状态（CHANGES 粘贴用）。"""
    rl = readerless_keys(table)
    lines = ["WP21 dataflow audit — readerless keys: "
             f"{len(rl)}/{len(table)} fields/keys"]
    for k in rl:
        v = table[k]
        lines.append(f"  READERLESS  {k}  [{v['tier']}]")
    annotated = sorted((BASELINE_PATH.name, k) for k in _load(BASELINE_PATH)) \
        + sorted((WHITELIST_PATH.name, k) for k in _load(WHITELIST_PATH))
    lines.append(f"  annotated: {len(annotated)} entries "
                 "(baseline expected-to-be-fixed / whitelist record-for-later)")
    return "\n".join(lines)


# -------------------------------------------------------------------- 测试 ----

def test_field_universe_extraction_is_not_vacuous():
    """抽取器哨兵：字段清单与遥测元组非空且含已知锚点——防止 ast 抽取静默坏掉、
    审计变成空集恒过的形式主义。"""
    ir = _dataclass_fields(TRAJECTORY, "IterationRecord")
    ev = _dataclass_fields(PARSE, "EvalResult")
    assert len(ir) >= 30 and "gate_spread_pct" in ir and "clock_drift" in ir
    assert len(ev) >= 20 and "fresh_recheck" in ev and "device_readings_ms" in ev
    assert _tuple_const(TRAJECTORY, "_SOL_HEADER_KEYS")
    assert _tuple_const(TIMING, "GPU_STATE_FIELDS")
    assert _tuple_const(DRIVER, "_WP11_TELEMETRY_FIELDS")  # 构成表消费契约在位
    table = audit()
    # 两类字段 + dict 载荷遥测键（SOL∪GPU_STATE，无重复）+ build_digest：动态对账防空集恒过
    telemetry_keys = sorted({k for dfn, tup, _, _ in TUPLES for k in _tuple_const(dfn, tup)})
    assert len(table) == len(ir) + len(ev) + len(telemetry_keys) + 1
    assert len(table) >= 55


def test_every_readerless_key_is_annotated():
    """核心拦截：任何无读者字段必须在基线清单（预期被修好，owner 包）或白名单
    （记录备用，理由）登记——新增的"机制在、数据流断"在此 fail。"""
    table = audit()
    baseline, whitelist = _load(BASELINE_PATH), _load(WHITELIST_PATH)
    rl = readerless_keys(table)
    unannotated = [k for k in rl if k not in baseline and k not in whitelist]
    assert not unannotated, (
        "新增无读者字段未登记（机制在、数据流断）——补消费方，或按"
        "'记录备用'入白名单/按'预期修复'入基线清单（带 owner 包）：\n"
        + "\n".join(f"  {k}  [{table[k]['tier']}]  e.g. {table[k]['hits']}"
                    for k in unannotated))


def test_annotations_are_current_and_substantiated():
    """登记的自我维护：键存在、理由/owner 齐全、基线与白名单不相交、且条目未过时
    （长出真读者的条目必须删行——WP15/WP19 合并后 WP23 逐条删空基线即'跑全绿'）。"""
    table = audit()
    baseline, whitelist = _load(BASELINE_PATH), _load(WHITELIST_PATH)
    universe = set(table)

    dup = set(baseline) & set(whitelist)
    assert not dup, f"基线与白名单不相交，重复登记: {sorted(dup)}"
    unknown = (set(baseline) | set(whitelist)) - universe
    assert not unknown, f"登记键不在字段宇宙（拼错或字段已删）: {sorted(unknown)}"

    thin = [k for k, v in baseline.items()
            if not v.get("owner") or not v.get("note")]
    assert not thin, f"基线条目缺 owner/note（防形式主义）: {thin}"
    thin = [k for k, v in whitelist.items() if not str(v).strip()]
    assert not thin, f"白名线条目缺理由（防形式主义）: {thin}"

    stale = [k for k in (set(baseline) | set(whitelist))
             if table[k]["status"] == "read"]
    assert not stale, (
        "登记条目已过时——消费方已落地，删行（基线条目清出即 WP23 '合并后全绿'）：\n"
        + "\n".join(f"  {k} -> {table[k]['hits']}" for k in sorted(stale)))


def test_build_digest_wiring():
    """Layer-1 digest 出口专项：build_digest 必须有生产调用点，或仍登记在基线
    （WP19 接线前）。两端都倒即 fail——防止 digest 链静默断裂回 F27 态。"""
    table = audit()
    baseline = _load(BASELINE_PATH)
    assert table[DIGEST_FN]["status"] == "read" or DIGEST_FN in baseline, (
        f"build_digest 无生产调用点且未登记基线：整条 Layer-1 digest 链断"
        f"（F27 态复发）；hits={table[DIGEST_FN]['hits']}")


def test_report_readerless_fields():
    """打印当前无读者清单（pytest -s 可见；CHANGES-WP21 粘贴源）。恒过。"""
    print(report_text(audit()))


if __name__ == "__main__":
    print(report_text(audit()))
