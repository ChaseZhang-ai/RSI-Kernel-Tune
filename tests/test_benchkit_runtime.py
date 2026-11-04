"""Tests for the card's bench engine (src/peregrine_kernel/card_template/eval/benchkit.py):
definition axes/shapes, the random input draws, workload views and the meta tools'
workload scoping (PK_LB_WORKLOAD_SCOPE) across the bench and the NCU window.

CPU only. Needs torch (skipped without it). The task suite's correctness/timing
runtime is covered by test_task_suite.py / test_multi_workload_perf.py / the fresh
recheck and precision tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "pk_template_benchkit", REPO_ROOT / "src" / "peregrine_kernel" / "card_template" / "eval" / "benchkit.py")
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)

RMS_DEF = {
    "name": "rmsnorm_h64", "op_type": "rmsnorm",
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 64}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
               "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
}


def _problem(**wl) -> dict:
    workload = {"uuid": "w0", "axes": {"batch_size": 5},
                "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}},
                "input_groups": [{"name": "plain", "kind": "generated", "seed": 1}]}
    workload.update(wl)
    return {"definition": json.loads(json.dumps(RMS_DEF)),
            "workload": workload, "dps": True, "_dir": "."}


# --- definitions and inputs -------------------------------------------------------------

def test_expr_axes_and_digit_dims() -> None:
    defn = {"axes": {"b": {"type": "var"}, "h": {"type": "const", "value": 8},
                     "t": {"type": "expr", "expression": "b * h // 2"}}}
    axes = lb.resolved_axes(defn, {"b": 3})
    assert axes == {"h": 8, "b": 3, "t": 12}
    assert lb.tensor_shapes({"x": {"shape": ["t", "4"]}, "s": {"shape": None}}, axes) == \
        {"x": (12, 4), "s": None}
    with pytest.raises(ValueError, match="not an integer"):
        lb.resolve_shape_expression("b / 2", {"b": 3})
    with pytest.raises(TypeError):
        lb.resolve_shape_expression("abs(b)", {"b": 3})


def test_name_heuristics() -> None:
    p = _problem()
    p["definition"]["inputs"] = {
        "input_layernorm_weight": {"shape": ["hidden_size"], "dtype": "float32"},
        "norm_bias": {"shape": ["hidden_size"], "dtype": "float32"},
        "attention_mask": {"shape": ["batch_size", "batch_size"], "dtype": "float32",
                           "description": "causal attention mask"},
        "rstd": {"shape": ["batch_size"], "dtype": "float32"},
        "up_proj": {"shape": ["hidden_size", "hidden_size"], "dtype": "float32"},
        "A_log": {"shape": ["hidden_size"], "dtype": "float32"},
    }
    p["workload"]["inputs"] = {}
    torch.manual_seed(0)
    nw, nb, mask, rstd, proj, a_log = lb.gen_inputs(p, "cpu")
    assert torch.equal(nw, torch.ones(64)) and torch.equal(nb, torch.zeros(64))
    assert mask[0, 1] == torch.finfo(torch.float32).min and mask[1, 0] == 0
    assert float(rstd.min()) >= 0.1
    assert float(proj.std()) < 0.3           # randn / sqrt(fan_in=64) ~ 0.125
    assert float(a_log.max()) <= -1.0 and float(a_log.min()) >= -5.0


def test_safetensors_input_is_shape_and_dtype_checked(tmp_path: Path) -> None:
    st = pytest.importorskip("safetensors.torch")
    st.save_file({"w": torch.ones(64)}, str(tmp_path / "b.safetensors"))
    p = _problem(inputs={
        "hidden_states": {"type": "random"},
        "weight": {"type": "safetensors", "path": "b.safetensors", "tensor_key": "w"}})
    p["_dir"] = str(tmp_path)
    assert torch.equal(lb.gen_inputs(p, "cpu")[1], torch.ones(64))


# --- workload groups ----------------------------------------------------------------------

def _group_problem(sizes=(3, 17, 64)) -> dict:
    p = _problem()
    p["group"] = []
    for i, b in enumerate(sizes):
        wl = {"uuid": f"w{i}", "axes": {"batch_size": b},
              "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}},
              "input_groups": [{"name": "plain", "kind": "generated", "seed": 1}]}
        p["group"].append({"index": 10 + i, "name": f"wl{10 + i}", "workload": wl})
    p["workload"] = p["group"][0]["workload"]
    return p


def test_workload_views_give_each_workload_its_own_shape() -> None:
    p = _group_problem()
    views = lb.workload_views(p)
    assert [v["_wl_name"] for v in views] == ["wl10", "wl11", "wl12"]
    assert [v["workload"]["axes"]["batch_size"] for v in views] == [3, 17, 64]
    assert [tuple(lb.allocate_outputs(v, "cpu")[0].shape) for v in views] == \
        [(3, 64), (17, 64), (64, 64)]
    assert lb.workload_views(_problem()) == [_problem()]


# --- the meta tools' workload scope ---------------------------------------------------------

def test_workload_scope_leaves_a_single_workload_card_alone() -> None:
    p = _problem()
    before = json.loads(json.dumps(p))
    assert lb.apply_workload_scope(p, env={lb.WORKLOAD_SCOPE_ENV: "representative"}) is None
    assert p == before
    g = _group_problem()
    assert lb.apply_workload_scope(g, env={}) is None and len(g["group"]) == 3
    assert lb.apply_workload_scope(g, env={lb.WORKLOAD_SCOPE_ENV: "all"}) is None


def test_workload_scope_names_one_workload_of_the_group() -> None:
    """The meta tools read each of meta_workloads in turn: a group entry's name narrows
    the card to it; a name the group lacks raises, never times the group."""
    g = _group_problem()
    rec = lb.apply_workload_scope(g, env={lb.WORKLOAD_SCOPE_ENV: "wl12"})
    assert rec == {"workloads": "wl12", "rep": "wl12", "of": 3}
    assert [x["name"] for x in g["group"]] == ["wl12"] \
        and g["workload"]["axes"]["batch_size"] == 64
    with pytest.raises(ValueError, match="names no workload"):
        lb.apply_workload_scope(_group_problem(), env={lb.WORKLOAD_SCOPE_ENV: "wl99"})


def test_named_scope_is_tagged_in_the_eval_json_and_is_no_gate(tmp_path, monkeypatch,
                                                               capsys) -> None:
    """A scoped run's JSON carries ``workload_scope``; the loop's gate parser refuses it:
    a meta reading, never a gate."""
    from peregrine_kernel.loop.parse import parse_eval
    p = _group_problem()
    p["definition"] = {**RMS_DEF,
                       "reference": (
                           "import torch\n\n"
                           "def run(hidden_states, weight):\n"
                           "    return hidden_states\n\n"
                           "def make_inputs(axes, group='plain', seed=0, device='cpu'):\n"
                           "    b = axes['batch_size']\n"
                           "    g = torch.Generator().manual_seed(seed)\n"
                           "    return {'hidden_states': torch.randn(b, 64, generator=g),\n"
                           "            'weight': torch.ones(64)}\n"),
                       "custom_inputs_entrypoint": "make_inputs"}
    path = tmp_path / "card" / "eval" / "problem" / "problem.json"
    path.parent.mkdir(parents=True)
    (tmp_path / "card" / "kernel.py").write_text(
        "def run(hidden_states, weight):\n    return hidden_states\n")
    p.pop("_dir", None)
    path.write_text(json.dumps(p))
    monkeypatch.setenv(lb.WORKLOAD_SCOPE_ENV, "wl11")
    assert lb.run_bench(["--mode", "correctness"], problem_path=path) == 0
    text = capsys.readouterr().out
    out = json.loads(text.split("=== JSON ===")[1])
    assert out["workload_scope"] == {"workloads": "wl11", "rep": "wl11", "of": 3}
    assert "WORKLOAD SCOPE: workload wl11 only (1 of 3" in text
    assert not parse_eval(text).correctness


def test_ncu_window_profiles_the_scoped_workload(tmp_path, monkeypatch, capsys) -> None:
    """eval/ncu_run.py follows PK_LB_WORKLOAD_SCOPE, so a meta tool's NCU (stall pricing,
    the role pass, ablation timing, the timeline launcher) reads the workload its bench
    timed; unset, a group card profiles its representative as before."""
    built: list = []
    monkeypatch.setattr(lb, "kernel_fence", lambda problem: None)
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (lambda *a: None, False))

    def fake_task_inputs(view, group, device):
        built.append(view["workload"]["axes"]["batch_size"])
        return []

    monkeypatch.setattr(lb, "task_inputs", fake_task_inputs)
    monkeypatch.setattr(lb.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(lb.torch.cuda.profiler, "start", lambda: None)
    monkeypatch.setattr(lb.torch.cuda.profiler, "stop", lambda: None)
    path = tmp_path / "card" / "eval" / "problem" / "problem.json"
    path.parent.mkdir(parents=True)
    (tmp_path / "card" / "kernel.py").write_text(
        "def run(hidden_states, weight):\n    return hidden_states\n")
    p = _group_problem()
    p.pop("_dir", None)
    path.write_text(json.dumps(p))
    monkeypatch.delenv(lb.WORKLOAD_SCOPE_ENV, raising=False)
    lb.ncu_window(problem_path=path)
    monkeypatch.setenv(lb.WORKLOAD_SCOPE_ENV, "wl12")
    lb.ncu_window(problem_path=path)
    # unscoped: every input group of every shape; scoped: the named shape only
    assert built == [3, 17, 64, 64]
    assert "ncu: workload wl12 (1 of 3" in capsys.readouterr().out


# --- precision default -------------------------------------------------------------------

def test_precision_measures_by_default_and_off_stops_it() -> None:
    assert lb.precision_on({}) is True                    # WP24: strict is the default
    assert lb.precision_on({"precision_contract": "official"}) is True
    assert lb.precision_on({"precision_contract": "off"}) is False
