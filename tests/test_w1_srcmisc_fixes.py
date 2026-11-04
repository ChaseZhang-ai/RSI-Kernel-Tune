# --- P3-6: the judge's axis check spans every shape, not only the first -------------

def test_judge_axis_check_spans_every_shapes_axes(tmp_path):
    """axes came from shapes[0] alone: an axis introduced by a later shape
    ('k' below, only on s2) read as undeclared on the definition that uses it —
    or, ordered the other way, a bogus one slipped past. The check now sees the
    union."""
    import json
    from peregrine_kernel import judge

    class _Shape:
        inputs = ()

        def __init__(self, name, axes):
            self.name, self.axes = name, axes

    class _Task:
        shapes = [_Shape("s1", {"m": 4}), _Shape("s2", {"k": 8})]

    d = tmp_path / "check"
    d.mkdir()
    good = {"inputs": {"x": {"shape": ["m", "k"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["k"], "dtype": "float32"}}}
    (d / judge.DEFINITION).write_text(json.dumps(good))
    (d / judge.JUDGE).write_text(
        "import torch\n\n\ndef run(x):\n    return x\n\n\n"
        "def make_inputs(axes, group, seed, device):\n    return {}\n")
    (d / judge.GROUPS).write_text("{}")
    full = judge.check_problems(d, _Task())
    assert not any("task's axes" in p for p in full), full

    bad = {"inputs": {"x": {"shape": ["zzz"], "dtype": "float32"}},
           "outputs": {"y": {"shape": ["k"], "dtype": "float32"}}}
    (d / judge.DEFINITION).write_text(json.dumps(bad))
    v = judge.check_problems(d, _Task())
    assert any("'zzz'" in p and "'k'" in p and "'m'" in p for p in v), v


# --- P3-7: a malformed prep shape lands in problems, not in a traceback -------------

def test_prep_report_holds_a_malformed_shape_without_crashing(tmp_path):
    """[int(s) for s in got["shape"]] raised straight out of check_prep_report
    on a hand-written bench that reported the shape as a string (or a list
    carrying a non-number): the whole prep block went unheld because the check
    died instead of naming the tensor. Bad shapes are problems now."""
    from test_window_contract import SYMMETRIZED
    from peregrine_kernel.window_contract import check_prep_report, parse_window_contract
    contract = parse_window_contract(SYMMETRIZED)
    gc = {"shape": [4, 32], "dtype": "torch.float32"}       # g_cumsum, always well-formed
    ok = {"outputs": {"A": {"shape": [4, 32, 64], "dtype": "bfloat16"}, "g_cumsum": gc}}
    assert check_prep_report(contract, ok) == []

    as_string = {"outputs": {"A": {"shape": "[4, 32, 64]", "dtype": "bfloat16"},
                             "g_cumsum": gc}}
    v = check_prep_report(contract, as_string)
    assert len(v) == 1 and "malformed shape" in v[0] and "'A'" in v[0], v

    junk_element = {"outputs": {"A": {"shape": [4, 32, {"b": 1}], "dtype": "bfloat16"},
                                "g_cumsum": gc}}
    v = check_prep_report(contract, junk_element)
    assert len(v) == 1 and "malformed shape" in v[0] and "[4, 32, {'b': 1}]" in v[0], v

    # a report with no shape at all keeps its old meaning: the mismatch path,
    # not the malformed one
    no_shape = {"outputs": {"A": {"dtype": "bfloat16"}, "g_cumsum": gc}}
    v = check_prep_report(contract, no_shape)
    assert v == ["prep produced 'A' with shape [], declared [4, 32, 64]"], v
