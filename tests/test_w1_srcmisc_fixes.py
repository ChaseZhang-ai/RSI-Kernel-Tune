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
