"""The cost model: JSON validation, and the one teacher turn ``--estimate`` spends.

The validation cases are the ones a mistyped model actually hits (a missing field, a
string where a number belongs, a negative byte count, a name the card does not have); the
estimate cases use a fake backend, so no agent runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from forgekernel.roofline import costmodel

ROOT = Path(__file__).resolve().parents[1]

# meta/*.md is operator-agnostic (invariant L1). The list is the one the repository's other
# prompt tests use; a cost-model prompt that names a kind of kernel has stopped being a
# prompt about counting and become a hint about the answer.
FORBIDDEN = (
    "attention", "softmax", "matmul", "gemm", "rmsnorm", "layernorm", "conv2d", "conv3d",
    "mla", "gqa", "moe", "rope", "dropout", "swiglu", "silu", "gelu", "mamba", "ssm",
    "transformer", "embedding", "head_dim", "seq_len", "kv_cache", "causal", "vocab",
    "hidden_size", "batch_size", "query", "key", "value",
)


def entry(name="wl0", flops=2.0e9, read=8.0e7, written=1.0e7, peak="bf16", **extra):
    return {"name": name, "flops": flops, "bytes_read": read, "bytes_written": written,
            "peak": peak, **extra}


def problem(workloads=(("wl0", {"B": 4}),)):
    group = [{"index": i, "name": name, "workload": {"uuid": name, "axes": axes},
              "sol_board": {"t_b_ms": 1.0, "t_sol_ms": 0.2}}
             for i, (name, axes) in enumerate(workloads)]
    payload = {"definition": {"name": "demo_op", "axes": {"B": {"type": "var"}},
                              "inputs": {"x": {"shape": ["B"], "dtype": "float32"}},
                              "outputs": {"y": {"shape": ["B"], "dtype": "float32"}},
                              "reference": "def run(x):\n    return x * 2\n"},
               "workload": group[0]["workload"], "sol_board": group[0]["sol_board"]}
    if len(group) > 1:
        payload["group"] = group
    return payload


# -- validation --------------------------------------------------------------------------


def test_a_valid_model_parses_from_a_dict_and_from_a_bare_list():
    wrapped = costmodel.parse({"workloads": [entry()]}, problem="p")
    bare = costmodel.parse([entry()], problem="p")
    assert wrapped.by_name()["wl0"].flops == 2.0e9
    assert bare.workloads[0].bytes_total == 9.0e7
    assert bare.workloads[0].peak == "bf16"


def test_peak_defaults_to_bf16_and_none_requires_no_flops():
    model = costmodel.parse({"workloads": [{"name": "a", "flops": 0, "bytes_read": 1,
                                            "bytes_written": 0}]})
    assert model.workloads[0].peak == "bf16"
    model = costmodel.parse({"workloads": [{"name": "a", "flops": 0, "bytes_read": 1,
                                            "bytes_written": 0, "peak": "none"}]})
    assert model.workloads[0].peak == "none"
    with pytest.raises(costmodel.CostModelError, match="no arithmetic"):
        costmodel.parse({"workloads": [{"name": "a", "flops": 5, "bytes_read": 1,
                                        "bytes_written": 0, "peak": "none"}]})


@pytest.mark.parametrize("broken, match", [
    ({"flops": 1.0, "bytes_read": 1.0, "bytes_written": 1.0}, "non-empty 'name'"),
    ({"name": "a", "bytes_read": 1.0, "bytes_written": 1.0}, "missing flops"),
    ({"name": "a", "flops": 1.0, "bytes_written": 1.0}, "missing bytes_read"),
    ({"name": "a", "flops": 1.0, "bytes_read": 1.0}, "missing bytes_written"),
    ({"name": "a", "flops": "1e9", "bytes_read": 1.0, "bytes_written": 1.0},
     "must be a number"),
    ({"name": "a", "flops": True, "bytes_read": 1.0, "bytes_written": 1.0},
     "must be a number"),
    ({"name": "a", "flops": -1.0, "bytes_read": 1.0, "bytes_written": 1.0},
     "must not be negative"),
    ({"name": "a", "flops": float("inf"), "bytes_read": 1.0, "bytes_written": 1.0},
     "must be finite"),
    ({"name": "a", "flops": 0.0, "bytes_read": 0.0, "bytes_written": 0.0},
     "nothing to build a floor from"),
    ({"name": "a", "flops": 1.0, "bytes_read": 1.0, "bytes_written": 1.0,
      "peak": "tf32"}, "is not one of"),
    ({"name": "a", "flops": 1.0, "bytes_read": 1.0, "bytes_written": 1.0,
      "basis": "not an object"}, "'basis' must be an object"),
])
def test_every_broken_entry_is_refused_by_name(broken, match):
    with pytest.raises(costmodel.CostModelError, match=match) as info:
        costmodel.parse({"workloads": [broken]})
    assert "workload 0" in str(info.value)


def test_a_duplicate_name_is_refused():
    with pytest.raises(costmodel.CostModelError, match="appears twice"):
        costmodel.parse({"workloads": [entry("a"), entry("a")]})


def test_the_wrong_envelope_is_refused():
    with pytest.raises(costmodel.CostModelError, match="no 'workloads'"):
        costmodel.parse({"entries": []})
    with pytest.raises(costmodel.CostModelError, match="non-empty list"):
        costmodel.parse({"workloads": []})
    with pytest.raises(costmodel.CostModelError, match="object or a list"):
        costmodel.parse("nope")


def test_load_reads_a_file_and_names_it_in_a_failure(tmp_path):
    good = tmp_path / "costs.json"
    good.write_text(json.dumps({"workloads": [entry()]}))
    model = costmodel.load(good)
    assert model.source == f"file {good}"
    assert model.workloads[0].name == "wl0"
    with pytest.raises(costmodel.CostModelError, match="No such file"):
        costmodel.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(costmodel.CostModelError, match="not JSON"):
        costmodel.load(bad)


def test_cover_reports_missing_and_extra_names():
    model = costmodel.parse({"workloads": [entry("wl0"), entry("other")]})
    problems = costmodel.cover(model, ["wl0", "wl1"])
    assert any("wl1" in p and "no entry" in p for p in problems)
    assert any("other" in p for p in problems)
    assert costmodel.cover(model, ["wl0", "other"]) == []


# -- --estimate: one teacher turn, with one retry -----------------------------------------


class FakeBackend:
    """``run_agent`` only: the teacher-turn seam, scripted."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.roles = []

    def run_agent(self, prompt, *, role="student"):
        self.prompts.append(prompt)
        self.roles.append(role)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


GOOD = json.dumps({"workloads": [entry()]})


def test_a_rejected_first_answer_is_fed_back_and_the_second_one_used():
    backend = FakeBackend(["sorry, here is prose", GOOD])
    model = costmodel.estimate(problem(), backend)
    assert len(backend.prompts) == 2
    assert backend.roles == ["teacher", "teacher"]
    assert "rejected" in backend.prompts[1]
    assert "my.ownership" not in backend.prompts[1]
    assert [a.get("ok") for a in model.attempts] == [None, True]
    assert model.source == "teacher estimate (2 attempt(s))"
    assert model.workloads[0].flops == 2.0e9


def test_two_bad_answers_are_an_error_not_a_guess():
    backend = FakeBackend(["no json", json.dumps({"workloads": [{"name": "wl0"}]})])
    with pytest.raises(costmodel.CostModelError, match="failed after 2 attempt"):
        costmodel.estimate(problem(), backend)
    assert len(backend.prompts) == 2


def test_a_turn_that_raises_costs_an_attempt_and_the_error_rides_the_feedback():
    backend = FakeBackend([RuntimeError("agent died"), GOOD])
    model = costmodel.estimate(problem(), backend)
    assert model.attempts[0]["error"].startswith("agent turn failed")
    assert "agent died" in backend.prompts[1]


def test_the_prompt_is_rendered_without_placeholders_and_names_no_operator():
    backend = FakeBackend([GOOD])
    costmodel.estimate(problem((("wl0", {"B": 4}), ("wl1", {"B": 64}))), backend,
                       target_profile="/x/B200.json")
    prompt = backend.prompts[0]
    lowered = prompt.lower()
    for word in FORBIDDEN:
        assert not re.search(rf"\b{re.escape(word)}\b", lowered), f"the prompt names {word!r}"
    for name in ("PROBLEM_NAME", "DEFINITION", "WORKLOADS", "N_WORKLOADS", "FEEDBACK",
                 "TARGET_PROFILE_PATH"):
        assert "{" + name + "}" not in prompt, f"{name} survived rendering"
    assert "demo_op" in prompt and "\"B\": 4" in prompt and "/x/B200.json" in prompt
    # the definition's own source is data, and it travels as-is
    assert "def run(x)" in prompt


def test_the_prompt_template_binds_every_placeholder_it_names():
    template = (ROOT / costmodel.PROMPT_FILE).read_text(encoding="utf-8")
    named = set(costmodel.unreplaced(template))
    bound = set(costmodel.prompt_bindings(problem()))
    assert named <= bound, f"nothing binds {sorted(named - bound)}"
    assert named == {"PROBLEM_NAME", "DEFINITION", "WORKLOADS", "N_WORKLOADS", "FEEDBACK",
                     "TARGET_PROFILE_PATH"}


def test_an_unbound_placeholder_in_a_prompt_file_is_refused(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("count {DEFINITION} and {NOBODY_BINDS_THIS}\n")
    with pytest.raises(costmodel.CostModelError, match="nothing binds"):
        costmodel.estimate(problem(), FakeBackend([GOOD]), prompt_path=prompt)


def test_a_missing_prompt_file_is_an_error(tmp_path):
    with pytest.raises(costmodel.CostModelError, match="No such file"):
        costmodel.estimate(problem(), FakeBackend([GOOD]), prompt_path=tmp_path / "no.md")


def test_json_in_prose_is_still_read():
    answer = ("Here is the model:\n```json\n" + GOOD + "\n```\nDone.")
    model = costmodel.estimate(problem(), FakeBackend([answer]))
    assert model.workloads[0].bytes_read == 8.0e7
