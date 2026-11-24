"""The target device profile: target/B200.json (the draft of step 0) and its Markdown rendering.

What is pinned about the draft is what was verified before it was written (the numbers of the
target GPU, the protocol of the board, what a submission must be) and that nothing that was not
measured is stated as a number. Task targethw completes the profile; these stay true then.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from forgekernel import runcontext, target_profile as tp
from forgekernel.loop import driver
from forgekernel.target_profile import (PROFILE_FILE, REQUIRED_KEYS, ProfileError, load_profile,
                                        missing_keys, profile_path, render_file, render_markdown)
from test_loop_driver import FakeBackend, _ctx

REPO = Path(__file__).resolve().parents[1]
HEADINGS = ["## The target GPU", "## How the board evaluates", "## What a submission must be",
            "## How the development machine may differ", "## Rules for the code you write"]
THROUGHPUT = ("fp8_dense_pflops", "fp8_sparse_pflops", "nvfp4_dense_pflops", "sfu_teraexp_s",
              "tmem_kb_per_sm")


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _leaves(item, (*path, i))
    else:
        yield path, value


# --- the draft says what was verified -------------------------------------------------------


def test_the_draft_is_at_its_place_and_complete():
    assert PROFILE_FILE == "target/B200.json"
    assert profile_path() == REPO / "target" / "B200.json"
    assert profile_path("/r") == Path("/r/target/B200.json")
    profile = load_profile()
    assert missing_keys(profile) == []
    assert missing_keys({k: 1 for k in REQUIRED_KEYS}) == []
    assert missing_keys({}) == list(REQUIRED_KEYS)
    assert REQUIRED_KEYS == ("gpu", "compute_capability", "arch", "sm_count", "memory",
                             "clocks_locked", "tensor_throughput", "evaluation", "submission",
                             "dev_machine_may_differ")


def test_the_draft_states_the_facts_of_the_target_that_were_verified():
    p = load_profile()
    assert p["gpu"] == "NVIDIA B200"
    assert (p["compute_capability"], p["arch"], p["sm_count"]) == ("10.0", "sm_100a", 148)
    assert p["memory"] == {"capacity_gb": 192, "type": "HBM3e", "bandwidth_tb_s": 8.0}
    assert (p["clocks_locked"]["sm_mhz"], p["clocks_locked"]["memory_mhz"]) == (1500, 3996)
    assert [p["tensor_throughput"][k] for k in THROUGHPUT] == [5, 10, 10, 5, 256]
    notes = " ".join(p["arch_notes"].values()).lower()
    assert "sm_100a" in notes and "sm_103a" in notes and "sm_100f" in notes
    assert "no forward compatibility" in notes
    sw = p["software"]
    assert (sw["cuda"], sw["python"], sw["torch"], sw["triton"], sw["cutlass_dsl"]) == \
        ("13.1.1", "3.12", "2.9.0", "fbtriton 3.7.1", "nvidia-cutlass-dsl 4.4.2")
    assert sw["cutlass_headers"] == "/usr/local/cutlass"
    protocol = p["evaluation"]["protocol"]
    assert (protocol["warm_up_calls"], protocol["timed_calls"]) == (10, 50)
    assert "flushed before every call" in protocol["l2"]
    assert "new address" in protocol["address"] and "median" in protocol["reading"]
    assert "first kernel" in protocol["stopwatch"] and "Host gaps" in protocol["stopwatch"]
    scoring = p["evaluation"]["scoring"]
    assert scoring["workload"] == "S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))"
    assert scoring["failed_or_rejected"] == 0 and "arithmetic mean" in scoring["problem"]
    assert (p["evaluation"]["time_limits"]["compile_s"],
            p["evaluation"]["time_limits"]["evaluation_s"]) == (120, 600)


def test_the_draft_says_what_a_submission_must_be():
    s = load_profile()["submission"]
    assert "submission.py" in s["form"] and "run(*args)" in s["form"]
    assert "positional" in s["arguments"] and "pre-allocated outputs" in s["arguments"]
    assert set(s["allowed_dependencies"]) == {"CUTLASS", "CuTe DSL", "Triton", "cuBLAS",
                                              "cuDNN", "PyTorch"}
    assert {"FlashInfer", "FlashAttention", "FlashMLA", "DeepGEMM"} <= set(s["not_importable"])
    assert any("torch.utils.cpp_extension" in lang for lang in s["languages"])
    assert len(s["prohibited"]) == 5 and "scores 0" in s["prohibited_consequence"]
    assert "named exactly" in s["collection"] and "scores 0" in s["collection"]


def test_the_draft_invents_nothing_that_was_not_measured():
    """The development machine's SM count, its clock lock and the board's live limits are not
    known: the draft says so and gives no number for them."""
    dev = load_profile()["tensor_throughput"]["development_machine"]
    assert dev["gpu"] == "NVIDIA B300"
    assert [dev[k] for k in THROUGHPUT] == [5, 10, 15, 10.7, 256]
    assert not isinstance(dev.get("sm_count"), (int, float))
    text = render_file()
    for said in ("not yet measured", "not published", "unknown", "to be measured"):
        assert said in text, said
    differences = {d["difference"].split(".")[0]: d for d in load_profile()["dev_machine_may_differ"]}
    assert "not yet measured" in differences["SM count"]["difference"]
    assert "not yet measured" in differences["Clocks"]["difference"]
    assert "not published" in load_profile()["evaluation"]["time_limits"]["basis"]


def test_every_difference_of_the_development_machine_is_listed_with_what_to_do():
    differences = load_profile()["dev_machine_may_differ"]
    by_topic = {d["difference"].split(".")[0]: d for d in differences}
    assert list(by_topic)[:6] == ["Compute capability", "SM count", "Memory",
                                  "Dense NVFP4 throughput", "SFU throughput", "Clocks"]
    assert "Software" in by_topic
    assert all({"difference", "effect", "what_to_do", "status"} <= set(d) for d in differences)
    assert all(d[k].strip() for d in differences for k in ("difference", "effect", "what_to_do"))
    assert "FK_TARGET_SMS" in by_topic["SM count"]["what_to_do"]
    assert "148" in by_topic["SM count"]["what_to_do"] and "288 GB" in by_topic["Memory"]["difference"]
    assert "optimistic" in by_topic["Dense NVFP4 throughput"]["effect"]
    assert "optimistic" in by_topic["SFU throughput"]["effect"]
    assert "sm_100a" in by_topic["Compute capability"]["what_to_do"]


def test_the_rules_tell_an_agent_what_the_background_file_requires():
    rules = load_profile()["rules"]
    text = "\n".join(rules)
    assert len(rules) >= 12 and len(set(rules)) == len(rules)
    assert all(isinstance(r, str) and r.strip() == r for r in rules)
    for must in ("CC 10.0", "sm_100a", "148 SMs", "1500 / 3996 MHz", "Never name an architecture",
                 "Never hard-code the SM count", "FK_TARGET_SMS", "submission.py", "run(*args)",
                 "positional", "torch.utils.cpp_extension", "torch.backends", "L2 cache is flushed",
                 "new address", "first to the last kernel", "pre-allocated", "every legal input",
                 "compiles from scratch", "100 seconds"):
        assert must in text, must
    # the development machine's architecture is never held up as something to build for
    assert "sm_103a" not in text and "sm_100f" not in text


# --- the rendering -------------------------------------------------------------------------


def test_the_same_profile_always_renders_to_the_same_text():
    assert render_file() == render_file() == render_markdown(load_profile())
    text = render_file()
    assert text.startswith("# Target device profile: NVIDIA B200\n\nThe GPU every submission")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert all(line == line.rstrip() for line in text.splitlines())
    assert [ln for ln in text.splitlines() if ln.startswith("## ")] == HEADINGS


def test_everything_in_the_profile_reaches_the_text():
    profile, text = load_profile(), render_file()
    for path, leaf in _leaves(profile):
        shown = leaf if isinstance(leaf, str) else str(leaf)
        assert shown in text, (path, leaf)             # a string leaf as it is, a number as read
    body = text[text.index(HEADINGS[-1]):]
    assert all(f"\n{n}. {rule}\n" in body for n, rule in enumerate(profile["rules"], 1))
    assert text.index("148") < text.index(HEADINGS[0]) + 600    # the SM count, among the facts
    assert "- **SM count**: 148" in text and "- **Arch**: `sm_100a`" in text
    assert "- **Capacity (GB)**: 192" in text and "- **Bandwidth (TB/s)**: 8.0" in text
    assert "- **FP8 dense (PFLOPS)**: 5" in text and "- **NVFP4 dense (PFLOPS)**: 15" in text
    assert "- **SM clock (MHz)**: 1500" in text and "- **Memory clock (MHz)**: 3996" in text
    last = text.splitlines()[-1]
    assert last.startswith("*Status: draft") and last.endswith(").*")


def test_a_profile_file_that_is_not_one_is_named(tmp_path):
    absent = tmp_path / "nope.json"
    with pytest.raises(ProfileError, match="cannot read it") as e:
        load_profile(absent)
    assert str(absent) in str(e.value)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProfileError, match="does not parse as JSON"):
        load_profile(bad)
    for not_an_object in ("[]", '"x"', "7"):
        bad.write_text(not_an_object, encoding="utf-8")
        with pytest.raises(ProfileError, match="a target profile is a JSON object"):
            load_profile(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"gpu": "G", "sm_count": 3}), encoding="utf-8")
    assert load_profile(good) == {"gpu": "G", "sm_count": 3}
    assert render_file(good) == render_markdown({"gpu": "G", "sm_count": 3})


@pytest.mark.parametrize("not_a_profile", [[], "x", 7, None])
def test_rendering_needs_an_object(not_a_profile):
    with pytest.raises(TypeError, match="a target profile is a JSON object"):
        render_markdown(not_a_profile)


def test_the_sections_come_in_the_order_an_agent_should_meet_them():
    scrambled = {"rules": ["one rule"], "submission": {"form": "a file"},
                 "evaluation": {"protocol": {"timed_calls": 50}},
                 "dev_machine_may_differ": [{"difference": "d" * 60, "what_to_do": "w" * 60}],
                 "gpu": "G", "sm_count": 4, "status": "a draft", "role": "graded here",
                 "version": 2, "profile": "p"}
    lines = render_markdown(scrambled).splitlines()
    assert lines[0] == "# Target device profile: G" and lines[2] == "graded here"
    assert [ln for ln in lines if ln.startswith("## ")] == HEADINGS
    assert lines[lines.index("## The target GPU") + 2] == "- **SM count**: 4"
    assert lines[-1] == "*Status: a draft*"
    assert lines[lines.index("---") - 3: lines.index("---") - 1] == ["_profile: p_", "_version: 2_"]


def test_a_key_nothing_knows_about_still_appears():
    """The profile will grow (task targethw adds what it measures): a new key shows up with no
    change to the renderer -- among the facts, or as a section of its own when it holds a list."""
    profile = {"gpu": "G", "measured": {"sm_count": 9, "clocks": {"sm_mhz": 1200}},
               "notes": {"deep": {"deeper": ["x"]}}, "findings": [{"rule": "a", "line": 3}],
               "evaluation": {"protocol": {}}}
    text = render_markdown(profile)
    heads = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert heads == ["## The target GPU", "## How the board evaluates", "## Notes", "## Findings"]
    assert "- **Measured**" in text and "  - **SM count**: 9" in text
    assert "    - **SM clock (MHz)**: 1200" in text
    assert "| Rule | Line |\n| --- | --- |\n| a | 3 |" in text
    assert "- **Protocol**: (nothing)" in text


def test_a_list_takes_the_form_it_reads_best_as():
    text = render_markdown({"gpu": "G", "evaluation": {
        "short": ["a", "b", "c"], "long": ["x" * 60, "y" * 70],
        "table": [{"rule": "r", "line": 3}, {"rule": "s", "line": 4, "note": "n|q"}],
        "blocks": [{"difference": "d" * 80, "effect": "e"}, {"difference": "f" * 80}],
        "empty": [], "nothing": {}, "mixed": [1, [2, 3], {"k": "v"}]}})
    lines = text.splitlines()
    assert "- **Short**: a; b; c" in lines
    long_at = lines.index("- **Long**")
    assert lines[long_at + 1: long_at + 3] == ["  - " + "x" * 60, "  - " + "y" * 70]
    table_at = lines.index("| Rule | Line | Note |")
    assert lines[table_at: table_at + 4] == ["| Rule | Line | Note |", "| --- | --- | --- |",
                                             "| r | 3 | unknown |", "| s | 4 | n\\|q |"]
    d, f = "  - **Difference: " + "d" * 80 + "**", "  - **Difference: " + "f" * 80 + "**"
    blocks_at = lines.index(d)
    assert lines[blocks_at: blocks_at + 3] == [d, "    - **Effect**: e", f]
    assert lines[blocks_at - 1] == "- **Blocks**"
    assert "- **Empty**: (nothing)" in lines and "- **Nothing**: (nothing)" in lines
    mixed_at = lines.index("- **Mixed**")
    assert lines[mixed_at: mixed_at + 7] == [
        "- **Mixed**", "  - 1", "  -", "    - 2", "    - 3", "  -", "    - **K**: v"]


def test_rules_are_numbered_and_nothing_else_is():
    text = render_markdown({"gpu": "G", "rules": ["first", "second"],
                            "submission": {"prohibited": ["a" * 60, "b" * 60]}})
    lines = text.splitlines()
    assert [ln for ln in lines if re.match(r"\d+\. ", ln)] == ["1. first", "2. second"]
    assert lines[lines.index("- **Prohibited**") + 1: lines.index("- **Prohibited**") + 3] == \
        ["  - " + "a" * 60, "  - " + "b" * 60]


def test_without_a_gpu_or_anything_else_it_still_renders_something():
    assert render_markdown({}) == "# Target device profile: unknown\n"
    assert render_markdown({"gpu": "G"}) == "# Target device profile: G\n"


def test_labels_and_values_are_shown_as_a_reader_meets_them():
    assert tp._label("fp8_dense_pflops") == "FP8 dense (PFLOPS)"
    assert tp._label("bandwidth_tb_s") == "Bandwidth (TB/s)"
    assert tp._label("tmem_kb_per_sm") == "TMEM (KB per SM)"
    assert tp._label("sfu_teraexp_s") == "SFU (TeraExp/s)"
    assert tp._label("sm_count") == "SM count" and tp._label("warm_up_calls") == "Warm up calls"
    assert tp._label("compile_s") == "Compile limit (s)" and tp._label("t_sol") == "T_SOL"
    assert tp._label("") == "" and tp._label("x_") == "X"
    assert [tp._scalar(v) for v in (None, True, False, 8.0, 1500, "a phrase", "fbtriton 3.7.1",
                                    "torch.Tensor")] == \
        ["unknown", "yes", "no", "8.0", "1500", "a phrase", "fbtriton 3.7.1", "torch.Tensor"]
    assert tp._scalar("sm_100a") == "`sm_100a`" and tp._scalar("/usr/local/cutlass") == \
        "`/usr/local/cutlass`"
    assert tp._scalar("run(*args)") == "`run(*args)`"


def test_the_rendered_profile_is_what_a_run_hands_its_agents(tmp_path, monkeypatch):
    """End to end over the seam the other tasks use: the rendered file as the target context
    file, named in the prompt an agent is given."""
    shown = tmp_path / "B200.md"
    shown.write_text(render_file(), encoding="utf-8")
    for run in ("run", "run2"):
        (tmp_path / run).mkdir()
    monkeypatch.setenv(*next(iter(runcontext.env_for({"target": shown}).items())))
    assert runcontext.path_for("target") == shown
    prompt = driver._student_prompt(_ctx(tmp_path / "run", FakeBackend([], [])), 0)
    assert f"Target device: `{shown}`" in prompt
    assert driver._bindings(_ctx(tmp_path / "run2", FakeBackend([], [])))["{TARGET_PROFILE_PATH}"] \
        == str(shown)
