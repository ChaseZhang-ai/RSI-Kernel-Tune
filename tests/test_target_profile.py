"""The target device profile: ``target/B200.json`` (the final profile task targethw delivered)
and the Markdown an agent reads.

What is pinned here is what was verified before the draft was written (the numbers of the target
GPU, the protocol of the board, what a submission must be), at the profile's final keys, plus
the rendering contract and the seam through which a run hands the profile to its agents.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rsi_kernel_tune import runcontext
from rsi_kernel_tune import target_profile as tp
from rsi_kernel_tune.loop import driver
from test_loop_driver import FakeBackend, _ctx

REPO = Path(__file__).resolve().parents[1]
HEADINGS = ["## The submission", "## Rules (hard)", "## Your development machine (probe)",
            "## Target vs development", "## Before you call a kernel done"]
# the profile's top-level shape (the JSON is the single source; a key added here is deliberate)
KNOWN_KEYS = {"schema", "name", "summary", "device", "development_gpu", "software",
              "evaluation", "submission", "rules", "codegen"}


# --- what the profile says about the target -------------------------------------------------


def test_the_profile_is_at_its_place_and_declares_its_schema():
    assert tp.PROFILE_FILE == "target/B200.json"
    assert tp.profile_path() == REPO / "target" / "B200.json"
    assert tp.profile_path("/r") == Path("/r/target/B200.json")
    profile = tp.load()
    assert profile["schema"] == "rsi_kernel_tune.target-profile/1"
    assert set(profile) == KNOWN_KEYS


def test_the_facts_of_the_target_that_were_verified():
    p = tp.load()
    d = p["device"]
    assert d["name"] == "NVIDIA B200"
    assert (d["compute_capability"], d["cuda_arch"], d["multi_processor_count"]) == \
        ([10, 0], "sm_100a", 148)
    assert (d["memory_gib"], d["memory_bandwidth_tb_s"]) == (192, 8.0)
    assert [d[k] for k in ("dense_fp8_pflops", "sparse_fp8_pflops", "dense_fp4_pflops",
                           "sparse_fp4_pflops", "sfu_tera_exp_s", "tmem_kb_per_sm")] == \
        [5.0, 10.0, 10.0, 20.0, 5.0, 256]
    assert (d["clocks"]["sm_mhz"], d["clocks"]["mem_mhz"], d["clocks"]["locked"]) == \
        (1500, 3996, True)
    note = d["accel_arch_note"]
    assert "sm_100a" in note and "sm_103" in note and "sm_100f" in note
    assert "no forward or backward compatibility" in note
    sw = p["software"]
    assert (sw["cuda"], sw["python"], sw["torch"], sw["triton"], sw["cutlass_dsl"]) == \
        ("13.1.1", "3.12", "2.9.0", "3.7.1", "4.4.2")
    assert sw["triton_distribution"] == "fbtriton" and sw["cutlass_headers"] == "/usr/local/cutlass"
    ev = p["evaluation"]
    assert (ev["warmup_calls"], ev["timed_calls"]) == (10, 50)
    assert ev["l2_flushed_per_call"] is True and ev["inputs_cloned_per_call"] is True
    assert "CUPTI" in ev["timer"] and "gaps" in ev["timer"]
    assert (ev["compile_timeout_s"], ev["job_timeout_s"]) == (120, 600)


def test_the_profile_says_what_a_submission_must_be():
    s = tp.load()["submission"]
    assert "submission.py" in s["track_format"]
    assert "run(*args)" in s["entry"] and "argument order" in s["entry"]
    assert "pre-allocated" in s["entry"]
    assert {"PyTorch operators", "Triton", "CuTe DSL", "CUTLASS headers"} <= \
        set(s["allowed_libraries"])
    assert any("cuBLAS" in lib and "cuDNN" in lib for lib in s["allowed_libraries"])
    assert s["embedded_only_libraries"] == ["FlashInfer", "FlashAttention", "FlashMLA", "DeepGEMM"]
    assert "torch.utils.cpp_extension" in s["embedded_source"]
    assert len(s["prohibited"]) == 5
    assert s["limits"] == {"upload_mb": 10, "archive_mb": 50, "extracted_mb": 100}


def test_the_profile_invents_nothing_that_was_not_measured():
    """The development machine's numbers are reference values and the probe's measurements live
    in ``target/dev.json`` (not committed): without one, the rendered text says so."""
    dev = tp.load()["development_gpu"]
    assert dev["name"] == "NVIDIA B300"
    assert (dev["compute_capability"], dev["cuda_arch"]) == ([10, 3], "sm_103a")
    assert [dev[k] for k in ("dense_fp4_pflops", "sfu_tera_exp_s")] == [15.0, 10.7]
    assert "Reference numbers" in dev["note"] and "target/dev.json" in dev["note"]
    text = tp.render_markdown(tp.load())
    assert "No probe result here" in text and "reference ones" in text


def test_the_profile_names_where_its_facts_come_from():
    p = tp.load()
    for block in ("device", "development_gpu", "software", "evaluation"):
        source = p[block]["source"]
        assert isinstance(source, list) and source, block
        assert all(isinstance(s, str) and s.strip() for s in source), block


def test_every_difference_of_the_development_machine_is_listed_with_what_to_do():
    text = tp.render_markdown(tp.load())
    for row in ("| compute capability |", "| cubin compatibility |", "| SMs |", "| memory |",
                "| HBM bandwidth |", "| dense FP4 |", "| SFU (EX2) |", "| clocks while timing |",
                "| stack |"):
        assert row in text, row
    assert "What that changes while you work here:" in text
    assert "optimistically fast" in text
    assert "torch.cuda.get_device_properties(0).multi_processor_count" in text
    assert "FK_TARGET_SMS" in text


def test_the_rules_tell_an_agent_what_the_background_file_requires():
    rules = tp.load()["rules"]
    assert len(rules) >= 12
    texts = [rule["text"] for rule in rules]
    assert len(set(texts)) == len(texts)
    assert [rule["id"] for rule in rules] == list(range(1, len(rules) + 1))
    joined = "\n".join(texts)
    for must in ("CC 10.0", "sm_100a", "148 SMs", "1500 / 3996 MHz", "Never name an architecture",
                 "Never hard-code the SM count", "FK_TARGET_SMS", "submission.py", "run(*args)",
                 "positional", "torch.utils.cpp_extension", "torch.backends",
                 "L2 cache is flushed", "new address", "first to the last kernel",
                 "pre-allocated", "compiles from scratch", "minute"):
        assert must in joined, must
    # the development machine's architecture is never held up as something to build for
    assert "sm_103a" not in joined and "sm_100f" not in joined


# --- the rendering --------------------------------------------------------------------------


def test_the_same_profile_always_renders_to_the_same_text():
    first = tp.render_markdown(tp.load())
    assert first == tp.render_markdown(tp.load())
    assert first.startswith("# Target hardware: NVIDIA B200 (graded) -- develop on another "
                            "Blackwell\n\nEverything you write is graded on NVIDIA B200")
    assert first.endswith("\n") and not first.endswith("\n\n")


def test_the_sections_come_in_the_order_an_agent_should_meet_them():
    heads = [ln for ln in tp.render_markdown(tp.load()).splitlines() if ln.startswith("## ")]
    assert heads == HEADINGS


def test_the_facts_and_rules_a_kernel_author_needs_reach_the_text():
    profile, text = tp.load(), tp.render_markdown(tp.load())
    for rule in profile["rules"]:
        assert f"\n{rule['id']}. {rule['text']}\n" in "\n" + text + "\n", rule["id"]
    for shown in ("NVIDIA B200", "148", "1500", "3996", "sm_100a", "192 GiB", "8 TB/s",
                  "CUDA 13.1.1", "torch 2.9.0", "triton 3.7.1 (fbtriton)",
                  "cutlass-dsl 4.4.2"):
        assert shown in text, shown


def test_rules_are_numbered_and_nothing_else_is():
    profile = tp.load()
    numbered = [ln for ln in tp.render_markdown(profile).splitlines()
                if re.match(r"\d+\. ", ln)]
    assert numbered == [f"{rule['id']}. {rule['text']}" for rule in profile["rules"]]


def test_the_table_compares_the_target_with_the_development_machine():
    text = tp.render_markdown(tp.load())
    assert "| quantity | target | development |" in text
    for row in ("| compute capability | 10.0 (sm_100a) | 10.3 (sm_103a) |",
                "| cubin compatibility | sm_100a only | sm_103a only: neither runs on the other |",
                "| SMs | 148 | 160 |", "| memory | 192 GiB | 288 GiB |",
                "| HBM bandwidth | 8 TB/s | 8 TB/s |",
                "| dense FP4 | 10 PFLOPS | 15 PFLOPS (1.5x) |",
                "| SFU (EX2) | 5 TeraExp/s | 10.7 TeraExp/s (~2x) |",
                "| clocks while timing | SM 1500 MHz / memory 3996 MHz (locked) | not recorded |"):
        assert row in text, row


def test_the_development_section_says_what_the_probe_measured():
    dev = {"gpu": {"name": "NVIDIA B300", "compute_capability": [10, 3], "cuda_arch": "sm_103a",
                   "multi_processor_count": 160, "memory_gib": 288, "clocks_locked": True,
                   "driver": "580.65"},
           "software": {"cuda_toolkit": "13.1.1", "torch": "2.9.0", "triton": "3.7.1",
                        "triton_is_fbtriton": True, "cutlass_dsl": "4.4.2"},
           "smokes": {"triton": {"run": True}}}
    text = tp.render_markdown(tp.load(), dev)
    assert "No probe result here" not in text
    assert ("- NVIDIA B300, CC 10.3 (sm_103a), 160 SMs, 288 GiB, driver 580.65, "
            "clocks locked") in text
    assert ("- measured stack: cuda 13.1.1, torch 2.9.0, triton 3.7.1 (fbtriton), "
            "cutlass-dsl 4.4.2") in text
    assert "- smokes: triton ok" in text
    reference = tp.render_markdown(tp.load())
    assert "- reference: NVIDIA B300, CC 10.3 (sm_103a), 160 SMs, 288 GiB" in reference


def test_the_formatting_helpers_read_as_a_reader_meets_them():
    assert [tp._fmt(v) for v in (None, True, False, 8.0, 1500, "a phrase")] == \
        ["?", "yes", "no", "8", "1500", "a phrase"]
    assert tp._cc_text([10, 0]) == "10.0" and tp._cc_text(None) == "?"
    assert tp._clocks_text({"sm_mhz": 1500, "mem_mhz": 3996, "locked": True}) == \
        "SM 1500 MHz / memory 3996 MHz (locked)"
    assert tp._clocks_text(None) == "not recorded"


def test_a_profile_file_that_is_not_one_is_named(tmp_path):
    with pytest.raises(tp.ProfileError) as e:
        tp.load(tmp_path)                       # tmp_path/target/B200.json is missing
    assert "B200.json" in str(e.value)
    (tmp_path / "target").mkdir()
    profile = tmp_path / "target" / "B200.json"
    profile.write_text("{not json", encoding="utf-8")
    with pytest.raises(tp.ProfileError, match="does not parse"):
        tp.load(tmp_path)
    for not_an_object in ("[]", '"x"', "7"):
        profile.write_text(not_an_object, encoding="utf-8")
        with pytest.raises(tp.ProfileError, match="a target profile is a JSON object"):
            tp.load(tmp_path)
    profile.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(tp.ProfileError, match="schema 'other'"):
        tp.load(tmp_path)
    good = {"schema": "rsi_kernel_tune.target-profile/1", "name": "X"}
    profile.write_text(json.dumps(good), encoding="utf-8")
    assert tp.load(tmp_path) == good


@pytest.mark.parametrize("not_a_profile", [[], "x", 7, None])
def test_rendering_needs_an_object(not_a_profile):
    with pytest.raises(TypeError, match="a target profile is a JSON object"):
        tp.render_markdown(not_a_profile)


def test_materialize_writes_the_rendered_profile_and_returns_its_path(tmp_path):
    out = tmp_path / "B200.rendered.md"
    got = tp.materialize(REPO, dev={}, out=out)
    assert got == out and out.is_file()
    assert out.read_text(encoding="utf-8") == tp.render_markdown(tp.load(REPO))


# --- the seam the other tasks use -----------------------------------------------------------


def test_the_rendered_profile_is_what_a_run_hands_its_agents(tmp_path, monkeypatch):
    """End to end over the seam the other tasks use: the rendered file as the target context
    file, named in the prompt an agent is given."""
    shown = tmp_path / "B200.md"
    shown.write_text(tp.render_markdown(tp.load(REPO)), encoding="utf-8")
    for run in ("run", "run2"):
        (tmp_path / run).mkdir()
    monkeypatch.setenv(*next(iter(runcontext.env_for({"target": shown}).items())))
    assert runcontext.path_for("target") == shown
    prompt = driver._student_prompt(_ctx(tmp_path / "run", FakeBackend([], [])), 0)
    assert f"Target device: `{shown}`" in prompt
    assert driver._bindings(_ctx(tmp_path / "run2", FakeBackend([], [])))["{TARGET_PROFILE_PATH}"] \
        == str(shown)
