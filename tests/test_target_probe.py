"""The probe: its parsing (against canned machine output), its program, and its delivery.

No GPU is needed for any of this: the machine's answers are canned strings, and the remote
path is exercised with a fake runner up to the call boundary (the tree it builds is real).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel.targethw import parity, probe, probe_program, smoke_sources

REPO = Path(__file__).resolve().parents[1]

SMI_QUERY = ("NVIDIA B300, 10.3, 288000, 580.65.06, 1500, 3996, 1500, 3996, 1965, 4000\n")
SMI_Q = """\
==============NVSMI LOG==============

Timestamp                                 : Mon Nov 24 10:00:00 2026

    Clocks
        Graphics                          : 1500 MHz
        SM                                : 1500 MHz
        Memory                            : 3996 MHz

    Applications Clocks
        Graphics                          : 1500 MHz
        Memory                            : 3996 MHz

    Max Clocks
        SM                                : 1965 MHz

    Clocks Event Reasons
        Idle                              : Not Active
        SW Power Cap                      : Active
"""
NVCC = """\
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Jun_09_02:43:40_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0
"""


def payload(**over) -> dict:
    """A probe program's document, as a B300 would print it."""
    base = {
        "schema": probe_program.SCHEMA,
        "host": {"hostname": "devbox", "platform": "Linux-6.18", "python": "3.12.3",
                 "executable": "/usr/bin/python3"},
        "nvidia_smi_query": {"rc": 0, "out": SMI_QUERY, "err": ""},
        "nvidia_smi_q": {"rc": 0, "out": SMI_Q, "err": ""},
        "nvcc": {"rc": 0, "out": NVCC, "err": ""},
        "python_info": {
            "python": "3.12.3",
            "torch": {"version": "2.9.0", "cuda": "13.1", "available": True},
            "torch_device": {"index": 0, "name": "NVIDIA B300", "compute_capability": [10, 3],
                             "multi_processor_count": 160, "memory_total_mib": 294912,
                             "l2_cache_mib": 126},
            "triton": {"version": "3.7.1", "distributions": ["fbtriton", "triton"]},
            "cutlass_dsl": {"version": "4.4.2", "import": "ok"},
            "cupti": {"version": "13.0.1"},
            "cuda_toolkit": {"version": None},
        },
        "smokes": {
            "triton": {"compile": True, "run": True, "ms": 0.012, "error": "", "detail": {}},
            "cutedsl": {"compile": True, "run": True, "ms": 0.021, "error": "", "detail": {}},
            "cuda_inline": {"compile": True, "run": True, "ms": 0.004, "error": "", "detail": {}},
        },
        "env": {"CUDA_HOME": "/usr/local/cuda"},
    }
    base.update(over)
    return base


# -- the parsers -----------------------------------------------------------------------

def test_parse_nvidia_smi_query_reads_every_column():
    got = probe.parse_nvidia_smi_query(SMI_QUERY)
    assert got["name"] == "NVIDIA B300"
    assert got["compute_capability"] == [10, 3]
    assert got["memory_total_mib"] == 288000
    assert got["driver"] == "580.65.06"
    assert got["current_sm_mhz"] == 1500 and got["app_sm_mhz"] == 1500
    assert got["max_sm_mhz"] == 1965


def test_parse_nvidia_smi_query_tolerates_a_missing_or_odd_answer():
    assert probe.parse_nvidia_smi_query("") == {}
    assert probe.parse_nvidia_smi_query("not,an,answer\n")["name"] == "not"
    got = probe.parse_nvidia_smi_query("NVIDIA X, N/A, N/A, 1, [N/A], [N/A], [N/A], [N/A], "
                                       "[N/A], [N/A]")
    assert got["name"] == "NVIDIA X" and "current_sm_mhz" not in got


def test_parse_nvidia_smi_q_reads_clocks_and_throttle():
    got = probe.parse_nvidia_smi_q(SMI_Q)
    assert got["clocks"] == {"sm_mhz": 1500, "mem_mhz": 3996}
    assert got["applications"] == {"sm_mhz": 1500, "mem_mhz": 3996}
    assert got["max_clocks"] == {"sm_mhz": 1965}
    assert got["throttle"].get("SW Power Cap") == "Active"
    assert probe.parse_nvidia_smi_q("") == {}


def test_parse_nvcc_version_reads_the_toolkit_and_release():
    got = probe.parse_nvcc_version(NVCC)
    assert got == {"toolkit": "13.3", "release": "V13.3.73"}
    assert probe.parse_nvcc_version("bash: nvcc: command not found") == {}


@pytest.mark.parametrize("clocks, locked", [
    ({"sm_mhz": 1500, "app_sm_mhz": 1500, "max_sm_mhz": 1965}, True),      # pinned below max
    ({"sm_mhz": 1965, "app_sm_mhz": 1965, "max_sm_mhz": 1965}, False),     # boosting at max
    ({"sm_mhz": 1500, "app_sm_mhz": 0, "max_sm_mhz": 1965}, None),         # cannot tell
    ({"sm_mhz": None, "app_sm_mhz": None, "max_sm_mhz": None}, None),
    ({"sm_mhz": 1800, "app_sm_mhz": 1500, "max_sm_mhz": 1965}, None),      # drifting
])
def test_clocks_locked_is_a_three_number_decision(clocks, locked):
    assert probe.clocks_locked(clocks) is locked


# -- the whole report --------------------------------------------------------------------

def test_a_full_report_becomes_target_dev_json():
    dev = probe.parse_report(payload(), mode="local", collected_at="2026-11-24T00:00:00Z")
    assert dev["schema"] == probe.SCHEMA
    gpu = dev["gpu"]
    assert gpu["name"] == "NVIDIA B300" and gpu["compute_capability"] == [10, 3]
    assert gpu["arch"] == "sm103" and gpu["cuda_arch"] == "sm_103a"
    assert gpu["multi_processor_count"] == 160 and gpu["memory_gib"] == 288.0
    assert gpu["clocks_locked"] is True and gpu["driver"] == "580.65.06"
    software = dev["software"]
    assert software["torch"] == "2.9.0" and software["triton"] == "3.7.1"
    assert software["triton_is_fbtriton"] is True
    assert software["triton_distribution"] == "fbtriton"
    assert software["cutlass_dsl"] == "4.4.2" and software["cuda_toolkit"] == "13.3"
    assert {k: v["run"] for k, v in dev["smokes"].items()} == {
        "triton": True, "cutedsl": True, "cuda_inline": True}
    assert dev["probe"]["smokes"] is True


def test_a_stock_triton_is_reported_as_such():
    doc = payload()
    doc["python_info"]["triton"] = {"version": "3.9.0", "distributions": ["triton"]}
    dev = probe.parse_report(doc)
    assert dev["software"]["triton_is_fbtriton"] is False
    assert dev["software"]["triton_distribution"] == "triton"


def test_nvidia_smi_missing_falls_back_to_torch():
    doc = payload(nvidia_smi_query={"rc": 127, "out": "", "err": "command not found"},
                  nvidia_smi_q={"rc": 127, "out": "", "err": "command not found"})
    dev = probe.parse_report(doc)
    gpu = dev["gpu"]
    assert gpu["name"] == "NVIDIA B300"                      # from torch
    assert gpu["compute_capability"] == [10, 3]
    assert "driver" not in gpu and "clocks" not in gpu       # nothing to read them from


def test_torch_not_importable_is_a_reported_field_not_a_crash():
    doc = payload()
    doc["python_info"] = {"python": "3.12.3", "torch": {"error": "ModuleNotFoundError: torch"},
                          "triton": {"error": "ModuleNotFoundError: triton"},
                          "cutlass_dsl": {"version": None, "error": "no module"},
                          "cupti": {"version": None}}
    doc["nvidia_smi_query"] = {"rc": 127, "out": "", "err": "command not found"}
    doc["smokes"] = {}          # a machine with no torch runs no smoke
    dev = probe.parse_report(doc)
    # nothing reports a device name, but nvidia-smi -q still gave the clocks
    assert "name" not in dev["gpu"] and dev["gpu"]["clocks"]["sm_mhz"] == 1500
    assert "torch_error" in dev["software"] and dev["software"]["torch"] is None
    assert "triton_error" in dev["software"]
    assert dev["smokes"]["triton"]["run"] is False


def test_no_device_and_no_smi_is_an_empty_gpu_with_a_reason():
    doc = payload()
    doc["python_info"] = {"python": "3.11.16", "torch": {"error": "no CUDA driver"}}
    doc["nvidia_smi_query"] = {"rc": 127, "out": "", "err": "command not found"}
    doc["nvidia_smi_q"] = {"rc": 127, "out": "", "err": "command not found"}
    dev = probe.parse_report(doc)
    assert dev["gpu"] == {"error": "no CUDA driver"}


def test_smokes_that_did_not_run_are_normalized():
    doc = payload(smokes={})
    dev = probe.parse_report(doc)
    for smoke in dev["smokes"].values():
        assert smoke["compile"] is False and smoke["run"] is False
        assert "no result" in smoke["error"]
    assert dev["probe"]["smokes"] is False


def test_parse_payload_takes_the_last_marker_line():
    stdout = ("noise\n"
              f"{probe.PAYLOAD_MARKER}{json.dumps({'a': 1})}\n"
              f"{probe.PAYLOAD_MARKER}{json.dumps({'a': 2})}\n")
    assert probe.parse_payload(stdout) == {"a": 2}


def test_parse_payload_without_a_marker_is_a_probe_error():
    with pytest.raises(probe.ProbeError, match="no JSON document"):
        probe.parse_payload("ssh: connect to host devbox port 22: refused")


# -- the program that is sent ----------------------------------------------------------

def test_the_probe_program_is_a_real_module_and_the_shipped_text_is_its_source():
    text = probe.program_source()
    assert "def collect(" in text and "def smoke_triton(" in text
    assert text == Path(probe_program.__file__).read_text(encoding="utf-8")
    assert "forgekernel" not in text.split("import")[0]      # standalone docstring, no import


def test_the_three_smoke_sources_are_parseable():
    for name in smoke_sources.NAMES:
        assert smoke_sources.compiles(name), name
    assert smoke_sources.cuda_source_compiles()


# -- delivery --------------------------------------------------------------------------

class FakeDeliver:
    def __init__(self, stdout: str, rc: int = 0, stderr: str = ""):
        self.stdout, self.rc, self.stderr = stdout, rc, stderr
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return probe.Delivered(self.rc, self.stdout, self.stderr)

    def describe(self) -> str:
        return "a fake machine"


def test_probe_writes_dev_json_and_compares_it(tmp_path):
    fake = FakeDeliver(probe.PAYLOAD_MARKER + json.dumps(payload()) + "\n")
    out = tmp_path / "dev.json"
    dev = probe.probe(deliver=fake, out=out, root=REPO)
    assert fake.calls and fake.calls[0]["timeout"] == probe.DEFAULT_TIMEOUT_S
    assert out.is_file()
    written = json.loads(out.read_text())
    assert written["gpu"]["compute_capability"] == [10, 3]
    assert written["parity"]["blocker"] == 0
    assert any("sm_count" in d for d in written["differences"])
    assert dev["_differences"] and dev["path"] == str(out)


def test_probe_on_a_machine_without_the_stack_has_blockers(tmp_path):
    doc = payload(smokes={})
    doc["python_info"] = {"python": "3.11.16", "torch": {"error": "no driver"}}
    doc["nvidia_smi_query"] = {"rc": 127, "out": "", "err": "command not found"}
    fake = FakeDeliver(probe.PAYLOAD_MARKER + json.dumps(doc) + "\n")
    dev = probe.probe(deliver=fake, out=tmp_path / "dev.json", root=REPO)
    assert dev["parity"]["blocker"] >= 1


def test_a_delivery_with_no_document_is_a_probe_error(tmp_path):
    fake = FakeDeliver("", rc=255, stderr="ssh: Could not resolve hostname devbox")
    with pytest.raises(probe.ProbeError):
        probe.probe(deliver=fake, out=tmp_path / "dev.json", root=REPO)


def test_the_remote_probe_builds_a_committed_job_tree():
    base, op_dir = probe.prepare_tree(include_smokes=False)
    try:
        assert (op_dir / "probe_program.py").read_text() == probe.program_source()
        entry = (op_dir / "gpu_entry.sh").read_text()
        assert "--no-smokes" in entry and "exec python3" in entry
        assert (op_dir / ".git").is_dir()
        import subprocess
        names = subprocess.run(["git", "-C", str(op_dir), "ls-files"],
                               capture_output=True, text=True).stdout.split()
        assert "probe_program.py" in names and "gpu_entry.sh" in names
        assert base.is_dir() and op_dir.parent.name == "operators"
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


class FakeMachine:
    remote_host = "devbox"
    remote_workdir = "/home/user/fk"
    remote_env_file = ""
    gpu = "0"


class FakeRunner:
    """Stands in for submit.direct.DirectRunner: writes a log, never touches ssh."""

    def __init__(self, cfg, log_text):
        self.cfg = cfg
        self.log_text = log_text
        self.calls = []

    def run_mode(self, branch, mode, log_file):
        self.calls.append((branch, mode))
        log_file.write("some job output\n")
        log_file.write(self.log_text)
        log_file.flush()
        from forgekernel.submit.jobs import STATUS_SUCCEEDED
        return STATUS_SUCCEEDED


def test_the_remote_probe_hands_the_tree_to_the_job_delivery(monkeypatch):
    made = {}
    log_line = probe.PAYLOAD_MARKER + json.dumps(payload()) + "\n"

    def factory(cfg):
        made["cfg"] = cfg
        made["runner"] = FakeRunner(cfg, log_line)
        return made["runner"]

    remote = probe.RemoteProbe(FakeMachine(), runner=factory)
    result = remote.run(timeout=321)
    assert result.rc == 0 and probe.PAYLOAD_MARKER in result.stdout
    cfg = made["cfg"]
    assert cfg.delivery == "ssh" and cfg.exec_host == "devbox"
    assert cfg.exec_workdir == "/home/user/fk" and cfg.exec_gpu == "0"
    assert cfg.timeout == 321 and cfg.entry_script == "gpu_entry.sh"
    assert cfg.branch == "main" and cfg.op_dir.endswith("target-probe")
    assert made["runner"].calls == [("main", "probe")]
    parsed = probe.parse_report(probe.parse_payload(result.stdout))
    assert parsed["gpu"]["arch"] == "sm103"


def test_the_remote_probe_surfaces_a_failed_job(monkeypatch):
    class Failed(FakeRunner):
        def run_mode(self, branch, mode, log_file):
            log_file.write("boom\n")
            return "Failed"

    def factory(cfg):
        return Failed(cfg, "")

    result = probe.RemoteProbe(FakeMachine(), runner=factory).run(timeout=5)
    assert result.rc == 1


def test_remote_without_an_ssh_host_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("FK_DELIVERY", "local")
    with pytest.raises(probe.ProbeError, match="REMOTE_HOST"):
        probe.probe(remote=True, root=tmp_path, out=tmp_path / "dev.json")


# -- parity ----------------------------------------------------------------------------

def test_parity_grades_a_matching_machine_cleanly():
    from forgekernel import target_profile

    doc = payload()
    dev = probe.parse_report(doc)
    differences = parity.compare(dev, target_profile.load(REPO))
    assert parity.summary(differences)[parity.BLOCKER] == 0
    # the B300 reference numbers are still a risk: FP4 and SFU differ
    keys = {d.key: d.level for d in differences}
    assert keys["sm_count"] == parity.RISK
    assert keys["dense_fp4"] == parity.RISK and keys["sfu"] == parity.RISK
    assert keys["compute_capability"] == parity.INFO


def test_parity_blocks_a_stack_that_cannot_run(tmp_path):
    from forgekernel import target_profile

    doc = payload(smokes={"triton": {"compile": False, "run": False, "ms": None,
                                     "error": "launch: RuntimeError: no kernel image",
                                     "detail": {}}})
    doc["python_info"]["cutlass_dsl"] = {"version": None, "error": "ModuleNotFoundError"}
    dev = probe.parse_report(doc)
    differences = parity.compare(dev, target_profile.load(REPO))
    blockers = {d.key for d in differences if d.level == parity.BLOCKER}
    assert {"smoke:triton", "smoke:cutedsl", "cutlass_dsl"} <= blockers
    assert parity.exit_code(differences) == 2
    lines = parity.lines(differences)
    assert lines and all(line.startswith((parity.BLOCKER, parity.RISK)) for line in lines)


def test_parity_skipped_smokes_are_info_not_blockers():
    from forgekernel import target_profile

    dev = probe.parse_report(payload(smokes={}))
    differences = parity.compare(dev, target_profile.load(REPO))
    smoke = [d for d in differences if d.key == "smoke:triton"][0]
    assert smoke.level == parity.INFO and "skipped" in smoke.dev


def test_parity_is_data_driven_over_the_profile():
    """A second profile (not B200): every threshold comes from the profile, not from code."""
    profile = {
        "device": {"name": "a smaller board", "compute_capability": [9, 0],
                   "multi_processor_count": 132, "memory_gib": 80,
                   "clocks": {"sm_mhz": 1980, "mem_mhz": 2619, "locked": True}},
        "software": {"torch": "2.9.0", "triton": "3.7.1", "triton_distribution": "fbtriton",
                     "cutlass_dsl": "4.4.2", "cuda": "13.1.1", "driver_min": "580"},
        "development_gpu": {},
    }
    dev = probe.parse_report(payload())
    differences = parity.compare(dev, profile)
    keys = {d.key: d.level for d in differences}
    assert keys["compute_capability"] == parity.INFO        # 10.3 is not older than 9.0
    assert keys["sm_count"] == parity.RISK and keys["clocks"] == parity.RISK
    assert "memory" not in keys                             # 288 GiB fits 80 GiB
    assert keys["driver"] if "driver" in keys else True


def test_parity_on_a_machine_with_no_gpu_is_a_blocker():
    from forgekernel import target_profile

    doc = payload()
    doc["python_info"] = {"torch": {"error": "no driver"}}
    doc["nvidia_smi_query"] = {"rc": 127, "out": "", "err": "command not found"}
    differences = parity.compare(probe.parse_report(doc), target_profile.load(REPO))
    assert differences[0].level == parity.BLOCKER and differences[0].key == "gpu"


# -- the profile the agents read, and how it reaches them ------------------------------

def test_the_rendered_profile_has_every_rule_and_the_development_differences():
    from forgekernel import target_profile

    profile = target_profile.load(REPO)
    markdown = target_profile.render_markdown(profile)
    for rule in profile["rules"]:
        assert f"{rule['id']}. {rule['text']}" in markdown
    assert "NVIDIA B300" in markdown and "sm_103a" in markdown
    assert "FP4" in markdown and "TeraExp/s" in markdown
    assert "FK_TARGET_SMS" in markdown and "submission.py" in markdown
    assert len(markdown.splitlines()) < 120     # a page and a half, not a document


def test_the_rendered_profile_is_stable():
    from forgekernel import target_profile

    profile = target_profile.load(REPO)
    dev = probe.parse_report(payload())
    assert target_profile.render_markdown(profile) == target_profile.render_markdown(profile)
    assert target_profile.render_markdown(profile, dev) == \
        target_profile.render_markdown(profile, dict(dev))


def test_materialize_writes_the_rendered_profile(tmp_path, monkeypatch):
    from forgekernel import target_profile

    monkeypatch.setattr(target_profile, "DEV_REL", Path("target") / "dev.json")
    out = target_profile.materialize(REPO, out=tmp_path / "B200.md")
    assert out.is_file()
    text = out.read_text()
    assert text == target_profile.render_markdown(target_profile.load(REPO),
                                                  target_profile.load_dev(REPO))
    assert "Rules (hard)" in text


def test_the_profile_reaches_a_prompt_through_the_placeholder(tmp_path):
    """materialize -> env_for -> the placeholder holds the file's *text*, not its path."""
    from forgekernel import runcontext, target_profile

    rendered = target_profile.render_markdown(target_profile.load(REPO),
                                              probe.parse_report(payload()))
    path = tmp_path / "B200.md"
    path.write_text(rendered)
    env = runcontext.env_for({"target": path})
    prompt = ("You are optimizing a GPU kernel.\n\n"
              "{TARGET_PROFILE_PATH}\n\nWrite submission.py.")
    text = runcontext.render(prompt, env)
    assert "{TARGET_PROFILE_PATH}" not in text
    assert "The kernel is graded on NVIDIA B200" in text
    assert str(path) not in text                  # the text, not a path the far side cannot open
    assert text.endswith("Write submission.py.")
    assert runcontext.paths(env)["target"] == path.resolve()


def test_an_unset_context_says_so_instead_of_leaving_a_placeholder():
    from forgekernel import runcontext

    text = runcontext.render("{TARGET_PROFILE_PATH}", {})
    assert "FK_TARGET_PROFILE_PATH is unset" in text


def test_a_context_path_that_cannot_be_read_is_an_error(tmp_path):
    from forgekernel import runcontext

    env = runcontext.env_for({"prior": tmp_path / "missing.md"})
    with pytest.raises(runcontext.ContextError, match="FK_PRIOR_PATH"):
        runcontext.render("{PRIOR_PATH}", env)
    with pytest.raises(KeyError):
        runcontext.env_for({"targert": tmp_path})       # a typo is refused, not ignored


def test_a_long_context_is_truncated_with_a_note(tmp_path):
    from forgekernel import runcontext

    path = tmp_path / "lessons.md"
    path.write_text("x" * (runcontext.MAX_CHARS + 100))
    text = runcontext.read("lessons", runcontext.env_for({"lessons": path}))
    assert "truncated" in text and len(text) < runcontext.MAX_CHARS + 200


# -- the command line ------------------------------------------------------------------

def test_the_target_command_is_discovered_and_its_subcommands_answer_help(capsys):
    from forgekernel import __main__ as cli
    from forgekernel import ext

    assert "target" in ext.commands()
    assert cli.main(["--help"]) == 0
    assert "forge target" in capsys.readouterr().out
    for sub in ("probe", "check", "compile-check", "show"):
        # argparse's --help exits 0, exactly as it does for `forge run --help`
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["target", sub, "--help"])
        assert exit_info.value.code == 0
        assert sub in capsys.readouterr().out


def test_target_show_prints_the_profile(capsys):
    from forgekernel import __main__ as cli

    assert cli.main(["target", "show"]) == 0
    out = capsys.readouterr().out
    assert "Rules (hard)" in out and "submission.py" in out


def test_target_check_exits_two_on_a_blocked_machine(tmp_path, capsys):
    from forgekernel import __main__ as cli

    doc = payload()
    doc["python_info"] = {"torch": {"error": "no driver"}}
    doc["nvidia_smi_query"] = {"rc": 127, "out": "", "err": "command not found"}
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(probe.parse_report(doc)))
    assert cli.main(["target", "check", "--dev", str(path)]) == 2
    out = capsys.readouterr().out
    assert "blocker" in out and "no GPU" in out
