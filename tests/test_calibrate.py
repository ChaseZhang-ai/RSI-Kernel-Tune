"""``forge calibrate``: the document, the probes and the delivery path (with fakes).

Nothing here needs a GPU: the probe scripts are checked by their ``--help``, the document
builder is fed the driver's own JSON shape, and both delivery paths are stopped at the
call boundary (the local one at ``subprocess``, the remote one at ``DirectRunner``).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel.ext import calibrate as calibrate_cmd
from forgekernel.roofline import calibrate

ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "src" / "forgekernel" / "roofline" / "probes"
PROBE_FILES = ("hbm.py", "l2.py", "tensor.py", "elementwise.py", "sfu.py", "launch.py",
               "run_all.py")


def load_by_path(name: str, path: Path):
    """Import a shipped script as a module (probes/ is not a package on purpose)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def document(*, probes=None, errors=None, machine=None):
    return {
        "machine": machine or {
            "gpu": "NVIDIA B300", "arch": "sm_103", "sm_count": 160, "memory_gib": 288,
            "driver": "580.1", "cuda": "13.1", "torch": "2.9.0", "hostname": "dev",
            "clocks": {"sm_mhz": 1500, "mem_mhz": 3996, "app_sm_mhz": 1500,
                       "max_sm_mhz": 1965, "locked": True},
        },
        "probes": probes if probes is not None else {
            "hbm": {"probe": "hbm", "entries": {
                "copy": {"value": 6300.0, "unit": "GB/s", "method": "copy",
                         "repeats": 7, "cv": 0.01}}},
            "tensor": {"probe": "tensor", "entries": {
                "bf16": {"value": 2400.0, "unit": "TFLOPS", "method": "m",
                         "repeats": 7, "cv": 0.02}}},
            "sfu": {"probe": "sfu", "entries": {
                "exp": {"value": 10200.0, "unit": "Gop/s", "method": "m",
                        "repeats": 7, "cv": 0.03,
                        "skipped": None}}},
        },
        "errors": errors or {},
        "seconds": 10.0,
    }


def payload_line(doc: dict) -> str:
    return calibrate.PAYLOAD_MARKER + json.dumps(doc)


# -- the probes as files -------------------------------------------------------------------


@pytest.mark.parametrize("name", PROBE_FILES)
def test_every_probe_script_has_help_and_needs_no_gpu_for_it(name, tmp_path):
    proc = subprocess.run([sys.executable, str(PROBES_DIR / name), "--help"],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_the_command_and_the_driver_agree_on_the_probe_names():
    driver = load_by_path("fk_calibrate_run_all", PROBES_DIR / "run_all.py")
    assert tuple(driver.PROBES) == tuple(calibrate.PROBES)
    assert all((PROBES_DIR / f"{name}.py").is_file() for name in calibrate.PROBES)


def test_a_probe_entry_reports_the_median_repeats_and_the_cv():
    hbm = load_by_path("fk_calibrate_hbm", PROBES_DIR / "hbm.py")
    steady = hbm.entry([5.0, 5.0, 5.0], "GB/s", "method")
    assert steady["value"] == 5.0
    assert steady["cv"] == 0.0
    assert steady["repeats"] == 3
    spread = hbm.entry([1.0, 2.0, 3.0], "GB/s", "method")
    assert spread["value"] == 2.0
    # population stdev of [1, 2, 3] is sqrt(2/3) = 0.8165; over the mean 2 -> 0.4082
    assert spread["cv"] == pytest.approx(0.4082482904638631)
    assert hbm.skipped("no triton")["value"] is None


def test_the_lock_state_is_read_from_the_clock_columns():
    driver = load_by_path("fk_calibrate_run_all_clocks", PROBES_DIR / "run_all.py")
    assert driver.clocks_locked({"sm_mhz": 1500, "app_sm_mhz": 1500,
                                 "max_sm_mhz": 1965}) is True
    assert driver.clocks_locked({"sm_mhz": 1965, "app_sm_mhz": 1965,
                                 "max_sm_mhz": 1965}) is False       # app clock = max
    assert driver.clocks_locked({"sm_mhz": 1965,
                                 "max_sm_mhz": 1965}) is False       # nothing pinned
    assert driver.clocks_locked({"sm_mhz": 1200, "app_sm_mhz": 1500,
                                 "max_sm_mhz": 1965}) is None        # does not match
    assert driver.clocks_locked({}) is None


def test_the_machine_block_parses_an_smi_line():
    driver = load_by_path("fk_calibrate_run_all_smi", PROBES_DIR / "run_all.py")
    parsed = driver.parse_smi("NVIDIA B300, 580.95.05, 1500, 3996, 1500, 1965")
    assert parsed == {"name": "NVIDIA B300", "driver": "580.95.05",
                      "current_sm_mhz": 1500, "current_mem_mhz": 3996,
                      "app_sm_mhz": 1500, "max_sm_mhz": 1965}
    assert driver.parse_smi("") == {}


# -- the document -------------------------------------------------------------------------


def test_parse_payload_takes_the_last_document_line():
    first = payload_line({"probes": {"a": {}}})
    second = payload_line(document())
    parsed = calibrate.parse_payload(f"noise\n{first}\nmore noise\n{second}\n")
    assert parsed["machine"]["gpu"] == "NVIDIA B300"
    with pytest.raises(calibrate.CalibrationError, match="no document"):
        calibrate.parse_payload("nothing here")


def test_parse_probe_payload():
    line = calibrate.PROBE_MARKER + json.dumps({"probe": "hbm", "entries": {}})
    assert calibrate.parse_probe_payload(line)["probe"] == "hbm"
    assert calibrate.parse_probe_payload("no json") is None


def test_build_produces_the_documented_schema():
    doc = calibrate.build(document(), delivery="ssh dev", when="2026-11-24T10:00:00+00:00",
                          seconds=10.0)
    assert doc["schema"] == calibrate.SCHEMA
    assert doc["gpu"] == "NVIDIA B300" and doc["arch"] == "sm_103"
    assert doc["sm_count"] == 160 and doc["clocks"]["locked"] is True
    assert doc["ts"] == "2026-11-24T10:00:00+00:00"
    assert doc["delivery"] == "ssh dev"
    assert set(doc["probes"]) == {"hbm", "tensor", "sfu"}
    assert doc["probes"]["hbm"]["entries"]["copy"]["value"] == 6300.0
    assert doc["errors"] == {}


def test_build_refuses_a_failed_run_or_a_document_without_a_gpu():
    with pytest.raises(calibrate.CalibrationError, match="every probe failed"):
        calibrate.build({"machine": {"gpu": "X"}, "probes": {},
                         "errors": {"hbm": {"rc": 1}}}, delivery="local")
    machine = document()["machine"]
    machine.pop("gpu")
    with pytest.raises(calibrate.CalibrationError, match="no GPU name"):
        calibrate.build({"machine": machine, "probes": document()["probes"]},
                        delivery="local")


@pytest.mark.parametrize("entry, match", [
    ({"unit": "GB/s", "repeats": 1}, "no value and no reason"),
    ({"value": "fast", "unit": "GB/s", "repeats": 1}, "not a number"),
    ({"value": 0.0, "unit": "GB/s", "repeats": 1}, "not positive"),
    ({"value": 1.0, "unit": "", "repeats": 1}, "without a unit"),
    ({"value": 1.0, "unit": "GB/s", "repeats": 0}, "repeats"),
    ({"value": 1.0, "unit": "GB/s", "repeats": 1, "cv": -1}, "cv"),
    ("nope", "not an entry object"),
])
def test_every_broken_entry_is_refused(entry, match):
    with pytest.raises(calibrate.CalibrationError, match=match):
        calibrate.validate_entry("hbm", "copy", entry)


def test_a_skipped_entry_keeps_its_reason():
    out = calibrate.validate_entry("tensor", "fp8", {"value": None, "unit": "",
                                                    "skipped": "no fp8 path"})
    assert out["value"] is None and out["skipped"] == "no fp8 path"


def test_load_round_trips_and_refuses_a_broken_file(tmp_path):
    doc = calibrate.build(document(), delivery="local")
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(doc))
    assert calibrate.load(path)["gpu"] == "NVIDIA B300"
    with pytest.raises(calibrate.CalibrationError, match="No such file"):
        calibrate.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(calibrate.CalibrationError, match="not JSON"):
        calibrate.load(bad)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": "other/1", "gpu": "X", "probes": {}}))
    with pytest.raises(calibrate.CalibrationError, match="rerun `forge calibrate`"):
        calibrate.load(wrong)


def test_default_out_is_the_gpu_and_the_date(tmp_path):
    doc = calibrate.build(document(), delivery="local", when="2026-11-24T10:00:00+00:00")
    assert calibrate.default_out(doc, directory=tmp_path) == \
        tmp_path / "nvidia-b300-2026-11-24.json"


def test_the_summary_names_every_reading_and_warns_about_unpinned_clocks():
    doc = calibrate.build(document(), delivery="local")
    lines = "\n".join(calibrate.summary_lines(doc))
    assert "NVIDIA B300" in lines and "160 SMs" in lines
    assert "hbm.copy" in lines and "6300 GB/s" in lines
    assert "tensor.bf16" in lines
    assert "not pinned" not in lines
    unlocked = document()
    unlocked["machine"]["clocks"]["locked"] = False
    lines = "\n".join(calibrate.summary_lines(calibrate.build(unlocked, delivery="local")))
    assert "not locked" in lines and "not pinned" in lines
    skipped = document()
    skipped["probes"]["tensor"]["entries"]["fp8"] = {"value": None, "unit": "",
                                                    "skipped": "no path"}
    lines = "\n".join(calibrate.summary_lines(calibrate.build(skipped, delivery="local")))
    assert "tensor.fp8" in lines and "skipped: no path" in lines
    failed = document(errors={"l2": {"error": "boom"}})
    lines = "\n".join(calibrate.summary_lines(calibrate.build(failed, delivery="local")))
    assert "l2: FAILED" in lines


# -- the job tree --------------------------------------------------------------------------


def test_the_entry_script_carries_the_budget_and_runs_the_driver():
    text = calibrate.entry_script(seconds=7.5, only=("hbm", "launch"), timeout=120.0)
    assert 'run_all.py --seconds 7.5 --only hbm,launch' in text
    assert "--timeout 120.0" in text
    assert "calibrate)" in text
    # python3 by default, FK_CALIBRATE_PYTHON when the job's interpreter must be named
    assert 'PY="${FK_CALIBRATE_PYTHON:-python3}"' in text
    assert 'exec "$PY" run_all.py' in text


def test_prepare_tree_ships_the_probes_as_a_committed_tree(tmp_path):
    base, op_dir = calibrate.prepare_tree(seconds=1.0, only=("hbm",), timeout=60.0)
    try:
        assert op_dir.is_dir()
        for name in PROBE_FILES:
            assert (op_dir / name).is_file(), name
        entry = op_dir / "gpu_entry.sh"
        assert entry.is_file() and entry.stat().st_mode & 0o111
        assert "--only hbm" in entry.read_text()
        head = subprocess.run(["git", "-C", str(op_dir), "rev-parse", "--abbrev-ref",
                               "HEAD"], capture_output=True, text=True)
        assert head.stdout.strip() == "main"
        status = subprocess.run(["git", "-C", str(op_dir), "status", "--porcelain"],
                                capture_output=True, text=True)
        assert status.stdout.strip() == ""                 # committed, nothing left over
        assert base.parent == Path("/tmp") or base.is_dir()
    finally:
        import shutil
        shutil.rmtree(op_dir.parents[2], ignore_errors=True)


# -- delivery (stopped at the call boundary) -----------------------------------------------


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_the_local_path_runs_the_entry_under_the_gpu_lock(monkeypatch):
    import contextlib
    import forgekernel.submit.direct as direct

    calls = {}
    monkeypatch.setattr(direct, "gpu_lock", lambda gpu: contextlib.nullcontext())
    real = calibrate.subprocess.run

    def fake_run(argv, **kw):
        # only the entry itself is faked: prepare_tree's git calls must really run
        if argv and argv[0] == "bash":
            calls["argv"] = argv
            calls["cwd"] = kw.get("cwd")
            calls["env"] = kw.get("env") or {}
            return Completed(0, payload_line(document()) + "\n")
        return real(argv, **kw)

    monkeypatch.setattr(calibrate.subprocess, "run", fake_run)
    delivered = calibrate.LocalRun(gpu="0").run(seconds=1.0, only=("hbm",), timeout=60.0)
    assert delivered.rc == 0
    assert calls["argv"][0] == "bash" and calls["argv"][1].endswith("gpu_entry.sh")
    assert calls["argv"][2] == "calibrate"
    # this machine's python3 need not be the interpreter holding torch
    assert calls["env"][calibrate.PYTHON_ENV] == sys.executable
    assert calibrate.parse_payload(delivered.stdout)["machine"]["gpu"] == "NVIDIA B300"


class FakeRunner:
    def __init__(self, cfg, *, status="succeeded", log="", boom=None):
        self.cfg = cfg
        self.status = status
        self.log = log
        self.boom = boom
        self.calls = []

    def run_mode(self, branch, mode, log_file):
        self.calls.append((branch, mode))
        if self.boom:
            raise self.boom
        log_file.write(self.log)
        return self.status


def test_the_remote_path_is_the_same_delivery_a_gpu_job_uses():
    from forgekernel.localenv import Machine
    from forgekernel.submit.jobs import STATUS_SUCCEEDED

    machine = Machine(delivery="ssh", gpu="0", remote_host="devbox",
                      remote_workdir="/work/fk")
    made = {}

    def factory(cfg):
        made["runner"] = FakeRunner(cfg, status=STATUS_SUCCEEDED,
                                    log=payload_line(document()) + "\n")
        return made["runner"]

    run = calibrate.RemoteRun(machine, runner=factory)
    delivered = run.run(seconds=1.0, only=("hbm",), timeout=900.0)
    assert delivered.rc == 0
    assert made["runner"].calls == [("main", "calibrate")]
    cfg = made["runner"].cfg
    assert cfg.delivery == "ssh" and cfg.exec_host == "devbox"
    assert cfg.exec_workdir == "/work/fk" and cfg.entry_script == "gpu_entry.sh"
    assert calibrate.parse_payload(delivered.stdout)["probes"]["hbm"]
    assert run.describe() == "ssh devbox"


def test_the_remote_job_carries_a_named_interpreter_when_one_is_set(monkeypatch):
    from forgekernel.localenv import Machine

    machine = Machine(delivery="ssh", gpu="0", remote_host="devbox",
                      remote_workdir="/work/fk")
    made = {}

    def factory(cfg):
        made["runner"] = FakeRunner(cfg, log="")
        return made["runner"]

    monkeypatch.setenv(calibrate.PYTHON_ENV, "/opt/venv/bin/python")
    calibrate.RemoteRun(machine, runner=factory).run(seconds=1.0, only=("hbm",),
                                                     timeout=10.0)
    assert made["runner"].cfg.env[calibrate.PYTHON_ENV] == "/opt/venv/bin/python"
    assert made["runner"].cfg.env["PYTHONUNBUFFERED"] == "1"
    monkeypatch.delenv(calibrate.PYTHON_ENV)
    calibrate.RemoteRun(machine, runner=factory).run(seconds=1.0, only=("hbm",),
                                                     timeout=10.0)
    assert calibrate.PYTHON_ENV not in made["runner"].cfg.env


def test_the_remote_path_reports_a_failed_job():
    from forgekernel.localenv import Machine

    machine = Machine(delivery="ssh", gpu="0", remote_host="devbox",
                      remote_workdir="/work/fk")
    run = calibrate.RemoteRun(machine, runner=lambda cfg: FakeRunner(cfg, boom=OSError("down")))
    delivered = run.run(seconds=1.0, only=("hbm",), timeout=10.0)
    assert delivered.rc == 1 and "down" in delivered.stderr


def test_resolve_delivery_follows_local_env_and_the_flags(monkeypatch):
    from forgekernel import localenv

    local = localenv.Machine(delivery="local", gpu="0")
    monkeypatch.setattr(localenv, "load_machine", lambda root=None: local)
    deliverer, name = calibrate.resolve_delivery()
    assert isinstance(deliverer, calibrate.LocalRun) and name == "this machine"

    remote = localenv.Machine(delivery="ssh", gpu="1", remote_host="host",
                              remote_workdir="/w")
    monkeypatch.setattr(localenv, "load_machine", lambda root=None: remote)
    deliverer, name = calibrate.resolve_delivery()
    assert isinstance(deliverer, calibrate.RemoteRun) and name == "ssh host"
    deliverer, _name = calibrate.resolve_delivery(local=True)
    assert isinstance(deliverer, calibrate.LocalRun)
    with pytest.raises(calibrate.CalibrationError, match="mutually exclusive"):
        calibrate.resolve_delivery(remote=True, local=True)


def test_the_ssh_delivery_without_a_host_is_a_clear_error(monkeypatch):
    from forgekernel import localenv

    broken = localenv.Machine(delivery="ssh", gpu="0")
    monkeypatch.setattr(localenv, "load_machine", lambda root=None: broken)
    with pytest.raises(calibrate.CalibrationError, match="REMOTE_HOST"):
        calibrate.resolve_delivery()


# -- the command ---------------------------------------------------------------------------


def test_calibrate_writes_validates_and_reports_the_path(tmp_path, monkeypatch):
    import contextlib

    class Stub(calibrate.LocalRun):
        def run(self, *, seconds, only, timeout):
            return calibrate.Delivered(0, payload_line(document()) + "\n", "")

    monkeypatch.setattr(calibrate, "resolve_delivery",
                        lambda **kw: (Stub(gpu="0"), "this machine"))
    out = tmp_path / "cal.json"
    doc = calibrate.calibrate(out=out, seconds=1.0, only=("hbm", "tensor", "sfu"))
    assert doc["path"] == str(out)
    assert calibrate.load(out)["probes"]["hbm"]["entries"]["copy"]["value"] == 6300.0
    assert doc["only"] == ["hbm", "tensor", "sfu"]


def test_calibrate_refuses_an_unknown_probe_and_an_empty_run(tmp_path, monkeypatch):
    with pytest.raises(calibrate.CalibrationError, match="unknown probe"):
        calibrate.calibrate(only=("nope",))

    class Silent(calibrate.LocalRun):
        def run(self, *, seconds, only, timeout):
            return calibrate.Delivered(1, "", "the driver could not start")

    monkeypatch.setattr(calibrate, "resolve_delivery",
                        lambda **kw: (Silent(gpu="0"), "this machine"))
    out = tmp_path / "cal.json"
    with pytest.raises(calibrate.CalibrationError, match="could not start"):
        calibrate.calibrate(out=out, only=("hbm",))


def test_the_public_command_prints_the_summary_and_the_path(tmp_path, monkeypatch, capsys):
    def fake(**kw):
        doc = calibrate.build(document(), delivery="this machine", seconds=1.0)
        path = Path(kw["out"])
        path.write_text(json.dumps(doc))
        doc["path"] = str(path)
        return doc

    monkeypatch.setattr(calibrate, "calibrate", fake)
    out = tmp_path / "cal.json"
    rc = calibrate_cmd.main(["--out", str(out), "--seconds", "1"])
    printed = capsys.readouterr()
    assert rc == 0
    assert "hbm.copy" in printed.out
    assert f"wrote {out}" in printed.out
    assert "forge roofline" in printed.out


def test_the_public_command_can_print_json_and_reports_an_error(monkeypatch, capsys):
    def fake(**kw):
        return calibrate.build(document(), delivery="this machine")

    monkeypatch.setattr(calibrate, "calibrate", fake)
    assert calibrate_cmd.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == calibrate.SCHEMA

    def broken(**kw):
        raise calibrate.CalibrationError("every probe failed (hbm: rc 1)")

    monkeypatch.setattr(calibrate, "calibrate", broken)
    assert calibrate_cmd.main([]) == 2
    assert "every probe failed" in capsys.readouterr().err
