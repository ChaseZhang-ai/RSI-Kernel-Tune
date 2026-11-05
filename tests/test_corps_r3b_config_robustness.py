"""BUG-CH-E-05 + BUG-CH-E-04 regression: the [loop] config face refuses readably.

Two faces of one gap:

- CH-E-05 — LoopContext construction ran bare ``int()``/``float()`` over ~18
  [loop] numeric keys. A mistyped value ("three", "3%") died with a ValueError
  traceback that named the value but never the key, on a face where the
  neighbouring knobs (_reuse_mode, _resolve_gate_window) already refuse with a
  readable SystemExit. Every numeric key now goes through _cfg_int/_cfg_float,
  which name the key.
- CH-E-04 — ``_load_loop_config`` read loop.toml without an encoding (locale
  dependent: a real legacy locale either crashed on the first non-ASCII byte or
  silently mojibake'd operator strings), and a TOML syntax error surfaced as a
  bare TOMLDecodeError. It now reads UTF-8 and converts both failures into a
  refusal that names the file.
"""

import sys
from pathlib import Path

import pytest

from rsi_kernel_tune.loop import cli


# -- the helpers ---------------------------------------------------------------------

def test_mistyped_int_knob_names_the_key():
    with pytest.raises(SystemExit, match=r"stagnation_k must be an integer.*'three'"):
        cli._cfg_int({"stagnation_k": "three"}, "stagnation_k", 3)


def test_mistyped_float_knob_names_the_key():
    with pytest.raises(SystemExit, match=r"reprice_min_gain_pct must be a number.*'3%'"):
        cli._cfg_float({"reprice_min_gain_pct": "3%"}, "reprice_min_gain_pct", 3.0)


def test_none_section_takes_the_default():
    assert cli._cfg_int(None, "stagnation_k", 3) == 3
    assert cli._cfg_float(None, "watermark_margin", 0.03) == 0.03


def test_numeric_strings_still_convert():
    """A TOML string that reads as a number keeps converting (today's behavior):
    only genuinely mistyped values are refused."""
    assert cli._cfg_int({"lever_pricing_rounds": "4"}, "lever_pricing_rounds", 3) == 4
    assert cli._cfg_float({"reanchor_band": "0.5"}, "reanchor_band", 0.06) == 0.5


def test_target_key_reports_its_qualified_name():
    """The lone [target] knob must be named with its section: '[loop]
    target.floor_margin' — and a mistyped value must not silently fall back to
    the default because the lookup key and the display name differ."""
    with pytest.raises(SystemExit, match=r"target\.floor_margin must be a number"):
        cli._cfg_float({"floor_margin": "wide"}, "floor_margin", 0,
                       label="target.floor_margin")


# -- LoopContext construction (end to end) -------------------------------------------

def test_launch_refuses_a_mistyped_loop_key_readably(tmp_path, monkeypatch):
    """The reported face: loop.toml says stagnation_k = "three" and launch died
    with `int() argument ... ` naming no key. Now the launch refuses naming the
    knob — same caliber as _reuse_mode / _resolve_gate_window next door."""
    import subprocess
    root = tmp_path / "repo"
    op_dir = root / "operators" / "demo"
    op_dir.mkdir(parents=True)
    kb = root / "hw_kb" / "cutedsl" / "sm90"
    kb.mkdir(parents=True)
    (kb / "index.md").write_text("# kb\n", encoding="utf-8")
    (op_dir / "loop.toml").write_text('[loop]\nstagnation_k = "three"\n',
                                      encoding="utf-8")
    (op_dir / "eval").mkdir()
    (op_dir / "eval" / "precision.py").write_text(
        "class PrecisionRecorder:\n    pass\n", encoding="utf-8")
    (op_dir / "eval" / "bench.py").write_text(
        "import precision\nprecision.PrecisionRecorder()\n", encoding="utf-8")
    ws = tmp_path / "ws"
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    subprocess.run(["git", "-C", str(ws), "remote", "add", "origin",
                    str(tmp_path / "origin.git")], check=True)
    monkeypatch.setattr(cli, "_repo_root", lambda: root)
    monkeypatch.setattr(cli, "job_env_preflight", lambda ctx, policy: None)
    with pytest.raises(SystemExit, match=r"stagnation_k must be an integer.*'three'"):
        cli.main(["demo", "--workspace", str(ws), "--oracle", str(tmp_path / "o.py"),
                  "--target-ms", "1.0", "--agent-cmd", "true",
                  "--allow-200k-window", "--allow-single-workload", "--hours", "0.1"])


# -- loop.toml loading (CH-E-04) -----------------------------------------------------

def test_loop_toml_is_read_as_utf8(tmp_path):
    """A non-ASCII operator string must survive byte-for-byte whatever the
    ambient locale claims."""
    op_dir = tmp_path / "op"
    op_dir.mkdir()
    (op_dir / "loop.toml").write_text(
        '[loop]\nref_name = "référence naïve café"\n', encoding="utf-8")
    cfg = cli._load_loop_config(op_dir)
    assert cfg["loop"]["ref_name"] == "référence naïve café"


def test_non_utf8_toml_is_a_readable_refusal(tmp_path):
    """One stray latin-1 byte (a card edited on a legacy machine) used to crash
    the launch with a bare UnicodeDecodeError naming no file; the UTF-8 read
    converts it into a refusal that does."""
    op_dir = tmp_path / "op"
    op_dir.mkdir()
    (op_dir / "loop.toml").write_bytes(b'[loop]\nref_name = "caf\xe9"\n')
    with pytest.raises(SystemExit, match=r"not valid UTF-8"):
        cli._load_loop_config(op_dir)


def test_broken_toml_is_a_readable_refusal_not_a_traceback(tmp_path):
    op_dir = tmp_path / "op"
    op_dir.mkdir()
    (op_dir / "loop.toml").write_text("[loop\nref_cmd = ", encoding="utf-8")
    with pytest.raises(SystemExit, match="loop.toml"):
        cli._load_loop_config(op_dir)


def test_missing_loop_toml_is_an_empty_config(tmp_path):
    assert cli._load_loop_config(tmp_path) == {}
