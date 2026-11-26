"""rsi_kernel_tune.ext: command extensions -- found, dispatched, kept apart, and reserved.

The fake extensions live in ``tmp_path`` and are added to the package's path; the real
ones are the first wave's own modules (one per task; ``FIRST_WAVE`` below).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import rsi_kernel_tune.loop.cli
import rsi_kernel_tune.submit.cli
from rsi_kernel_tune import __main__ as forge_main
from rsi_kernel_tune import ext, launch
from rsi_kernel_tune.task import TaskError, load_task
from test_task import MINIMAL, _kb, _task

REPO = Path(__file__).resolve().parents[1]

HELLO = ('COMMAND = "hello"\n\n\ndef main(argv):\n    print("hello", *argv)\n    return 7\n')
BROKEN = "import no_such_module_for_a_test\n"        # fails to import
SYNTAX = "def main(:\n"                              # does not even parse
MISNAMED = 'COMMAND = "other"\n\n\ndef main(argv):\n    return 0\n'
NO_MAIN = 'COMMAND = "silent"\n'                    # no main
STIFF = 'COMMAND = "stiff"\nmain = 3\n'             # a main that is not callable


def _plug(monkeypatch, tmp_path, **modules) -> None:
    """Write each ``name = text`` as a module under tmp_path and add tmp_path to the package's
    path: from here on the package lists them."""
    for name, text in modules.items():
        (tmp_path / f"{name}.py").write_text(text, encoding="utf-8")
    # once, however many times a test adds a module
    monkeypatch.setattr(ext, "__path__", list(dict.fromkeys([*ext.__path__, str(tmp_path)])))
    importlib.invalidate_caches()


@pytest.fixture(autouse=True)
def _no_fake_extension_stays_imported():
    """A module imported by one test must not answer for another."""
    before = {k for k in sys.modules if k.startswith("rsi_kernel_tune.ext.")}
    yield
    for name in [k for k in sys.modules if k.startswith("rsi_kernel_tune.ext.") and k not in before]:
        del sys.modules[name]


def _with(name: str, text: str = HELLO) -> str:
    return text.replace("hello", name)


# --- the directory and its modules are the commands ---------------------------------------

# the extension commands the task plan writes (00-背景与约定 §7/§13; module name == command).
# Step 0 shipped none on purpose; each task that adds an extension command adds its module
# here. The second wave added `shapes`, `lessons`, `calibrate` and `roofline`; the set stays
# a closed list so a command nobody planned still fails this test.
PLANNED = {"bench", "lib", "target", "prior", "campaign", "evolve", "board",
           "shapes", "lessons", "calibrate", "roofline"}
FIRST_WAVE = PLANNED          # the name the first wave's tests know


def test_the_package_ships_only_first_wave_commands():
    found = ext.commands()
    shipped = {p.stem for p in (REPO / "src" / "rsi_kernel_tune" / "ext").iterdir()
               if p.suffix == ".py" and not p.name.startswith("_")}
    assert set(found) == shipped          # discovery finds exactly the modules that are there
    assert set(found) <= PLANNED
    assert ext.command_names() == frozenset(shipped)


# --- found ---------------------------------------------------------------------------------


def test_a_module_in_the_directory_is_a_command_before_anything_imports_it(tmp_path, monkeypatch):
    _plug(monkeypatch, tmp_path, hello=HELLO, zeta=_with("zeta"), _helper=_with("_helper"))
    found = ext.commands()
    # `_helper` is a helper, not a command; the shipped modules stay listed too
    assert [n for n in found if n in {"hello", "zeta", "_helper"}] == ["hello", "zeta"]
    assert found["hello"] == ext.Extension("hello", "rsi_kernel_tune.ext.hello")
    assert found["hello"].module == found["hello"].load().__name__
    assert "rsi_kernel_tune.ext._helper" not in sys.modules
    assert ext.command_names() == frozenset(found)
    # listed again on each call: a module added later is a command the moment it is there
    _plug(monkeypatch, tmp_path, late=_with("late"))
    assert [n for n in ext.commands() if n in {"hello", "zeta", "late"}] == ["hello", "late",
                                                                             "zeta"]


def test_what_cannot_be_listed_is_no_command_and_breaks_nothing_that_asks(tmp_path, monkeypatch):
    def _explode():
        raise OSError("directory gone")

    monkeypatch.setattr(ext, "commands", _explode)
    assert ext.command_names() == frozenset()
    assert "none could be listed (OSError: directory gone)" in ext.usage(("run",))
    assert "Extension commands" in forge_main._usage()       # the usage still prints
    assert load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path)).name == "rmsnorm"
    assert forge_main.main(["--help"]) == 0


# --- dispatched ----------------------------------------------------------------------------


def test_forge_runs_the_command_with_what_follows_it(tmp_path, monkeypatch, capsys):
    _plug(monkeypatch, tmp_path, hello=HELLO)
    assert "rsi_kernel_tune.ext.hello" not in sys.modules
    assert forge_main.main(["hello", "a", "b"]) == 7
    assert capsys.readouterr().out == "hello a b\n"
    assert sys.modules["rsi_kernel_tune.ext.hello"].COMMAND == "hello"     # imported by the call
    assert forge_main.main(["hello"]) == 7 and capsys.readouterr().out == "hello\n"


def test_an_extension_may_answer_nothing_and_that_is_success(tmp_path, monkeypatch):
    _plug(monkeypatch, tmp_path, quiet=_with("quiet").replace("return 7", "return None"))
    assert forge_main.main(["quiet"]) == 0
    _plug(monkeypatch, tmp_path, loud=_with("loud").replace("return 7", "return 3"))
    assert forge_main.main(["loud"]) == 3


def test_what_a_loaded_extensions_main_raises_is_its_own(tmp_path, monkeypatch):
    _plug(monkeypatch, tmp_path, crasher=_with("crasher").replace("return 7", 'raise RuntimeError("boom")'))
    with pytest.raises(RuntimeError, match="boom"):
        forge_main.main(["crasher"])


def test_the_usage_text_names_the_commands_without_importing_them(tmp_path, monkeypatch, capsys):
    _plug(monkeypatch, tmp_path, zeta=_with("zeta"), hello=HELLO)
    assert forge_main.main(["--help"]) == 0
    out = capsys.readouterr().out
    # the built-in usage as it was, then one line per extension command, in name order
    assert "forge new tasks/<op>" in out and "forge run .artifacts/<op>_<id>" in out
    tail = out.split("Extension commands (src/rsi_kernel_tune/ext/, one module each):", 1)[1]
    lines = [line.split()[1] for line in tail.splitlines() if line.startswith("  forge ")]
    assert lines == sorted(lines)                     # one line per command, by name
    assert {"hello", "zeta"} <= set(lines)
    assert set(ext.command_names()) <= set(lines)     # whatever is found is listed
    for name in ("hello", "zeta"):
        assert f"(`forge {name} --help`)" in out
    assert out == forge_main._usage() + "\n"
    assert "rsi_kernel_tune.ext.hello" not in sys.modules and "rsi_kernel_tune.ext.zeta" not in sys.modules
    for word in ([], ["help"], ["-h"], ["--help"]):
        assert forge_main.main(word) == 0
        assert "forge hello ..." in capsys.readouterr().out
    assert capsys.readouterr().err == ""


# --- kept apart ----------------------------------------------------------------------------


def test_a_broken_extension_breaks_nothing_else_and_says_what_is_wrong_with_it(
        tmp_path, monkeypatch, capsys):
    _plug(monkeypatch, tmp_path, hello=HELLO, broken=BROKEN, syntax=SYNTAX, misnamed=MISNAMED,
          silent=NO_MAIN, stiff=STIFF)
    assert forge_main.main(["--help"]) == 0                   # the usage still prints
    assert "forge broken ..." in capsys.readouterr().out      # ... and still lists it
    assert forge_main.main(["--version"]) == 0                # so does a built-in
    assert forge_main.main(["hello"]) == 7                    # and a sound extension
    capsys.readouterr()
    for name, why in (("broken", "ModuleNotFoundError: No module named 'no_such_module_for_a_test'"),
                      ("syntax", "SyntaxError: "),
                      ("misnamed", "TypeError: COMMAND is 'other' in rsi_kernel_tune.ext.misnamed"),
                      ("silent", "AttributeError: module 'rsi_kernel_tune.ext.silent' has no attribute 'main'"),
                      ("stiff", "TypeError: main is not callable")):
        assert forge_main.main([name, "arg"]) == 2, name
        err = capsys.readouterr()
        assert err.err.startswith(f"[forge] extension command '{name}' failed to load: {why}"), name
        assert err.err.count("\n") == 1 and err.out == ""
    assert forge_main.main(["hello"]) == 7                    # still fine after the failures
    assert "rsi_kernel_tune.ext.broken" not in sys.modules        # nothing half-imported is left behind


def test_a_built_in_command_wins_over_an_extension_of_the_same_name(tmp_path, monkeypatch, capsys):
    _plug(monkeypatch, tmp_path, run=_with("run"), hello=HELLO)
    monkeypatch.setitem(launch.COMMANDS, "run", lambda argv: 42)
    assert forge_main.main(["run", "anything"]) == 42         # the built-in, not the extension
    assert forge_main.main(["hello"]) == 7
    assert "forge run ..." not in ext.usage((*forge_main.BUILTIN_COMMANDS, "help"))
    assert forge_main.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "forge hello ..." in out and "forge run ..." not in out
    assert "forge run .artifacts" in out                      # the built-in's own line stays


def test_the_operators_own_commands_win_as_the_second_word(tmp_path, monkeypatch):
    _plug(monkeypatch, tmp_path, hello=HELLO)
    seen: list = []
    monkeypatch.setattr(rsi_kernel_tune.loop.cli, "main", lambda argv: seen.append(argv) or 3)
    assert forge_main.main(["hello", "loop", "--x"]) == 3     # `forge <op> loop`: a built-in
    assert seen == [["hello", "--x"]]                         # handed on without the word
    assert "rsi_kernel_tune.ext.hello" not in sys.modules


def test_an_unknown_first_word_still_goes_to_the_operator_commands(tmp_path, monkeypatch):
    _plug(monkeypatch, tmp_path, hello=HELLO)
    seen: list = []
    monkeypatch.setattr(rsi_kernel_tune.submit.cli, "main", lambda argv: seen.append(argv) or 5)
    assert forge_main.main(["someop", "submit", "--mode", "all"]) == 5
    assert seen == [["someop", "submit", "--mode", "all"]]


# --- reserved ------------------------------------------------------------------------------


def test_a_task_may_not_take_an_extensions_name(tmp_path, monkeypatch):
    _plug(monkeypatch, tmp_path, hello=HELLO)
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, MINIMAL, name="hello"), kb_root=_kb(tmp_path))
    assert e.value.problems == [
        "'hello' is a forge extension command (src/rsi_kernel_tune/ext/hello.py); name the task otherwise"]
    assert "rsi_kernel_tune.ext.hello" not in sys.modules       # refused by name alone
    # a task may still be named anything else, and forge's own commands are refused as before
    assert load_task(_task(tmp_path, MINIMAL, name="hell"), kb_root=_kb(tmp_path)).name == "hell"
    with pytest.raises(TaskError, match="one of forge's own commands"):
        load_task(_task(tmp_path, MINIMAL, name="run"), kb_root=_kb(tmp_path))


def test_a_task_keeps_its_name_when_no_extension_claims_it(tmp_path, monkeypatch):
    # reserved is whatever is in the directory now: the same task loads once it is empty again
    assert load_task(_task(tmp_path, MINIMAL, name="hello"), kb_root=_kb(tmp_path)).name == "hello"
