"""The graded hardware (B200) versus the one you develop on: probe, compare, compile-check.

  forge target probe [--out FILE] [--remote] [--no-smokes]
                                  What this machine (or the ssh host of local.env) is: GPU,
                                  clocks, stack, and three smokes compiled and run for real.
                                  Writes target/dev.json (not committed).
  forge target check [--dev FILE] [--target FILE] [--json]
                                  Differences between the development machine and the target,
                                  graded blocker / risk / info, each with what to do about it.
                                  Exit 2 when something blocks.
  forge target compile-check PATH [--arch sm100] [--json]
                                  Compile a submission.py (or a task directory) for the target
                                  architecture without a target GPU: embedded CUDA through
                                  nvcc, Triton kernels through triton.compile, CuTe DSL
                                  statically, plus the rules an exported submission must pass.
                                  Exit 2 on a failure.
  forge target show [--json] [--out FILE]
                                  The target profile as the agents get it (Markdown).

The profile itself is target/B200.json; the agents read it through
``runcontext.env_for({"target": target_profile.materialize(...)})``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMAND = "target"


def cmd_probe(args) -> int:
    from forgekernel import target_profile
    from forgekernel.targethw import parity, probe

    try:
        profile = target_profile.load(args.root)
    except target_profile.ProfileError as e:
        sys.stderr.write(f"forge target probe: {e}\n")
        return 2
    try:
        dev = probe.probe(remote=args.remote, out=args.out, root=args.root, profile=profile,
                          include_smokes=not args.no_smokes, timeout=args.timeout)
    except probe.ProbeError as e:
        sys.stderr.write(f"forge target probe: {e}\n")
        return 2
    if args.json:
        print(json.dumps({k: v for k, v in dev.items() if not k.startswith("_")}, indent=2))
        return 0
    print(_probe_summary(dev))
    path = dev.get("path")
    if path:
        print(f"\nwrote {path}")
    counts = (dev.get("parity") or {})
    if counts.get(parity.BLOCKER):
        print(f"\n{counts[parity.BLOCKER]} blocker(s): `forge target check` says what and why")
    return 0


def _probe_summary(dev: dict) -> str:
    gpu = dev.get("gpu") or {}
    software = dev.get("software") or {}
    lines = [f"machine   {dev.get('delivery') or dev.get('mode')}"]
    if gpu:
        cc = gpu.get("compute_capability") or []
        cc_text = ".".join(str(x) for x in cc) if cc else "?"
        clocks = gpu.get("clocks") or {}
        lock = {True: "locked", False: "not locked", None: "lock unknown"}.get(
            gpu.get("clocks_locked"), "lock unknown")
        lines.append(f"gpu       {gpu.get('name') or '?'}  CC {cc_text} "
                     f"({gpu.get('cuda_arch') or '?'})  {gpu.get('multi_processor_count') or '?'} "
                     f"SMs  {gpu.get('memory_gib') or '?'} GiB")
        if clocks:
            lines.append(f"clocks    SM {clocks.get('sm_mhz') or '?'} MHz / memory "
                         f"{clocks.get('mem_mhz') or '?'} MHz ({lock})")
        if gpu.get("driver"):
            lines.append(f"driver    {gpu['driver']}")
    else:
        lines.append(f"gpu       none reported ({gpu.get('error') or 'no device'})")
    stack = ", ".join(f"{k} {v}" for k, v in (
        ("cuda", software.get("cuda_toolkit")), ("torch", software.get("torch")),
        ("triton", software.get("triton")), ("cutlass-dsl", software.get("cutlass_dsl")),
        ("cupti", software.get("cupti"))) if v)
    lines.append(f"stack     {stack or '?'}")
    if software.get("triton") and not software.get("triton_is_fbtriton"):
        lines.append("          (a stock triton: the board runs the fbtriton fork)")
    for name, smoke in (dev.get("smokes") or {}).items():
        if smoke.get("run"):
            state = f"ok, {smoke.get('ms'):.3f} ms" if isinstance(smoke.get("ms"), (int, float)) \
                else "ok"
        elif smoke.get("compile"):
            state = "compiles, does not run"
        else:
            state = "FAILED"
        detail = f" -- {smoke.get('error')}" if smoke.get("error") else ""
        lines.append(f"smoke {name:11s} {state}{detail}")
    counts = dev.get("parity") or {}
    if counts:
        lines.append(f"parity    {counts.get('blocker', 0)} blocker, {counts.get('risk', 0)} "
                     f"risk, {counts.get('info', 0)} info")
    return "\n".join(lines)


def cmd_check(args) -> int:
    from forgekernel import target_profile
    from forgekernel.targethw import parity, probe

    try:
        profile = target_profile.load(args.target_root)
    except target_profile.ProfileError as e:
        sys.stderr.write(f"forge target check: {e}\n")
        return 2
    dev = _load_dev(args)
    if dev is None:
        sys.stderr.write("forge target check: no probe result (run `forge target probe` first, "
                         "or pass --dev FILE)\n")
        return 2
    differences = parity.compare(dev, profile)
    if args.json:
        print(json.dumps({"device": profile.get("device", {}).get("name"),
                          "differences": [d.to_dict() for d in differences],
                          "summary": parity.summary(differences)}, indent=2))
    else:
        print(f"{profile.get('device', {}).get('name', 'target')} vs "
              f"{dev.get('delivery') or dev.get('mode') or 'development machine'}")
        if not differences:
            print("no differences worth reporting")
        for level in parity.LEVELS:
            group = [d for d in differences if d.level == level]
            if not group:
                continue
            print(f"\n{level} ({len(group)})")
            for d in group:
                print(f"  {d.key}: target {d.target}, development {d.dev}")
                print(f"    -> {d.advice}")
        counts = parity.summary(differences)
        print(f"\n{counts['blocker']} blocker, {counts['risk']} risk, {counts['info']} info")
    return parity.exit_code(differences)


def _load_dev(args) -> dict | None:
    from forgekernel import target_profile

    path = Path(args.dev) if getattr(args, "dev", None) else None
    if path is None:
        return target_profile.load_dev(args.root)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        sys.stderr.write(f"forge target check: {path}: {e}\n")
        return None


def cmd_compile_check(args) -> int:
    from forgekernel.targethw import compile_check

    try:
        report = compile_check.check_file(args.path, arch=args.arch, timeout=args.timeout)
    except compile_check.CompileCheckError as e:
        sys.stderr.write(f"forge target compile-check: {e}\n")
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return report.exit_code()
    for line in report.lines():
        print(line)
    if report.failed:
        print(f"\n{len(report.failed)} failure(s): this submission would not build or would be "
              f"refused on the target")
    elif report.unknown:
        print(f"\n{len(report.unknown)} check(s) unknown on this machine (see the reasons above)")
    return report.exit_code()


def cmd_show(args) -> int:
    from forgekernel import target_profile

    try:
        profile = target_profile.load(args.root)
    except target_profile.ProfileError as e:
        sys.stderr.write(f"forge target show: {e}\n")
        return 2
    if args.json:
        text = json.dumps(profile, indent=2) + "\n"
    else:
        text = target_profile.render_markdown(profile, target_profile.load_dev(args.root))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(text.splitlines())} lines)")
        return 0
    sys.stdout.write(text)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge target",
        description="The graded GPU (B200) versus the one you develop on: probe the machine, "
                    "compare the two, and check a submission compiles for the target.")
    p.add_argument("--root", default=None, help="repository root (default: this checkout)")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    probe_p = sub.add_parser(
        "probe", help="measure this machine (or the ssh host of local.env) and write "
                      "target/dev.json")
    probe_p.add_argument("--out", default=None, help="where to write the result")
    probe_p.add_argument("--remote", action="store_true",
                         help="probe the ssh host local.env names, through the GPU-job delivery")
    probe_p.add_argument("--local", action="store_true", help="this machine (the default)")
    probe_p.add_argument("--no-smokes", action="store_true",
                         help="skip the three smoke kernels (faster; no compile/run evidence)")
    probe_p.add_argument("--timeout", type=int, default=900,
                         help="seconds the probe may run (default 900)")
    probe_p.add_argument("--json", action="store_true", help="the full result as JSON")
    probe_p.set_defaults(func=cmd_probe)

    check_p = sub.add_parser(
        "check", help="compare target/dev.json with the target profile (blocker / risk / info)")
    check_p.add_argument("--dev", default=None, help="a probe result file (default target/dev.json)")
    check_p.add_argument("--target", dest="target_root", default=None,
                         help="the target profile's repository root")
    check_p.add_argument("--json", action="store_true", help="the differences as JSON")
    check_p.set_defaults(func=cmd_check)

    cc_p = sub.add_parser(
        "compile-check", help="compile a submission for the target architecture (no target GPU)")
    cc_p.add_argument("path", help="a submission.py, a kernel file, or a task directory")
    cc_p.add_argument("--arch", default="sm100",
                      help="the target architecture (default sm100 = B200)")
    cc_p.add_argument("--timeout", type=int, default=900, help="seconds per compile (default 900)")
    cc_p.add_argument("--json", action="store_true", help="the conclusions as JSON")
    cc_p.set_defaults(func=cmd_compile_check)

    show_p = sub.add_parser("show", help="the target profile as the agents read it")
    show_p.add_argument("--json", action="store_true", help="the profile itself")
    show_p.add_argument("--out", default=None, help="write it here instead of stdout")
    show_p.set_defaults(func=cmd_show)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
