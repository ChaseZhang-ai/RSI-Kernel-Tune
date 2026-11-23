"""Development machine vs graded machine: what differs, and what that costs.

``compare(dev, profile)`` turns the probe's ``target/dev.json`` and the profile's
``target/B200.json`` into a list of ``Difference``s, each with a level and an action:

``blocker``  the board's software stack cannot compile or run on this machine (a missing
             runtime, a missing device, a driver below the board image's floor). Nothing
             measured here is worth anything until it is fixed.
``risk``     the stack works but a reading taken here is optimistic or does not transfer:
             different SM count, different FP4 / SFU throughput, unlocked or different
             clocks, other library versions.
``info``     expected and harmless: the compute capability differs (that is why the card
             compiles for the device and the submission derives its arch), python version,
             memory size.

The comparison is data-driven: every threshold comes from the profile, so the same code
serves a future target by editing ``target/B200.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCKER = "blocker"
RISK = "risk"
INFO = "info"
LEVELS = (BLOCKER, RISK, INFO)
_ORDER = {level: i for i, level in enumerate(LEVELS)}

CLOCK_TOLERANCE_MHZ = 50


@dataclass(frozen=True)
class Difference:
    level: str             # blocker | risk | info
    key: str
    target: str
    dev: str
    advice: str

    def to_dict(self) -> dict:
        return {"level": self.level, "key": self.key, "target": self.target, "dev": self.dev,
                "advice": self.advice}

    def __str__(self) -> str:
        return f"[{self.level}] {self.key}: target {self.target}, development {self.dev} -- {self.advice}"


def _cc(cc) -> tuple[int, int] | None:
    if isinstance(cc, (list, tuple)) and len(cc) >= 2:
        try:
            return int(cc[0]), int(cc[1])
        except (TypeError, ValueError):
            return None
    return None


def _text(value, dash: str = "unknown") -> str:
    if value is None or value == "":
        return dash
    return str(value)


def _version_tuple(text) -> tuple[int, ...]:
    out = []
    for part in str(text or "").replace("-", ".").split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _at_least(got, want) -> bool | None:
    """Is ``got`` >= ``want``? None when either side is not a version."""
    g, w = _version_tuple(got), _version_tuple(want)
    if not g or not w:
        return None
    return g >= w


def _mm(text) -> tuple[int, int] | None:
    parts = _version_tuple(text)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def compare(dev: dict, profile: dict) -> list[Difference]:
    """Every difference worth reporting, blockers first."""
    target = profile.get("device") or {}
    reference = profile.get("development_gpu") or {}
    board_sw = profile.get("software") or {}
    gpu = dev.get("gpu") or {}
    software = dev.get("software") or {}
    smokes = dev.get("smokes") or {}
    out: list[Difference] = []

    def add(level: str, key: str, target_text: str, dev_text: str, advice: str) -> None:
        out.append(Difference(level, key, target_text, dev_text, advice))

    # -- the stack that has to work ------------------------------------------------
    for name in ("triton", "cutedsl", "cuda_inline"):
        smoke = smokes.get(name)
        if smoke is None:
            add(INFO, f"smoke:{name}", "runs on the board", "not run by this probe",
                "re-run `forge target probe` to cover this path")
            continue
        if not smoke.get("compile"):
            add(BLOCKER, f"smoke:{name}", "compiles on the board", "does not compile here",
                f"{name} path unusable on this machine: {_text(smoke.get('error'), 'no error text')}")
        elif not smoke.get("run"):
            add(BLOCKER, f"smoke:{name}", "runs on the board", "compiles but does not run here",
                f"{name} path unusable on this machine: {_text(smoke.get('error'), 'no error text')}")
        else:
            add(INFO, f"smoke:{name}", "compiles and runs", "compiles and runs",
                "path is usable; its timings are still development readings")

    # -- the device ----------------------------------------------------------------
    if not gpu or not gpu.get("name"):
        add(BLOCKER, "gpu", "a CUDA device", "none detected",
            "no GPU on this machine: nothing can be measured; point local.env at the dev host")
        return sorted(out, key=lambda d: _ORDER.get(d.level, 9))

    dev_cc, target_cc = _cc(gpu.get("compute_capability")), _cc(target.get("compute_capability"))
    if dev_cc is None:
        add(RISK, "compute_capability", _cc_text(target_cc), "not reported",
            "the probe could not read the device's capability; treat every reading as unverified")
    elif target_cc is not None:
        if dev_cc < target_cc:
            add(BLOCKER, "compute_capability", _cc_text(target_cc), _cc_text(dev_cc),
                "an older device cannot run the target's features: developing here cannot "
                "validate the graded build")
        elif dev_cc != target_cc:
            add(INFO, "compute_capability", _cc_text(target_cc), _cc_text(dev_cc),
                "expected: the card compiles for this device (dev_arch) and a submission "
                "derives its gencode at run time")

    dev_sms = gpu.get("multi_processor_count")
    target_sms = target.get("multi_processor_count")
    if dev_sms and target_sms and int(dev_sms) != int(target_sms):
        add(RISK, "sm_count", f"{target_sms}", f"{dev_sms}",
            f"grid sizes tuned here do not transfer; export FK_TARGET_SMS={target_sms} to "
            f"simulate the target's SM count (kernel code must fall back to the device query "
            f"when it is unset)")

    dev_mem = gpu.get("memory_gib")
    if dev_mem and target.get("memory_gib") and float(dev_mem) < float(target["memory_gib"]):
        add(RISK, "memory", f"{target['memory_gib']} GiB", f"{dev_mem} GiB",
            "a workspace that fits here may not fit on the target; the overflow shows up as "
            "an allocation failure on the board")

    dev_clocks = gpu.get("clocks") or {}
    target_clocks = target.get("clocks") or {}
    if dev_clocks.get("sm_mhz") is None:
        add(INFO, "clocks", _clocks(target_clocks), "not reported",
            "the probe could not read the clocks; lock them before comparing two readings")
    elif not dev_clocks.get("locked"):
        add(RISK, "clocks", _clocks(target_clocks), f"{_clocks(dev_clocks)} (not locked)",
            "unlocked clocks drift between runs; lock SM and memory clocks (nvidia-smi -lgc "
            "-lmc) before you compare kernels")
    elif target_clocks.get("sm_mhz") and abs(int(dev_clocks.get("sm_mhz") or 0)
                                            - int(target_clocks["sm_mhz"])) > CLOCK_TOLERANCE_MHZ:
        add(RISK, "clocks", _clocks(target_clocks), _clocks(dev_clocks),
            "a different clock changes the compute:bandwidth balance, so the best tile here "
            "may not be the best tile there")

    driver_floor = board_sw.get("driver_min")
    if driver_floor:
        ok = _at_least(gpu.get("driver"), driver_floor)
        if ok is False:
            add(BLOCKER, "driver", f">= {driver_floor}", _text(gpu.get("driver")),
                "the board image requires a newer driver: the CUDA runtime will refuse or "
                "silently fall back")

    # -- the software stack --------------------------------------------------------
    board_torch = board_sw.get("torch")
    if board_torch:
        if not software.get("torch"):
            add(BLOCKER, "torch", board_torch, "not installed",
                "torch is the harness's only runtime; install the board's version")
        elif _version_tuple(software.get("torch")) != _version_tuple(board_torch):
            add(RISK, "torch", board_torch, _text(software.get("torch")),
                "torch's kernels (SDPA, _scaled_mm, cuBLAS heuristics) differ between "
                "versions: a reading is from another stack")

    board_triton = board_sw.get("triton")
    if board_triton:
        if not software.get("triton"):
            add(BLOCKER, "triton", board_triton, "not installed",
                "the Triton path cannot be exercised here")
        else:
            if _version_tuple(software.get("triton")) != _version_tuple(board_triton):
                add(RISK, "triton", board_triton, _text(software.get("triton")),
                    "a different Triton compiles different code: readings and target-compile "
                    "results are indicative, not equal")
            if not software.get("triton_is_fbtriton"):
                add(RISK, "triton_distribution", "fbtriton", "stock triton",
                    "the board runs the fbtriton fork; confirm the constructs you rely on "
                    "exist there (`forge target compile-check` uses the installed one)")

    board_dsl = board_sw.get("cutlass_dsl")
    if board_dsl:
        if not software.get("cutlass_dsl"):
            add(BLOCKER, "cutlass_dsl", board_dsl, "not installed",
                "CuTe DSL kernels cannot be compiled or run here")
        elif _version_tuple(software.get("cutlass_dsl")) != _version_tuple(board_dsl):
            add(RISK, "cutlass_dsl", board_dsl, _text(software.get("cutlass_dsl")),
                "CuTe DSL's API and code generation change between releases")

    cuda = software.get("cuda_toolkit") or software.get("cuda")
    if board_sw.get("cuda") and cuda:
        ok = _at_least(cuda, board_sw["cuda"])
        if ok is False:
            add(RISK, "cuda_toolkit", board_sw["cuda"], _text(cuda),
                "an older toolkit may not know the target architecture; a gencode it cannot "
                "emit fails the target compile check")
    if not software.get("nvcc"):
        add(RISK, "nvcc", "present (CUDA 13.1.1)", "not found",
            "inline-CUDA submissions cannot be compiled locally and the target compile check "
            "degrades to a static check")

    board_py = _mm(board_sw.get("python"))
    dev_py = _mm(software.get("python"))
    if board_py and dev_py and board_py != dev_py:
        add(INFO, "python", _text(board_sw.get("python")), _text(software.get("python")),
            "a different interpreter is harmless unless a dependency pins it")

    if software.get("cupti") is None and software.get("cupti_checked"):
        add(RISK, "cupti", _text(board_sw.get("cupti_python")), "not installed",
            "NCU / CUPTI metrics are unavailable here: local timings use another basis")

    # -- the quantities the two GPUs differ in --------------------------------------
    if reference.get("dense_fp4_pflops") and target.get("dense_fp4_pflops"):
        add(RISK, "dense_fp4", f"{target['dense_fp4_pflops']} PFLOPS",
            f"{reference['dense_fp4_pflops']} PFLOPS",
            "FP4-bound kernels read optimistic here: report the local number as a lower bound")
    if reference.get("sfu_tera_exp_s") and target.get("sfu_tera_exp_s"):
        add(RISK, "sfu", f"{target['sfu_tera_exp_s']} TeraExp/s",
            f"{reference['sfu_tera_exp_s']} TeraExp/s",
            "exp/ex2-heavy kernels read optimistic here: report the local number as a lower bound")

    return sorted(out, key=lambda d: _ORDER.get(d.level, 9))


def _cc_text(cc: tuple[int, int] | None) -> str:
    return f"{cc[0]}.{cc[1]}" if cc else "unknown"


def _clocks(clocks: dict) -> str:
    sm, mem = clocks.get("sm_mhz"), clocks.get("mem_mhz")
    if sm is None and mem is None:
        return "not recorded"
    return f"SM {_text(sm)} MHz / memory {_text(mem)} MHz"


def summary(differences: list[Difference]) -> dict[str, int]:
    return {level: sum(1 for d in differences if d.level == level) for level in LEVELS}


def exit_code(differences: list[Difference]) -> int:
    """0 when nothing blocks, 2 when something does (a shell can gate on it)."""
    return 2 if any(d.level == BLOCKER for d in differences) else 0


def lines(differences: list[Difference], *, levels: tuple = (BLOCKER, RISK)) -> list[str]:
    """One line per difference of the given levels (for the rendered profile)."""
    return [f"{d.level}: {d.key}: target {d.target}, development {d.dev}"
            for d in differences if d.level in levels]
