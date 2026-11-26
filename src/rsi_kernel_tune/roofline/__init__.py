"""Measured peaks and the distance from them: ``forge calibrate`` + ``forge roofline``.

Three pieces, three files, one flow:

* ``calibrate`` -- measure the machine's peaks (device and L2 bandwidth, dense tensor
  throughput per precision, the scalar and special-function pipes, launch latency) as
  standalone probes delivered through this project's GPU-job machinery, and archive them
  as one JSON document. The peaks of the **development** machine are measured, never
  assumed;
* ``analyze`` -- the target machine's peaks (the graded GPU's official figures, from
  ``target/B200.json`` and ``peaks.toml``) and the development machine's measured ones,
  against a cost model (FLOPs and bytes per workload, from a file or one teacher turn):
  the floor each workload cannot beat, its bottleneck class and how far a reading is
  from it;
* ``render`` -- that analysis as the Markdown context file a prompt reads
  (``{ROOFLINE_PATH}``).

The floor arithmetic is ``skills/roofline/roofline_tools.py``, shared with the skill so
``forge roofline`` and a GPU job's ``roofline_tools.py distance`` cannot disagree.

Nothing here changes the loop's gate (``loop/parse.py``'s ``device_median_ms``): the
roofline is a reading *about* a reading, an orchestration-layer input, not a target. And
nothing here belongs in ``hw_kb/``: an entry there is certified by a probe on the
architecture it describes, while a roofline is a model with efficiency factors in it.

::

    from rsi_kernel_tune.roofline import calibrate, analyze, costmodel, render
    doc = calibrate.calibrate(seconds=10)                  # or calibrate.load(path)
    peaks = analyze.peaks_from_calibration(doc)
    target = analyze.peaks_from_profile(json.loads(open("target/B200.json").read()),
                                        analyze.load_peaks())
    model = costmodel.load("costs.json")                    # or costmodel.estimate(...)
    result = analyze.analyze(problem_json, target, model, dev_peaks=peaks)
    print(render.render(result))
"""

from __future__ import annotations

from rsi_kernel_tune.roofline import analyze, calibrate, costmodel, render

__all__ = ["analyze", "calibrate", "costmodel", "render"]
