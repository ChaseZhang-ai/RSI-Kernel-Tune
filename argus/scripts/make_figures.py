#!/usr/bin/env python3
"""Generate the repository figures (docs/assets/*.png).

Deterministic, dependency-light (matplotlib only), and committed next to the figures so
every diagram in the README can be regenerated:

    python3 scripts/make_figures.py

Three figures: the system overview (figure1), the layered verification gate (figure2), and
the local ablation results (figure3).
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "assets"

INK = "#1F2937"
BLUE = "#3B6FA0"
BLUE_FILL = "#E8F0F8"
GREEN = "#3E7C59"
GREEN_FILL = "#E9F3EC"
ORANGE = "#C97B2D"
ORANGE_FILL = "#FBF0E2"
RED = "#B04A3F"
RED_FILL = "#F9ECEA"
GRAY = "#6B7280"
GRAY_FILL = "#F3F4F6"


def box(ax, x, y, w, h, title, body="", edge=BLUE, fill=BLUE_FILL, title_size=10,
        body_size=8.4, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 2)) if dashed else "solid"))
    cy = y + h * (0.72 if body else 0.5)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=title_size,
            color=INK, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center",
                fontsize=body_size, color=GRAY)


def arrow(ax, start, end, label="", color=INK, style="-|>", rad=0.0, dashed=False,
          label_dy=0.018, label_size=8.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13, linewidth=1.3, color=color,
        linestyle=(0, (4, 2)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=label_size, color=color)


def canvas(w=12.0, h=6.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[make_figures] wrote {out}")


# --------------------------------------------------------------------------- #
# figure 1: the system overview
# --------------------------------------------------------------------------- #

def figure_overview():
    fig, ax = canvas(12.4, 5.6)
    ax.text(0.5, 0.97, "ARGUS wraps the CudaForge agent loop in verification, attribution "
            "and a population", ha="center", va="center", fontsize=12.5,
            color=INK, fontweight="bold")

    # top row: the generation pipeline
    box(ax, 0.015, 0.66, 0.155, 0.17, "Operator task",
        "definition\nworkloads · anchors")
    box(ax, 0.205, 0.66, 0.155, 0.17, "Coder agent",
        "seed · rewrite", edge=GRAY, fill=GRAY_FILL)
    box(ax, 0.395, 0.66, 0.135, 0.17, "Candidate", "kernel + config")
    box(ax, 0.565, 0.66, 0.185, 0.17, "Verifier  (C1)",
        "tolerance · probes\nprecision · fingerprint", edge=GREEN, fill=GREEN_FILL)
    box(ax, 0.83, 0.66, 0.155, 0.17, "Verified kernel",
        "promoted / submitted", edge=GREEN, fill=GREEN_FILL)

    # bottom row: the evidence loop
    box(ax, 0.565, 0.22, 0.185, 0.17, "Evaluation",
        "board-protocol timing\nNCU counters", edge=ORANGE, fill=ORANGE_FILL)
    box(ax, 0.395, 0.22, 0.135, 0.17, "Attribution (C2)",
        "bottleneck\ndelta · levers ≤ x")
    box(ax, 0.205, 0.22, 0.155, 0.17, "Judge agent",
        "one lever, its ceiling", edge=GRAY, fill=GRAY_FILL)
    box(ax, 0.015, 0.22, 0.155, 0.17, "Population (C3)",
        "K lineages · UCB\nprereq filter · head parent")

    arrow(ax, (0.170, 0.745), (0.205, 0.745), "seed")
    arrow(ax, (0.360, 0.745), (0.395, 0.745))
    arrow(ax, (0.530, 0.745), (0.565, 0.745), "outputs")
    arrow(ax, (0.750, 0.745), (0.830, 0.745), "hard gate passes", color=GREEN)
    arrow(ax, (0.657, 0.66), (0.657, 0.39), "run · time · profile")
    arrow(ax, (0.565, 0.305), (0.530, 0.305), "counters", label_dy=0.105)
    arrow(ax, (0.395, 0.305), (0.360, 0.305), "causal context", label_dy=0.105)
    arrow(ax, (0.170, 0.305), (0.205, 0.305), "selects the arm", label_dy=0.105)
    arrow(ax, (0.282, 0.39), (0.282, 0.66), "propose", rad=-0.0)

    ax.text(0.5, 0.055,
            "the Coder/Judge seam is CudaForge's (reproduced); everything around it is ARGUS. "
            "The verified head becomes the next parent; a candidate that fails a hard check "
            "scores 0 and is never a parent.",
            ha="center", fontsize=8.6, color=GRAY, style="italic")
    save(fig, "figure1-overview.png")


# --------------------------------------------------------------------------- #
# figure 2: the layered verification gate
# --------------------------------------------------------------------------- #

def figure_verification():
    fig, ax = canvas(11.6, 6.2)
    ax.text(0.5, 0.968, "The verification gate: six layers, each catching what a "
            "single-shape allclose cannot see", ha="center", va="center",
            fontsize=12.5, color=INK, fontweight="bold")

    layers = [
        ("1 · static scan", "delegation to the reference · in-kernel narrowing · caches · "
         "clock tampering", RED, RED_FILL, "catches: a narrowing hidden in the source"),
        ("2 · board tolerance, all workloads", "|o−r| ≤ atol + rtol·|r| · matched ratio ≥ 0.99 "
         "· every workload", GREEN, GREEN_FILL, "catches: a kernel tuned to one shape"),
        ("3 · differential probes", "memoisation on perturbed inputs · identity · degenerate "
         "variance", GREEN, GREEN_FILL, "catches: computes once, replays the answer"),
        ("4 · precision contract", "candidate error against the fp64 oracle, relative to the "
         "reference's own error", GREEN, GREEN_FILL, "catches: tf32 accumulation that passes 1e-2"),
        ("5 · metamorphic laws", "scale-invariant norm · shift-invariant softmax · row "
         "equivariance", ORANGE, ORANGE_FILL, "catches: wrong off the sampled shapes"),
        ("6 · counter fingerprint", "a time drop no counter explains · bandwidth above peak · "
         "vanished stores", ORANGE, ORANGE_FILL, "catches: reports 3× faster, counters flat"),
    ]
    y = 0.815
    h = 0.098
    gap = 0.024
    for title, body, edge, fill, catches in layers:
        ax.add_patch(FancyBboxPatch(
            (0.09, y - h), 0.86, h, boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.3, edgecolor=edge, facecolor=fill))
        ax.text(0.115, y - h * 0.34, title, ha="left", va="center", fontsize=9.6,
                color=INK, fontweight="bold")
        ax.text(0.115, y - h * 0.72, body, ha="left", va="center", fontsize=8.1,
                color=GRAY)
        ax.text(0.915, y - h * 0.5, catches, ha="right", va="center", fontsize=7.9,
                color=edge, style="italic")
        y -= h + gap

    ax.add_patch(FancyBboxPatch(
        (0.09, 0.015), 0.86, 0.082, boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1.4, edgecolor=INK, facecolor="#FFFFFF"))
    ax.text(0.115, 0.070, "Verdict: accept iff every hard layer passes; the soft layers "
            "only raise flags.", ha="left", va="center", fontsize=9.2, color=INK,
            fontweight="bold")
    ax.text(0.115, 0.038, "Local check: 4/4 realistic hacks pass the single-shape allclose; "
            "0/4 pass this gate; the honest kernel passes both.",
            ha="left", va="center", fontsize=8.2, color=GRAY, style="italic")
    save(fig, "figure2-verification.png")


# --------------------------------------------------------------------------- #
# figure 3: local ablation results
# --------------------------------------------------------------------------- #

def figure_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    labels = ["honest", "tf32 accum", "shape-specialised", "memoised", "fake-fast"]
    cudaforge = [1, 1, 1, 1, 1]
    argus = [1, 0, 0, 0, 0]
    xs = range(len(labels))
    ax.bar([x - 0.18 for x in xs], cudaforge, width=0.34, color="#C0CBD6",
           edgecolor=GRAY, label="CudaForge gate (allclose, 1 shape)")
    ax.bar([x + 0.18 for x in xs], argus, width=0.34, color=GREEN, edgecolor=GREEN,
           label="ARGUS gate (six layers)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8.4, rotation=14, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["rejected", "accepted"], fontsize=8.4)
    ax.set_ylim(0, 1.5)
    ax.set_title("Which gate accepts which kernel", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.0))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    names = ["gate off\n(same chain)", "gate on\n(chain)", "gate on\n(population)"]
    vals = [0.0000, 0.7942, 0.9412]
    colors = [RED, "#C0CBD6", GREEN]
    bars = ax.bar(names, vals, width=0.55, color=colors,
                  edgecolor=["none"] * 3)
    ax.axhline(0.9412, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.text(2.42, 0.9412, "landscape optimum", fontsize=8.0, color=GRAY,
            va="center", ha="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center",
                fontsize=8.8, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("effective board score", fontsize=9.0)
    ax.set_title("Equal budget (40 evals, 6 seeds, mean)", fontsize=10.5, color=INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle("Local ablation on a rugged synthetic landscape — logic, not hardware "
                 "magnitudes", fontsize=11.0, color=GRAY, y=0.02, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "figure3-ablation.png")


def main():
    figure_overview()
    figure_verification()
    figure_ablation()


if __name__ == "__main__":
    main()
