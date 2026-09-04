"""Render assets/fairness_audit.png from the real fairness-audit output.

Reads the `fairness` block straight out of demo_output.json (the committed
output of an actual `python pipeline.py demo` run against the real 150k-row
GMSC test set -- see src/fairness.py and pipeline.py). No illustrative or
made-up numbers: if demo_output.json changes (e.g. after a re-run), re-run
this script and the chart updates to match.

Usage:
    python scripts/generate_fairness_chart.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
DEMO_OUTPUT = ROOT / "demo_output.json"
OUT_PATH = ROOT / "assets" / "fairness_audit.png"

# -- palette (dataviz skill reference palette, light mode) ------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"       # categorical slot 1 -- reference-rate group
CRITICAL = "#d03b3b"   # status: fails the four-fifths threshold
GOOD = "#0ca30c"

FOUR_FIFTHS_RULE = 0.80


def draw_bar(ax, x, height, width, color):
    """A column growing from a square baseline."""
    ax.add_patch(Rectangle((x - width / 2, 0), width, height, linewidth=0, facecolor=color, zorder=3))


def main():
    demo = json.loads(DEMO_OUTPUT.read_text())
    fairness = demo["fairness"]
    rates = fairness["group_decline_rates"]
    ratio = fairness["four_fifths_ratio"]
    passes = fairness["passes_four_fifths_rule"]

    under_40 = rates["age_under_40"]
    plus_40 = rates["age_40_plus"]
    reference = max(under_40, plus_40)
    threshold = FOUR_FIFTHS_RULE * reference

    groups = [("Under 40", under_40, BLUE), ("40 and older", plus_40, CRITICAL)]

    OUT_PATH.parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ymax = reference * 1.35
    xs = [0.3, 0.9]
    width = 0.34

    for x, (label, rate, color) in zip(xs, groups):
        draw_bar(ax, x, rate, width, color)
        ax.text(
            x,
            rate + ymax * 0.025,
            f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=INK_PRIMARY,
        )
        ax.text(x, -ymax * 0.045, label, ha="center", va="top", fontsize=11, color=INK_SECONDARY)

    # four-fifths threshold line: the minimum rate the lower group would
    # need to clear 80% of the higher (reference) group's rate.
    ax.axhline(threshold, color=INK_SECONDARY, linewidth=1.5, linestyle=(0, (5, 3)), zorder=2)
    ax.text(
        1.18,
        threshold + ymax * 0.02,
        "four-fifths threshold",
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=INK_SECONDARY,
    )
    ax.text(
        1.18,
        threshold - ymax * 0.02,
        f"80% of reference = {threshold * 100:.1f}%",
        ha="left",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )

    # status chip, top-right -- color plus icon + label, never color alone
    verdict = "FAILS" if not passes else "PASSES"
    vcolor = CRITICAL if not passes else GOOD
    icon = "✕" if not passes else "✓"
    chip_text = f"{icon}  {verdict} the four-fifths rule -- ratio {ratio:.2f} (needs ≥ 0.80)"
    ax.text(
        1.63,
        ymax * 1.0,
        chip_text,
        ha="right",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=vcolor,
    )

    # gridlines / axis
    for gy in [0.01, 0.02, 0.03, 0.04]:
        if gy <= ymax:
            ax.axhline(gy, color=GRIDLINE, linewidth=1, zorder=1)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=2)

    ax.set_ylim(-ymax * 0.09, ymax)
    ax.set_xlim(0, 1.65)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    ax.set_yticklabels([f"{v * 100:.0f}%" for v in [0, 0.01, 0.02, 0.03, 0.04]], color=INK_MUTED, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.text(
        0.045,
        0.965,
        "Decline rate by age group -- four-fifths fairness audit",
        fontsize=15,
        fontweight="bold",
        color=INK_PRIMARY,
        va="top",
    )
    fig.text(
        0.045,
        0.905,
        "Real trained-model output on the GMSC test set  ·  src/fairness.py  ·  demo_output.json",
        fontsize=9.5,
        color=INK_MUTED,
        va="top",
    )
    fig.savefig(OUT_PATH, facecolor=SURFACE)
    print(f"wrote {OUT_PATH}  (under_40={under_40:.4f}, 40_plus={plus_40:.4f}, ratio={ratio:.4f})")


if __name__ == "__main__":
    main()
