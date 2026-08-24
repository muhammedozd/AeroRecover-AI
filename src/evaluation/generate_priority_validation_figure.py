"""Generate the IEEE paper figure for observed propagation rates by priority."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.paper_figures.paper_style import (  # noqa: E402
    PAPER_COLORS,
    apply_paper_style,
    save_figure,
)


PRIORITIES = ("P1 Critical", "P2 High", "P3 Monitor", "P4 Normal")
OBSERVED_RATES = np.array([90.162950, 69.517142, 25.054820, 0.437218])
FLIGHT_COUNTS = np.array([39_951, 17_997, 45_604, 728_470])


def main() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.7), facecolor=PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])

    positions = np.arange(len(PRIORITIES))
    bars = ax.bar(
        positions,
        OBSERVED_RATES,
        width=0.58,
        color="#16324F",
        edgecolor=PAPER_COLORS["text"],
        linewidth=0.4,
        rasterized=False,
        zorder=3,
    )

    for bar, rate in zip(bars, OBSERVED_RATES):
        # Keep the P4 value legible above its sub-1% bar without changing the data.
        label_y = max(float(rate) + 1.15, 2.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=PAPER_COLORS["text"],
            rasterized=False,
            clip_on=False,
        )

    tick_labels = [
        f"{priority}\nn={count:,}"
        for priority, count in zip(PRIORITIES, FLIGHT_COUNTS)
    ]
    ax.set_xticks(positions, tick_labels)
    ax.set_ylabel("Observed propagation rate (%)", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="x", labelsize=7.4, length=0, pad=4)
    ax.tick_params(axis="y", labelsize=8, length=2.5, width=0.6)

    ax.grid(
        axis="y",
        color=PAPER_COLORS["light_border"],
        linewidth=0.45,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PAPER_COLORS["border"])
        ax.spines[side].set_linewidth(0.65)

    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.23, top=0.985)
    save_figure(fig, "priority_observed_propagation_rate")
    plt.close(fig)


if __name__ == "__main__":
    main()
