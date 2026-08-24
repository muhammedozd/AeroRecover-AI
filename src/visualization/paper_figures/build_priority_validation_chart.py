"""Build the single-column IEEE priority validation comparison chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from paper_style import PAPER_COLORS, apply_paper_style, save_figure


PRIORITY_LABELS = ["P1\nCritical", "P2\nHigh", "P3\nMonitor", "P4\nNormal"]
OBSERVED_PROPAGATION_RATE = np.array([90.162950, 69.517142, 25.054820, 0.437218])
MEAN_PREDICTED_PROBABILITY = np.array([90.204041, 69.457932, 29.395485, 0.543847])


def add_value_labels(ax, bars, x_offset_points: float) -> None:
    for bar in bars:
        value = float(bar.get_height())
        # Keep sub-1% P4 labels visibly separated from the baseline.
        y = max(value + 1.1, 2.3)
        ax.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(x_offset_points, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=PAPER_COLORS["text"],
            clip_on=False,
        )


def main() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.5), facecolor=PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])

    positions = np.arange(len(PRIORITY_LABELS))
    width = .34
    bar_style = {"edgecolor": PAPER_COLORS["muted_text"], "linewidth": .4, "zorder": 3}
    observed_bars = ax.bar(
        positions - width / 2,
        OBSERVED_PROPAGATION_RATE,
        width,
        color=PAPER_COLORS["navy"],
        label="Observed propagation rate",
        **bar_style,
    )
    predicted_bars = ax.bar(
        positions + width / 2,
        MEAN_PREDICTED_PROBABILITY,
        width,
        color="#9FB6C8",
        label="Mean predicted probability",
        **bar_style,
    )

    # Shift paired labels slightly outwards so near-identical values remain distinct.
    add_value_labels(ax, observed_bars, x_offset_points=-2.5)
    add_value_labels(ax, predicted_bars, x_offset_points=2.5)

    ax.set_ylabel("Rate (%)", fontsize=8.5)
    ax.set_xticks(positions, PRIORITY_LABELS)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", labelsize=8, length=2.5, width=.6)
    ax.grid(axis="y", color=PAPER_COLORS["light_border"], linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PAPER_COLORS["border"])
    ax.spines["bottom"].set_color(PAPER_COLORS["border"])
    ax.spines["left"].set_linewidth(.65)
    ax.spines["bottom"].set_linewidth(.65)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.7,
        handlelength=1.2,
        handletextpad=.45,
        labelspacing=.25,
        borderaxespad=.25,
    )
    fig.subplots_adjust(left=.16, right=.985, bottom=.20, top=.98)
    save_figure(fig, "priority_validation_comparison")
    plt.close(fig)


if __name__ == "__main__":
    main()
