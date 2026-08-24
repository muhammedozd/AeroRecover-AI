"""Build the scientifically correct AeroRecover AI architecture figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_style import PAPER_COLORS, apply_paper_style, save_figure


def process_box(ax, x, y, width, height, label, fill, accent=False, fontsize=8.7):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor=fill,
        edgecolor=PAPER_COLORS["bronze"] if accent else PAPER_COLORS["steel_blue"],
        linewidth=1.05, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=fontsize, color=PAPER_COLORS["text"], zorder=3)
    return patch


def routed_arrow(ax, vertices, *, dashed=False, color=None, width=1.0):
    path = Path(vertices, [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1))
    patch = FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=8.5,
        linewidth=width, linestyle=(0, (3, 2)) if dashed else "solid",
        color=color or PAPER_COLORS["steel_blue"], zorder=.5,
        capstyle="butt", joinstyle="miter",
    )
    ax.add_patch(patch)


def vertical_arrow(ax, upper, lower, *, color=None, dashed=False):
    x = upper.get_x() + upper.get_width() / 2
    routed_arrow(
        ax,
        [(x, upper.get_y() - .003), (x, lower.get_y() + lower.get_height() + .003)],
        color=color, dashed=dashed,
    )


def main() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(13.8, 4.75), facecolor=PAPER_COLORS["background"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    data_fill, model_fill, decision_fill = "#F4F7F9", "#E7EFF5", "#E7F0EF"
    ax.text(.16, .91, "DATA AND ROTATION", ha="center", fontsize=9, fontweight="bold",
            color=PAPER_COLORS["navy"])
    ax.text(.48, .91, "PREDICTION AND GRAPH", ha="center", fontsize=9, fontweight="bold",
            color=PAPER_COLORS["navy"])
    ax.text(.84, .91, "EXPLAINABILITY AND DECISION SUPPORT", ha="center", fontsize=9,
            fontweight="bold", color=PAPER_COLORS["navy"])

    w, h = .25, .105
    bts = process_box(ax, .035, .72, w, h, "BTS Flight Records", data_fill)
    rotation = process_box(ax, .035, .55, w, h, "Rotation Construction", data_fill)
    features = process_box(ax, .035, .38, w, h, "Operational Features", data_fill)
    prediction = process_box(ax, .355, .72, w, h, "XGBoost Prediction", model_fill)
    graph = process_box(ax, .355, .55, w, h, "Scored Aircraft-Rotation Graph", model_fill, fontsize=8.2)
    multihop = process_box(ax, .355, .38, w, h, "Multi-Hop Tracing", model_fill)
    shap = process_box(ax, .785, .72, .18, h, "SHAP Explanations", decision_fill)
    assessment = process_box(
        ax, .785, .47, .18, .145, "Likelihood–Impact–Urgency\nAssessment",
        decision_fill, fontsize=8.1,
    )
    priority = process_box(
        ax, .785, .28, .18, h, "P1–P4 Operational Priority",
        decision_fill, accent=True, fontsize=8.1,
    )

    # Main physical prediction pipeline.
    vertical_arrow(ax, bts, rotation)
    vertical_arrow(ax, rotation, features)
    routed_arrow(ax, [(.285, .4325), (.325, .4325), (.325, .7725), (.355, .7725)])
    vertical_arrow(ax, prediction, graph)
    vertical_arrow(ax, graph, multihop)

    # Parallel model-explanation branch; it does not originate from multi-hop tracing.
    routed_arrow(ax, [(.605, .7725), (.785, .7725)])

    # Compact assessment inputs, with thin dashed provenance lines.
    inputs = [
        process_box(ax, .625, .575, .145, .042, "Likelihood: model probability", "#FFFFFF", fontsize=6.8),
        process_box(ax, .625, .515, .145, .042, "Impact: downstream edge count", "#FFFFFF", fontsize=6.8),
        process_box(ax, .625, .455, .145, .042, "Urgency: turnaround conditions", "#FFFFFF", fontsize=6.8),
    ]
    routed_arrow(ax, [(.605, .75), (.615, .75), (.615, .596), (.625, .596)], dashed=True, width=.75)
    routed_arrow(ax, [(.605, .4325), (.615, .4325), (.615, .536), (.625, .536)], dashed=True, width=.75)
    routed_arrow(ax, [(.285, .405), (.31, .405), (.31, .505), (.615, .505), (.615, .476), (.625, .476)],
                 dashed=True, width=.75)
    for input_box in inputs:
        y = input_box.get_y() + input_box.get_height() / 2
        routed_arrow(ax, [(.77, y), (.785, y)], dashed=True, width=.75)

    vertical_arrow(ax, assessment, priority, color=PAPER_COLORS["bronze"])

    # Output band receives explanation, propagation, and operational-priority outputs.
    band = process_box(
        ax, .25, .075, .60, .085, "Historical Replay and Decision Report",
        "#F4F7F9", fontsize=8.8,
    )
    routed_arrow(ax, [(.48, .38), (.48, .16)])
    routed_arrow(ax, [(.875, .28), (.875, .20), (.80, .20), (.80, .16)])
    routed_arrow(ax, [(.965, .7725), (.978, .7725), (.978, .118), (.85, .118)])

    ax.text(.5, .022, "Historical decision support; not live operational commands.",
            ha="center", va="center", fontsize=8, color=PAPER_COLORS["muted_text"])
    save_figure(fig, "aerorecover_system_architecture")
    plt.close(fig)


if __name__ == "__main__":
    main()
