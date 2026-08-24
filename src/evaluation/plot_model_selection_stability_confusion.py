"""Build the three-panel validation/locked-test evaluation figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "results" / "baseline_model_comparison.csv"
MONTHLY_PATH = PROJECT_ROOT / "results" / "final_test_monthly_metrics.csv"
FINAL_TEST_PATH = PROJECT_ROOT / "results" / "final_test_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
PDF_PATH = OUTPUT_DIR / "model_selection_stability_confusion.pdf"
PNG_PATH = OUTPUT_DIR / "model_selection_stability_confusion.png"

NAVY = "#17324D"
BLUE = "#3B78A8"
TEAL = "#2A9D8F"
RISK = "#D95D5D"
LIGHT_GRID = "#D9E0E6"
MODEL_ORDER = ["Logistic Regression", "XGBoost", "XGBoost Operational"]
MONTH_ORDER = ["November", "December"]

EXPECTED_BASELINE = {
    "Logistic Regression": {"F1": 0.7846264560182086, "PR_AUC": 0.8347289051464617},
    "XGBoost": {"F1": 0.8134342998391336, "PR_AUC": 0.8797689032263415},
    "XGBoost Operational": {"F1": 0.8156498266924328, "PR_AUC": 0.8797689032263415},
}
EXPECTED_MONTHLY = {
    "November": {"sample_count": 403_594, "precision": 0.7499093582564557,
                 "recall": 0.8358403304746083, "f1": 0.7905465664415849},
    "December": {"sample_count": 401_532, "precision": 0.7506452253325392,
                 "recall": 0.8475361452534838, "f1": 0.7961536436856236},
}
EXPECTED_CONFUSION = {
    "true_negative": 742_344, "false_positive": 13_744,
    "false_negative": 7_737, "true_positive": 41_301,
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    null_counts = frame[list(columns)].isna().sum()
    if null_counts.any():
        raise ValueError(f"{source} contains unexpected missing values:\n{null_counts[null_counts.gt(0)]}")


def verify_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"Unexpected {label}: {actual}; expected {expected}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for path in (BASELINE_PATH, MONTHLY_PATH, FINAL_TEST_PATH):
        require_file(path)
    baseline = pd.read_csv(BASELINE_PATH)
    monthly = pd.read_csv(MONTHLY_PATH)
    with FINAL_TEST_PATH.open(encoding="utf-8") as stream:
        final_test = json.load(stream)

    require_columns(baseline, {"MODEL", "THRESHOLD", "F1", "PR_AUC"}, BASELINE_PATH)
    require_columns(
        monthly, {"month", "sample_count", "threshold", "precision", "recall", "f1"}, MONTHLY_PATH,
    )
    missing_json = sorted(set(EXPECTED_CONFUSION) - set(final_test))
    if missing_json:
        raise ValueError(f"{FINAL_TEST_PATH} is missing required keys: {missing_json}")
    if any(final_test[key] is None for key in EXPECTED_CONFUSION):
        raise ValueError(f"{FINAL_TEST_PATH} contains null confusion counts")

    selected = baseline.loc[baseline["MODEL"].isin(MODEL_ORDER)].set_index("MODEL").reindex(MODEL_ORDER)
    if selected.index.tolist() != MODEL_ORDER or selected[["F1", "PR_AUC"]].isna().any().any():
        raise ValueError("Baseline input does not contain exactly the required model rows.")
    for model, metrics in EXPECTED_BASELINE.items():
        for metric, expected in metrics.items():
            verify_close(float(selected.loc[model, metric]), expected, f"{model} {metric}")
    verify_close(float(selected.loc["XGBoost Operational", "THRESHOLD"]), 0.46,
                 "XGBoost Operational threshold")

    monthly_selected = monthly.set_index("month").reindex(MONTH_ORDER)
    if monthly_selected[["sample_count", "precision", "recall", "f1"]].isna().any().any():
        raise ValueError("Monthly input does not contain both November and December rows.")
    for month, metrics in EXPECTED_MONTHLY.items():
        for metric, expected in metrics.items():
            verify_close(float(monthly_selected.loc[month, metric]), expected, f"{month} {metric}")
        verify_close(float(monthly_selected.loc[month, "threshold"]), 0.46, f"{month} threshold")
    for key, expected in EXPECTED_CONFUSION.items():
        if int(final_test[key]) != expected:
            raise ValueError(f"Unexpected {key}: {final_test[key]}; expected {expected}")
    verify_close(float(final_test.get("threshold", np.nan)), 0.46, "final-test threshold")
    if final_test.get("evaluation_period") != "2023-11-01/2023-12-31":
        raise ValueError("Final-test JSON does not identify the locked November-December period.")
    return selected, monthly_selected, final_test


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9, width=0.7)
    axis.set_axisbelow(True)


def draw_validation_panel(axis: plt.Axes, baseline: pd.DataFrame) -> None:
    y = np.arange(len(MODEL_ORDER))
    height = 0.28
    f1 = baseline["F1"].to_numpy()
    pr_auc = baseline["PR_AUC"].to_numpy()
    bars_f1 = axis.barh(y + height / 2, f1, height, color=BLUE, label="F1")
    bars_pr = axis.barh(y - height / 2, pr_auc, height, color=TEAL, label="PR-AUC")
    for bars in (bars_f1, bars_pr):
        for bar in bars:
            axis.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                      f"{bar.get_width():.3f}", va="center", fontsize=8.5, color=NAVY)
    axis.set_yticks(y, MODEL_ORDER)
    axis.invert_yaxis()
    axis.set_xlim(0.70, 0.90)
    axis.set_xticks(np.arange(0.70, 0.901, 0.05))
    axis.set_xlabel("Score (truncated at 0.70)", fontsize=9.5)
    axis.set_title("(a) Validation model comparison", loc="left", fontsize=11.5, pad=10)
    axis.grid(axis="x", color=LIGHT_GRID, linewidth=0.6)
    axis.legend(loc="lower right", ncol=2, frameon=False, fontsize=9)
    axis.text(0.5, -0.24, r"Operational threshold $\tau$ = 0.46",
              transform=axis.transAxes, ha="center", fontsize=8.5, color="#566573")
    axis.plot([0.0, 0.012], [-0.018, 0.018], transform=axis.transAxes,
              color=NAVY, linewidth=1.0, clip_on=False)
    axis.plot([0.018, 0.030], [-0.018, 0.018], transform=axis.transAxes,
              color=NAVY, linewidth=1.0, clip_on=False)
    style_axis(axis)


def draw_monthly_panel(axis: plt.Axes, monthly: pd.DataFrame) -> None:
    metrics = [("precision", "Precision", NAVY), ("recall", "Recall", TEAL), ("f1", "F1", BLUE)]
    x = np.arange(len(MONTH_ORDER))
    width = 0.22
    for index, (column, label, color) in enumerate(metrics):
        positions = x + (index - 1) * width
        values = monthly[column].to_numpy()
        bars = axis.bar(positions, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}",
                      ha="center", va="bottom", fontsize=8.2, rotation=0)
    month_labels = [
        f"November\n(n={int(monthly.loc['November', 'sample_count']):,})",
        f"December\n(n={int(monthly.loc['December', 'sample_count']):,})",
    ]
    axis.set_xticks(x, month_labels)
    axis.set_ylim(0.70, 0.90)
    axis.set_yticks(np.arange(0.70, 0.901, 0.05))
    axis.set_ylabel("Score (truncated at 0.70)", fontsize=9.5)
    axis.set_title("(b) Locked-test monthly stability", loc="left", fontsize=11.5, pad=10)
    axis.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)
    axis.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
    axis.text(0.5, -0.24, r"Frozen threshold $\tau$ = 0.46", transform=axis.transAxes,
              ha="center", fontsize=8.5, color=NAVY)
    axis.plot([-0.018, 0.018], [0.0, 0.012], transform=axis.transAxes,
              color=NAVY, linewidth=1.0, clip_on=False)
    axis.plot([-0.018, 0.018], [0.020, 0.032], transform=axis.transAxes,
              color=NAVY, linewidth=1.0, clip_on=False)
    style_axis(axis)


def draw_confusion_panel(axis: plt.Axes, final_test: dict) -> None:
    matrix = np.array([
        [final_test["true_negative"], final_test["false_positive"]],
        [final_test["false_negative"], final_test["true_positive"]],
    ], dtype=int)
    cmap = LinearSegmentedColormap.from_list("academic_blues", ["#EEF5F8", BLUE, NAVY])
    norm = LogNorm(vmin=matrix.min(), vmax=matrix.max())
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(1.5, -0.5)
    for row in range(2):
        for column in range(2):
            axis.add_patch(Rectangle(
                (column - 0.5, row - 0.5), 1, 1,
                facecolor=cmap(norm(matrix[row, column])), edgecolor="none",
            ))
    abbreviations = np.array([["TN", "FP"], ["FN", "TP"]])
    threshold = np.sqrt(matrix.min() * matrix.max())
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > threshold else NAVY
            axis.text(column, row, f"{abbreviations[row, column]}\n{matrix[row, column]:,}",
                      ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    axis.set_xticks([0, 1], ["Predicted\nNegative", "Predicted\nPositive"])
    axis.set_yticks([0, 1], ["Actual Negative", "Actual Positive"])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", top=False, labeltop=True, bottom=False, labelbottom=False, pad=5)
    axis.set_title("(c) Locked-test confusion matrix", loc="left", fontsize=11.5, pad=10)
    axis.text(0.5, -0.12, "Log-scaled color intensity", transform=axis.transAxes,
              ha="center", fontsize=8.5, color="#566573")
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(NAVY)


def main() -> None:
    baseline, monthly, final_test = load_inputs()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
        "xtick.color": NAVY, "ytick.color": NAVY,
        "text.color": NAVY, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.5, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.08, 0.84], wspace=0.52)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    draw_validation_panel(axes[0], baseline)
    draw_monthly_panel(axes[1], monthly)
    draw_confusion_panel(axes[2], final_test)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.25)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(PNG_PATH, format="png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {PDF_PATH}")
    print(f"Saved: {PNG_PATH}")


if __name__ == "__main__":
    main()
