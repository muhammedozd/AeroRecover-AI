"""Generate the two-panel IEEE validation-threshold analysis figure."""
from __future__ import annotations

from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_rotation_model import MODEL_COLUMNS, create_time_masks, prepare_features  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_propagation_2023_time_split.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
SELECTED_THRESHOLD = 0.46
# One empirical evaluation per percentage point; no interpolation or smoothing.
THRESHOLDS = np.linspace(0.01, 0.99, 99)
NAVY, MUTED_BLUE, DARK_TEAL = "#183B5B", "#6688A3", "#287271"
REFERENCE_GRAY, GRID_GRAY, AXIS_GRAY = "#777777", "#D9DDE1", "#555555"


def validation_predictions(pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Score only September-October rows using the canonical project split."""
    probabilities, targets = [], []
    for chunk in pd.read_csv(DATA_PATH, usecols=MODEL_COLUMNS, chunksize=150_000, low_memory=False):
        _, validation_mask, _ = create_time_masks(chunk)
        if not validation_mask.any():
            continue
        features, target, _, _ = prepare_features(chunk)
        probabilities.append(pipeline.predict_proba(features.loc[validation_mask])[:, 1])
        targets.append(target.loc[validation_mask].to_numpy(dtype=np.int8, copy=True))
    if not probabilities:
        raise RuntimeError("No September-October 2023 validation rows were found.")
    y_true, y_probability = np.concatenate(targets), np.concatenate(probabilities)
    print(f"Validation samples scored: {len(y_true):,}")
    return y_true, y_probability


def threshold_metrics(y_true: np.ndarray, y_probability: np.ndarray) -> dict[str, np.ndarray]:
    """Compute empirical classification and operational rates at each threshold."""
    names = ("precision", "recall", "f1", "fpr", "fnr", "alert_rate")
    values = {name: np.empty(THRESHOLDS.size, dtype=float) for name in names}
    positive, negative, n = y_true == 1, y_true != 1, y_true.size
    for index, threshold in enumerate(THRESHOLDS):
        predicted = y_probability >= threshold
        tp = np.count_nonzero(predicted & positive)
        fp = np.count_nonzero(predicted & negative)
        tn = np.count_nonzero(~predicted & negative)
        fn = np.count_nonzero(~predicted & positive)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values["precision"][index], values["recall"][index] = precision, recall
        values["f1"][index] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values["fpr"][index] = fp / (fp + tn) if fp + tn else 0.0
        values["fnr"][index] = fn / (fn + tp) if fn + tp else 0.0
        values["alert_rate"][index] = (tp + fp) / n
    return values


def selected_metric_values(metrics: dict[str, np.ndarray]) -> tuple[int, dict[str, float]]:
    index = int(np.flatnonzero(np.isclose(THRESHOLDS, SELECTED_THRESHOLD))[0])
    selected = {name: float(series[index]) for name, series in metrics.items()}
    print(f"Exact validation metrics at threshold = {SELECTED_THRESHOLD:.2f}")
    for name in ("precision", "recall", "f1", "fpr", "fnr", "alert_rate"):
        print(f"  {name:10s} = {selected[name]:.6f}")
    return index, selected


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=9, pad=5)
    ax.set(xlim=(0.01, 0.99), ylim=(0.0, 1.0), xlabel="Decision threshold")
    ax.set_xticks(np.arange(0.1, 1.0, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(True, color=GRID_GRAY, linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.axvline(SELECTED_THRESHOLD, color=REFERENCE_GRAY, linewidth=0.8,
               linestyle=(0, (4, 3)), zorder=2)
    for spine in ax.spines.values():
        spine.set_color(AXIS_GRAY)
        spine.set_linewidth(0.65)
    ax.tick_params(width=0.6, length=3, color=AXIS_GRAY)


def annotation_box() -> dict[str, object]:
    return dict(boxstyle="round,pad=0.25", facecolor="white",
                edgecolor="#BBBBBB", linewidth=0.5)


def plot_analysis(metrics: dict[str, np.ndarray], selected_index: int,
                  selected: dict[str, float]) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True, facecolor="white")
    classification = (("Precision", "precision", NAVY), ("Recall", "recall", MUTED_BLUE),
                      ("F1", "f1", DARK_TEAL))
    for label, key, color in classification:
        axes[0].plot(THRESHOLDS, metrics[key], color=color, linewidth=1.65, label=label)
        axes[0].plot(SELECTED_THRESHOLD, metrics[key][selected_index], marker="o",
                     markersize=3.5, color=color, linestyle="none", zorder=4)
    style_axis(axes[0], "(a) Validation classification trade-off")
    axes[0].set_ylabel("Rate")
    axes[0].legend(loc="lower left", frameon=False, ncol=3, handlelength=1.5, columnspacing=0.8)
    axes[0].text(0.975, 0.96,
                 "threshold = 0.46\nprecision = 0.791\nrecall = 0.842\nF1 = 0.816",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=6.8,
                 linespacing=1.2, bbox=annotation_box(), zorder=5)

    operational = (("False-positive rate", "fpr", NAVY),
                   ("False-negative rate", "fnr", MUTED_BLUE),
                   ("Alert rate", "alert_rate", DARK_TEAL))
    for label, key, color in operational:
        axes[1].plot(THRESHOLDS, metrics[key], color=color, linewidth=1.65, label=label)
    style_axis(axes[1], "(b) Operational alert trade-off")
    axes[1].legend(loc="upper center", frameon=False, ncol=1, handlelength=1.6)
    axes[1].text(0.975, 0.04,
                 f"threshold = 0.46\nFPR = {selected['fpr']:.3f}\n"
                 f"FNR = {selected['fnr']:.3f}\nalert rate = {selected['alert_rate']:.3f}",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6.8,
                 linespacing=1.2, bbox=annotation_box(), zorder=5)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.9, wspace=0.13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "validation_threshold_analysis.pdf"
    png_path = OUTPUT_DIR / "validation_threshold_analysis.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for path in (pdf_path, png_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure output was not created: {path}")
        print(f"Created {path} ({path.stat().st_size:,} bytes)")


def main() -> None:
    pipeline = joblib.load(MODEL_PATH)
    if not {"preprocessor", "classifier"}.issubset(pipeline.named_steps):
        raise ValueError("Frozen model is not the expected fitted pipeline.")
    y_true, y_probability = validation_predictions(pipeline)
    metrics = threshold_metrics(y_true, y_probability)
    selected_index, selected = selected_metric_values(metrics)
    plot_analysis(metrics, selected_index, selected)


if __name__ == "__main__":
    main()
