"""Generate ROC, precision-recall, and calibration plots for the locked test set."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.models.train_rotation_model import (
    MODEL_COLUMNS,
    create_time_masks,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset_2023.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_propagation_2023_time_split.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "paper_figures"
)


def load_final_test_data():
    """Load the locked November-December 2023 test data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Rotation dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        usecols=MODEL_COLUMNS,
        low_memory=False,
    )

    (
        features,
        target,
        _,
        _,
    ) = prepare_features(data)

    (
        _,
        _,
        test_mask,
    ) = create_time_masks(data)

    test_features = features.loc[
        test_mask
    ].copy()

    test_target = target.loc[
        test_mask
    ].copy()

    print(
        "Locked test samples:",
        f"{len(test_features):,}",
    )

    print(
        "Actual propagation rate:",
        f"{test_target.mean():.4%}",
    )

    return (
        test_features,
        test_target,
    )


def load_model():
    """Load the frozen time-split XGBoost model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded:",
        MODEL_PATH,
    )

    return model


def calculate_calibration_bins(
    probabilities,
    actual_values,
    bin_count: int = 10,
) -> pd.DataFrame:
    """
    Divide predictions into equal-frequency groups.

    Each bin contains approximately the same number
    of test samples.
    """

    calibration_data = pd.DataFrame(
        {
            "PREDICTED_PROBABILITY": probabilities,
            "ACTUAL_PROPAGATION": actual_values,
        }
    )

    calibration_data["PROBABILITY_BIN"] = pd.qcut(
        calibration_data[
            "PREDICTED_PROBABILITY"
        ],
        q=bin_count,
        duplicates="drop",
    )

    calibration_summary = (
        calibration_data
        .groupby(
            "PROBABILITY_BIN",
            observed=True,
        )
        .agg(
            MEAN_PREDICTED_PROBABILITY=(
                "PREDICTED_PROBABILITY",
                "mean",
            ),
            OBSERVED_PROPAGATION_RATE=(
                "ACTUAL_PROPAGATION",
                "mean",
            ),
            SAMPLE_COUNT=(
                "ACTUAL_PROPAGATION",
                "size",
            ),
        )
        .reset_index()
    )

    return calibration_summary


def build_evaluation_figures(
    test_target,
    test_probabilities,
    calibration_summary: pd.DataFrame,
):
    """Build three separate IEEE single-column evaluation figures."""

    false_positive_rate, true_positive_rate, _ = roc_curve(
        test_target,
        test_probabilities,
    )

    precision_values, recall_values, _ = precision_recall_curve(
        test_target,
        test_probabilities,
    )

    roc_auc = roc_auc_score(
        test_target,
        test_probabilities,
    )

    average_precision = average_precision_score(
        test_target,
        test_probabilities,
    )

    baseline_rate = float(
        test_target.mean()
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    figures = {}

    # ROC curve
    roc_figure, roc_axis = plt.subplots(
        figsize=(3.5, 2.8)
    )

    roc_axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#2563EB",
        linewidth=1.8,
        label=f"XGBoost (AUC = {roc_auc:.4f})",
    )

    roc_axis.plot(
        [0, 1],
        [0, 1],
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.1,
        label="Random classifier",
    )

    roc_axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Receiver Operating Characteristic",
    )

    roc_axis.legend(
        loc="lower right",
        frameon=False,
    )

    roc_axis.grid(alpha=0.25)
    roc_figure.tight_layout(pad=0.7)

    figures["roc"] = roc_figure

    # Precision-recall curve
    pr_figure, pr_axis = plt.subplots(
        figsize=(3.5, 2.8)
    )

    pr_axis.plot(
        recall_values,
        precision_values,
        color="#D97706",
        linewidth=1.8,
        label=f"XGBoost (AP = {average_precision:.4f})",
    )

    pr_axis.axhline(
        baseline_rate,
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.1,
        label=f"Test prevalence ({baseline_rate:.3f})",
    )

    pr_axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curve",
    )

    pr_axis.legend(
        loc="lower left",
        frameon=False,
    )

    pr_axis.grid(alpha=0.25)
    pr_figure.tight_layout(pad=0.7)

    figures["precision_recall"] = pr_figure

    # Quantile calibration
    calibration_figure, calibration_axis = plt.subplots(
        figsize=(3.5, 2.8)
    )

    calibration_axis.plot(
        [0, 1],
        [0, 1],
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.1,
        label="Perfect calibration",
    )

    calibration_axis.plot(
        calibration_summary[
            "MEAN_PREDICTED_PROBABILITY"
        ],
        calibration_summary[
            "OBSERVED_PROPAGATION_RATE"
        ],
        color="#059669",
        marker="o",
        markersize=4,
        linewidth=1.7,
        label="XGBoost",
    )

    calibration_axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed propagation rate",
        title="Quantile Calibration",
    )

    calibration_axis.legend(
        loc="upper left",
        frameon=False,
    )

    calibration_axis.grid(alpha=0.25)
    calibration_figure.tight_layout(pad=0.7)

    figures["calibration"] = calibration_figure

    return (
        figures,
        roc_auc,
        average_precision,
    )

def save_individual_figures(
    test_target,
    test_probabilities,
    calibration_summary: pd.DataFrame,
) -> None:
    """Save each evaluation chart as a separate publication figure."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        test_target,
        test_probabilities,
    )

    precision_values, recall_values, _ = precision_recall_curve(
        test_target,
        test_probabilities,
    )

    roc_auc = roc_auc_score(
        test_target,
        test_probabilities,
    )

    average_precision = average_precision_score(
        test_target,
        test_probabilities,
    )

    baseline_rate = float(
        test_target.mean()
    )

    # ROC
    figure, axis = plt.subplots(
        figsize=(3.5, 3.0)
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#2563EB",
        linewidth=2.2,
        label=f"XGBoost (AUC = {roc_auc:.4f})",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label="Random classifier",
    )

    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )

    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=7,
    )

    axis.grid(alpha=0.25)
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "final_test_roc_curve.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIR / "final_test_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Precision-recall
    figure, axis = plt.subplots(
        figsize=(3.5, 3.0)
    )

    axis.plot(
        recall_values,
        precision_values,
        color="#D97706",
        linewidth=2.2,
        label=f"XGBoost (AP = {average_precision:.4f})",
    )

    axis.axhline(
        baseline_rate,
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label=f"Test prevalence ({baseline_rate:.3f})",
    )

    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precision",
    )

    axis.legend(
        loc="lower left",
        frameon=False,
        fontsize=7,
    )

    axis.grid(alpha=0.25)
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "final_test_precision_recall_curve.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIR / "final_test_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Calibration
    figure, axis = plt.subplots(
        figsize=(3.5, 3.0)
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label="Perfect calibration",
    )

    axis.plot(
        calibration_summary[
            "MEAN_PREDICTED_PROBABILITY"
        ],
        calibration_summary[
            "OBSERVED_PROPAGATION_RATE"
        ],
        color="#059669",
        marker="o",
        markersize=5,
        linewidth=2,
        label="XGBoost",
    )

    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed propagation rate",
    )

    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=7,
    )

    axis.grid(alpha=0.25)
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "final_test_calibration_curve.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIR / "final_test_calibration_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nIndividual paper figures saved:")
    print(OUTPUT_DIR / "final_test_roc_curve.pdf")
    print(OUTPUT_DIR / "final_test_precision_recall_curve.pdf")
    print(OUTPUT_DIR / "final_test_calibration_curve.pdf")


def save_outputs(
    figures: dict,
    calibration_summary: pd.DataFrame,
) -> None:
    """Save each plot as a separate vector PDF and PNG."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_names = {
        "roc": "final_test_roc_curve",
        "precision_recall":
            "final_test_precision_recall_curve",
        "calibration":
            "final_test_calibration_curve",
    }

    for figure_key, output_name in output_names.items():
        figure = figures[figure_key]

        pdf_path = OUTPUT_DIR / f"{output_name}.pdf"
        png_path = OUTPUT_DIR / f"{output_name}.png"

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        figure.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(figure)

        print("Saved:", pdf_path)
        print("Saved:", png_path)

    calibration_path = (
        OUTPUT_DIR
        / "final_test_calibration_bins.csv"
    )

    calibration_summary.to_csv(
        calibration_path,
        index=False,
    )

    print("Saved:", calibration_path)


def main():
    test_features, test_target = (
        load_final_test_data()
    )

    model = load_model()

    test_probabilities = (
        model.predict_proba(
            test_features
        )[:, 1]
    )

    calibration_summary = (
        calculate_calibration_bins(
            probabilities=test_probabilities,
            actual_values=test_target.to_numpy(),
            bin_count=10,
        )
    )

    print("\nCalibration bins")
    print("-" * 90)

    print(
        calibration_summary.to_string(
            index=False
        )
    )

    (
        figures,
        roc_auc,
        average_precision,
    ) = build_evaluation_figures(
        test_target=test_target,
        test_probabilities=test_probabilities,
        calibration_summary=calibration_summary,
    )

    print("\nFinal-test curve metrics")
    print("-" * 60)

    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.6f}"
    )

    save_outputs(
        figures=figures,
        calibration_summary=calibration_summary,
    )

    save_individual_figures(
        test_target=test_target,
        test_probabilities=test_probabilities,
        calibration_summary=calibration_summary,
    )


if __name__ == "__main__":
    main()