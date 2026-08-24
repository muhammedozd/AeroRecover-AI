"""Build IEEE paper SHAP panels from the frozen validation-period model."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.rotation_features import build_rotation_model_features  # noqa: E402
from src.models.rotation_model_contract import (  # noqa: E402
    FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, load_model_pipeline,
)
from src.visualization.paper_figures.paper_style import (  # noqa: E402
    PAPER_COLORS,
    apply_paper_style,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv"
VALIDATION_START = pd.Timestamp("2023-09-01")
VALIDATION_END = pd.Timestamp("2023-11-01")
MAX_SAMPLES = 5_000
RANDOM_STATE = 42
SHAP_RECONSTRUCTION_TOLERANCE = 1e-5

DISPLAY_NAMES = {
    "PREV_DEST": "Previous destination",
    "PREV_DELAY_LEVEL": "Previous delay level",
    "ROTATION_POSITION": "Rotation position",
    "PREV_ARR_DELAY": "Previous arrival delay",
    "PREV_ARR_MIN": "Previous arrival time",
    "PREV_CRS_ARR_MIN": "Previous scheduled arrival time",
    "PLANNED_TURNAROUND": "Planned turnaround",
    "TURN_BUFFER": "Turnaround buffer",
    "PREV_DELAY_RATIO": "Previous-delay ratio",
    "HAS_BUFFER": "Turnaround buffer available",
    "IS_SHORT_TURN": "Short turnaround",
    "PREV_DELAYED": "Previous flight delayed",
}

DEPENDENCE_SPECS = (
    ("PREV_DELAY_RATIO", "Previous-delay ratio", "(a)"),
    ("TURN_BUFFER", "Turnaround buffer (min)", "(b)"),
    ("PREV_ARR_DELAY", "Previous arrival delay (min)", "(c)"),
    ("PLANNED_TURNAROUND", "Planned turnaround (min)", "(d)"),
)


def load_validation_sample() -> pd.DataFrame:
    """Select one reproducible sample exclusively from September-October 2023."""
    validation_chunks: list[pd.DataFrame] = []
    usecols = RAW_FEATURE_COLUMNS
    for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=150_000, low_memory=False):
        features, dates = build_rotation_model_features(chunk)
        validation_mask = dates.ge(VALIDATION_START) & dates.lt(VALIDATION_END)
        if validation_mask.any():
            validation_chunks.append(features.loc[validation_mask].copy())

    if not validation_chunks:
        raise RuntimeError("No September-October 2023 validation rows were found.")
    validation = pd.concat(validation_chunks, ignore_index=True)
    sample = validation.sample(
        n=min(MAX_SAMPLES, len(validation)), random_state=RANDOM_STATE, replace=False
    ).reset_index(drop=True)
    print(f"Validation population (2023-09-01 to 2023-10-31): {len(validation):,}")
    print(f"Shared SHAP sample for all panels: {len(sample):,} (random_state={RANDOM_STATE})")
    return sample


def transformed_source_features(preprocessor) -> list[str]:
    """Map every fitted transformed column to its original input feature."""
    mapping: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        output_slice = preprocessor.output_indices_[name]
        expected_count = output_slice.stop - output_slice.start
        one_hot = getattr(transformer, "named_steps", {}).get("one_hot")
        if one_hot is not None and hasattr(one_hot, "categories_"):
            local = [str(column) for column, categories in zip(columns, one_hot.categories_)
                     for _ in categories]
        elif expected_count == len(columns):
            local = [str(column) for column in columns]
        else:
            raise ValueError(f"Unsupported transformed feature mapping for {name}.")
        if len(local) != expected_count:
            raise ValueError(f"Unexpected transformed column count for {name}.")
        mapping.extend(local)
    return mapping


def compute_and_validate_shap(sample: pd.DataFrame, pipeline):
    """Compute TreeSHAP and verify that contributions reconstruct raw margins."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(sample)
    source_features = transformed_source_features(preprocessor)
    if transformed.shape[1] != len(source_features):
        raise ValueError("Transformed data and source-feature mapping do not align.")

    booster = classifier.get_booster()
    dmatrix = xgb.DMatrix(transformed)
    contributions = booster.predict(dmatrix, pred_contribs=True)
    shap_values = contributions[:, :-1]
    base_values = contributions[:, -1]
    raw_margins = booster.predict(dmatrix, output_margin=True)
    probabilities = pipeline.predict_proba(sample)[:, 1]
    reconstructed_margins = base_values + shap_values.sum(axis=1)
    reconstructed_probabilities = 1.0 / (1.0 + np.exp(-raw_margins))

    margin_error = float(np.max(np.abs(reconstructed_margins - raw_margins)))
    probability_error = float(np.max(np.abs(reconstructed_probabilities - probabilities)))
    if margin_error >= SHAP_RECONSTRUCTION_TOLERANCE:
        raise ValueError(f"SHAP raw-margin reconstruction failed: max error={margin_error:.8g}.")
    if probability_error >= SHAP_RECONSTRUCTION_TOLERANCE:
        raise ValueError(f"Raw-margin probability check failed: max error={probability_error:.8g}.")

    print("SHAP output scale verified: raw margin (log-odds), not probability contribution.")
    print(f"Maximum |base + sum(SHAP) - raw margin|: {margin_error:.8g}")
    print(f"Maximum |sigmoid(raw margin) - predict_proba|: {probability_error:.8g}")
    return shap_values, source_features


def grouped_importance(shap_values: np.ndarray, source_features: list[str]) -> pd.Series:
    """Aggregate transformed-column mean absolute SHAP values by source feature."""
    # Group importance is the sum of transformed-column mean(abs(SHAP)) values,
    # so one-hot levels return to the original categorical feature as requested.
    transformed_importance = pd.Series(np.abs(shap_values).mean(axis=0))
    grouped = transformed_importance.groupby(pd.Series(source_features), sort=False).sum()
    return grouped.sort_values(ascending=False)


def style_axes(ax, *, grid_axis: str) -> None:
    ax.set_facecolor(PAPER_COLORS["background"])
    ax.grid(axis=grid_axis, color=PAPER_COLORS["light_border"], linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PAPER_COLORS["border"])
        ax.spines[side].set_linewidth(0.65)
    ax.tick_params(axis="both", labelsize=7.5, length=2.3, width=0.55)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.02, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=PAPER_COLORS["text"], rasterized=False)


def draw_importance(ax, grouped: pd.Series, *, panel_label: str | None = None) -> None:
    top_descending = grouped.head(8)
    top = top_descending.iloc[::-1]
    top_three = set(top_descending.head(3).index)
    colors = [PAPER_COLORS["navy"] if feature in top_three else "#7893AA"
              for feature in top.index]
    labels = [DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())
              for feature in top.index]
    positions = np.arange(len(top))
    bars = ax.barh(positions, top.to_numpy(), height=0.62, color=colors,
                   edgecolor="white", linewidth=0.35, rasterized=False, zorder=3)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Mean absolute SHAP value", fontsize=8)
    value_offset = float(top.max()) * 0.018
    for bar, value in zip(bars, top.to_numpy()):
        ax.text(float(value) + value_offset, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", ha="left", va="center", fontsize=7,
                color=PAPER_COLORS["text"], rasterized=False)
    ax.set_xlim(0, float(top.max()) * 1.17)
    ax.margins(y=0.04)
    style_axes(ax, grid_axis="x")
    if panel_label:
        add_panel_label(ax, panel_label)


def feature_shap_values(feature: str, shap_values: np.ndarray,
                        source_features: list[str]) -> np.ndarray:
    indices = [index for index, source in enumerate(source_features) if source == feature]
    if len(indices) != 1:
        raise ValueError(f"Expected one transformed {feature} column, found {len(indices)}.")
    return shap_values[:, indices[0]]


def draw_dependence(ax, sample: pd.DataFrame, shap_values: np.ndarray,
                    source_features: list[str], feature: str, x_label: str,
                    *, panel_label: str | None = None,
                    show_y_label: bool = True) -> int:
    x = pd.to_numeric(sample[feature], errors="raise").to_numpy()
    y = feature_shap_values(feature, shap_values, source_features)
    lower, upper = np.nanpercentile(x, [0.5, 99.5])
    outside = int(np.count_nonzero((x < lower) | (x > upper)))
    ax.scatter(x, y, s=6, color=PAPER_COLORS["steel_blue"], alpha=0.22,
               edgecolors="none", rasterized=False, zorder=2)
    ax.axhline(0, color=PAPER_COLORS["muted_text"], linewidth=0.65,
               linestyle=(0, (3, 2)), rasterized=False, zorder=1)
    ax.set_xlim(float(lower), float(upper))
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel("SHAP contribution to model output" if show_y_label else "", fontsize=8)
    style_axes(ax, grid_axis="y")
    if panel_label:
        add_panel_label(ax, panel_label)
    print(f"{DISPLAY_NAMES[feature]} visual x limits (0.5th, 99.5th percentiles): "
          f"({lower:.6g}, {upper:.6g}); observations outside: {outside:,}")
    return outside


def save_exact_figure(fig, stem: str) -> None:
    """Save without a tight bbox so the requested physical canvas is preserved."""
    output_dir = PROJECT_ROOT / "results" / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    for path in (pdf, png):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure output was not created: {path}")
        print(f"Created {path} ({path.stat().st_size:,} bytes)")


def save_importance_figure(grouped: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.7), facecolor="white")
    draw_importance(ax, grouped)
    fig.subplots_adjust(left=0.43, right=0.985, bottom=0.18, top=0.98)
    save_exact_figure(fig, "shap_grouped_importance_compact")
    plt.close(fig)


def save_operational_dependence_panels(sample: pd.DataFrame, shap_values: np.ndarray,
                                       source_features: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), constrained_layout=True,
                             facecolor="white")
    for index, (ax, (feature, x_label, panel_label)) in enumerate(
        zip(axes.flat, DEPENDENCE_SPECS)
    ):
        draw_dependence(ax, sample, shap_values, source_features, feature, x_label,
                        panel_label=panel_label, show_y_label=index in (0, 2))
    save_exact_figure(fig, "shap_operational_dependence_panels")
    plt.close(fig)


def main() -> None:
    apply_paper_style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8})
    pipeline = load_model_pipeline()
    if not {"preprocessor", "classifier"}.issubset(pipeline.named_steps):
        raise ValueError("Frozen model is not the expected fitted preprocessing pipeline.")
    sample = load_validation_sample()
    shap_values, source_features = compute_and_validate_shap(sample, pipeline)
    grouped = grouped_importance(shap_values, source_features)
    print("Grouped importance: sum of transformed-column mean(abs(SHAP)) by source feature.")
    for feature, value in grouped.items():
        print(f"  {DISPLAY_NAMES.get(feature, feature)}: {value:.8f}")
    save_importance_figure(grouped)
    save_operational_dependence_panels(sample, shap_values, source_features)


if __name__ == "__main__":
    main()
