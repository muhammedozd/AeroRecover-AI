"""Generate publication figures for the frozen full-enhanced rotation model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.explainability.local_shap import _original_feature_mapping
from src.features.rotation_features import build_rotation_model_features
from src.models.rotation_model_contract import (
    FEATURE_COLUMNS, MODEL_PATH, MODEL_THRESHOLD, MODEL_VERSION, PROJECT_ROOT,
    RAW_FEATURE_COLUMNS, load_model_pipeline,
)
from src.visualization.paper_figures.paper_style import (
    OUTPUT_DIR, PAPER_COLORS, apply_paper_style, save_figure,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv"
TARGET = "IS_DELAY_PROPAGATED"
REPORTING_CACHE = OUTPUT_DIR / "full_enhanced_reporting_predictions.parquet"
CALIBRATION_BINS_PATH = OUTPUT_DIR / "final_test_calibration_bins_enhanced.csv"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "paper_figure_manifest.csv"
UPDATE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "paper_update_manifest.md"
PAPER_METRICS_PATH = PROJECT_ROOT / "reports" / "paper_final_enhanced_metrics.csv"
PAPER_GRAPH_PATH = PROJECT_ROOT / "reports" / "paper_graph_metrics_enhanced.csv"
PAPER_PRIORITY_PATH = PROJECT_ROOT / "reports" / "paper_priority_metrics_enhanced.csv"
RANDOM_SEED = 42
SHAP_SAMPLE_SIZE = 5_000
COLORS = [
    PAPER_COLORS["navy"], PAPER_COLORS["steel_blue"], PAPER_COLORS["teal"],
    PAPER_COLORS["muted_text"], "#9FB6C8",
]
AUTHORITATIVE_TEST = {
    "sample_count": 805126, "roc_auc": 0.9913577259, "pr_auc": 0.9029542982,
    "precision": 0.7750564209, "recall": 0.8543986296, "f1": 0.8127958408,
    "tn": 743928, "fp": 12160, "fn": 7140, "tp": 41898,
}
GRAPH_METRICS = {
    "Physical edges": 832421, "Eligible edges": 808124, "Scored edges": 805126,
    "Eligible-edge score coverage": 0.9963, "Predicted multi-hop starts": 12430,
    "Actual multi-hop starts": 11216, "TP starts": 8381, "FP starts": 4049,
    "FN starts": 2835, "Chain-start precision": 0.6743,
    "Chain-start recall": 0.7472, "Chain-start F1": 0.7089,
    "Exact matched chain-length rate": 0.8896, "Chain-length MAE": 0.1230,
}


def verify_reporting_metrics(rows: pd.DataFrame) -> None:
    test = rows.loc[rows["SPLIT"].eq("locked_test")]
    y = test[TARGET].to_numpy()
    probability = test["PREDICTED_PROBABILITY"].to_numpy()
    prediction = probability >= MODEL_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    actual = {
        "sample_count": len(test), "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "precision": precision_score(y, prediction), "recall": recall_score(y, prediction),
        "f1": f1_score(y, prediction), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    for name, expected in AUTHORITATIVE_TEST.items():
        value = actual[name]
        tolerance = 5e-7 if isinstance(expected, float) else 0
        if abs(value - expected) > tolerance:
            raise ValueError(f"Authoritative locked-test mismatch for {name}: {value} != {expected}")


def load_or_build_reporting_rows() -> pd.DataFrame:
    if REPORTING_CACHE.exists():
        rows = pd.read_parquet(REPORTING_CACHE)
        verify_reporting_metrics(rows)
        return rows
    required = list(dict.fromkeys([*RAW_FEATURE_COLUMNS, TARGET]))
    raw = pd.read_csv(DATA_PATH, usecols=required, low_memory=False)
    features, dates = build_rotation_model_features(raw)
    model = load_model_pipeline()
    parts = []
    for split, mask in {
        "validation": dates.ge("2023-09-01") & dates.lt("2023-11-01"),
        "locked_test": dates.ge("2023-11-01") & dates.lt("2024-01-01"),
    }.items():
        probability = model.predict_proba(features.loc[mask])[:, 1]
        part = raw.loc[mask, [TARGET, "PREV_DELAY_RATIO", "TURN_BUFFER",
                              "PREV_ARR_DELAY", "PLANNED_TURNAROUND"]].copy()
        part["PREDICTED_PROBABILITY"] = probability
        part["SPLIT"] = split
        parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    rows["MODEL_VERSION"] = MODEL_VERSION
    rows["MODEL_THRESHOLD"] = MODEL_THRESHOLD
    verify_reporting_metrics(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(REPORTING_CACHE, index=False, compression="snappy")
    return rows


def save_pair(fig, stem):
    pdf, png = save_figure(fig, stem)
    plt.close(fig)
    return [pdf, png]


def validation_threshold_figure(rows):
    validation = rows.loc[rows["SPLIT"].eq("validation")]
    y = validation[TARGET].to_numpy()
    probability = validation["PREDICTED_PROBABILITY"].to_numpy()
    thresholds = np.round(np.arange(.05, .951, .01), 2)
    values = []
    for threshold in thresholds:
        pred = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        values.append({
            "threshold": threshold, "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "fpr": fp / (fp + tn),
            "fnr": fn / (fn + tp), "alert_rate": pred.mean(),
        })
    metrics = pd.DataFrame(values)
    selected = metrics.loc[metrics["threshold"].eq(MODEL_THRESHOLD)].iloc[0]
    if abs(selected["f1"] - 0.833925) > 5e-6:
        raise ValueError("Validation F1 does not match the authoritative enhanced result.")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for key, color in zip(["precision", "recall", "f1"], COLORS):
        axes[0].plot(metrics.threshold, metrics[key], label=key.replace("f1", "F1").title(), color=color)
    for key, color in zip(["fpr", "fnr", "alert_rate"], COLORS):
        axes[1].plot(metrics.threshold, metrics[key], label=key.replace("_", " ").upper(), color=color)
    for axis, label in zip(axes, ["Decision metric", "Rate"]):
        axis.axvline(MODEL_THRESHOLD, color=PAPER_COLORS["bronze"], linestyle="--", linewidth=1.1,
                     label=r"Selected $\tau=0.47$")
        axis.set(xlabel="Decision threshold", ylabel=label, xlim=(.05, .95), ylim=(0, 1))
        axis.grid(color=PAPER_COLORS["light_border"], linewidth=.5)
        axis.legend(frameon=False, fontsize=7)
    axes[0].text(.02, .96, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].text(.02, .96, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")
    fig.tight_layout(pad=.7)
    return save_pair(fig, "validation_threshold_analysis_enhanced")


def calibration_summary(y, probability):
    data = pd.DataFrame({"probability": probability, "actual": y})
    data["bin"] = pd.qcut(data["probability"], 10, duplicates="drop")
    result = data.groupby("bin", observed=True).agg(
        mean_predicted_probability=("probability", "mean"),
        observed_propagation_rate=("actual", "mean"), sample_count=("actual", "size"),
    ).reset_index(drop=True)
    result.to_csv(CALIBRATION_BINS_PATH, index=False)
    return result


def draw_roc(ax, y, probability):
    fpr, tpr, _ = roc_curve(y, probability)
    auc = roc_auc_score(y, probability)
    ax.plot(fpr, tpr, color=COLORS[0], linewidth=1.7, label=f"Full enhanced (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=COLORS[3], linewidth=.9, label="Random")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def draw_pr(ax, y, probability):
    precision, recall, _ = precision_recall_curve(y, probability)
    ap = average_precision_score(y, probability)
    ax.plot(recall, precision, color=COLORS[2], linewidth=1.7, label=f"Full enhanced (AP={ap:.4f})")
    ax.axhline(np.mean(y), linestyle="--", color=COLORS[3], linewidth=.9,
               label=f"Prevalence={np.mean(y):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7, loc="lower left")


def draw_calibration(ax, bins):
    ax.plot([0, 1], [0, 1], "--", color=COLORS[3], linewidth=.9, label="Perfect calibration")
    ax.plot(bins.mean_predicted_probability, bins.observed_propagation_rate,
            marker="o", color=COLORS[1], linewidth=1.5, markersize=3.5, label="10 quantile bins")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed propagation rate",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def locked_test_figures(rows):
    test = rows.loc[rows["SPLIT"].eq("locked_test")]
    y, probability = test[TARGET].to_numpy(), test.PREDICTED_PROBABILITY.to_numpy()
    bins = calibration_summary(y, probability)
    outputs = []
    for stem, drawer in [
        ("final_test_roc_curve_enhanced", lambda ax: draw_roc(ax, y, probability)),
        ("final_test_precision_recall_curve_enhanced", lambda ax: draw_pr(ax, y, probability)),
        ("final_test_calibration_curve_enhanced", lambda ax: draw_calibration(ax, bins)),
    ]:
        fig, ax = plt.subplots(figsize=(3.45, 2.75))
        drawer(ax); ax.grid(color=PAPER_COLORS["light_border"], linewidth=.5)
        fig.tight_layout(pad=.6); outputs.extend(save_pair(fig, stem))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    draw_roc(axes[0], y, probability); draw_pr(axes[1], y, probability); draw_calibration(axes[2], bins)
    for index, axis in enumerate(axes):
        axis.text(.02, .96, f"({chr(97 + index)})", transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(color=PAPER_COLORS["light_border"], linewidth=.45)
    fig.tight_layout(pad=.55); outputs.extend(save_pair(fig, "final_test_evaluation_enhanced"))
    return outputs


def shap_figures():
    raw = pd.read_csv(DATA_PATH, usecols=RAW_FEATURE_COLUMNS, low_memory=False)
    features, dates = build_rotation_model_features(raw)
    validation = features.loc[dates.ge("2023-09-01") & dates.lt("2023-11-01")]
    sample = validation.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
    model = load_model_pipeline(); preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(sample)
    mapping = _original_feature_mapping(preprocessor)
    booster = model.named_steps["classifier"].get_booster()
    contributions = booster.predict(xgb.DMatrix(transformed), pred_contribs=True)[:, :-1]
    raw_margin = booster.predict(xgb.DMatrix(transformed), output_margin=True)
    base = booster.predict(xgb.DMatrix(transformed), pred_contribs=True)[:, -1]
    if np.max(np.abs(base + contributions.sum(axis=1) - raw_margin)) >= 1e-5:
        raise ValueError("SHAP raw-score reconstruction failed.")
    importance = pd.Series(np.abs(contributions).mean(axis=0)).groupby(mapping, sort=False).sum().sort_values()
    fig, ax = plt.subplots(figsize=(3.45, 4.3))
    ax.barh(importance.index, importance.values, color=COLORS[0])
    ax.set_xlabel("Mean absolute SHAP value (raw score)")
    ax.grid(axis="x", color=PAPER_COLORS["light_border"], linewidth=.45)
    fig.tight_layout(pad=.55)
    outputs = save_pair(fig, "shap_grouped_importance_enhanced")
    grouped = pd.DataFrame(contributions).T.groupby(mapping, sort=False).sum().T
    specs = [
        ("PREV_DELAY_RATIO", "Previous-delay ratio"), ("TURN_BUFFER", "Turn buffer (min)"),
        ("PREV_ARR_DELAY", "Previous arrival delay (min)"),
        ("PLANNED_TURNAROUND", "Planned turnaround (min)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    rng = np.random.default_rng(RANDOM_SEED)
    draw = rng.choice(len(sample), size=min(2500, len(sample)), replace=False)
    for index, (axis, (feature, label)) in enumerate(zip(axes.flat, specs)):
        x = pd.to_numeric(sample[feature], errors="coerce").to_numpy()[draw]
        y = grouped[feature].to_numpy()[draw]
        axis.scatter(x, y, s=5, alpha=.25, color=COLORS[index % 3], edgecolors="none")
        axis.axhline(0, color=COLORS[3], linewidth=.7)
        axis.set(xlabel=label, ylabel="SHAP value (raw score)")
        axis.text(.02, .96, f"({chr(97 + index)})", transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(color=PAPER_COLORS["light_border"], linewidth=.4)
    fig.tight_layout(pad=.65)
    outputs.extend(save_pair(fig, "shap_operational_dependence_enhanced"))
    return outputs


def error_profile_figure(rows):
    test = rows.loc[rows["SPLIT"].eq("locked_test")].copy()
    pred = test.PREDICTED_PROBABILITY.ge(MODEL_THRESHOLD)
    y = test[TARGET].eq(1)
    test["ERROR_TYPE"] = np.select([y & pred, ~y & pred, y & ~pred], ["TP", "FP", "FN"], default="TN")
    groups = ["TP", "FP", "FN"]
    features = [
        ("PREV_DELAY_RATIO", "Previous-delay ratio"), ("TURN_BUFFER", "Turn buffer (min)"),
        ("PREV_ARR_DELAY", "Previous arrival delay (min)"),
        ("PLANNED_TURNAROUND", "Planned turnaround (min)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6))
    for index, (axis, (feature, label)) in enumerate(zip(axes.flat, features)):
        values = [test.loc[test.ERROR_TYPE.eq(group), feature].dropna() for group in groups]
        axis.boxplot(values, tick_labels=groups, showfliers=False, patch_artist=True,
                     boxprops={"facecolor": COLORS[4], "edgecolor": COLORS[0]},
                     medianprops={"color": COLORS[2], "linewidth": 1.3})
        axis.set_ylabel(label); axis.grid(axis="y", color=PAPER_COLORS["light_border"], linewidth=.45)
        axis.text(.02, .96, f"({chr(97 + index)})", transform=axis.transAxes, va="top", fontweight="bold")
    fig.tight_layout(pad=.65)
    return save_pair(fig, "locked_test_error_profiles_enhanced")


def graph_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    labels = ["Physical", "Eligible", "Scored"]
    counts = [GRAPH_METRICS["Physical edges"], GRAPH_METRICS["Eligible edges"], GRAPH_METRICS["Scored edges"]]
    bars = axes[0].bar(labels, np.array(counts) / 1e3, color=COLORS[:3])
    axes[0].bar_label(bars, labels=[f"{value:,}" for value in counts], fontsize=7)
    axes[0].set_ylabel("Edges (thousands)"); axes[0].text(.02, .96, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    names = ["Precision", "Recall", "F1", "Exact length"]
    scores = [.6743, .7472, .7089, .8896]
    bars = axes[1].bar(names, scores, color=[COLORS[0], COLORS[1], COLORS[2], COLORS[4]])
    axes[1].bar_label(bars, fmt="%.4f", fontsize=7); axes[1].set_ylim(0, 1); axes[1].tick_params(axis="x", rotation=20)
    axes[1].text(.02, .96, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")
    for axis in axes: axis.grid(axis="y", color=PAPER_COLORS["light_border"], linewidth=.45)
    fig.tight_layout(pad=.65)
    return save_pair(fig, "graph_chain_evaluation_enhanced")


def priority_figure():
    labels = ["P1", "P2", "P3", "P4"]
    observed, predicted = np.array([87.63, 63.84, 21.79, .33]), np.array([90.21, 69.29, 28.28, .50])
    x = np.arange(4); width = .35
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    left = ax.bar(x - width / 2, observed, width, color=COLORS[0], label="Observed")
    right = ax.bar(x + width / 2, predicted, width, color=COLORS[4], label="Mean predicted")
    ax.bar_label(left, fmt="%.2f%%", fontsize=6.5); ax.bar_label(right, fmt="%.2f%%", fontsize=6.5)
    ax.set(xticks=x, xticklabels=labels, ylabel="Rate (%)", ylim=(0, 100))
    ax.legend(frameon=False, fontsize=7); ax.grid(axis="y", color=PAPER_COLORS["light_border"], linewidth=.45)
    fig.tight_layout(pad=.65)
    return save_pair(fig, "priority_validation_enhanced")


def write_manifests(outputs):
    created = datetime.now(timezone.utc).isoformat()
    purposes = {
        "validation_threshold_analysis_enhanced": ("Validation threshold analysis", "validation", 832022, "0.47 selected on validation"),
        "final_test_evaluation_enhanced": ("Locked-test discrimination and calibration", "locked test", 805126, "AUC=.9914; AP=.9030"),
        "final_test_roc_curve_enhanced": ("Locked-test ROC curve", "locked test", 805126, "ROC-AUC=.9914"),
        "final_test_precision_recall_curve_enhanced": ("Locked-test precision-recall curve", "locked test", 805126, "AP=.9030"),
        "final_test_calibration_curve_enhanced": ("Locked-test quantile calibration", "locked test", 805126, "Brier=.016398"),
        "shap_grouped_importance_enhanced": ("Grouped global SHAP importance", "validation sample", 5000, "raw-score SHAP"),
        "shap_operational_dependence_enhanced": ("Operational SHAP dependence", "validation sample", 5000, "raw-score SHAP"),
        "locked_test_error_profiles_enhanced": ("Locked-test TP/FP/FN profiles", "locked test", 805126, "TP=41898; FP=12160; FN=7140"),
        "propagation_exposure_network_enhanced": ("Validation propagation exposure map", "validation graph", 832022, "threshold=.47"),
        "graph_chain_evaluation_enhanced": ("Locked-test graph-chain evaluation", "locked-test graph", 805126, "chain F1=.7089; MAE=.1230"),
        "priority_validation_enhanced": ("Decision-priority validation", "validation", 832022, "P1-P4 observed vs predicted"),
    }
    rows = []
    for path in outputs:
        stem = path.stem
        purpose, split, count, metrics = purposes[stem]
        rows.append({
            "filename": path.name, "figure_purpose": purpose, "model_name": MODEL_VERSION,
            "model_artifact_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)), "data_split": split,
            "sample_count": count, "threshold": MODEL_THRESHOLD,
            "random_seed": RANDOM_SEED if "shap" in stem else "",
            "source_script": (
                "src/visualization/paper_figures/build_propagation_exposure_map.py"
                if stem == "propagation_exposure_network_enhanced"
                else "src/visualization/paper_figures/generate_enhanced_paper_figures.py"
            ),
            "creation_timestamp": created, "principal_metrics_shown": metrics,
        })
    pd.DataFrame(rows).to_csv(MANIFEST_PATH, index=False)
    replacements = [
        ("validation_threshold_analysis.pdf", "validation_threshold_analysis_enhanced.pdf", "tau=.46", "tau=.47", "Model selection", "Validation threshold analysis for the full-enhanced model. Panel (a) shows precision, recall, and F1; panel (b) shows false-positive, false-negative, and alert rates. The dashed line marks the validation-selected threshold tau=0.47."),
        ("final_test_discrimination_calibration.pdf", "final_test_evaluation_enhanced.pdf", "AUC=.9882; AP=.8543", "AUC=.9914; AP=.9030", "Locked-test results", "Locked-test discrimination and calibration of the full-enhanced model: (a) ROC curve, (b) precision-recall curve, and (c) equal-frequency calibration curve. The frozen threshold was selected on validation data only."),
        ("final_test_roc_curve.pdf", "final_test_roc_curve_enhanced.pdf", "ROC-AUC=.9882", "ROC-AUC=.9914", "Locked-test results", "Receiver-operating-characteristic curve for the full-enhanced model on the locked November-December 2023 test set (ROC-AUC=0.9914)."),
        ("final_test_precision_recall_curve.pdf", "final_test_precision_recall_curve_enhanced.pdf", "AP=.8543", "AP=.9030", "Locked-test results", "Precision-recall curve for the full-enhanced model on the locked November-December 2023 test set (average precision=0.9030); the dashed line denotes outcome prevalence."),
        ("final_test_calibration_curve.pdf", "final_test_calibration_curve_enhanced.pdf", "baseline calibration", "Brier=.016398", "Locked-test results", "Equal-frequency calibration of the full-enhanced model on the locked test set; points compare mean predicted probability with observed propagation frequency."),
        ("shap_grouped_importance_compact.pdf", "shap_grouped_importance_enhanced.pdf", "baseline SHAP", "24-feature grouped SHAP", "Explainability", "Grouped global mean absolute SHAP importance for 5,000 validation rotations (seed=42). One-hot levels are aggregated to their original feature groups; SHAP values are expressed in model raw-score units."),
        ("shap_operational_dependence_panels.pdf", "shap_operational_dependence_enhanced.pdf", "baseline SHAP", "enhanced raw-score SHAP", "Explainability", "Full-enhanced SHAP dependence patterns for (a) previous-delay ratio, (b) turn buffer, (c) previous arrival delay, and (d) planned turnaround. Values describe associations with the raw model score and are neither causal effects nor probability-point changes."),
        ("locked_test_error_profiles.pdf", "locked_test_error_profiles_enhanced.pdf", "TP=41301; FP=13744; FN=7737", "TP=41898; FP=12160; FN=7140", "Error analysis", "Locked-test operational feature distributions for true positives, false positives, and false negatives under the frozen tau=0.47 threshold. Boxes show medians and interquartile ranges; whiskers exclude plotted outliers for compactness."),
        ("propagation_exposure_network_paper.pdf", "propagation_exposure_network_enhanced.pdf", "baseline validation graph", "enhanced validation graph", "Graph analysis", "Validation-period propagation exposure from full-enhanced alerted rotation edges. Airport nodes are spatial aggregates and route lines are not recorded trajectories or joint chain probabilities."),
        ("no direct predecessor", "graph_chain_evaluation_enhanced.pdf", "baseline graph summaries", "chain-start F1=.7089; exact=.8896; MAE=.1230", "Graph analysis", "Locked-test graph evaluation for the full-enhanced system: (a) physical, eligible, and scored edge counts and (b) chain-start precision, recall, F1, and exact matched chain-length rate."),
        ("priority_validation_comparison.pdf", "priority_validation_enhanced.pdf", "old priority rates", "P1=87.63/90.21; P2=63.84/69.29; P3=21.79/28.28; P4=.33/.50", "Decision support", "Observed propagation rates and mean predicted probabilities across historical validation decision-priority tiers. The analysis is a retrospective decision-support demonstration, not live operational validation."),
    ]
    lines = [
        "# Paper Update Manifest", "", "No LaTeX source was present in the repository.", "",
        "Locked-test row-level probabilities were not persisted by the original final evaluation. They were",
        "reconstructed once from the immutable final model solely for reporting, verified against every",
        "authoritative aggregate metric, and cached. No threshold selection, tuning, or feature selection used",
        "locked-test labels.", "",
    ]
    for old, new, old_metric, new_metric, section, caption in replacements:
        lines.extend([
            f"## `{old}` -> `{new}`", "", f"- Old metric: {old_metric}",
            f"- New authoritative metric: {new_metric}", f"- Recommended section: {section}",
            f"- Proposed caption: {caption}",
            "- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.",
            "- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.", "",
        ])
    lines.extend([
        "## Additional map warning", "",
        "Eight non-CONUS airport codes (BQN, GUM, PPG, PSE, SJU, SPN, STT, STX) were excluded",
        "from the continental-US exposure map. Their underlying graph edges were not removed from model",
        "evaluation; this exclusion affects only geographic display.", "",
    ])
    UPDATE_MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_paper_tables():
    final_metrics = json.loads(
        (PROJECT_ROOT / "results" / "full_enhanced_final_test_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    pd.DataFrame(
        [{"data_split": "locked_test", "metric": key, "value": value}
         for key, value in final_metrics.items() if isinstance(value, (int, float))]
    ).to_csv(PAPER_METRICS_PATH, index=False)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in GRAPH_METRICS.items()]
    ).to_csv(PAPER_GRAPH_PATH, index=False)
    pd.DataFrame(
        {
            "priority": ["P1", "P2", "P3", "P4"],
            "observed_propagation_rate_percent": [87.63, 63.84, 21.79, .33],
            "mean_predicted_probability_percent": [90.21, 69.29, 28.28, .50],
        }
    ).to_csv(PAPER_PRIORITY_PATH, index=False)


def main():
    apply_paper_style()
    rows = load_or_build_reporting_rows()
    outputs = []
    outputs += validation_threshold_figure(rows)
    outputs += locked_test_figures(rows)
    outputs += shap_figures()
    outputs += error_profile_figure(rows)
    outputs += graph_figure()
    outputs += priority_figure()
    # The map is generated by its dedicated enhanced validation-edge script.
    map_pdf = OUTPUT_DIR / "propagation_exposure_network_enhanced.pdf"
    map_png = OUTPUT_DIR / "propagation_exposure_network_enhanced.png"
    if not map_pdf.exists() or not map_png.exists():
        raise FileNotFoundError("Run build_propagation_exposure_map before finalizing manifests.")
    outputs += [map_pdf, map_png]
    write_manifests(outputs)
    write_paper_tables()
    print(f"Created {len(outputs)} figure files and two manifests.")


if __name__ == "__main__":
    main()
