"""Post-hoc error analysis for the frozen November-December 2023 test predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluate_locked_test_graph_policy import (
    EXPECTED_TEST_SAMPLE_COUNT,
    OPERATIONAL_THRESHOLD,
    load_locked_test_rotations,
    score_rotations,
)
from src.graph.score_graph_edges import add_target_flight_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_OUTPUT_PATH = PROJECT_ROOT / "reports" / "locked_test_error_feature_profiles.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "reports" / "locked_test_error_analysis_summary.txt"
FIGURE_OUTPUT_PATH = PROJECT_ROOT / "figures" / "locked_test_error_profiles.pdf"
FIGURE_NOTES_PATH = PROJECT_ROOT / "reports" / "locked_test_error_figure_notes.txt"

EXPECTED_CONFUSION_COUNTS = {
    "TN": 742_344,
    "FP": 13_744,
    "FN": 7_737,
    "TP": 41_301,
}
FEATURES = [
    "PREV_ARR_DELAY",
    "TURN_BUFFER",
    "PREV_DELAY_RATIO",
    "PLANNED_TURNAROUND",
    "ROTATION_POSITION",
]
ERROR_ORDER = ["TN", "FP", "FN", "TP"]


def label_errors(y_true: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    """Create the required row-level prediction and error fields in memory."""
    predicted_label = probabilities.ge(OPERATIONAL_THRESHOLD).astype(np.int8)
    true_values = y_true.astype(np.int8).reset_index(drop=True)
    labels = np.select(
        [
            true_values.eq(0) & predicted_label.eq(0),
            true_values.eq(0) & predicted_label.eq(1),
            true_values.eq(1) & predicted_label.eq(0),
            true_values.eq(1) & predicted_label.eq(1),
        ],
        ERROR_ORDER,
        default="INVALID",
    )
    return pd.DataFrame({
        "y_true": true_values,
        "predicted_probability": probabilities.to_numpy(),
        "predicted_label": predicted_label.to_numpy(),
        "error_type": labels,
    })


def verify_confusion_counts(error_rows: pd.DataFrame) -> dict[str, int]:
    counts = error_rows["error_type"].value_counts().reindex(ERROR_ORDER, fill_value=0)
    actual = {name: int(counts[name]) for name in ERROR_ORDER}
    if actual != EXPECTED_CONFUSION_COUNTS:
        differences = {
            name: actual[name] - EXPECTED_CONFUSION_COUNTS[name] for name in ERROR_ORDER
        }
        raise ValueError(
            "Locked-test confusion counts do not match the frozen reference. "
            f"Expected={EXPECTED_CONFUSION_COUNTS}; actual={actual}; differences={differences}. "
            "Analysis stopped before writing reports."
        )
    if sum(actual.values()) != EXPECTED_TEST_SAMPLE_COUNT:
        raise AssertionError("Verified confusion counts do not sum to the locked-test population.")
    return actual


def build_feature_profiles(analysis_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for error_type in ERROR_ORDER:
        group = analysis_data.loc[analysis_data["error_type"].eq(error_type)]
        for feature in FEATURES:
            values = group[feature]
            rows.append({
                "error_type": error_type,
                "feature": feature,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "standard_deviation": float(values.std(ddof=1)),
                "percentile_25": float(values.quantile(0.25)),
                "percentile_75": float(values.quantile(0.75)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })
    return pd.DataFrame(rows)


def build_contrasts(profiles: pd.DataFrame) -> pd.DataFrame:
    """Rank descriptive group gaps using pooled-scale standardized mean differences."""
    indexed = profiles.set_index(["error_type", "feature"])
    rows = []
    for left, right in [("FP", "TN"), ("FN", "TP")]:
        for feature in FEATURES:
            left_row = indexed.loc[(left, feature)]
            right_row = indexed.loc[(right, feature)]
            pooled_scale = np.sqrt(
                (left_row["standard_deviation"] ** 2 + right_row["standard_deviation"] ** 2) / 2
            )
            mean_difference = left_row["mean"] - right_row["mean"]
            standardized = mean_difference / pooled_scale if pooled_scale > 0 else np.nan
            rows.append({
                "comparison": f"{left}_vs_{right}",
                "feature": feature,
                "left_mean": float(left_row["mean"]),
                "right_mean": float(right_row["mean"]),
                "mean_difference": float(mean_difference),
                "standardized_mean_difference": float(standardized),
                "absolute_standardized_difference": float(abs(standardized)),
            })
    return pd.DataFrame(rows)


def feature_finding(row: pd.Series) -> str:
    left, right = row["comparison"].split("_vs_")
    direction = "higher" if row["mean_difference"] > 0 else "lower"
    return (
        f"- {left} cases were characterized by {direction} mean {row['feature']} than "
        f"{right} cases ({row['left_mean']:.3f} vs {row['right_mean']:.3f}; "
        f"standardized descriptive difference={row['standardized_mean_difference']:.3f})."
    )


def build_summary(
    counts: dict[str, int], profiles: pd.DataFrame, contrasts: pd.DataFrame,
) -> str:
    false_positive_rate = counts["FP"] / (counts["FP"] + counts["TN"])
    false_negative_rate = counts["FN"] / (counts["FN"] + counts["TP"])
    findings = [
        f"- The false-positive rate was {false_positive_rate:.6f} "
        f"({counts['FP']:,} / {counts['FP'] + counts['TN']:,} observed negatives).",
        f"- The false-negative rate was {false_negative_rate:.6f} "
        f"({counts['FN']:,} / {counts['FN'] + counts['TP']:,} observed positives).",
    ]
    material = contrasts.loc[
        contrasts["absolute_standardized_difference"].ge(0.20)
    ].sort_values("absolute_standardized_difference", ascending=False)
    findings.extend(feature_finding(row) for _, row in material.head(4).iterrows())
    if len(findings) < 4:
        findings.append(
            "- Remaining feature contrasts had absolute standardized descriptive differences "
            "below 0.20 and were not elevated as important findings."
        )

    fp_tn = contrasts.loc[contrasts["comparison"].eq("FP_vs_TN")].sort_values(
        "absolute_standardized_difference", ascending=False
    )
    fn_tp = contrasts.loc[contrasts["comparison"].eq("FN_vs_TP")].sort_values(
        "absolute_standardized_difference", ascending=False
    )
    return (
        "LOCKED TEST POST-HOC ERROR ANALYSIS\n"
        "===================================\n"
        "Period: 2023-11-01 through 2023-12-31 (inclusive)\n"
        "Frozen model: xgboost_propagation_2023_time_split.pkl\n"
        "Fixed threshold: tau=0.46\n"
        "No retraining, tuning, feature changes, threshold changes, or policy changes were performed.\n"
        "All statements below are descriptive associations, not causal conclusions.\n\n"
        f"Verified counts: TN={counts['TN']:,}, FP={counts['FP']:,}, "
        f"FN={counts['FN']:,}, TP={counts['TP']:,}\n"
        f"False-positive rate: {false_positive_rate:.6f}\n"
        f"False-negative rate: {false_negative_rate:.6f}\n\n"
        "Most distinct FP vs TN feature differences\n"
        "------------------------------------------\n"
        f"{fp_tn.to_string(index=False)}\n\n"
        "Most distinct FN vs TP feature differences\n"
        "------------------------------------------\n"
        f"{fn_tp.to_string(index=False)}\n\n"
        "Key data-supported findings\n"
        "---------------------------\n"
        + "\n".join(findings[:6])
        + "\n"
    )


def build_publication_figure(analysis_data: pd.DataFrame) -> dict[str, object]:
    """Create a compact vector PDF from TP/FP/FN locked-test distributions."""
    import matplotlib.pyplot as plt

    groups = ["TP", "FP", "FN"]
    colors = {"TP": "#0072B2", "FP": "#D55E00", "FN": "#009E73"}
    line_styles = {"TP": "-", "FP": "--", "FN": ":"}
    specifications = [
        ("PREV_DELAY_RATIO", "Previous-delay ratio"),
        ("TURN_BUFFER", "Turnaround buffer (min)"),
    ]
    selected = analysis_data.loc[analysis_data["error_type"].isin(groups)].copy()
    missing = selected[[name for name, _ in specifications]].isna().sum()
    if missing.any():
        raise ValueError(f"Figure inputs contain missing values; figure not generated:\n{missing[missing.gt(0)]}")
    display_report: dict[str, object] = {}
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3))
    for panel, (axis, (feature, label)) in enumerate(zip(axes, specifications)):
        if feature == "TURN_BUFFER":
            group_quantiles = selected.groupby("error_type")[feature].quantile([0.01, 0.99])
            lower = float(group_quantiles.xs(0.01, level=1).min())
            upper = float(group_quantiles.xs(0.99, level=1).max())
            limit_method = "union of TP/FP/FN group-specific 1st-99th percentile ranges"
        else:
            lower, upper = selected[feature].quantile([0.01, 0.99]).tolist()
            limit_method = "pooled TP/FP/FN 1st-99th percentile range"
        bins = np.linspace(lower, upper, 36)
        group_exclusions = {}
        for group in groups:
            original_values = selected.loc[selected["error_type"].eq(group), feature]
            displayed_values = original_values.loc[original_values.between(lower, upper)]
            lower_excluded_count = int(original_values.lt(lower).sum())
            upper_excluded_count = int(original_values.gt(upper).sum())
            excluded_count = lower_excluded_count + upper_excluded_count
            group_exclusions[group] = {
                "total_count": len(original_values),
                "excluded_count": excluded_count,
                "excluded_percentage": excluded_count / len(original_values) * 100,
                "lower_excluded_count": lower_excluded_count,
                "lower_excluded_percentage": lower_excluded_count / len(original_values) * 100,
                "upper_excluded_count": upper_excluded_count,
                "upper_excluded_percentage": upper_excluded_count / len(original_values) * 100,
            }
            axis.hist(
                displayed_values, bins=bins, density=True, histtype="step", linewidth=1.25,
                color=colors[group], linestyle=line_styles[group],
                label=f"{group} (n={len(original_values):,})",
            )
        display_report[feature] = {
            "lower_limit": float(lower), "upper_limit": float(upper),
            "limit_method": limit_method, "groups": group_exclusions,
        }
        axis.set_xlim(lower, upper)
        axis.set_xlabel(label, fontsize=7)
        axis.set_ylabel("Density", fontsize=7)
        axis.tick_params(labelsize=7, width=0.6, length=2.5)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        axis.set_title(f"({chr(97 + panel)})", loc="left", pad=3, fontsize=8, fontweight="bold")
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, fontsize=7, handlelength=2.0, columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.76, wspace=0.48)
    FIGURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT_PATH, format="pdf")
    plt.close(fig)
    return display_report


def format_figure_notes(display_report: dict[str, object]) -> str:
    labels = {
        "PREV_DELAY_RATIO": "Previous-delay ratio",
        "TURN_BUFFER": "Turnaround buffer (min)",
    }
    lines = [
        "LOCKED TEST ERROR FIGURE DISPLAY NOTES",
        "======================================",
        "The figure uses only original November-December 2023 locked-test records.",
        "Group definitions are unchanged: TP=41,301, FP=13,744, FN=7,737.",
        "No values were winsorized or moved to an axis boundary.",
        "Histograms were calculated from original in-range values only; observations outside",
        "the shared panel display limits were excluded from visualization as reported below.",
        "",
    ]
    for feature, report in display_report.items():
        lines.append(
            f"{labels[feature]} display limits: "
            f"[{report['lower_limit']:.6f}, {report['upper_limit']:.6f}]"
        )
        lines.append(f"Limit method: {report['limit_method']}.")
        for group in ["TP", "FP", "FN"]:
            group_report = report["groups"][group]
            lines.append(
                f"- {group}: excluded {group_report['excluded_count']:,} of "
                f"{group_report['total_count']:,} ({group_report['excluded_percentage']:.6f}%); "
                f"lower tail {group_report['lower_excluded_count']:,} "
                f"({group_report['lower_excluded_percentage']:.6f}%), upper tail "
                f"{group_report['upper_excluded_count']:,} "
                f"({group_report['upper_excluded_percentage']:.6f}%)"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    rotations, X_test, y_test = load_locked_test_rotations()
    rotations = add_target_flight_id(rotations)
    scored = score_rotations(rotations, X_test, y_test)
    error_rows = label_errors(
        y_true=y_test,
        probabilities=scored["PROPAGATION_PROBABILITY"],
    )
    counts = verify_confusion_counts(error_rows)

    missing = X_test[FEATURES].isna().sum()
    if missing.any():
        raise ValueError(
            "Profile features contain missing locked-test values; no reports were written:\n"
            f"{missing[missing.gt(0)]}"
        )
    analysis_data = X_test[FEATURES].reset_index(drop=True).join(error_rows)
    profiles = build_feature_profiles(analysis_data)
    contrasts = build_contrasts(profiles)
    summary = build_summary(counts, profiles, contrasts)

    display_report = build_publication_figure(analysis_data)
    figure_notes = format_figure_notes(display_report)

    PROFILE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(PROFILE_OUTPUT_PATH, index=False)
    SUMMARY_OUTPUT_PATH.write_text(summary, encoding="utf-8")
    FIGURE_NOTES_PATH.write_text(figure_notes, encoding="utf-8")
    print("\n" + summary)
    print("Row-level error data was kept in memory and not saved because it has 805,126 rows.")
    print(f"Saved: {PROFILE_OUTPUT_PATH}")
    print(f"Saved: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved: {FIGURE_OUTPUT_PATH}")
    print(f"Saved: {FIGURE_NOTES_PATH}")


if __name__ == "__main__":
    main()
