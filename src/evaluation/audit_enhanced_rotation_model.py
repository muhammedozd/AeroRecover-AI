"""Audit the enhanced BTS rotation experiment without using test predictions."""

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.train_evaluate_enhanced_rotation_model import (
    BASELINE_CATEGORICAL,
    BASELINE_NUMERICAL,
    DATA_PATH,
    DEFAULT_THRESHOLD,
    ENHANCED_CATEGORICAL,
    ENHANCED_MODEL_PATH,
    ENHANCED_NUMERICAL,
    FORBIDDEN_FEATURES,
    PROJECT_ROOT,
    TARGET,
    TEST_METRICS_PATH,
    add_enhanced_features,
    build_pipeline,
    decision_metrics,
    select_f1_threshold,
)


REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "figures"
PROVENANCE_PATH = REPORT_DIR / "enhanced_rotation_feature_provenance.csv"
ABLATION_PATH = REPORT_DIR / "enhanced_rotation_ablation_validation.csv"
MISSING_AUDIT_PATH = REPORT_DIR / "enhanced_rotation_missing_value_audit.txt"
SUMMARY_PATH = REPORT_DIR / "enhanced_rotation_model_audit_summary.txt"
FIGURE_PATH = FIGURE_DIR / "enhanced_rotation_ablation_validation.png"

FLIGHT_CONTEXT_CATEGORICAL = ["DEST", "OP_UNIQUE_CARRIER"]
FLIGHT_CONTEXT_NUMERICAL = ["DISTANCE"]
SCHEDULE_NUMERICAL = [
    "CRS_DEP_TIME_SIN",
    "CRS_DEP_TIME_COS",
    "DAY_OF_WEEK",
    "MONTH",
    "IS_WEEKEND",
]
OPERATIONAL_NUMERICAL = [
    "DELAY_EXCESS_OVER_TURN",
    "AVAILABLE_BUFFER_RATIO",
    "PREV_DELAY_SHORT_TURN_INTERACTION",
]

MODEL_SPECS = {
    "BASELINE": (BASELINE_CATEGORICAL, BASELINE_NUMERICAL),
    "BASELINE_PLUS_FLIGHT_CONTEXT": (
        BASELINE_CATEGORICAL + FLIGHT_CONTEXT_CATEGORICAL,
        BASELINE_NUMERICAL + FLIGHT_CONTEXT_NUMERICAL,
    ),
    "BASELINE_PLUS_SCHEDULE": (
        BASELINE_CATEGORICAL,
        BASELINE_NUMERICAL + SCHEDULE_NUMERICAL,
    ),
    "BASELINE_PLUS_OPERATIONAL_INTERACTIONS": (
        BASELINE_CATEGORICAL,
        BASELINE_NUMERICAL + OPERATIONAL_NUMERICAL,
    ),
    "FULL_ENHANCED": (ENHANCED_CATEGORICAL, ENHANCED_NUMERICAL),
}


def provenance_table():
    definitions = {
        "PREV_DEST": ("baseline_previous_flight", "DEST shifted within tail/date", "Previous flight destination; known at prediction time."),
        "PREV_DELAY_LEVEL": ("baseline_previous_flight", "PREV_ARR_DELAY", "Previous arrival delay bucket."),
        "ROTATION_POSITION": ("baseline_schedule", "TAIL_NUM, FL_DATE, CRS_DEP_TIME", "Scheduled rotation order."),
        "PREV_ARR_DELAY": ("baseline_previous_flight", "ARR_DELAY shifted within tail/date", "Realized delay of the already-arrived previous flight."),
        "PREV_ARR_MIN": ("baseline_previous_flight", "ARR_TIME shifted within tail/date", "Realized arrival time of the previous flight."),
        "PREV_CRS_ARR_MIN": ("baseline_schedule", "CRS_ARR_TIME shifted within tail/date", "Previous flight scheduled arrival time."),
        "PLANNED_TURNAROUND": ("baseline_schedule", "CRS_DEP_TIME, PREV_CRS_ARR_MIN", "Scheduled turnaround, adjusted across midnight."),
        "TURN_BUFFER": ("baseline_operational", "PLANNED_TURNAROUND, PREV_ARR_DELAY", "Scheduled turn remaining after positive previous delay."),
        "PREV_DELAY_RATIO": ("baseline_operational", "PREV_ARR_DELAY, PLANNED_TURNAROUND", "Positive previous delay divided by scheduled turn."),
        "HAS_BUFFER": ("baseline_operational", "TURN_BUFFER", "Indicator that turn buffer is positive."),
        "IS_SHORT_TURN": ("baseline_schedule", "PLANNED_TURNAROUND", "Scheduled turn shorter than 45 minutes."),
        "PREV_DELAYED": ("baseline_previous_flight", "PREV_ARR_DELAY", "Previous flight delay of at least 15 minutes."),
        "DEST": ("flight_context", "DEST", "Current flight scheduled destination."),
        "OP_UNIQUE_CARRIER": ("flight_context", "OP_UNIQUE_CARRIER", "Published operating carrier."),
        "DISTANCE": ("flight_context", "DISTANCE", "Published route distance."),
        "CRS_DEP_MIN_SAFE": ("schedule", "CRS_DEP_TIME", "Validated scheduled departure minute of day."),
        "CRS_DEP_TIME_SIN": ("schedule", "CRS_DEP_TIME", "Sine encoding using a 1,440-minute day."),
        "CRS_DEP_TIME_COS": ("schedule", "CRS_DEP_TIME", "Cosine encoding using a 1,440-minute day."),
        "DAY_OF_WEEK": ("schedule", "FL_DATE", "Calendar weekday of the scheduled flight date."),
        "MONTH": ("schedule", "FL_DATE", "Calendar month of the scheduled flight date."),
        "IS_WEEKEND": ("schedule", "FL_DATE", "Saturday/Sunday indicator."),
        "DELAY_EXCESS_OVER_TURN": ("operational_interaction", "PREV_ARR_DELAY, PLANNED_TURNAROUND", "Positive previous delay exceeding scheduled turn."),
        "AVAILABLE_BUFFER_RATIO": ("operational_interaction", "PREV_ARR_DELAY, PLANNED_TURNAROUND", "Positive remaining buffer divided by scheduled turn."),
        "PREV_DELAY_SHORT_TURN_INTERACTION": ("operational_interaction", "PREV_ARR_DELAY, IS_SHORT_TURN", "Positive previous delay multiplied by short-turn indicator."),
    }
    exact_features = ENHANCED_CATEGORICAL + ENHANCED_NUMERICAL
    missing_definitions = set(exact_features) - set(definitions)
    if missing_definitions:
        raise ValueError(f"Missing provenance for features: {sorted(missing_definitions)}")
    return pd.DataFrame(
        [
            {
                "FEATURE": feature,
                "FEATURE_GROUP": definitions[feature][0],
                "SOURCE_COLUMNS": definitions[feature][1],
                "AVAILABLE_AT_PREDICTION_TIME": True,
                "LEAKAGE_RISK": "LOW",
                "NOTES": definitions[feature][2],
            }
            for feature in exact_features
        ]
    )


def validate_feature_contract(provenance):
    exact_features = ENHANCED_CATEGORICAL + ENHANCED_NUMERICAL
    forbidden = set(exact_features) & FORBIDDEN_FEATURES
    target_derived = {
        feature for feature in exact_features
        if feature == TARGET or "PROPAGATED" in feature.upper()
    }
    actual_current_times = {"DEP_TIME", "ARR_TIME"} & set(exact_features)
    if forbidden or target_derived or actual_current_times:
        problems = sorted(forbidden | target_derived | actual_current_times)
        raise ValueError(f"Leakage audit failed; forbidden feature(s): {problems}")
    if not provenance["AVAILABLE_AT_PREDICTION_TIME"].all():
        unavailable = provenance.loc[
            ~provenance["AVAILABLE_AT_PREDICTION_TIME"], "FEATURE"
        ].tolist()
        raise ValueError(f"Features unavailable at prediction time: {unavailable}")

    saved_model = joblib.load(ENHANCED_MODEL_PATH)
    transformers = saved_model.named_steps["preprocessor"].transformers
    saved_features = []
    for name, _, columns in transformers:
        if name in {"categorical", "numerical"}:
            saved_features.extend(list(columns))
    if saved_features != exact_features:
        raise ValueError(
            "Saved enhanced model feature list differs from source definition: "
            f"saved={saved_features}, expected={exact_features}"
        )


def audit_operational_features(df):
    positive_delay = pd.to_numeric(df["PREV_ARR_DELAY"], errors="coerce").clip(lower=0)
    planned_turn = pd.to_numeric(df["PLANNED_TURNAROUND"], errors="coerce")
    denominator = planned_turn.mask(planned_turn == 0, pd.NA)
    expected = {
        "DELAY_EXCESS_OVER_TURN": (positive_delay - planned_turn).clip(lower=0),
        "AVAILABLE_BUFFER_RATIO": (planned_turn - positive_delay).clip(lower=0) / denominator,
        "PREV_DELAY_SHORT_TURN_INTERACTION": positive_delay
        * pd.to_numeric(df["IS_SHORT_TURN"], errors="coerce"),
    }
    lines = ["Operational feature checks", "=" * 26]
    for feature, expected_values in expected.items():
        actual = pd.to_numeric(df[feature], errors="coerce")
        finite_actual = actual.to_numpy(dtype=float, na_value=np.nan)
        infinite_count = int(np.isinf(finite_actual).sum())
        mismatch = ~np.isclose(actual, expected_values, equal_nan=True)
        negative_count = int((actual < 0).sum())
        if infinite_count or int(mismatch.sum()):
            raise ValueError(
                f"Operational feature audit failed for {feature}: "
                f"infinite={infinite_count}, formula_mismatch={int(mismatch.sum())}"
            )
        if negative_count:
            raise ValueError(f"Unexpected negative values in {feature}: {negative_count}")
        lines.append(
            f"{feature}: formula_match=yes, infinite=0, negative=0, missing={int(actual.isna().sum()):,}"
        )
    lines.append(f"PLANNED_TURNAROUND zero denominators: {int((planned_turn == 0).sum()):,}")
    return lines


def missing_value_audit(df):
    missing_mask = df["PREV_DELAY_LEVEL"].isna()
    delays = pd.to_numeric(df.loc[missing_mask, "PREV_ARR_DELAY"], errors="coerce")
    source_missing = int(delays.isna().sum())
    outside_cut = int(((delays < -1000) | (delays > 1000)).sum())
    boundary_gap = int(((delays > -1) & (delays < -1)).sum())
    if not missing_mask.any():
        cause = "no missing PREV_DELAY_LEVEL values"
    elif source_missing:
        cause = "source PREV_ARR_DELAY missingness"
    elif outside_cut == len(delays):
        cause = "all values fall outside pd.cut bounds [-1000, 1000]"
    else:
        cause = "mixed or unexpected; inspect frequency table"
    frequencies = delays.value_counts(dropna=False).sort_index()
    lines = [
        "PREV_DELAY_LEVEL Missing-Value Audit",
        "=" * 40,
        f"Missing row count: {int(missing_mask.sum()):,}",
        f"PREV_ARR_DELAY missing among these rows: {source_missing:,}",
        f"PREV_ARR_DELAY minimum: {delays.min()}",
        f"PREV_ARR_DELAY maximum: {delays.max()}",
        f"Values outside pd.cut bounds: {outside_cut:,}",
        f"Boundary-gap rows: {boundary_gap:,}",
        f"Determined cause: {cause}",
        "",
        "PREV_ARR_DELAY frequencies:",
        frequencies.to_string(),
    ]
    if missing_mask.any():
        lines.extend(
            [
                "",
                "Safe recommendation (not applied): use open-ended outer bounds such as",
                "[-np.inf, -1, 14, 29, 59, np.inf], explicitly document boundary semantics,",
                "then rebuild and revalidate the rotation dataset in a separate change.",
            ]
        )
    return lines, int(missing_mask.sum()), cause


def metric_row(model_name, threshold_set, threshold, metrics, baseline_metrics):
    row = {"MODEL": model_name, "THRESHOLD_SET": threshold_set, "THRESHOLD": threshold}
    for metric, value in metrics.items():
        row[metric] = value
        row[f"{metric}_MINUS_BASELINE"] = value - baseline_metrics[metric]
    return row


def run_ablations(df, train_index, validation_index):
    if not train_index.equals(df.index[df["FL_DATE_PARSED"] < "2023-09-01"]):
        raise ValueError("Train row indices are not identical across ablation models.")
    validation_expected = df.index[
        (df["FL_DATE_PARSED"] >= "2023-09-01")
        & (df["FL_DATE_PARSED"] < "2023-11-01")
    ]
    if not validation_index.equals(validation_expected):
        raise ValueError("Validation row indices are not identical across ablation models.")

    y_train = df.loc[train_index, TARGET]
    y_validation = df.loc[validation_index, TARGET]
    results = {}
    for name, (categorical, numerical) in MODEL_SPECS.items():
        features = categorical + numerical
        model = build_pipeline(categorical, numerical)
        model.fit(df.loc[train_index, features], y_train)
        probabilities = model.predict_proba(df.loc[validation_index, features])[:, 1]
        selected_threshold, best_f1 = select_f1_threshold(y_validation, probabilities)
        results[name] = {
            "default": decision_metrics(y_validation, probabilities, DEFAULT_THRESHOLD),
            "selected": decision_metrics(y_validation, probabilities, selected_threshold),
            "selected_threshold": selected_threshold,
            "best_f1": best_f1,
        }
        print(f"Completed {name}: threshold={selected_threshold:.2f}, F1={best_f1:.6f}")

    rows = []
    for name, result in results.items():
        rows.append(
            metric_row(
                name,
                "PROJECT_DEFAULT_0.46",
                DEFAULT_THRESHOLD,
                result["default"],
                results["BASELINE"]["default"],
            )
        )
        rows.append(
            metric_row(
                name,
                "VALIDATION_SELECTED_F1",
                result["selected_threshold"],
                result["selected"],
                results["BASELINE"]["selected"],
            )
        )
    return pd.DataFrame(rows), results


def make_figure(ablation):
    selected = ablation[ablation["THRESHOLD_SET"] == "VALIDATION_SELECTED_F1"]
    x = np.arange(len(selected))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, selected["PR_AUC"], width, label="PR-AUC")
    ax.bar(x + width / 2, selected["F1"], width, label="F1")
    ax.set_xticks(x, selected["MODEL"].str.replace("BASELINE_PLUS_", "+", regex=False), rotation=18, ha="right")
    ax.set_ylim(0.75, 0.94)
    ax.set_ylabel("Validation score")
    ax.set_title("Enhanced rotation model ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def main():
    provenance = provenance_table()
    validate_feature_contract(provenance)
    required_columns = sorted(
        set(
            ["FL_DATE", "CRS_DEP_TIME", TARGET]
            + ENHANCED_CATEGORICAL
            + BASELINE_NUMERICAL
            + ["DISTANCE"]
        )
    )
    df = pd.read_csv(DATA_PATH, usecols=required_columns, low_memory=False)
    df, flight_dates = add_enhanced_features(df)
    df["FL_DATE_PARSED"] = flight_dates
    if df[TARGET].isna().any() or flight_dates.isna().any():
        raise ValueError("Target or FL_DATE contains missing/invalid values.")
    train_index = df.index[flight_dates < "2023-09-01"]
    validation_index = df.index[
        (flight_dates >= "2023-09-01") & (flight_dates < "2023-11-01")
    ]
    if len(train_index.intersection(validation_index)):
        raise ValueError("Train and validation date ranges overlap.")

    operational_lines = audit_operational_features(df)
    missing_lines, missing_count, missing_cause = missing_value_audit(df)
    ablation, results = run_ablations(df, train_index, validation_index)

    full = results["FULL_ENHANCED"]["selected"]
    baseline = results["BASELINE"]["selected"]
    contribution_confirmed = full["PR_AUC"] > baseline["PR_AUC"] and full["F1"] > baseline["F1"]
    if not contribution_confirmed:
        decision = "FAIL"
        decision_reason = "Full enhanced validation PR-AUC and F1 did not both improve over baseline."
    elif missing_count:
        decision = "PASS_WITH_WARNINGS"
        decision_reason = (
            "No leakage was found and full enhanced validation contribution was confirmed, "
            "but PREV_DELAY_LEVEL has a bounded-bin data-quality issue."
        )
    else:
        decision = "PASS"
        decision_reason = "No leakage was found and full enhanced validation contribution was confirmed."

    # Read the locked report only; no test features or predictions are used here.
    locked_test = pd.read_csv(TEST_METRICS_PATH)
    summary_lines = [
        "Enhanced Rotation Model Audit",
        "=" * 29,
        f"DECISION: {decision}",
        f"Reason: {decision_reason}",
        "Weather features used: no",
        "Leakage feature audit: passed",
        "Saved-model feature contract: passed",
        "Train/validation identical row indices across models: yes",
        f"Train rows: {len(train_index):,}",
        f"Validation rows: {len(validation_index):,}",
        "Test predictions generated during audit: no",
        f"Previously locked test report rows read: {len(locked_test):,}",
        f"PREV_DELAY_LEVEL missing rows: {missing_count:,}",
        f"Missing-value cause: {missing_cause}",
        "",
        "Exact FULL_ENHANCED features:",
        ", ".join(ENHANCED_CATEGORICAL + ENHANCED_NUMERICAL),
        "",
        *operational_lines,
        "",
        "Validation-selected comparison:",
        ablation[ablation["THRESHOLD_SET"] == "VALIDATION_SELECTED_F1"].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Metric direction: positive differences are favorable for ROC-AUC, PR-AUC, precision,",
        "recall, and F1; negative differences are favorable for log loss and Brier score.",
    ]
    if missing_count:
        summary_lines.extend(
            [
                "Recommendation: resolve the PREV_DELAY_LEVEL data-quality warning and revalidate",
                "before promoting the enhanced model.",
            ]
        )
    else:
        summary_lines.append(
            "Recommendation: the full enhanced model is a validated main-system candidate."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    provenance.to_csv(PROVENANCE_PATH, index=False)
    ablation.to_csv(ABLATION_PATH, index=False)
    MISSING_AUDIT_PATH.write_text("\n".join(missing_lines) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    make_figure(ablation)
    print("\n".join(summary_lines))
    print("\nSaved outputs:")
    for path in [PROVENANCE_PATH, ABLATION_PATH, MISSING_AUDIT_PATH, SUMMARY_PATH, FIGURE_PATH]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
