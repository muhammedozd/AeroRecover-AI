"""Train and compare baseline and enhanced BTS rotation models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.features.rotation_features import build_rotation_model_features
from src.models.rotation_model_contract import (
    CATEGORICAL_FEATURES as ACTIVE_CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES as ACTIVE_NUMERICAL_FEATURES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models" / "experiments"

VALIDATION_METRICS_PATH = REPORT_DIR / "enhanced_rotation_validation_metrics.csv"
TEST_METRICS_PATH = REPORT_DIR / "enhanced_rotation_test_metrics.csv"
SUMMARY_PATH = REPORT_DIR / "enhanced_rotation_experiment_summary.txt"
BASELINE_MODEL_PATH = MODEL_DIR / "bts_baseline_rotation_model.pkl"
ENHANCED_MODEL_PATH = MODEL_DIR / "bts_enhanced_rotation_model.pkl"

TARGET = "IS_DELAY_PROPAGATED"
DEFAULT_THRESHOLD = 0.46
RANDOM_STATE = 42

BASELINE_CATEGORICAL = ["PREV_DEST", "PREV_DELAY_LEVEL"]
BASELINE_NUMERICAL = [
    "ROTATION_POSITION",
    "PREV_ARR_DELAY",
    "PREV_ARR_MIN",
    "PREV_CRS_ARR_MIN",
    "PLANNED_TURNAROUND",
    "TURN_BUFFER",
    "PREV_DELAY_RATIO",
    "HAS_BUFFER",
    "IS_SHORT_TURN",
    "PREV_DELAYED",
]
ENHANCED_CATEGORICAL = ACTIVE_CATEGORICAL_FEATURES
ENHANCED_NUMERICAL = ACTIVE_NUMERICAL_FEATURES

FORBIDDEN_FEATURES = {
    "DEP_DELAY",
    "ARR_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "PROPAGATED_DELAY_MINUTES",
    "ACTUAL_TURNAROUND",
    "RECOVERY_MARGIN",
    "DELAYED",
    "DEP_TIME",
    "ARR_TIME",
}


def add_enhanced_features(df):
    """Create features available before the current flight departs."""
    features, flight_dates = build_rotation_model_features(df)
    result = df.copy()
    for column in features.columns:
        result[column] = features[column]
    return result, flight_dates


def build_pipeline(categorical_features, numerical_features):
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical_features),
            ("numerical", numerical_pipeline, numerical_features),
        ]
    )
    classifier = XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def decision_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "LOG_LOSS": log_loss(y_true, probabilities, labels=[0, 1]),
        "BRIER_SCORE": brier_score_loss(y_true, probabilities),
        "PRECISION": precision_score(y_true, predictions, zero_division=0),
        "RECALL": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "TRUE_NEGATIVE": int(tn),
        "FALSE_POSITIVE": int(fp),
        "FALSE_NEGATIVE": int(fn),
        "TRUE_POSITIVE": int(tp),
    }


def select_f1_threshold(y_true, probabilities):
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def comparison_frame(baseline, enhanced, threshold_label, baseline_threshold, enhanced_threshold):
    rows = []
    for metric in baseline:
        rows.append(
            {
                "THRESHOLD_SET": threshold_label,
                "BASELINE_THRESHOLD": baseline_threshold,
                "ENHANCED_THRESHOLD": enhanced_threshold,
                "METRIC": metric,
                "BASELINE": baseline[metric],
                "ENHANCED": enhanced[metric],
                "ENHANCED_MINUS_BASELINE": enhanced[metric] - baseline[metric],
            }
        )
    return pd.DataFrame(rows)


def validate_experiment(df, flight_dates, baseline_features, enhanced_features):
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")
    if flight_dates.isna().any():
        raise ValueError("FL_DATE contains invalid or missing dates.")
    if not set(baseline_features).issubset(enhanced_features):
        raise ValueError("Enhanced features do not contain all baseline features.")
    leaked = (set(baseline_features) | set(enhanced_features)) & FORBIDDEN_FEATURES
    if leaked:
        raise ValueError(f"Forbidden leakage features found: {sorted(leaked)}")
    derived_target_columns = {
        column for column in baseline_features + enhanced_features
        if column == TARGET or "PROPAGATED" in column
    }
    if derived_target_columns:
        raise ValueError(f"Target-derived features found: {sorted(derived_target_columns)}")
    numeric_values = df[ENHANCED_NUMERICAL].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric_values.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Enhanced numerical features contain infinite values.")

    train = flight_dates < "2023-09-01"
    validation = (flight_dates >= "2023-09-01") & (flight_dates < "2023-11-01")
    test = flight_dates >= "2023-11-01"
    membership = train.astype(int) + validation.astype(int) + test.astype(int)
    if len(df) != int(train.sum() + validation.sum() + test.sum()):
        raise ValueError("Time split sizes do not sum to the dataset size.")
    if not membership.eq(1).all():
        raise ValueError("Time splits overlap or leave rows unassigned.")
    return train, validation, test


def format_metric_block(title, frame):
    return title + "\n" + frame.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def main():
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
    baseline_features = BASELINE_CATEGORICAL + BASELINE_NUMERICAL
    enhanced_features = ENHANCED_CATEGORICAL + ENHANCED_NUMERICAL
    train_mask, validation_mask, test_mask = validate_experiment(
        df, flight_dates, baseline_features, enhanced_features
    )

    split_indices = {
        "train": df.index[train_mask],
        "validation": df.index[validation_mask],
        "test": df.index[test_mask],
    }
    # Both models select from these exact same index objects.
    baseline_model = build_pipeline(BASELINE_CATEGORICAL, BASELINE_NUMERICAL)
    enhanced_model = build_pipeline(ENHANCED_CATEGORICAL, ENHANCED_NUMERICAL)
    y_train = df.loc[split_indices["train"], TARGET]
    baseline_model.fit(df.loc[split_indices["train"], baseline_features], y_train)
    enhanced_model.fit(df.loc[split_indices["train"], enhanced_features], y_train)

    y_validation = df.loc[split_indices["validation"], TARGET]
    baseline_validation_prob = baseline_model.predict_proba(
        df.loc[split_indices["validation"], baseline_features]
    )[:, 1]
    enhanced_validation_prob = enhanced_model.predict_proba(
        df.loc[split_indices["validation"], enhanced_features]
    )[:, 1]

    baseline_threshold, baseline_best_f1 = select_f1_threshold(
        y_validation, baseline_validation_prob
    )
    enhanced_threshold, enhanced_best_f1 = select_f1_threshold(
        y_validation, enhanced_validation_prob
    )
    validation_default = comparison_frame(
        decision_metrics(y_validation, baseline_validation_prob, DEFAULT_THRESHOLD),
        decision_metrics(y_validation, enhanced_validation_prob, DEFAULT_THRESHOLD),
        "PROJECT_DEFAULT_0.46",
        DEFAULT_THRESHOLD,
        DEFAULT_THRESHOLD,
    )
    validation_selected = comparison_frame(
        decision_metrics(y_validation, baseline_validation_prob, baseline_threshold),
        decision_metrics(y_validation, enhanced_validation_prob, enhanced_threshold),
        "VALIDATION_SELECTED_F1",
        baseline_threshold,
        enhanced_threshold,
    )
    validation_metrics = pd.concat(
        [validation_default, validation_selected], ignore_index=True
    )

    # The locked validation thresholds are applied to the test set once.
    y_test = df.loc[split_indices["test"], TARGET]
    baseline_test_prob = baseline_model.predict_proba(
        df.loc[split_indices["test"], baseline_features]
    )[:, 1]
    enhanced_test_prob = enhanced_model.predict_proba(
        df.loc[split_indices["test"], enhanced_features]
    )[:, 1]
    test_metrics = comparison_frame(
        decision_metrics(y_test, baseline_test_prob, baseline_threshold),
        decision_metrics(y_test, enhanced_test_prob, enhanced_threshold),
        "LOCKED_VALIDATION_F1",
        baseline_threshold,
        enhanced_threshold,
    )

    missing_counts = df[enhanced_features].isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    summary_parts = [
        "Enhanced BTS Rotation Model Experiment",
        "=" * 38,
        f"Dataset rows: {len(df):,}",
        f"Train rows: {len(split_indices['train']):,}",
        f"Validation rows: {len(split_indices['validation']):,}",
        f"Test rows: {len(split_indices['test']):,}",
        "Both models used identical row indices: yes",
        f"Baseline validation-selected threshold: {baseline_threshold:.2f} (F1={baseline_best_f1:.6f})",
        f"Enhanced validation-selected threshold: {enhanced_threshold:.2f} (F1={enhanced_best_f1:.6f})",
        "",
        "Missing values in generated/enhanced feature set (non-zero only):",
        missing_counts.to_string() if not missing_counts.empty else "None",
        "",
        format_metric_block("Validation metrics", validation_metrics),
        "",
        format_metric_block("Test metrics", test_metrics),
        "",
        "Interpretation: positive differences are better for ROC-AUC, PR-AUC, precision, recall, and F1;",
        "negative differences are better for log loss and Brier score. Accuracy is not used as a primary metric.",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    validation_metrics.to_csv(VALIDATION_METRICS_PATH, index=False)
    test_metrics.to_csv(TEST_METRICS_PATH, index=False)
    SUMMARY_PATH.write_text("\n".join(summary_parts) + "\n", encoding="utf-8")
    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    joblib.dump(enhanced_model, ENHANCED_MODEL_PATH)

    print("\n".join(summary_parts))
    print("\nSaved outputs:")
    for path in [
        VALIDATION_METRICS_PATH,
        TEST_METRICS_PATH,
        SUMMARY_PATH,
        BASELINE_MODEL_PATH,
        ENHANCED_MODEL_PATH,
    ]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
