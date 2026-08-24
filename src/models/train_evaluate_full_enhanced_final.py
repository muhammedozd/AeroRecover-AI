"""Train the approved full-enhanced model and evaluate the locked test once."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.features.rotation_features import build_rotation_model_features
from src.models.rotation_model_contract import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MODEL_THRESHOLD,
    MODEL_VERSION,
    NUMERICAL_FEATURES,
    RAW_FEATURE_COLUMNS,
    validate_pipeline_contract,
)

from src.models.train_evaluate_enhanced_rotation_model import (
    BASELINE_CATEGORICAL,
    BASELINE_NUMERICAL,
    DATA_PATH,
    FORBIDDEN_FEATURES,
    PROJECT_ROOT,
    TARGET,
    build_pipeline,
    decision_metrics,
)


FROZEN_THRESHOLD = MODEL_THRESHOLD
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "full_enhanced_final_test_metrics.json"
MONTHLY_PATH = RESULTS_DIR / "full_enhanced_final_test_monthly_metrics.csv"
COMPARISON_PATH = RESULTS_DIR / "baseline_vs_full_enhanced_locked_test.csv"
CONTRACT_PATH = RESULTS_DIR / "full_enhanced_model_feature_contract.json"
BASELINE_METRICS_PATH = RESULTS_DIR / "final_test_metrics.json"
BASELINE_COMPARISON_PATH = PROJECT_ROOT / "reports" / "enhanced_rotation_test_metrics.csv"

EXPECTED_FEATURES = FEATURE_COLUMNS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract():
    if len(EXPECTED_FEATURES) != 24 or len(set(EXPECTED_FEATURES)) != 24:
        raise ValueError(
            f"FULL_ENHANCED feature contract must contain 24 unique features; "
            f"found {len(EXPECTED_FEATURES)}."
        )
    leaked = set(EXPECTED_FEATURES) & FORBIDDEN_FEATURES
    target_derived = {
        feature for feature in EXPECTED_FEATURES
        if feature == TARGET or "PROPAGATED" in feature.upper()
    }
    if leaked or target_derived:
        raise ValueError(
            f"Feature contract contains leakage columns: {sorted(leaked | target_derived)}"
        )


def metric_payload(y_true, probabilities, threshold):
    payload = decision_metrics(y_true, probabilities, threshold)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "sample_count": int(len(y_true)),
        "actual_propagations": int(np.asarray(y_true).sum()),
        "actual_propagation_rate": float(np.asarray(y_true).mean()),
        "alert_count": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(payload["PRECISION"]),
        "recall": float(payload["RECALL"]),
        "f1": float(payload["F1"]),
        "roc_auc": float(payload["ROC_AUC"]),
        "pr_auc": float(payload["PR_AUC"]),
        "log_loss": float(payload["LOG_LOSS"]),
        "brier_score": float(payload["BRIER_SCORE"]),
        "true_negative": int(payload["TRUE_NEGATIVE"]),
        "false_positive": int(payload["FALSE_POSITIVE"]),
        "false_negative": int(payload["FALSE_NEGATIVE"]),
        "true_positive": int(payload["TRUE_POSITIVE"]),
    }


def saved_model_features(model):
    selected = []
    for name, _, columns in model.named_steps["preprocessor"].transformers:
        if name in {"categorical", "numerical"}:
            selected.extend(list(columns))
    return selected


def main():
    validate_contract()
    required_columns = sorted(
        set(
            ["FL_DATE", "CRS_DEP_TIME", TARGET]
            + RAW_FEATURE_COLUMNS
        )
    )
    df = pd.read_csv(DATA_PATH, usecols=required_columns, low_memory=False)
    features, flight_dates = build_rotation_model_features(df)
    if flight_dates.isna().any() or df[TARGET].isna().any():
        raise ValueError("FL_DATE or target contains missing/invalid values.")

    train_mask = (flight_dates >= "2023-01-01") & (flight_dates < "2023-09-01")
    locked_test_mask = (flight_dates >= "2023-11-01") & (flight_dates < "2024-01-01")
    if (train_mask & locked_test_mask).any():
        raise ValueError("Training and locked-test rows overlap.")
    train_index = df.index[train_mask]
    test_index = df.index[locked_test_mask]
    if not len(train_index) or not len(test_index):
        raise ValueError("Training or locked-test split is empty.")

    model = build_pipeline(CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    model.fit(features.loc[train_index], df.loc[train_index, TARGET])

    # This is the only locked-test prediction pass in the final workflow.
    test_probabilities = model.predict_proba(features.loc[test_index])[:, 1]
    y_test = df.loc[test_index, TARGET]
    overall = metric_payload(y_test, test_probabilities, FROZEN_THRESHOLD)
    overall.update(
        {
            "evaluation_period": "2023-11-01/2023-12-31",
            "training_period": "2023-01-01/2023-08-31",
            "model": MODEL_VERSION,
            "feature_count": len(EXPECTED_FEATURES),
            "threshold_source": "frozen validation selection",
            "locked_test_prediction_passes": 1,
        }
    )

    test_months = flight_dates.loc[test_index].dt.to_period("M")
    monthly_rows = []
    for period in sorted(test_months.unique()):
        positions = np.flatnonzero((test_months == period).to_numpy())
        row = metric_payload(
            y_test.iloc[positions], test_probabilities[positions], FROZEN_THRESHOLD
        )
        row["month"] = period.strftime("%B")
        monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)

    baseline = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    baseline_report = pd.read_csv(BASELINE_COMPARISON_PATH)
    baseline_report_values = {
        row["METRIC"].lower(): row["BASELINE"]
        for _, row in baseline_report.iterrows()
    }
    comparison_metrics = [
        "roc_auc", "pr_auc", "log_loss", "brier_score", "precision", "recall", "f1",
        "true_negative", "false_positive", "false_negative", "true_positive",
    ]
    comparison_rows = []
    for metric in comparison_metrics:
        baseline_value = baseline.get(metric, baseline_report_values.get(metric, np.nan))
        if pd.isna(baseline_value):
            raise ValueError(f"Missing locked-test baseline metric: {metric}")
        enhanced_value = overall[metric]
        comparison_rows.append(
            {
                "metric": metric,
                "baseline_threshold": baseline["threshold"],
                "full_enhanced_threshold": FROZEN_THRESHOLD,
                "baseline": baseline_value,
                "full_enhanced": enhanced_value,
                "full_enhanced_minus_baseline": enhanced_value - baseline_value,
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    reloaded_model = joblib.load(MODEL_PATH)
    validate_pipeline_contract(reloaded_model)
    actual_features = saved_model_features(reloaded_model)
    if actual_features != EXPECTED_FEATURES:
        raise ValueError(
            f"Saved model feature contract mismatch: expected={EXPECTED_FEATURES}, "
            f"actual={actual_features}"
        )
    contract = {
        "contract_name": "FULL_ENHANCED_24_FEATURES",
        "feature_count": len(EXPECTED_FEATURES),
        "features": EXPECTED_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "saved_model_features": actual_features,
        "contract_verified": True,
        "frozen_threshold": FROZEN_THRESHOLD,
        "training_rows": int(len(train_index)),
        "locked_test_rows": int(len(test_index)),
        "model_sha256": sha256(MODEL_PATH),
    }
    METRICS_PATH.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    monthly.to_csv(MONTHLY_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(overall, indent=2))
    print("\nFeature contract verified: 24/24")
    print("Locked-test prediction passes: 1")
    print("Saved outputs:")
    for path in [MODEL_PATH, METRICS_PATH, MONTHLY_PATH, COMPARISON_PATH, CONTRACT_PATH]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
