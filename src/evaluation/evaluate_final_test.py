"""Evaluate the frozen model on the locked final test period."""

import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.evaluate_rotation_model import (
    load_dataset,
    load_model,
)
from src.models.train_rotation_model import (
    create_time_masks,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OPERATIONAL_THRESHOLD = 0.46
EXPECTED_TEST_SAMPLE_COUNT = 805_126

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_test_metrics.json"
)

MONTHLY_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_test_monthly_metrics.csv"
)

def prepare_test_data(data):
    (
        X,
        y,
        _,
        _,
    ) = prepare_features(data)

    _, _, test_mask = create_time_masks(
        data
    )

    test_dates = pd.to_datetime(
        data.loc[test_mask, "FL_DATE"],
        format="%Y-%m-%d",
    )

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    return (
        X_test,
        y_test,
        test_dates,
    )

def calculate_metrics(
    y_true,
    probabilities,
    threshold=OPERATIONAL_THRESHOLD,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    return {
        "sample_count": int(len(y_true)),
        "actual_propagations": int(y_true.sum()),
        "actual_propagation_rate": float(y_true.mean()),
        "alert_count": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }

def main():
    data = load_dataset()

    (
        X_test,
        y_test,
        test_dates,
    ) = prepare_test_data(data)

    if len(X_test) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            "Unexpected final test sample count: "
            f"{len(X_test):,}"
        )

    if not test_dates.between(
        pd.Timestamp("2023-11-01"),
        pd.Timestamp("2023-12-31"),
    ).all():
        raise ValueError(
            "Final test contains dates outside "
            "November-December 2023."
        )

    model = load_model()

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    final_metrics = calculate_metrics(
        y_true=y_test,
        probabilities=probabilities,
    )

    final_metrics["evaluation_period"] = (
        "2023-11-01/2023-12-31"
    )
    final_metrics["model"] = (
        "xgboost_propagation_2023_time_split"
    )

    monthly_results = []

    for month_number, month_name in [
        (11, "November"),
        (12, "December"),
    ]:
        month_mask = (
            test_dates.dt.month == month_number
        )

        month_metrics = calculate_metrics(
            y_true=y_test.loc[month_mask],
            probabilities=probabilities[
                month_mask.to_numpy()
            ],
        )

        month_metrics["month"] = month_name
        monthly_results.append(
            month_metrics
        )

    monthly_results_df = pd.DataFrame(
        monthly_results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            final_metrics,
            output_file,
            indent=2,
        )

    monthly_results_df.to_csv(
        MONTHLY_RESULTS_PATH,
        index=False,
    )

    print("\nFINAL LOCKED TEST RESULTS")
    print("-" * 60)

    for metric_name, metric_value in final_metrics.items():
        print(
            f"{metric_name}: {metric_value}"
        )

    print("\nMONTHLY TEST RESULTS")
    print("-" * 60)
    print(
        monthly_results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {RESULTS_PATH}"
    )
    print(
        f"Saved to: {MONTHLY_RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()