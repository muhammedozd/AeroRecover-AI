from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset.csv"

TARGET_COLUMN = "IS_DELAY_PROPAGATED"




def score_previous_delay_ratio(value):
    if value < 0.20:
        return 0
    elif value < 0.40:
        return 1
    elif value < 0.60:
        return 2
    else:
        return 3

def score_previous_arrival_delay(value):
    if value < 15:
        return 0
    elif value < 30:
        return 1
    elif value < 60:
        return 2
    else:
        return 3

def score_turn_buffer(value):
    if value >= 30:
        return 0
    elif value >= 20:
        return 1
    elif value >= 10:
        return 2
    else:
        return 3

def score_planned_turnaround(value):
    if 60 <= value < 180:
        return 0
    elif 30 <= value < 60:
        return 1
    elif value < 30:
        return 2
    else:
        return 1

def calculate_operational_risk_score(flight):
    score = 0

    score += score_previous_delay_ratio(flight["PREV_DELAY_RATIO"])

    score += score_previous_arrival_delay(flight["PREV_ARR_DELAY"])

    score += score_turn_buffer(flight["TURN_BUFFER"])

    score += score_planned_turnaround(flight["PLANNED_TURNAROUND"])

    return score

 

   

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset could not be found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print(f"Dataset path: {DATA_PATH}")
    print(f"Dataset shape: {data.shape}")

    return data


def analyze_binned_feature(
    data: pd.DataFrame,
    feature_name: str,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    required_columns = [feature_name, TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    analysis_data = data[required_columns].dropna().copy()

    range_column = f"{feature_name}_RANGE"

    analysis_data[range_column] = pd.cut(
        analysis_data[feature_name],
        bins=bins,
        labels=labels,
        right=False,
    )

    summary = (
        analysis_data
        .groupby(range_column, observed=True)[TARGET_COLUMN]
        .agg(["count", "sum", "mean"])
    )

    summary = summary.rename(
        columns={
            "count": "flight_count",
            "sum": "propagated_flights",
            "mean": "propagation_rate",
        }
    )

    summary["propagation_rate"] *= 100

    return summary


def analyze_previous_delay_ratio(
    data: pd.DataFrame,
) -> pd.DataFrame:
    bins = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        float("inf"),
    ]

    labels = [
        "0.0-0.2",
        "0.2-0.4",
        "0.4-0.6",
        "0.6-0.8",
        "0.8+",
    ]

    return analyze_binned_feature(
        data=data,
        feature_name="PREV_DELAY_RATIO",
        bins=bins,
        labels=labels,
    )


def analyze_previous_arrival_delay(
    data: pd.DataFrame,
) -> pd.DataFrame:
    bins = [
        float("-inf"),
        0,
        15,
        30,
        60,
        120,
        float("inf"),
    ]

    labels = [
        "<0",
        "0-15",
        "15-30",
        "30-60",
        "60-120",
        "120+",
    ]

    return analyze_binned_feature(
        data=data,
        feature_name="PREV_ARR_DELAY",
        bins=bins,
        labels=labels,
    )


def analyze_turn_buffer(
    data: pd.DataFrame,
) -> pd.DataFrame:
    bins = [
        float("-inf"),
        0,
        10,
        20,
        30,
        60,
        float("inf"),
    ]

    labels = [
        "<0",
        "0-10",
        "10-20",
        "20-30",
        "30-60",
        "60+",
    ]

    return analyze_binned_feature(
        data=data,
        feature_name="TURN_BUFFER",
        bins=bins,
        labels=labels,
    )


def analyze_planned_turnaround(
    data: pd.DataFrame,
) -> pd.DataFrame:
    bins = [
        float("-inf"),
        30,
        45,
        60,
        90,
        120,
        180,
        float("inf"),
    ]

    labels = [
        "<30",
        "30-45",
        "45-60",
        "60-90",
        "90-120",
        "120-180",
        "180+",
    ]

    return analyze_binned_feature(
        data=data,
        feature_name="PLANNED_TURNAROUND",
        bins=bins,
        labels=labels,
    )


def calculate_roc_metrics(
    data: pd.DataFrame,
):
    y_true = data[TARGET_COLUMN]

    y_score = data["OPERATIONAL_RISK_SCORE"]

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
    )

    auc = roc_auc_score(
        y_true,
        y_score,
    )

    return (
        fpr,
        tpr,
        thresholds,
        auc,
    )





def plot_roc_curve(
    fpr,
    tpr,
    auc,
):
    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Operational Risk Score (AUC = {auc:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Operational Risk Score")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def print_analysis(
    title: str,
    summary: pd.DataFrame,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(summary)



def analyze_operational_risk_score(
    data: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        data
        .groupby("OPERATIONAL_RISK_SCORE")[TARGET_COLUMN]
        .agg(["count", "sum", "mean"])
    )

    summary = summary.rename(
        columns={
            "count": "flight_count",
            "sum": "propagated_flights",
            "mean": "propagation_rate",
        }
    )

    summary["propagation_rate"] *= 100

    return summary


def calculate_optimal_threshold(
    fpr,
    tpr,
    thresholds,
):
    youden_index = tpr - fpr

    best_index = youden_index.argmax()

    best_threshold = thresholds[best_index]
    best_sensitivity = tpr[best_index]
    best_specificity = 1 - fpr[best_index]
    best_youden_index = youden_index[best_index]

    return (
        best_threshold,
        best_sensitivity,
        best_specificity,
        best_youden_index,
    )


def main() -> None:
    data = load_data()

    data["OPERATIONAL_RISK_SCORE"] = data.apply(
    calculate_operational_risk_score,
    axis=1,
)

    print(
    data[
        [
            "PREV_DELAY_RATIO",
            "PREV_ARR_DELAY",
            "TURN_BUFFER",
            "PLANNED_TURNAROUND",
            "OPERATIONAL_RISK_SCORE",
        ]
    ].head()
)

    ratio_summary = analyze_previous_delay_ratio(data)

    arrival_delay_summary = analyze_previous_arrival_delay(data)

    turn_buffer_summary = analyze_turn_buffer(data)

    turnaround_summary = analyze_planned_turnaround(data)

    risk_score_summary = analyze_operational_risk_score(data)

    fpr, tpr, thresholds, auc = calculate_roc_metrics(
    data,

)


    best_threshold, sensitivity, specificity, youden_index = (
    calculate_optimal_threshold(
        fpr,
        tpr,
        thresholds,
    )
    )

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Youden Index: {youden_index:.4f}")

    print(f"AUC Score: {auc:.4f}")
    plot_roc_curve(
    fpr,
    tpr,
    auc,
)

    print_analysis(
        "PREV_DELAY_RATIO ANALYSIS",
        ratio_summary,
    )

    print_analysis(
        "PREV_ARR_DELAY ANALYSIS",
        arrival_delay_summary,
    )

    print_analysis(
        "TURN_BUFFER ANALYSIS",
        turn_buffer_summary,
    )

    print_analysis(
        "PLANNED_TURNAROUND ANALYSIS",
        turnaround_summary,
    )


    print_analysis(
    "OPERATIONAL RISK SCORE ANALYSIS",
    risk_score_summary,
)


    

if __name__ == "__main__":
    main()

