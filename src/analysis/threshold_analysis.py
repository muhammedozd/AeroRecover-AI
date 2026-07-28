from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset.csv"

TARGET_COLUMN = "IS_DELAY_PROPAGATED"


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


def print_analysis(
    title: str,
    summary: pd.DataFrame,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(summary)


def main() -> None:
    data = load_data()

    ratio_summary = analyze_previous_delay_ratio(data)

    arrival_delay_summary = analyze_previous_arrival_delay(data)

    turn_buffer_summary = analyze_turn_buffer(data)

    turnaround_summary = analyze_planned_turnaround(data)

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


if __name__ == "__main__":
    main()