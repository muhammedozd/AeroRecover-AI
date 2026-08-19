"""Analyze decision-support priority errors."""

from ast import main
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETAILS_PATH = (
    PROJECT_ROOT
    / "results"
    / "decision_policy_validation_details.parquet"
)


def load_evaluation_data() -> pd.DataFrame:
    if not DETAILS_PATH.exists():
        raise FileNotFoundError(
            f"Detailed DSS evaluation not found: {DETAILS_PATH}"
        )

    evaluation_data = pd.read_parquet(
        DETAILS_PATH
    )

    print(
        "Detailed evaluation rows:",
        f"{len(evaluation_data):,}",
    )

    return evaluation_data

def identify_policy_error_groups(
    evaluation_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    normal_propagations = evaluation_data[
        (evaluation_data["PRIORITY"] == "P4_NORMAL")
        & (evaluation_data["ACTUAL_PROPAGATION"] == 1)
    ].copy()

    critical_non_events = evaluation_data[
        (evaluation_data["PRIORITY"] == "P1_CRITICAL")
        & (evaluation_data["ACTUAL_PROPAGATION"] == 0)
    ].copy()

    monitored_propagations = evaluation_data[
        (evaluation_data["PRIORITY"] == "P3_MONITOR")
        & (evaluation_data["ACTUAL_PROPAGATION"] == 1)
    ].copy()

    return (
        normal_propagations,
        critical_non_events,
        monitored_propagations,
    )



def summarize_error_group(
    error_group: pd.DataFrame,
) -> pd.DataFrame:

    analysis_columns = [
        "PROPAGATION_PROBABILITY",
        "PREV_ARR_DELAY",
        "TURN_BUFFER",
        "PREV_DELAY_RATIO",
        "PLANNED_TURNAROUND",
        "EDGE_COUNT",
    ]

    summary = error_group[
        analysis_columns
    ].agg([
        "count",
        "mean",
        "median",
        "min",
        "max",
    ])

    return summary

def main() -> None:
    evaluation_data = load_evaluation_data()

    (
        normal_propagations,
        critical_non_events,
        monitored_propagations,
    ) = identify_policy_error_groups(
        evaluation_data
    )

    print("\nDSS policy error groups")
    print("-" * 60)

    print(
        "P4 actual propagations:",
        f"{len(normal_propagations):,}",
    )

    print(
        "P1 non-propagations:",
        f"{len(critical_non_events):,}",
    )

    print(
        "P3 actual propagations:",
        f"{len(monitored_propagations):,}",
    )
    print("\nP4 actual propagation characteristics")
    print("-" * 60)
    print(
    summarize_error_group(
        normal_propagations
    ).to_string()
)

    print("\nP1 non-propagation characteristics")
    print("-" * 60)
    print(
    summarize_error_group(
        critical_non_events
    ).to_string()
)

    print("\nP3 actual propagation characteristics")
    print("-" * 60)
    print(
        summarize_error_group(
            monitored_propagations
        ).to_string()
    )
    error_summary = pd.concat(
    {
        "P4_ACTUAL_PROPAGATION":
            summarize_error_group(normal_propagations),
        "P1_NON_PROPAGATION":
            summarize_error_group(critical_non_events),
        "P3_ACTUAL_PROPAGATION":
            summarize_error_group(monitored_propagations),
    },
    names=[
        "ERROR_GROUP",
        "STATISTIC",
    ],
)

    output_path = (
    PROJECT_ROOT
    / "results"
    / "decision_policy_error_analysis.csv"
)

    error_summary.to_csv(output_path)

    print("\nError analysis saved:")
    print(output_path)

if __name__ == "__main__":
    main()
