"""Evaluate the DSS priority policy on validation graph edges."""

from pathlib import Path

import pandas as pd

from src.decision_support.contracts import (
    FlightDecisionInput,
)
from src.decision_support.assessment_service import (
    assess_flight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORED_EDGES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "scored_tail_edges_2023_validation.parquet"
)

CHAIN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predicted_chain_summary_validation.parquet"
)

def load_validation_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    scored_edges = pd.read_parquet(
        SCORED_EDGES_PATH,
        columns=[
            "SOURCE_FLIGHT_ID",
            "PROPAGATION_PROBABILITY",
            "ACTUAL_PROPAGATION",
            "PREV_ARR_DELAY",
            "TURN_BUFFER",
            "PREV_DELAY_RATIO",
            "PLANNED_TURNAROUND",
        ],
    )

    chain_summary = pd.read_parquet(
        CHAIN_SUMMARY_PATH,
        columns=[
            "START_FLIGHT_ID",
            "EDGE_COUNT",
        ],
    )

    print(
        "Validation samples:",
        f"{len(scored_edges):,}",
    )

    print(
        "Propagation rate:",
        f"{scored_edges['ACTUAL_PROPAGATION'].mean():.4f}",
    )

    decision_columns = [
    "PROPAGATION_PROBABILITY",
    "PREV_ARR_DELAY",
    "TURN_BUFFER",
    "PREV_DELAY_RATIO",
    "PLANNED_TURNAROUND",
]

    print("\nMissing decision inputs:")
    print(scored_edges[decision_columns].isna().sum())

    all_edge_count = len(scored_edges)

    scored_edges = (
    scored_edges
    .dropna(subset=decision_columns)
    .copy()
)

    excluded_edge_count = (
    all_edge_count - len(scored_edges)
)

    print(
    "DSS-evaluable samples:",
    f"{len(scored_edges):,}",
)

    print(
    "Excluded unmatched edges:",
    f"{excluded_edge_count:,}",
)

    return scored_edges, chain_summary

def attach_chain_lengths(
    scored_edges: pd.DataFrame,
    chain_summary: pd.DataFrame,
) -> pd.DataFrame:
    evaluation_data = scored_edges.merge(
        chain_summary,
        left_on="SOURCE_FLIGHT_ID",
        right_on="START_FLIGHT_ID",
        how="left",
        validate="one_to_one",
    )

    evaluation_data["EDGE_COUNT"] = (
        evaluation_data["EDGE_COUNT"]
        .fillna(0)
        .astype(int)
    )

    return evaluation_data

def apply_decision_policy(
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    evaluation_data = evaluation_data.copy()

    assessments = []

    for row in evaluation_data.itertuples(
        index=False
    ):
        flight_input = FlightDecisionInput(
            propagation_probability=float(
                row.PROPAGATION_PROBABILITY
            ),
            previous_arrival_delay=float(
                row.PREV_ARR_DELAY
            ),
            turn_buffer=float(
                row.TURN_BUFFER
            ),
            previous_delay_ratio=float(
                row.PREV_DELAY_RATIO
            ),
            planned_turnaround=float(
                row.PLANNED_TURNAROUND
            ),
            downstream_edge_count=int(
                row.EDGE_COUNT
            ),
        )

        assessments.append(
            assess_flight(flight_input)
        )

    evaluation_data["LIKELIHOOD"] = [
        assessment.likelihood.value
        for assessment in assessments
    ]

    evaluation_data["IMPACT"] = [
        assessment.impact.value
        for assessment in assessments
    ]

    evaluation_data["URGENCY"] = [
        assessment.urgency.value
        for assessment in assessments
    ]

    evaluation_data["PRIORITY"] = [
        assessment.priority.name
        for assessment in assessments
    ]

    return evaluation_data

def summarize_priority_policy(
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    priority_summary = (
        evaluation_data
        .groupby("PRIORITY")[
            "ACTUAL_PROPAGATION"
        ]
        .agg(["count", "sum", "mean"])
    )

    priority_summary = priority_summary.rename(
        columns={
            "count": "FLIGHT_COUNT",
            "sum": "ACTUAL_PROPAGATIONS",
            "mean": "ACTUAL_PROPAGATION_RATE",
        }
    )

    priority_summary[
        "ACTUAL_PROPAGATION_RATE"
    ] *= 100

    additional_metrics = (
        evaluation_data
        .groupby("PRIORITY")
        .agg(
            MEAN_PREDICTED_PROBABILITY=(
                "PROPAGATION_PROBABILITY",
                "mean",
            ),
            MEAN_EDGE_COUNT=(
                "EDGE_COUNT",
                "mean",
            ),
        )
    )

    priority_summary = priority_summary.join(
        additional_metrics
    )
    priority_summary[
        "MEAN_PREDICTED_PROBABILITY"
    ] *= 100
    return priority_summary

def main() -> None:
    scored_edges, chain_summary = (
        load_validation_data()
    )

    print(
        "Scored edge columns:",
        scored_edges.columns.tolist(),
    )

    print(
        "Chain summary columns:",
        chain_summary.columns.tolist(),
    )

    evaluation_data = attach_chain_lengths(
        scored_edges=scored_edges,
        chain_summary=chain_summary,
    )

    evaluation_data = apply_decision_policy(
        evaluation_data
    )

    priority_summary = summarize_priority_policy(
        evaluation_data
    )

    priority_order = [
        "P1_CRITICAL",
        "P2_HIGH",
        "P3_MONITOR",
        "P4_NORMAL",
    ]

    priority_summary = priority_summary.reindex(
        priority_order
    )

    print("\nDSS priority policy evaluation")
    print("-" * 80)
    print(priority_summary.to_string())

    output_path = (
    PROJECT_ROOT
    / "results"
    / "decision_policy_validation.csv"
)

    output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

    priority_summary.to_csv(output_path)

    print("\nDecision-policy results saved:")
    print(output_path)


if __name__ == "__main__":
    main()
