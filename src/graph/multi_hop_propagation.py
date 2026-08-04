"""Trace multi-hop delay propagation across flight graph edges."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORED_EDGES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "scored_tail_edges_2023_validation.parquet"
)

F1_OPTIMAL_THRESHOLD = 0.46

def load_scored_edges() -> pd.DataFrame:
    edge_columns = [
        "SOURCE_FLIGHT_ID",
        "TARGET_FLIGHT_ID",
        "TAIL_NUM",
        "FL_DATE",
        "CONNECTION_AIRPORT",
        "PLANNED_CONNECTION_MINUTES",
        "PROPAGATION_PROBABILITY",
        "PROPAGATION_ALERT",
        "ACTUAL_PROPAGATION",
    ]

    edges = pd.read_parquet(
        SCORED_EDGES_PATH,
        columns=edge_columns,
    )

    initial_count = len(edges)

    edges = edges.dropna(
        subset=[
            "PROPAGATION_PROBABILITY",
        ]
    ).copy()

    print(
        "All validation edges:",
        f"{initial_count:,}",
    )

    print(
        "Scored validation edges:",
        f"{len(edges):,}",
    )

    return edges


def validate_graph_structure(
    edges: pd.DataFrame,
) -> None:
    duplicate_sources = (
        edges["SOURCE_FLIGHT_ID"]
        .duplicated()
        .sum()
    )

    duplicate_targets = (
        edges["TARGET_FLIGHT_ID"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate source flights:",
        f"{duplicate_sources:,}",
    )

    print(
        "Duplicate target flights:",
        f"{duplicate_targets:,}",
    )

    if duplicate_sources > 0:
        raise ValueError(
            "A source flight has multiple outgoing "
            "tail connections."
        )

def trace_domino_path(
    edges: pd.DataFrame,
    start_flight_id: str,
    max_hops: int = 5,
) -> pd.DataFrame:
    edge_lookup = edges.set_index(
        "SOURCE_FLIGHT_ID"
    )

    current_flight_id = start_flight_id
    cumulative_probability = 1.0
    path_rows = []

    for hop in range(
        1,
        max_hops + 1,
    ):
        if current_flight_id not in edge_lookup.index:
            break

        current_edge = edge_lookup.loc[
            current_flight_id
        ]

        target_flight_id = current_edge[
            "TARGET_FLIGHT_ID"
        ]

        local_probability = float(
            current_edge[
                "PROPAGATION_PROBABILITY"
            ]
        )

        cumulative_probability *= (
            local_probability
        )

        path_rows.append(
            {
                "HOP": hop,
                "SOURCE_FLIGHT_ID": current_flight_id,
                "TARGET_FLIGHT_ID": target_flight_id,
                "LOCAL_PROBABILITY": local_probability,
                "CUMULATIVE_PROBABILITY": cumulative_probability,
                "PROPAGATION_ALERT": current_edge[
                    "PROPAGATION_ALERT"
                ],
                "ACTUAL_PROPAGATION": current_edge[
                    "ACTUAL_PROPAGATION"
                ],
                "CONNECTION_AIRPORT": current_edge[
                    "CONNECTION_AIRPORT"
                ],
                "PLANNED_CONNECTION_MINUTES": current_edge[
                    "PLANNED_CONNECTION_MINUTES"
                ],
            }
        )

        current_flight_id = target_flight_id

    return pd.DataFrame(
        path_rows
    )

def main():
    scored_edges = load_scored_edges()

    validate_graph_structure(
        scored_edges
    )

    start_flight_id = (
        "20230904_G4_218_DEN_AVL_1038_190NV"
    )

    domino_path = trace_domino_path(
        edges=scored_edges,
        start_flight_id=start_flight_id,
        max_hops=5,
    )

    print("\nDomino propagation path")
    print("-" * 100)

    print(
        domino_path.to_string(
            index=False
        )
    )

if __name__ == "__main__":
    main()