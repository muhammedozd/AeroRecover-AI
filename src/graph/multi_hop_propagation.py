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


def find_chain_starts(
    edges: pd.DataFrame,
    signal_column: str,
) -> list[str]:
    active_edges = edges[
        edges[signal_column] == 1
    ].copy()

    active_target_ids = set(
        active_edges["TARGET_FLIGHT_ID"]
    )

    chain_starts = active_edges.loc[
        ~active_edges["SOURCE_FLIGHT_ID"].isin(
            active_target_ids
        ),
        "SOURCE_FLIGHT_ID",
    ].tolist()

    return chain_starts


def calculate_chain_length(
    edge_lookup: pd.DataFrame,
    start_flight_id: str,
    max_hops: int = 20,
):
    current_flight_id = start_flight_id
    edge_count = 0
    cumulative_probability = 1.0

    while (
        edge_count < max_hops
        and current_flight_id in edge_lookup.index
    ):
        current_edge = edge_lookup.loc[
            current_flight_id
        ]

        cumulative_probability *= float(
            current_edge[
                "PROPAGATION_PROBABILITY"
            ]
        )

        edge_count += 1

        current_flight_id = current_edge[
            "TARGET_FLIGHT_ID"
        ]

    return (
        edge_count,
        cumulative_probability,
        current_flight_id,
    )

def build_chain_summary(
    edge_lookup: pd.DataFrame,
    chain_starts: list[str],
) -> pd.DataFrame:
    chain_rows = []

    for start_flight_id in chain_starts:
        (
            edge_count,
            cumulative_probability,
            end_flight_id,
        ) = calculate_chain_length(
            edge_lookup=edge_lookup,
            start_flight_id=start_flight_id,
        )

        chain_rows.append({
            "START_FLIGHT_ID": start_flight_id,
            "END_FLIGHT_ID": end_flight_id,
            "EDGE_COUNT": edge_count,
            "FLIGHT_COUNT": edge_count + 1,
            "CUMULATIVE_PROBABILITY":
                cumulative_probability,
        })

    return pd.DataFrame(chain_rows)

def main():
    scored_edges = load_scored_edges()

    validate_graph_structure(
        scored_edges
    )

    predicted_chain_starts = find_chain_starts(
        edges=scored_edges,
        signal_column="PROPAGATION_ALERT",
    )

    actual_chain_starts = find_chain_starts(
        edges=scored_edges,
        signal_column="ACTUAL_PROPAGATION",
    )
    

    print(
        "Predicted chain starts:",
        f"{len(predicted_chain_starts):,}",
    )

    predicted_edges = scored_edges[
        scored_edges["PROPAGATION_ALERT"] == 1
    ].copy()

    predicted_edge_lookup = (
        predicted_edges.set_index(
            "SOURCE_FLIGHT_ID"
        )
    )

    predicted_chains = build_chain_summary(
    edge_lookup=predicted_edge_lookup,
    chain_starts=predicted_chain_starts,
)

    predicted_length_distribution = (
    predicted_chains["EDGE_COUNT"]
    .value_counts()
    .sort_index()
)

    print("\nPredicted chain length distribution")
    print("-" * 60)
    print(predicted_length_distribution)

    actual_edges = scored_edges[
    scored_edges["ACTUAL_PROPAGATION"] == 1
].copy()

    actual_edge_lookup = actual_edges.set_index(
    "SOURCE_FLIGHT_ID"
)

    actual_chains = build_chain_summary(
    edge_lookup=actual_edge_lookup,
    chain_starts=actual_chain_starts,
)

    actual_length_distribution = (
    actual_chains["EDGE_COUNT"]
    .value_counts()
    .sort_index()
)

    print("\nActual chain length distribution")
    print("-" * 60)
    print(actual_length_distribution)

    predicted_domino_starts = set(
    predicted_chains.loc[
        predicted_chains["EDGE_COUNT"] >= 2,
        "START_FLIGHT_ID",
    ]
)

    actual_domino_starts = set(
    actual_chains.loc[
        actual_chains["EDGE_COUNT"] >= 2,
        "START_FLIGHT_ID",
    ]
)

    print(
    "Actual chain starts:",
    f"{len(actual_chain_starts):,}",
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