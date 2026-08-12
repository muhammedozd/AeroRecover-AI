"""Build the temporal aircraft rotation graph."""

from pathlib import Path

import pandas as pd

from src.data.load_flights import load_flights

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
)

NODES_PATH = (
    GRAPH_OUTPUT_DIR
    / "flight_nodes_2023.csv"
)

EDGES_PATH = (
    GRAPH_OUTPUT_DIR
    / "tail_edges_2023.parquet"
)


GRAPH_COLUMNS = [
    "FL_DATE",
    "TAIL_NUM",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "DEP_DELAY",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
]

def load_graph_flights() -> pd.DataFrame:
    flights = load_flights(
        columns=GRAPH_COLUMNS,
    )

    initial_count = len(flights)

    flights = flights.dropna(
        subset=[
            "FL_DATE",
            "TAIL_NUM",
            "OP_UNIQUE_CARRIER",
            "OP_CARRIER_FL_NUM",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
        ]
    ).copy()

    flights = flights[
        (flights["CANCELLED"] == 0)
        & (flights["DIVERTED"] == 0)
    ].copy()

    flights["FL_DATE"] = pd.to_datetime(
        flights["FL_DATE"],
        format="%m/%d/%Y %I:%M:%S %p",
    )

    print(f"Initial flights: {initial_count:,}")
    print(f"Valid graph flights: {len(flights):,}")

    return flights

def build_flight_nodes(
    flights: pd.DataFrame,
) -> pd.DataFrame:
    nodes = flights.copy()

    nodes["CRS_DEP_TIME"] = (
        nodes["CRS_DEP_TIME"]
        .astype(int)
    )

    nodes["FLIGHT_NUMBER"] = (
        nodes["OP_CARRIER_FL_NUM"]
        .astype(int)
        .astype(str)
    )

    nodes["DEP_TIME_TEXT"] = (
        nodes["CRS_DEP_TIME"]
        .astype(str)
        .str.zfill(4)
    )

    nodes["FLIGHT_ID"] = (
    nodes["FL_DATE"].dt.strftime("%Y%m%d")
    + "_"
    + nodes["OP_UNIQUE_CARRIER"].astype(str)
    + "_"
    + nodes["FLIGHT_NUMBER"]
    + "_"
    + nodes["ORIGIN"].astype(str)
    + "_"
    + nodes["DEST"].astype(str)
    + "_"
    + nodes["DEP_TIME_TEXT"]
    + "_"
    + nodes["TAIL_NUM"].astype(str)
)

    duplicate_count = (
    nodes["FLIGHT_ID"]
    .duplicated()
    .sum()
)

    print(f"Flight nodes: {len(nodes):,}")
    print(
    "Duplicate flight IDs: "
    f"{duplicate_count:,}"
)
    print(
    "Duplicate flight IDs: "
    f"{duplicate_count:,}"
)

    return nodes

def hhmm_to_minutes(
    time_values: pd.Series,
) -> pd.Series:
    integer_times = time_values.astype(int)

    hours = integer_times // 100
    minutes = integer_times % 100

    return hours * 60 + minutes

def build_tail_edges(
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    ordered_nodes = nodes.sort_values(
        by=[
            "TAIL_NUM",
            "FL_DATE",
            "CRS_DEP_TIME",
        ]
    ).copy()

    tail_groups = ordered_nodes.groupby(
        [
            "TAIL_NUM",
            "FL_DATE",
        ],
        sort=False,
    )

    ordered_nodes["NEXT_FLIGHT_ID"] = (
        tail_groups["FLIGHT_ID"]
        .shift(-1)
    )

    ordered_nodes["NEXT_ORIGIN"] = (
        tail_groups["ORIGIN"]
        .shift(-1)
    )

    ordered_nodes["NEXT_CRS_DEP_TIME"] = (
    tail_groups["CRS_DEP_TIME"]
    .shift(-1)
)

    valid_connection_mask = (
    ordered_nodes["NEXT_FLIGHT_ID"].notna()
    & ordered_nodes["DEST"].eq(
        ordered_nodes["NEXT_ORIGIN"]
    )
)
    edges = ordered_nodes.loc[
    valid_connection_mask,
    [
        "FLIGHT_ID",
        "NEXT_FLIGHT_ID",
        "TAIL_NUM",
        "FL_DATE",
        "DEST",
        "CRS_ARR_TIME",
        "NEXT_CRS_DEP_TIME",
    ],
].copy()

    edges = edges.rename(
    columns={
        "FLIGHT_ID": "SOURCE_FLIGHT_ID",
        "NEXT_FLIGHT_ID": "TARGET_FLIGHT_ID",
        "DEST": "CONNECTION_AIRPORT",
        "CRS_ARR_TIME": "SOURCE_CRS_ARR_TIME",
        "NEXT_CRS_DEP_TIME": "TARGET_CRS_DEP_TIME",
    }
)

    edges["SOURCE_ARR_MIN"] = hhmm_to_minutes(
    edges["SOURCE_CRS_ARR_TIME"]
)

    edges["TARGET_DEP_MIN"] = hhmm_to_minutes(
    edges["TARGET_CRS_DEP_TIME"]
)

    edges["PLANNED_CONNECTION_MINUTES"] = (
    edges["TARGET_DEP_MIN"]
    - edges["SOURCE_ARR_MIN"]
)

# Önce bütün aday edge'lerin istatistiklerini göster
    print("\nPlanned connection statistics")
    print("-" * 60)

    print(
    edges["PLANNED_CONNECTION_MINUTES"]
    .describe()
)

    print(
    "Non-positive connections:",
    (
        edges["PLANNED_CONNECTION_MINUTES"]
        <= 0
    ).sum(),
)

    print(
    "Connections longer than 240 minutes:",
    (
        edges["PLANNED_CONNECTION_MINUTES"]
        > 240
    ).sum(),
)

# Yalnızca zamansal olarak geçerli edge'leri tut
    edges = edges[
    edges["PLANNED_CONNECTION_MINUTES"] > 0
].copy()

# Model kullanımına uygun olanları işaretle
    edges["IS_PROPAGATION_EDGE"] = (
    edges["PLANNED_CONNECTION_MINUTES"]
    .between(1, 240)
    .astype(int)
)

    edges["EDGE_TYPE"] = "TAIL_CONNECTION"

    print(
    "Retained physical tail edges:",
    f"{len(edges):,}",
)

    print(
    "Propagation-eligible edges:",
    f"{edges['IS_PROPAGATION_EDGE'].sum():,}",
)

    return edges

#Snappy, büyük veri tablolarında sık kullanılan
#hızlı ve kayıpsız bir sıkıştırma algoritmasıdır.
def save_tail_edges(
    edges: pd.DataFrame,
) -> None:
    GRAPH_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edges.to_parquet(
        EDGES_PATH,
        index=False,
        compression="snappy",
    )

    print(f"Tail edges saved: {EDGES_PATH}")


if __name__ == "__main__":
    graph_flights = load_graph_flights()

    flight_nodes = build_flight_nodes(
        graph_flights
    )

    tail_edges = build_tail_edges(
        flight_nodes
    )

    print("\nTail edge samples")
    print("-" * 80)

    print(
        tail_edges[
            [
                "SOURCE_FLIGHT_ID",
                "TARGET_FLIGHT_ID",
                "TAIL_NUM",
                "CONNECTION_AIRPORT",
                "SOURCE_CRS_ARR_TIME",
                "TARGET_CRS_DEP_TIME",
            ]
        ].head(10)
    )

    print(
        "Edge table shape:",
        tail_edges.shape,
    )

    save_tail_edges(
    tail_edges
)
