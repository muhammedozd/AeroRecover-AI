"""Attach model propagation probabilities to graph edges."""

from pathlib import Path

import joblib
import pandas as pd

from src.models.train_rotation_model import (
    MODEL_COLUMNS,
    create_time_masks,
    prepare_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROTATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset_2023.csv"
)

EDGES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "tail_edges_2023.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_propagation_2023_time_split.pkl"
)

SCORED_EDGES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "scored_tail_edges_2023_validation.parquet"
)

F1_OPTIMAL_THRESHOLD = 0.46

FLIGHT_ID_COLUMNS = [
    "FL_DATE",
    "TAIL_NUM",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
]

#   *, listenin içindeki elemanları dışarı çıkarır.
#dict.fromkeys tekrarlayan elemanları kaldırır
def load_validation_rotations():
    required_columns = list(
        dict.fromkeys(
            [
                *MODEL_COLUMNS,
                *FLIGHT_ID_COLUMNS,
            ]
        )
    )

    data = pd.read_csv(
        ROTATION_DATA_PATH,
        usecols=required_columns,
    )

    X, y, _, _ = prepare_features(data)

    _, validation_mask, _ = create_time_masks(
        data
    )

    validation_rotations = data.loc[
        validation_mask
    ].copy()

    X_validation = X.loc[
        validation_mask
    ]

    y_validation = y.loc[
        validation_mask
    ]

    print(
        "Validation rotations:",
        f"{len(validation_rotations):,}",
    )

    return (
        validation_rotations,
        X_validation,
        y_validation,
    )


def add_target_flight_id(
    rotations: pd.DataFrame,
) -> pd.DataFrame:
    rotations = rotations.copy()

    flight_dates = pd.to_datetime(
        rotations["FL_DATE"],
        format="%Y-%m-%d",
    )

    flight_numbers = (
        rotations["OP_CARRIER_FL_NUM"]
        .astype(int)
        .astype(str)
    )

    departure_times = (
        rotations["CRS_DEP_TIME"]
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )

    rotations["TARGET_FLIGHT_ID"] = (
        flight_dates.dt.strftime("%Y%m%d")
        + "_"
        + rotations["OP_UNIQUE_CARRIER"].astype(str)
        + "_"
        + flight_numbers
        + "_"
        + rotations["ORIGIN"].astype(str)
        + "_"
        + rotations["DEST"].astype(str)
        + "_"
        + departure_times
        + "_"
        + rotations["TAIL_NUM"].astype(str)
    )

    duplicate_count = (
        rotations["TARGET_FLIGHT_ID"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate target flight IDs:",
        f"{duplicate_count:,}",
    )

    return rotations


def score_validation_rotations(
    rotations: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    model = joblib.load(
        MODEL_PATH
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    scored_rotations = rotations[
        [
            "TARGET_FLIGHT_ID",
        ]
    ].copy()

    scored_rotations[
        "PROPAGATION_PROBABILITY"
    ] = probabilities

    scored_rotations[
        "PROPAGATION_ALERT"
    ] = (
        probabilities
        >= F1_OPTIMAL_THRESHOLD
    ).astype(int)

    scored_rotations[
        "ACTUAL_PROPAGATION"
    ] = y_validation.to_numpy()

    print(
        "Scored validation rotations:",
        f"{len(scored_rotations):,}",
    )

    print(
        "Mean propagation probability:",
        f"{probabilities.mean():.4f}",
    )

    print(
        "Propagation alerts:",
        f"{scored_rotations['PROPAGATION_ALERT'].sum():,}",
    )

    return scored_rotations


def load_validation_edges() -> pd.DataFrame:
    edge_columns = [
        "SOURCE_FLIGHT_ID",
        "TARGET_FLIGHT_ID",
        "TAIL_NUM",
        "FL_DATE",
        "CONNECTION_AIRPORT",
        "PLANNED_CONNECTION_MINUTES",
        "IS_PROPAGATION_EDGE",
    ]

    validation_edges = pd.read_parquet(
        EDGES_PATH,
        columns=edge_columns,
        filters=[
            (
                "FL_DATE",
                ">=",
                pd.Timestamp("2023-09-01"),
            ),
            (
                "FL_DATE",
                "<",
                pd.Timestamp("2023-11-01"),
            ),
            (
                "IS_PROPAGATION_EDGE",
                "==",
                1,
            ),
        ],
    )

    print(
        "Validation graph edges:",
        f"{len(validation_edges):,}",
    )

    return validation_edges

#on Pandas’a iki tabloyu hangi ortak sütuna göre birleştireceğini söyler.
#how=left sol taraftaki tabloyu korur.
def attach_scores_to_edges(
    validation_edges: pd.DataFrame,
    scored_rotations: pd.DataFrame,
) -> pd.DataFrame:
    scored_edges = validation_edges.merge(
        scored_rotations,
        on="TARGET_FLIGHT_ID",
        how="left",
        validate="one_to_one",
    )

    scored_edges["MODEL_VERSION"] = (
        "xgboost_2023_time_split"
    )

    matched_count = (
        scored_edges[
            "PROPAGATION_PROBABILITY"
        ]
        .notna()
        .sum()
    )

    unmatched_count = (
        scored_edges[
            "PROPAGATION_PROBABILITY"
        ]
        .isna()
        .sum()
    )

    match_rate = (
        matched_count
        / len(scored_edges)
        * 100
    )

    print(
        "Matched scored edges:",
        f"{matched_count:,}",
    )

    print(
        "Unmatched edges:",
        f"{unmatched_count:,}",
    )

    print(
        "Edge score match rate:",
        f"{match_rate:.2f}%",
    )

    return scored_edges


def save_scored_edges(
    scored_edges: pd.DataFrame,
) -> None:
    scored_edges.to_parquet(
        SCORED_EDGES_PATH,
        index=False,
        compression="snappy",
    )

    print(
        "Scored graph edges saved:",
        SCORED_EDGES_PATH,
    )

def main():
    (
        validation_rotations,
        X_validation,
        y_validation,
    ) = load_validation_rotations()

    validation_rotations = (
        add_target_flight_id(
            validation_rotations
        )
    )

    scored_rotations = (
        score_validation_rotations(
            validation_rotations,
            X_validation,
            y_validation,
        )
    )

    validation_edges = (
        load_validation_edges()
    )

    scored_edges = attach_scores_to_edges(
        validation_edges,
        scored_rotations,
    )

    print("\nScored graph edge samples")
    print("-" * 100)

    print(
        scored_edges[
            [
                "SOURCE_FLIGHT_ID",
                "TARGET_FLIGHT_ID",
                "PLANNED_CONNECTION_MINUTES",
                "PROPAGATION_PROBABILITY",
                "PROPAGATION_ALERT",
                "ACTUAL_PROPAGATION",
            ]
        ]
        .dropna(
            subset=[
                "PROPAGATION_PROBABILITY"
            ]
        )
        .head(10)
        .to_string(index=False)
    )

    
    save_scored_edges(
    scored_edges
)


if __name__ == "__main__":
    main()