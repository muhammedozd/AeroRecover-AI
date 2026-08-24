"""Attach full-enhanced propagation scores to validation graph edges."""

import json

import pandas as pd

from src.features.rotation_features import build_rotation_model_features
from src.models.rotation_model_contract import (
    MODEL_THRESHOLD, MODEL_VERSION, PROJECT_ROOT, RAW_FEATURE_COLUMNS,
    load_model_pipeline,
)

ROTATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv"
EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "graph" / "tail_edges_2023.parquet"
SCORED_EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "graph" / "scored_tail_edges_2023_validation_full_enhanced.parquet"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "full_enhanced_graph_scoring_summary.json"
TARGET = "IS_DELAY_PROPAGATED"
FLIGHT_ID_COLUMNS = [
    "FL_DATE", "TAIL_NUM", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM",
    "ORIGIN", "DEST", "CRS_DEP_TIME",
]


def load_validation_rotations():
    required = list(dict.fromkeys([*RAW_FEATURE_COLUMNS, *FLIGHT_ID_COLUMNS, TARGET]))
    data = pd.read_csv(ROTATION_DATA_PATH, usecols=required, low_memory=False)
    features, dates = build_rotation_model_features(data)
    validation_mask = dates.ge("2023-09-01") & dates.lt("2023-11-01")
    rotations = data.loc[validation_mask].copy()
    X_validation = features.loc[validation_mask]
    y_validation = data.loc[validation_mask, TARGET]
    if not rotations.index.equals(X_validation.index) or not rotations.index.equals(y_validation.index):
        raise ValueError("Validation rotations, features, and target use different row indices.")
    return rotations, X_validation, y_validation


def add_target_flight_id(rotations: pd.DataFrame) -> pd.DataFrame:
    result = rotations.copy()
    dates = pd.to_datetime(result["FL_DATE"], format="%Y-%m-%d", errors="raise")
    flight_numbers = result["OP_CARRIER_FL_NUM"].astype(int).astype(str)
    departure_times = result["CRS_DEP_TIME"].astype(int).astype(str).str.zfill(4)
    result["TARGET_FLIGHT_ID"] = (
        dates.dt.strftime("%Y%m%d") + "_" + result["OP_UNIQUE_CARRIER"].astype(str)
        + "_" + flight_numbers + "_" + result["ORIGIN"].astype(str)
        + "_" + result["DEST"].astype(str) + "_" + departure_times
        + "_" + result["TAIL_NUM"].astype(str)
    )
    duplicates = int(result["TARGET_FLIGHT_ID"].duplicated().sum())
    if duplicates:
        raise ValueError(f"Duplicate validation TARGET_FLIGHT_ID values: {duplicates}")
    return result


def score_validation_rotations(rotations, features, target):
    model = load_model_pipeline()
    probabilities = model.predict_proba(features)[:, 1]
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Graph scoring produced probabilities outside [0, 1].")
    scored = rotations[
        ["TARGET_FLIGHT_ID", "PREV_ARR_DELAY", "TURN_BUFFER",
         "PREV_DELAY_RATIO", "PLANNED_TURNAROUND"]
    ].copy()
    scored["PROPAGATION_PROBABILITY"] = probabilities
    scored["PROPAGATION_ALERT"] = (probabilities >= MODEL_THRESHOLD).astype("int8")
    scored["ACTUAL_PROPAGATION"] = target.to_numpy()
    return scored


def load_validation_edges() -> pd.DataFrame:
    columns = [
        "SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID", "TAIL_NUM", "FL_DATE",
        "CONNECTION_AIRPORT", "PLANNED_CONNECTION_MINUTES", "IS_PROPAGATION_EDGE",
    ]
    return pd.read_parquet(
        EDGES_PATH, columns=columns,
        filters=[
            ("FL_DATE", ">=", pd.Timestamp("2023-09-01")),
            ("FL_DATE", "<", pd.Timestamp("2023-11-01")),
            ("IS_PROPAGATION_EDGE", "==", 1),
        ],
    )


def attach_scores_to_edges(validation_edges, scored_rotations):
    duplicate_source_ids = int(validation_edges["SOURCE_FLIGHT_ID"].duplicated().sum())
    duplicate_target_ids = int(validation_edges["TARGET_FLIGHT_ID"].duplicated().sum())
    duplicate_edge_ids = int(validation_edges.duplicated(["SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID"]).sum())
    if duplicate_source_ids or duplicate_target_ids or duplicate_edge_ids:
        raise ValueError(
            "Duplicate graph IDs found: "
            f"source={duplicate_source_ids}, target={duplicate_target_ids}, edge={duplicate_edge_ids}."
        )
    scored_edges = validation_edges.merge(
        scored_rotations, on="TARGET_FLIGHT_ID", how="left", validate="one_to_one"
    )
    scored_edges["MODEL_VERSION"] = MODEL_VERSION
    scored_edges["MODEL_THRESHOLD"] = MODEL_THRESHOLD
    matched = int(scored_edges["PROPAGATION_PROBABILITY"].notna().sum())
    missing = int(scored_edges["PROPAGATION_PROBABILITY"].isna().sum())
    summary = {
        "model_version": MODEL_VERSION,
        "threshold": MODEL_THRESHOLD,
        "validation_edge_count": int(len(scored_edges)),
        "matched_score_count": matched,
        "missing_score_count": missing,
        "match_rate": matched / len(scored_edges) if len(scored_edges) else 0.0,
        "duplicate_source_flight_ids": duplicate_source_ids,
        "duplicate_target_flight_ids": duplicate_target_ids,
        "duplicate_edge_pairs": duplicate_edge_ids,
    }
    return scored_edges, summary


def main():
    rotations, features, target = load_validation_rotations()
    rotations = add_target_flight_id(rotations)
    scored_rotations = score_validation_rotations(rotations, features, target)
    scored_edges, summary = attach_scores_to_edges(load_validation_edges(), scored_rotations)
    SCORED_EDGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_edges.to_parquet(SCORED_EDGES_PATH, index=False, compression="snappy")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Scored graph edges saved:", SCORED_EDGES_PATH)


if __name__ == "__main__":
    main()
