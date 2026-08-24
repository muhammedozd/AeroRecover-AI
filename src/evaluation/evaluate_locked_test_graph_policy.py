"""Evaluate frozen graph and P1-P4 policy logic on the locked test period."""

from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.evaluate_decision_policy import (
    apply_decision_policy, attach_chain_lengths, summarize_priority_policy,
)
from src.evaluation.evaluate_graph_propagation import (
    build_chains, evaluate_chain_matching,
)
from src.graph.multi_hop_propagation import validate_graph_structure
from src.graph.score_graph_edges import add_target_flight_id, attach_scores_to_edges
from src.models.train_rotation_model import MODEL_COLUMNS, create_time_masks, prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROTATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv"
EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "graph" / "tail_edges_2023.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_propagation_2023_time_split.pkl"
GRAPH_OUTPUT_PATH = PROJECT_ROOT / "reports" / "locked_test_graph_evaluation.csv"
PRIORITY_OUTPUT_PATH = PROJECT_ROOT / "reports" / "locked_test_priority_evaluation.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "reports" / "locked_test_graph_evaluation_summary.txt"
TEST_START = pd.Timestamp("2023-11-01")
TEST_END_EXCLUSIVE = pd.Timestamp("2024-01-01")
EXPECTED_TEST_SAMPLE_COUNT = 805_126
OPERATIONAL_THRESHOLD = 0.46
FLIGHT_ID_COLUMNS = [
    "TAIL_NUM", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME",
]
EDGE_COLUMNS = [
    "SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID", "TAIL_NUM", "FL_DATE",
    "CONNECTION_AIRPORT", "PLANNED_CONNECTION_MINUTES", "IS_PROPAGATION_EDGE",
]
DECISION_COLUMNS = [
    "PROPAGATION_PROBABILITY", "PREV_ARR_DELAY", "TURN_BUFFER",
    "PREV_DELAY_RATIO", "PLANNED_TURNAROUND",
]
PRIORITY_ORDER = ["P1_CRITICAL", "P2_HIGH", "P3_MONITOR", "P4_NORMAL"]


def require_columns(actual: list[str], required: list[str], source: str) -> None:
    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_locked_test_rotations() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    required = list(dict.fromkeys([*MODEL_COLUMNS, *FLIGHT_ID_COLUMNS]))
    header = pd.read_csv(ROTATION_DATA_PATH, nrows=0)
    require_columns(header.columns.tolist(), required, str(ROTATION_DATA_PATH))
    chunks = []
    for chunk in pd.read_csv(
        ROTATION_DATA_PATH, usecols=required, chunksize=250_000, low_memory=False,
    ):
        _, _, test_mask = create_time_masks(chunk)
        if test_mask.any():
            chunks.append(chunk.loc[test_mask].copy())
    if not chunks:
        raise ValueError("No November-December 2023 locked-test rotations found.")
    rotations = pd.concat(chunks, ignore_index=True)
    dates = pd.to_datetime(rotations["FL_DATE"], format="%Y-%m-%d", errors="raise")
    if len(rotations) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"Unexpected locked-test sample count: {len(rotations):,}; expected "
            f"{EXPECTED_TEST_SAMPLE_COUNT:,}. No metrics were produced."
        )
    if not dates.ge(TEST_START).all() or not dates.lt(TEST_END_EXCLUSIVE).all():
        raise ValueError("Locked-test rotations contain dates outside November-December 2023.")
    X_test, y_test, _, _ = prepare_features(rotations)
    return rotations, X_test, y_test


def load_test_edges() -> pd.DataFrame:
    require_columns(pq.ParquetFile(EDGES_PATH).schema.names, EDGE_COLUMNS, str(EDGES_PATH))
    edges = pd.read_parquet(
        EDGES_PATH, columns=EDGE_COLUMNS,
        filters=[("FL_DATE", ">=", TEST_START), ("FL_DATE", "<", TEST_END_EXCLUSIVE)],
    )
    dates = pd.to_datetime(edges["FL_DATE"], errors="raise")
    if edges.empty or not dates.ge(TEST_START).all() or not dates.lt(TEST_END_EXCLUSIVE).all():
        raise ValueError("Physical edge filter did not produce only November-December 2023 rows.")
    return edges


def score_rotations(rotations: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features and model_features != X_test.columns.tolist():
        raise ValueError("Frozen model feature schema does not match test features. No metrics were produced.")
    probabilities = model.predict_proba(X_test)[:, 1]
    scored = rotations[[
        "TARGET_FLIGHT_ID", "PREV_ARR_DELAY", "TURN_BUFFER",
        "PREV_DELAY_RATIO", "PLANNED_TURNAROUND",
    ]].copy()
    scored["PROPAGATION_PROBABILITY"] = probabilities
    scored["PROPAGATION_ALERT"] = (probabilities >= OPERATIONAL_THRESHOLD).astype(int)
    scored["ACTUAL_PROPAGATION"] = y_test.to_numpy()
    return scored


def metric_table(metrics: dict[str, int | float | str]) -> pd.DataFrame:
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])


def main() -> None:
    rotations, X_test, y_test = load_locked_test_rotations()
    rotations = add_target_flight_id(rotations)
    if rotations["TARGET_FLIGHT_ID"].duplicated().any():
        raise ValueError("Locked-test rotations contain duplicate target flight IDs.")
    physical_edges = load_test_edges()
    eligible_edges = physical_edges.loc[physical_edges["IS_PROPAGATION_EDGE"].eq(1)].copy()
    scored_edges_all = attach_scores_to_edges(
        eligible_edges, score_rotations(rotations, X_test, y_test),
    )
    scored_edges = scored_edges_all.dropna(subset=["PROPAGATION_PROBABILITY"]).copy()
    if len(scored_edges) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"Unexpected scored test-edge count: {len(scored_edges):,}; expected "
            f"{EXPECTED_TEST_SAMPLE_COUNT:,}. No metrics were produced."
        )
    require_columns(scored_edges.columns.tolist(), DECISION_COLUMNS, "scored test edges")
    missing_inputs = scored_edges[DECISION_COLUMNS].isna().sum()
    if missing_inputs.any():
        raise ValueError(f"Missing locked-test decision inputs:\n{missing_inputs[missing_inputs.gt(0)]}")
    validate_graph_structure(scored_edges)

    predicted_chains = build_chains(scored_edges, "PROPAGATION_ALERT")
    observed_chains = build_chains(scored_edges, "ACTUAL_PROPAGATION")
    matching_metrics, _ = evaluate_chain_matching(predicted_chains, observed_chains)
    graph_metrics: dict[str, int | float | str] = {
        "evaluation_period": "2023-11-01/2023-12-31",
        "threshold": OPERATIONAL_THRESHOLD,
        "locked_test_samples": len(rotations),
        "physical_test_edges": len(physical_edges),
        "propagation_eligible_test_edges": len(eligible_edges),
        "scored_test_edges": len(scored_edges),
        "edge_score_coverage": len(scored_edges) / len(physical_edges),
        "eligible_edge_score_coverage": len(scored_edges) / len(eligible_edges),
        "observed_positive_edges": int(scored_edges["ACTUAL_PROPAGATION"].sum()),
        "predicted_positive_edges_at_tau_0_46": int(scored_edges["PROPAGATION_ALERT"].sum()),
        "predicted_chain_summaries": len(predicted_chains),
        "observed_chain_summaries": len(observed_chains),
        **matching_metrics,
    }
    matched_count = int(graph_metrics["correctly_matched_chain_count"])
    graph_metrics["underprediction_rate_conditional_on_matched_starts"] = (
        int(graph_metrics["underestimated_matched_chains"]) / matched_count
        if matched_count else 0.0
    )
    graph_metrics["overprediction_rate_conditional_on_matched_starts"] = (
        int(graph_metrics["overestimated_matched_chains"]) / matched_count
        if matched_count else 0.0
    )

    evaluation_data = apply_decision_policy(attach_chain_lengths(scored_edges, predicted_chains))
    priority_summary = summarize_priority_policy(evaluation_data).reindex(PRIORITY_ORDER)
    if priority_summary["FLIGHT_COUNT"].isna().any():
        missing = priority_summary.index[priority_summary["FLIGHT_COUNT"].isna()].tolist()
        raise ValueError(f"Locked test produced no samples for priorities: {missing}")
    priority_output = priority_summary.reset_index().rename(columns={
        "PRIORITY": "priority", "FLIGHT_COUNT": "sample_count",
        "ACTUAL_PROPAGATIONS": "observed_positive_count",
        "ACTUAL_PROPAGATION_RATE": "observed_propagation_rate_percent",
        "MEAN_PREDICTED_PROBABILITY": "mean_predicted_probability_percent",
        "MEAN_EDGE_COUNT": "mean_downstream_edge_count",
    })[["priority", "sample_count", "observed_positive_count",
        "observed_propagation_rate_percent", "mean_predicted_probability_percent",
        "mean_downstream_edge_count"]]
    priority_output["sample_count"] = priority_output["sample_count"].astype(int)
    priority_output["observed_positive_count"] = priority_output[
        "observed_positive_count"
    ].astype(int)

    GRAPH_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph_output = metric_table(graph_metrics)
    graph_output.to_csv(GRAPH_OUTPUT_PATH, index=False)
    priority_output.to_csv(PRIORITY_OUTPUT_PATH, index=False)
    summary = (
        "LOCKED TEST GRAPH AND P1-P4 POLICY EVALUATION\n"
        "================================================\n"
        "Frozen model: xgboost_propagation_2023_time_split.pkl\n"
        "Fixed threshold: tau=0.46\n"
        "Period verified: 2023-11-01 through 2023-12-31 (inclusive)\n"
        f"Sample count verified: {len(rotations):,}\n"
        "No model retraining or test-set tuning was performed.\n"
        "Edge-score coverage uses all physical test edges as its denominator; "
        "eligible-edge coverage is also reported separately.\n"
        "Chain-start matching metrics use multi-hop chains (EDGE_COUNT >= 2); "
        "all predicted/observed chain-summary counts are also reported.\n"
        "Conditional exact-length rate is calculated only over correctly matched chain starts.\n\n"
        f"GRAPH METRICS\n-------------\n{graph_output.to_string(index=False)}\n\n"
        f"P1-P4 POLICY METRICS\n---------------------\n{priority_output.to_string(index=False)}\n"
    )
    SUMMARY_OUTPUT_PATH.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print("Outputs:")
    for path in (GRAPH_OUTPUT_PATH, PRIORITY_OUTPUT_PATH, SUMMARY_OUTPUT_PATH):
        print(path)


if __name__ == "__main__":
    main()
