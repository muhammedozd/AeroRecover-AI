"""Build and evaluate full-enhanced validation propagation chains."""

import json

import pandas as pd

from src.models.rotation_model_contract import MODEL_THRESHOLD, MODEL_VERSION, PROJECT_ROOT


SCORED_EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "graph" / "scored_tail_edges_2023_validation_full_enhanced.parquet"
PREDICTED_CHAINS_PATH = PROJECT_ROOT / "data" / "processed" / "predicted_chains_validation_full_enhanced.parquet"
PREDICTED_CHAIN_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "predicted_chain_summary_validation_full_enhanced.parquet"
METRICS_PATH = PROJECT_ROOT / "reports" / "full_enhanced_multi_hop_validation_metrics.json"
EDGE_COLUMNS = [
    "SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID", "TAIL_NUM", "FL_DATE",
    "CONNECTION_AIRPORT", "PLANNED_CONNECTION_MINUTES",
    "PROPAGATION_PROBABILITY", "PROPAGATION_ALERT", "ACTUAL_PROPAGATION",
]


def load_scored_edges() -> pd.DataFrame:
    edges = pd.read_parquet(SCORED_EDGES_PATH, columns=EDGE_COLUMNS)
    initial_count = len(edges)
    edges = edges.dropna(subset=["PROPAGATION_PROBABILITY"]).copy()
    print(f"Scored validation edges: {len(edges):,}/{initial_count:,}")
    return edges


def validate_graph_structure(edges: pd.DataFrame) -> None:
    duplicate_sources = int(edges["SOURCE_FLIGHT_ID"].duplicated().sum())
    duplicate_targets = int(edges["TARGET_FLIGHT_ID"].duplicated().sum())
    duplicate_pairs = int(edges.duplicated(["SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID"]).sum())
    if duplicate_sources or duplicate_targets or duplicate_pairs:
        raise ValueError(
            "Duplicate graph IDs found: "
            f"source={duplicate_sources}, target={duplicate_targets}, edge={duplicate_pairs}."
        )


def find_chain_starts(edges: pd.DataFrame, signal_column: str) -> list[str]:
    active = edges.loc[edges[signal_column].eq(1)]
    active_targets = set(active["TARGET_FLIGHT_ID"])
    return active.loc[
        ~active["SOURCE_FLIGHT_ID"].isin(active_targets), "SOURCE_FLIGHT_ID"
    ].tolist()


def trace_chain_edges(edge_lookup: pd.DataFrame, start_flight_id: str, max_hops: int = 20):
    current = start_flight_id
    rows = []
    cumulative_probability = 1.0
    for hop in range(1, max_hops + 1):
        if current not in edge_lookup.index:
            break
        edge = edge_lookup.loc[current]
        cumulative_probability *= float(edge["PROPAGATION_PROBABILITY"])
        row = edge.to_dict()
        row.update(
            {
                "START_FLIGHT_ID": start_flight_id,
                "HOP": hop,
                "LOCAL_PROBABILITY": float(edge["PROPAGATION_PROBABILITY"]),
                "CUMULATIVE_PROBABILITY": cumulative_probability,
            }
        )
        rows.append(row)
        current = edge["TARGET_FLIGHT_ID"]
    return rows, current, cumulative_probability


def build_chain_outputs(edges: pd.DataFrame, signal_column: str):
    active = edges.loc[edges[signal_column].eq(1)].copy()
    lookup = active.set_index("SOURCE_FLIGHT_ID", drop=False)
    edge_rows = []
    summaries = []
    for start in find_chain_starts(edges, signal_column):
        rows, end, cumulative = trace_chain_edges(lookup, start)
        edge_rows.extend(rows)
        edge_count = len(rows)
        summaries.append(
            {
                "START_FLIGHT_ID": start,
                "END_FLIGHT_ID": end,
                "EDGE_COUNT": edge_count,
                "FLIGHT_COUNT": edge_count + 1,
                "CUMULATIVE_PROBABILITY": cumulative,
            }
        )
    return pd.DataFrame(edge_rows), pd.DataFrame(summaries)


def build_chain_summary(edge_lookup: pd.DataFrame, chain_starts: list[str]) -> pd.DataFrame:
    """Compatibility helper used by validation evaluators."""
    summaries = []
    for start in chain_starts:
        rows, end, cumulative = trace_chain_edges(edge_lookup, start)
        summaries.append(
            {"START_FLIGHT_ID": start, "END_FLIGHT_ID": end, "EDGE_COUNT": len(rows),
             "FLIGHT_COUNT": len(rows) + 1, "CUMULATIVE_PROBABILITY": cumulative}
        )
    return pd.DataFrame(summaries)


def evaluate_chain_matching(predicted: pd.DataFrame, actual: pd.DataFrame):
    predicted = predicted.loc[predicted["EDGE_COUNT"].ge(2)].copy()
    actual = actual.loc[actual["EDGE_COUNT"].ge(2)].copy()
    predicted_starts = set(predicted["START_FLIGHT_ID"])
    actual_starts = set(actual["START_FLIGHT_ID"])
    tp = len(predicted_starts & actual_starts)
    fp = len(predicted_starts - actual_starts)
    fn = len(actual_starts - predicted_starts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = predicted[["START_FLIGHT_ID", "EDGE_COUNT"]].merge(
        actual[["START_FLIGHT_ID", "EDGE_COUNT"]], on="START_FLIGHT_ID",
        suffixes=("_PREDICTED", "_ACTUAL"), validate="one_to_one",
    )
    errors = matched["EDGE_COUNT_PREDICTED"] - matched["EDGE_COUNT_ACTUAL"]
    return {
        "model_version": MODEL_VERSION,
        "threshold": MODEL_THRESHOLD,
        "predicted_multi_hop_starts": len(predicted_starts),
        "actual_multi_hop_starts": len(actual_starts),
        "true_positive_starts": tp,
        "false_positive_starts": fp,
        "false_negative_starts": fn,
        "chain_start_precision": precision,
        "chain_start_recall": recall,
        "chain_start_f1": f1,
        "exact_length_rate": float(errors.eq(0).mean()) if tp else 0.0,
        "edge_count_mae": float(errors.abs().mean()) if tp else 0.0,
    }


def main():
    edges = load_scored_edges()
    validate_graph_structure(edges)
    predicted_edges, predicted_summary = build_chain_outputs(edges, "PROPAGATION_ALERT")
    _, actual_summary = build_chain_outputs(edges, "ACTUAL_PROPAGATION")
    metrics = evaluate_chain_matching(predicted_summary, actual_summary)
    PREDICTED_CHAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predicted_edges.to_parquet(PREDICTED_CHAINS_PATH, index=False, compression="snappy")
    predicted_summary.to_parquet(PREDICTED_CHAIN_SUMMARY_PATH, index=False, compression="snappy")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Predicted chain edges saved:", PREDICTED_CHAINS_PATH)
    print("Predicted chain summary saved:", PREDICTED_CHAIN_SUMMARY_PATH)


if __name__ == "__main__":
    main()
