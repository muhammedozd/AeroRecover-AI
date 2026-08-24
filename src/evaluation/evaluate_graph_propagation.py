import pandas as pd

from src.graph.multi_hop_propagation import (
    build_chain_summary,
    find_chain_starts,
    load_scored_edges,
    validate_graph_structure,
)

def build_chains(
    edges: pd.DataFrame,
    signal_column: str,
) -> pd.DataFrame:
    signal_edges = edges[
        edges[signal_column] == 1
    ].copy()

    edge_lookup = signal_edges.set_index(
        "SOURCE_FLIGHT_ID"
    )

    chain_starts = find_chain_starts(
        edges=edges,
        signal_column=signal_column,
    )

    return build_chain_summary(
        edge_lookup=edge_lookup,
        chain_starts=chain_starts,
    )


def evaluate_chain_matching(
    predicted_chains: pd.DataFrame,
    observed_chains: pd.DataFrame,
) -> tuple[dict[str, int | float], pd.DataFrame]:
    """Compute the validation multi-hop start and length metrics."""
    predicted_multi_hop = predicted_chains[
        predicted_chains["EDGE_COUNT"] >= 2
    ].copy()
    observed_multi_hop = observed_chains[
        observed_chains["EDGE_COUNT"] >= 2
    ].copy()
    predicted_starts = set(predicted_multi_hop["START_FLIGHT_ID"])
    observed_starts = set(observed_multi_hop["START_FLIGHT_ID"])
    tp = len(predicted_starts & observed_starts)
    fp = len(predicted_starts - observed_starts)
    fn = len(observed_starts - predicted_starts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = predicted_multi_hop[["START_FLIGHT_ID", "EDGE_COUNT"]].merge(
        observed_multi_hop[["START_FLIGHT_ID", "EDGE_COUNT"]],
        on="START_FLIGHT_ID", how="inner",
        suffixes=("_PREDICTED", "_OBSERVED"), validate="one_to_one",
    )
    if len(matched) != tp:
        raise AssertionError("Matched chain count does not equal true positives.")
    matched["EDGE_COUNT_ERROR"] = (
        matched["EDGE_COUNT_PREDICTED"] - matched["EDGE_COUNT_OBSERVED"]
    )
    matched["ABS_EDGE_COUNT_ERROR"] = matched["EDGE_COUNT_ERROR"].abs()
    exact = int(matched["EDGE_COUNT_ERROR"].eq(0).sum())
    metrics: dict[str, int | float] = {
        "predicted_multi_hop_starts": len(predicted_starts),
        "observed_multi_hop_starts": len(observed_starts),
        "true_positive_starts": tp,
        "false_positive_starts": fp,
        "false_negative_starts": fn,
        "chain_start_precision": precision,
        "chain_start_recall": recall,
        "chain_start_f1": f1,
        "correctly_matched_chain_count": len(matched),
        "exact_length_matches": exact,
        "conditional_exact_length_rate": exact / tp if tp else 0.0,
        "edge_count_mae": float(matched["ABS_EDGE_COUNT_ERROR"].mean()) if tp else 0.0,
        "underestimated_matched_chains": int(matched["EDGE_COUNT_ERROR"].lt(0).sum()),
        "overestimated_matched_chains": int(matched["EDGE_COUNT_ERROR"].gt(0).sum()),
    }
    return metrics, matched


def main():
    scored_edges = load_scored_edges()

    validate_graph_structure(
        scored_edges
    )

    predicted_chains = build_chains(
        edges=scored_edges,
        signal_column="PROPAGATION_ALERT",
    )

    actual_chains = build_chains(
        edges=scored_edges,
        signal_column="ACTUAL_PROPAGATION",
    )

    

    print("\nGraph evaluation data")
    print("-" * 40)
    print(
        f"Predicted chain rows: "
        f"{len(predicted_chains):,}"
    )
    print(
        f"Actual chain rows: "
        f"{len(actual_chains):,}"
    )

    predicted_multi_hop = predicted_chains[
        predicted_chains["EDGE_COUNT"] >= 2
    ].copy()

    actual_multi_hop = actual_chains[
        actual_chains["EDGE_COUNT"] >= 2
    ].copy()

    predicted_domino_starts = set(
        predicted_multi_hop["START_FLIGHT_ID"]
    )

    actual_domino_starts = set(
        actual_multi_hop["START_FLIGHT_ID"]
    )

    true_positive_starts = (
        predicted_domino_starts
        & actual_domino_starts
    )

    false_positive_starts = (
        predicted_domino_starts
        - actual_domino_starts
    )

    false_negative_starts = (
        actual_domino_starts
        - predicted_domino_starts
    )

    tp = len(true_positive_starts)
    fp = len(false_positive_starts)
    fn = len(false_negative_starts)

#assert, Python anahtar kelimesidir.Verilen koşul doğru değilse AssertionError hatası oluşturur.
    assert tp + fp == len(predicted_domino_starts)
    assert tp + fn == len(actual_domino_starts)

    print("\nMulti-hop start matching")
    print("-" * 40)
    print(f"True positives: {tp:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    matched_chains = (
    predicted_multi_hop[
            ["START_FLIGHT_ID", "EDGE_COUNT"]
        ]
        .merge(
            actual_multi_hop[
                ["START_FLIGHT_ID", "EDGE_COUNT"]
            ],
            on="START_FLIGHT_ID",
            how="inner",
            suffixes=("_PREDICTED", "_ACTUAL"),
            validate="one_to_one",
        )
    )

    assert len(matched_chains) == tp

    matched_chains["EDGE_COUNT_ERROR"] = (
        matched_chains["EDGE_COUNT_PREDICTED"]
        - matched_chains["EDGE_COUNT_ACTUAL"]
    )

    matched_chains["ABS_EDGE_COUNT_ERROR"] = (
    matched_chains["EDGE_COUNT_ERROR"].abs()
)

    exact_length_matches = (
        matched_chains["EDGE_COUNT_ERROR"] == 0
    ).sum()
    exact_length_rate = (
        exact_length_matches
        / len(matched_chains)
    )
    edge_count_mae = (
        matched_chains[
            "ABS_EDGE_COUNT_ERROR"
        ].mean()
    )

    underestimated_chains = (
        matched_chains["EDGE_COUNT_ERROR"] < 0
    ).sum()

    overestimated_chains = (
        matched_chains["EDGE_COUNT_ERROR"] > 0
    ).sum()

    assert (
        exact_length_matches
        + underestimated_chains
        + overestimated_chains
        == len(matched_chains)
    )

    print(f"Exact length matches: {exact_length_matches:,}")
    print(f"Exact length rate: {exact_length_rate:.6f}")
    print(f"Edge-count MAE: {edge_count_mae:.6f}")

    print(
        f"Underestimated chains: "
        f"{underestimated_chains:,}"
    )

    print(
        f"Overestimated chains: "
        f"{overestimated_chains:,}"
    )
    print("\nMatched chain lengths")
    print("-" * 40)
    print(
        f"Matched chains: "
        f"{len(matched_chains):,}"
    )


    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")


    print("\nMulti-hop chain data")
    print("-" * 40)

    print(
        f"Predicted multi-hop starts: "
        f"{len(predicted_domino_starts):,}"
    )

    print(
        f"Actual multi-hop starts: "
        f"{len(actual_domino_starts):,}"
    )
    
    
    

if __name__ == "__main__":
    main()
