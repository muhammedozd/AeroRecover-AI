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