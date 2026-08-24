"""Build the historical validation replay dataset."""

from pathlib import Path

import pandas as pd

from src.models.rotation_model_contract import MODEL_THRESHOLD, MODEL_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORED_EDGES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "scored_tail_edges_2023_validation_full_enhanced.parquet"
)

CHAIN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predicted_chain_summary_validation_full_enhanced.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "decision_support_replay_validation_full_enhanced.parquet"
)

def load_source_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not SCORED_EDGES_PATH.exists():
        raise FileNotFoundError(
            f"Scored edges file not found: {SCORED_EDGES_PATH}"
        )

    if not CHAIN_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Chain summary file not found: {CHAIN_SUMMARY_PATH}"
        )

    scored_edges = pd.read_parquet(
        SCORED_EDGES_PATH
    )

    chain_summary = pd.read_parquet(
        CHAIN_SUMMARY_PATH
    )

    return scored_edges, chain_summary

def build_replay_dataset(
    scored_edges: pd.DataFrame,
    chain_summary: pd.DataFrame,
) -> pd.DataFrame:
    scored_columns = [
        "SOURCE_FLIGHT_ID",
        "TARGET_FLIGHT_ID",
        "PROPAGATION_PROBABILITY",
        "PREV_ARR_DELAY",
        "TURN_BUFFER",
        "PREV_DELAY_RATIO",
        "PLANNED_TURNAROUND",
    ]

    replay_data = chain_summary.merge(
        scored_edges[scored_columns],
        left_on="START_FLIGHT_ID",
        right_on="SOURCE_FLIGHT_ID",
        how="left",
        validate="one_to_one",
    )
   #PROPAGATION_PROBABILITY, sağdaki scored_edges tablosundan gelir.
    unmatched_rows = (
        replay_data["PROPAGATION_PROBABILITY"]
        .isna()
        .sum()
    )

    if unmatched_rows > 0:
        raise ValueError(
            f"{unmatched_rows:,} replay rows could not "
            "be matched to scored graph edges."
        )

    replay_data["MODEL_VERSION"] = MODEL_VERSION
    replay_data["MODEL_THRESHOLD"] = MODEL_THRESHOLD
    return replay_data

def save_replay_dataset(
    replay_data: pd.DataFrame,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay_data.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    print(
        "Replay dataset saved:",
        OUTPUT_PATH,
    )

    

def main() -> None:
    scored_edges, chain_summary = load_source_data()

    replay_data = build_replay_dataset(
        scored_edges=scored_edges,
        chain_summary=chain_summary,
    )

    save_replay_dataset(
        replay_data
    )

    print(
        "Replay dataset rows:",
        f"{len(replay_data):,}",
    )


if __name__ == "__main__":
    main()


