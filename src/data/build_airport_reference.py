"""Build a reproducible US airport coordinate reference from OurAirports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reference" / "us_airport_coordinates.parquet"
METADATA_PATH = OUTPUT_PATH.with_suffix(".metadata.json")
SOURCE_COLUMNS = [
    "iata_code", "latitude_deg", "longitude_deg", "name", "municipality", "iso_country", "type"
]
TYPE_PRIORITY = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}


def build_airport_reference(source: str = SOURCE_URL) -> pd.DataFrame:
    """Load, validate, and deterministically select US IATA airport records."""
    try:
        airports = pd.read_csv(source, usecols=SOURCE_COLUMNS)
    except Exception as exc:
        raise RuntimeError(f"Could not read the OurAirports source: {exc}") from exc

    missing_columns = sorted(set(SOURCE_COLUMNS) - set(airports.columns))
    if missing_columns:
        raise ValueError(f"OurAirports schema is missing columns: {', '.join(missing_columns)}")

    airports = airports.loc[airports["iso_country"].eq("US"), SOURCE_COLUMNS].copy()
    airports["iata_code"] = airports["iata_code"].astype("string").str.strip().str.upper()
    airports = airports.dropna(subset=["iata_code", "latitude_deg", "longitude_deg"])
    airports = airports.loc[airports["iata_code"].ne("")].copy()
    airports["type_priority"] = airports["type"].map(TYPE_PRIORITY).fillna(99).astype(int)
    airports = airports.sort_values(["iata_code", "type_priority", "name"], kind="stable")

    selected = airports.groupby("iata_code", sort=True, as_index=False).first()
    best_priority_counts = airports.groupby("iata_code")["type_priority"].transform("min")
    best = airports.loc[airports["type_priority"].eq(best_priority_counts)]
    unresolved = best.loc[best["iata_code"].duplicated(keep=False), "iata_code"].unique().tolist()
    if unresolved:
        raise ValueError(
            "Duplicate IATA codes remain after airport-type prioritization: "
            + ", ".join(sorted(unresolved))
        )

    selected = selected.drop(columns="type_priority")
    duplicate_count = int(selected["iata_code"].duplicated().sum())
    missing_coordinate_count = int(selected[["latitude_deg", "longitude_deg"]].isna().any(axis=1).sum())
    if duplicate_count or missing_coordinate_count:
        raise ValueError("Airport reference failed post-build integrity validation.")
    return selected


def save_airport_reference(airports: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    airports.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    metadata = {
        "source_url": SOURCE_URL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(airports),
        "duplicate_iata_count": int(airports["iata_code"].duplicated().sum()),
        "missing_coordinate_count": int(
            airports[["latitude_deg", "longitude_deg"]].isna().any(axis=1).sum()
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Airport rows: {metadata['row_count']:,}")
    print(f"Duplicate IATA codes: {metadata['duplicate_iata_count']:,}")
    print(f"Missing coordinates: {metadata['missing_coordinate_count']:,}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")


def main() -> None:
    save_airport_reference(build_airport_reference())


if __name__ == "__main__":
    main()
