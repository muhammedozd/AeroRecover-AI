"""Load the raw BTS flight dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLIGHTS_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "T_ONTIME_REPORTING.csv"


def load_flights(csv_path: Path = FLIGHTS_CSV_PATH) -> pd.DataFrame:
    """Load flight records from a CSV file into a pandas DataFrame."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Uçuş veri dosyası bulunamadı: {csv_path}. "
            "CSV dosyasını data/raw/flights.csv konumuna yerleştirin."
        )

    return pd.read_csv(csv_path)
