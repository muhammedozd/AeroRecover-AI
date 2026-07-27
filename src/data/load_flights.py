"""Load the raw BTS flight dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLIGHTS_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bts"
    / "BTS-JAN-2023.csv"
)


def load_flights(csv_path: Path = FLIGHTS_CSV_PATH) -> pd.DataFrame:
    """Load flight records from a CSV file into a pandas DataFrame."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Uçuş veri dosyası bulunamadı: {csv_path}. "
            "CSV dosyasını data/raw/bts klasörüne yerleştirin."
        )

    return pd.read_csv(csv_path)

if __name__ == "__main__":
    df = load_flights()

    print(f"Satır sayısı : {df.shape[0]}")
    print(f"Sütun sayısı : {df.shape[1]}")
    #print(df.head())
    #print(df.info())
    print(df.isnull().sum())
    print(df.describe())
    print(df.columns)
    print(df.columns.tolist())



