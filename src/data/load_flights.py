"""Load the raw BTS flight dataset."""

from pathlib import Path

import pandas as pd

FLIGHT_DTYPES = {
    "FL_DATE": "string",
    "TAIL_NUM": "string",
    "OP_UNIQUE_CARRIER": "string",
    "OP_CARRIER_FL_NUM": "float32",
    "ORIGIN": "string",
    "DEST": "string",
    "CRS_DEP_TIME": "float32",
    "CRS_ARR_TIME": "float32",
    "DEP_DELAY": "float32",
    "ARR_DELAY": "float32",
    "CANCELLED": "float32",
    "DIVERTED": "float32",
    "DISTANCE": "float32",
    "DEP_TIME": "float32",
    "ARR_TIME": "float32",
    "LATE_AIRCRAFT_DELAY": "float32",
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLIGHTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bts_2023"
)


def load_flights(
    columns: list[str] | None = None,
    data_dir: Path = FLIGHTS_DIR,
) -> pd.DataFrame:
    """Load and combine monthly BTS flight files."""

    csv_files = sorted(
        data_dir.glob("bts_2023_*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"Bu klasörde BTS CSV dosyası bulunamadı: {data_dir}"
        )

    if len(csv_files) != 12:
        raise ValueError(
            f"12 dosya bekleniyordu, {len(csv_files)} dosya bulundu."
        )


#.items() bir Python sözlüğündeki anahtar ve değerleri birlikte almak için kullanılır.
    selected_dtypes = {
    column: dtype
    for column, dtype in FLIGHT_DTYPES.items()
    if columns is None or column in columns
}

    monthly_frames = []

    for file_path in csv_files:
        print(f"Yükleniyor: {file_path.name}")

#usecols=columns Yalnızca istediğimiz sütunları okur.
#hazırladığımız veri tipi sözlüğünü pandasa gönderir.
#low_memory=False Bellek kullanımını optimize eder ve veri tiplerini otomatik olarak belirler.
        monthly_df = pd.read_csv(
            file_path,
            usecols=columns,
            low_memory=False,
             dtype=selected_dtypes,
        )

        monthly_frames.append(monthly_df)

#concat() data çerçeveleri birleştirir.


    flights_df = pd.concat(
        monthly_frames,
        ignore_index=True,
    )

    return flights_df

if __name__ == "__main__":
    test_columns = [
        "YEAR",
        "MONTH",
        "FL_DATE",
        "TAIL_NUM",
    ]

    df = load_flights(columns=test_columns)

    print(f"Satır sayısı: {len(df):,}")
    print(f"Sütun sayısı: {len(df.columns)}")
    print(f"İlk ay: {df['MONTH'].min()}")
    print(f"Son ay: {df['MONTH'].max()}")
    print(df.columns.tolist())



