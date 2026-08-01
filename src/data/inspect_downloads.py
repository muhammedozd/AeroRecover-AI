from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "incoming_2023"

 #glob("*.csv") içerisinde .csv olan dosyaları alır ve bunları sıralar. Bu, veri dizinindeki tüm CSV dosyalarını bulmak için kullanılır.
csv_files = sorted(DATA_DIR.glob("*.csv"))

required_columns = [
    "YEAR",
    "MONTH",
    "FL_DATE",
    "TAIL_NUM",
    "CANCELLED",
    "DIVERTED",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
]


flight_key = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
]




results = []

for file_path in csv_files:
    df = pd.read_csv(
        file_path,
        usecols=required_columns,
        low_memory=False,
    )
   #len(csv_files) bulunan dosya sayısını verir.


    duplicate_rows = df.duplicated(
    subset=flight_key,
    keep=False,
).sum()

    
    flight_dates = pd.to_datetime(df["FL_DATE"])
    

    valid_mask = (
    df["TAIL_NUM"].notna()
    & (df["CANCELLED"] == 0)
    & (df["DIVERTED"] == 0)
)
    results.append({
        "file": file_path.name,
        "month": int(df["MONTH"].iloc[0]),
        "first_date": flight_dates.min().date(),
        "last_date": flight_dates.max().date(),
        "rows": len(df),
        "missing_tail": int(df["TAIL_NUM"].isna().sum()),
        "cancelled": int((df["CANCELLED"] == 1).sum()),
        "diverted": int((df["DIVERTED"] == 1).sum()),
        "valid_flights": int(valid_mask.sum()),
        "excluded_flights": int((~valid_mask).sum()),

        "duplicate_rows": int(duplicate_rows),
    })


report = pd.DataFrame(results).sort_values("month")

report["missing_tail_rate"] = (
    report["missing_tail"] / report["rows"] * 100
).round(2)

report["cancelled_rate"] = (
    report["cancelled"] / report["rows"] * 100
).round(2)

report["diverted_rate"] = (
    report["diverted"] / report["rows"] * 100
).round(2)

print(report.to_string(index=False))
print(f"\nToplam uçuş kaydı: {report['rows'].sum():,}")

report["valid_flight_rate"] = (
    report["valid_flights"] / report["rows"] * 100
).round(2)