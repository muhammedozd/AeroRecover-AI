from src.data.load_flights import load_flights
import pandas as pd

from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset.csv"

)
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)
def build_rotations():
    df = load_flights()


    rotation_columns = [
        "FL_DATE",
        "TAIL_NUM",
        "OP_UNIQUE_CARRIER",
        "OP_CARRIER_FL_NUM",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "DEP_DELAY",
        "ARR_DELAY",
        "CANCELLED",
        "DIVERTED",
        "DISTANCE",
        "DEP_TIME",
        "ARR_TIME",
        "LATE_AIRCRAFT_DELAY",

    ]

    rotation_df = df[rotation_columns].copy()
    rotation_df = rotation_df.dropna(subset=["TAIL_NUM"])

    rotation_df = rotation_df[
    (rotation_df["CANCELLED"] == 0) &
    (rotation_df["DIVERTED"] == 0)
]
    rotation_df["FL_DATE"] = pd.to_datetime(
    rotation_df["FL_DATE"],
    format="%m/%d/%Y %I:%M:%S %p"
)
    rotation_df = rotation_df.sort_values(
    by=["TAIL_NUM", "FL_DATE", "CRS_DEP_TIME"]
)

    rotation_df["ROTATION_POSITION"] = (
    rotation_df
    .groupby(["TAIL_NUM", "FL_DATE"])
    .cumcount()
    + 1
)
    rotation_df["PREV_DEST"] = (
    rotation_df
    .groupby(["TAIL_NUM", "FL_DATE"])["DEST"]
    .shift(1)
)
    rotation_df["IS_CONNECTED"] = (
    rotation_df["ORIGIN"] == rotation_df["PREV_DEST"]
)
    rotation_df["PREV_ARR_DELAY"] = (
    rotation_df
    .groupby(["TAIL_NUM", "FL_DATE"])["ARR_DELAY"]
    .shift(1)
)

    rotation_df["CRS_DEP_MIN"] = (
    rotation_df["CRS_DEP_TIME"] // 100 * 60
    + rotation_df["CRS_DEP_TIME"] % 100
)
    rotation_df["CRS_ARR_MIN"] = (
    rotation_df["CRS_ARR_TIME"] // 100 * 60
    + rotation_df["CRS_ARR_TIME"] % 100
)

    rotation_df["DEP_MIN"] = (
    rotation_df["DEP_TIME"] // 100 * 60
    + rotation_df["DEP_TIME"] % 100
)

    rotation_df["ARR_MIN"] = (
    rotation_df["ARR_TIME"] // 100 * 60
    + rotation_df["ARR_TIME"] % 100
)
    
    rotation_df["PREV_ARR_MIN"] = (
    rotation_df
    .groupby(["TAIL_NUM", "FL_DATE"])["ARR_MIN"]
    .shift(1)
)

    # Önceki uçuşun planlanan varış zamanı
    rotation_df["PREV_CRS_ARR_MIN"] = (
    rotation_df
    .groupby(["TAIL_NUM", "FL_DATE"])["CRS_ARR_MIN"]
    .shift(1)
)

# Planlanan turnaround süresi
    rotation_df["PLANNED_TURNAROUND"] = (
    rotation_df["CRS_DEP_MIN"]
    - rotation_df["PREV_CRS_ARR_MIN"]
)

# Gece yarısını aşan uçuşlar için düzeltme
    rotation_df.loc[
    rotation_df["PLANNED_TURNAROUND"] < 0,
    "PLANNED_TURNAROUND"
] += 1440

    print(rotation_df["PLANNED_TURNAROUND"].describe())
    print(rotation_df["PLANNED_TURNAROUND"].value_counts().head(20))
    print(
    rotation_df[rotation_df["PLANNED_TURNAROUND"] > 300]
    ["PLANNED_TURNAROUND"]
    .value_counts()
)
    print(
    (rotation_df["PLANNED_TURNAROUND"] > 300).sum()
)

    rotation_df["TURN_BUFFER"] = (
    rotation_df["PLANNED_TURNAROUND"]
    - rotation_df["PREV_ARR_DELAY"].clip(lower=0)
)
   
    rotation_df["PREV_DELAY_RATIO"] = (
    rotation_df["PREV_ARR_DELAY"].clip(lower=0)
    / rotation_df["PLANNED_TURNAROUND"].replace(0, pd.NA)
)
    rotation_df["HAS_BUFFER"] = (
    rotation_df["TURN_BUFFER"] > 0
).astype(int)
    rotation_df["PREV_DELAY_LEVEL"] = pd.cut(
    rotation_df["PREV_ARR_DELAY"],
    bins=[-1000, -1, 14, 29, 59, 1000],
    labels=[
        "Early",
        "OnTime",
        "Minor",
        "Moderate",
        "Severe"
    ]
)

    rotation_df["IS_SHORT_TURN"] = (
    rotation_df["PLANNED_TURNAROUND"] < 45
).astype(int)

    rotation_df["ACTUAL_TURNAROUND"] = (
    rotation_df["DEP_MIN"]
    - rotation_df["PREV_ARR_MIN"]
)
    rotation_df.loc[
    rotation_df["ACTUAL_TURNAROUND"] < 0,
    "ACTUAL_TURNAROUND"
] += 1440

    rotation_df["VALID_ROTATION"] = (
    rotation_df["PREV_DEST"]
    == rotation_df["ORIGIN"]
)
    rotation_df.loc[
    rotation_df["VALID_ROTATION"] == False,
    "ACTUAL_TURNAROUND"
] = pd.NA
    
    rotation_df = rotation_df[
    rotation_df["ACTUAL_TURNAROUND"] <= 240
].copy()
    
    rotation_df = rotation_df[
    rotation_df["VALID_ROTATION"]
].copy()

    rotation_df["PREV_DELAYED"] = (
    rotation_df["PREV_ARR_DELAY"] >= 15
).astype(int)

    
    rotation_df["PROPAGATED_DELAY_MINUTES"] = (
    rotation_df["LATE_AIRCRAFT_DELAY"]
    .fillna(0)
    .clip(lower=0)
)
    rotation_df["IS_DELAY_PROPAGATED"] = (
    rotation_df["PROPAGATED_DELAY_MINUTES"] >= 15
).astype(int)




    rotation_df["TURNAROUND_GROUP"] = pd.cut(
    rotation_df["ACTUAL_TURNAROUND"],
    bins=[0, 30, 60, 90, 120, 240],
    labels=[
        "0-30",
        "31-60",
        "61-90",
        "91-120",
        "120-240"
    ]
)
    rotation_df["DELAYED"] = (
    rotation_df["DEP_DELAY"] >= 15
).astype(int)
    
    turnaround_analysis = (
    rotation_df
    .groupby("TURNAROUND_GROUP", observed=True)["DELAYED"]
    .agg(["count", "mean"])

)
    turnaround_analysis["delay_rate_percent"] = (
    turnaround_analysis["mean"] * 100
)
    long_turnaround_df = rotation_df[
    rotation_df["TURNAROUND_GROUP"] == "120-240"
]
    carrier_analysis = (
    long_turnaround_df
    .groupby("OP_UNIQUE_CARRIER")["DELAYED"]
    .agg(["count", "mean"])
    .sort_values("count", ascending=False)
)
    carrier_analysis["delay_rate_percent"] = (
    carrier_analysis["mean"] * 100
)
    rotation_df["RECOVERY_MARGIN"] = (
        rotation_df["ACTUAL_TURNAROUND"]
        - rotation_df["PREV_ARR_DELAY"]
    )
    rotation_df["RECOVERY_GROUP"] = pd.cut(
        rotation_df["RECOVERY_MARGIN"],
        bins=[-2000, 0, 30, 60, 90, 300],
        labels=[
            "<0",
            "0-30",
            "31-60",
            "61-90",
            ">90"
        ]
        )
    recovery_analysis = (
        rotation_df
        .groupby("RECOVERY_GROUP", observed=True)["DELAYED"]
        .agg(["count", "mean"])
    )
    
    recovery_analysis["delay_rate_percent"] = (
        recovery_analysis["mean"] * 100
    )
    
    print(recovery_analysis)
        
    #print(rotation_df["ACTUAL_TURNAROUND"].describe())
    #print(rotation_df["ACTUAL_TURNAROUND"].quantile([0.90, 0.95, 0.99]))

    #print("Rotasyon verisi boyutu:", rotation_df.shape)
    #print(rotation_df.head())

    #print("Temizlenmiş rotasyon verisi boyutu:", rotation_df.shape)
    #print("Eksik tail number sayisi:", rotation_df["TAIL_NUM"].isna().sum())
    #print("İptal edilen uçuş sayisi:", rotation_df["CANCELLED"].sum())
    #print("Yönlendirilen uçuş sayisi:", rotation_df["DIVERTED"].sum())
    print(
        rotation_df[
        [
            "PREV_ARR_DELAY",
            "LATE_AIRCRAFT_DELAY",
            "PROPAGATED_DELAY_MINUTES",
            "IS_DELAY_PROPAGATED"
        ]
        ].head(10)
)
    print(
        rotation_df["IS_DELAY_PROPAGATED"]
    .value_counts(normalize=True)
)
    return rotation_df



if __name__ == "__main__":
    rotation_df = build_rotations()


    rotation_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("=" * 50)
    print("Rotation dataset successfully created.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total samples: {len(rotation_df):,}")
    print("=" * 50)