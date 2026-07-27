from src.data.load_flights import load_flights
import pandas as pd

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
    
    print(rotation_df["ACTUAL_TURNAROUND"].describe())
    print(rotation_df["ACTUAL_TURNAROUND"].quantile([0.90, 0.95, 0.99]))

    #print("Rotasyon verisi boyutu:", rotation_df.shape)
    #print(rotation_df.head())

    #print("Temizlenmiş rotasyon verisi boyutu:", rotation_df.shape)
    #print("Eksik tail number sayisi:", rotation_df["TAIL_NUM"].isna().sum())
    #print("İptal edilen uçuş sayisi:", rotation_df["CANCELLED"].sum())
    #print("Yönlendirilen uçuş sayisi:", rotation_df["DIVERTED"].sum())


if __name__ == "__main__":
    build_rotations()