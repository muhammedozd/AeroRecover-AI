from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset.csv"
)


df = pd.read_csv(DATA_PATH)

print("Veri boyutu:", df.shape)
print(df[["PREV_ARR_DELAY", "IS_DELAY_PROPAGATED"]].head())


delay_bins = [-1000, 0, 15, 30, 60, 120, 10000]
delay_labels = [
    "Early",
    "0-15",
    "15-30",
    "30-60",
    "60-120",
    "120+"
]

df["PREV_DELAY_GROUP"] = pd.cut(
    df["PREV_ARR_DELAY"],
    bins=delay_bins,
    labels=delay_labels
)

analysis = (
    df
    .groupby("PREV_DELAY_GROUP", observed=True)["IS_DELAY_PROPAGATED"]
    .agg(["count", "mean"])
)

analysis["propagation_rate"] = analysis["mean"] * 100

print(analysis)

buffer_bins = [-1000, 0, 15, 30, 60, 120, 10000]

buffer_labels = [
    "<0",
    "0-15",
    "15-30",
    "30-60",
    "60-120",
    "120+"
]

df["TURN_BUFFER_GROUP"] = pd.cut(
    df["TURN_BUFFER"],
    bins=buffer_bins,
    labels=buffer_labels
)

buffer_analysis = (
    df
    .groupby("TURN_BUFFER_GROUP", observed=True)["IS_DELAY_PROPAGATED"]
    .agg(["count", "mean"])
)

buffer_analysis["propagation_rate"] = (
    buffer_analysis["mean"] * 100
)

print(buffer_analysis)


turnaround_bins = [0, 30, 45, 60, 90, 120, 180, 10000]

turnaround_labels = [
    "0-30",
    "30-45",
    "45-60",
    "60-90",
    "90-120",
    "120-180",
    "180+"
]

df["PLANNED_TURNAROUND_GROUP"] = pd.cut(
    df["PLANNED_TURNAROUND"],
    bins=turnaround_bins,
    labels=turnaround_labels,
    include_lowest=True
)

turnaround_analysis = (
    df
    .groupby(
        "PLANNED_TURNAROUND_GROUP",
        observed=True
    )["IS_DELAY_PROPAGATED"]
    .agg(["count", "mean"])
)

turnaround_analysis["propagation_rate"] = (
    turnaround_analysis["mean"] * 100
)

print(turnaround_analysis)


rotation_bins = [0, 1, 2, 3, 4, 5, 10, 100]

rotation_labels = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6-10",
    "10+"
]

df["ROTATION_GROUP"] = pd.cut(
    df["ROTATION_POSITION"],
    bins=rotation_bins,
    labels=rotation_labels,
    include_lowest=True
)

rotation_analysis = (
    df
    .groupby("ROTATION_GROUP", observed=True)["IS_DELAY_PROPAGATED"]
    .agg(["count", "mean"])
)

rotation_analysis["propagation_rate"] = (
    rotation_analysis["mean"] * 100
)

print(rotation_analysis)