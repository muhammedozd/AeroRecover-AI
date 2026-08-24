from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEATHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather-2023"
    / "ATL_2023_NOAA.csv"
)

CLEAN_WEATHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather-2023"
    / "ATL_weather_2023_clean.csv"
)

SELECTED_COLUMNS = [
    "STATION",
    "DATE",
    "NAME",
    "REPORT_TYPE",
    "CALL_SIGN",
    "QUALITY_CONTROL",
    "WND",
    "CIG",
    "VIS",
    "TMP",
    "DEW",
    "SLP",
    "AA1",
    "MW1",
]

AVIATION_REPORT_TYPES = [
    "FM-15",
    "FM-16",
]


print("Weather file:", WEATHER_PATH)
print("File exists:", WEATHER_PATH.exists())

weather_df = pd.read_csv(
    WEATHER_PATH,
    usecols=SELECTED_COLUMNS,
)

weather_df["DATE"] = pd.to_datetime(
    weather_df["DATE"],
    utc=True,
    errors="raise",
)

weather_df["CALL_SIGN"] = (
    weather_df["CALL_SIGN"]
    .str.strip()
)

aviation_weather_df = weather_df[
    weather_df["REPORT_TYPE"].isin(
        AVIATION_REPORT_TYPES
    )
].copy()


wind_parts = (
    aviation_weather_df["WND"]
    .str.split(",", expand=True)
)

wind_parts.columns = [
    "WIND_DIRECTION_RAW",
    "WIND_DIRECTION_QUALITY",
    "WIND_TYPE",
    "WIND_SPEED_RAW",
    "WIND_SPEED_QUALITY",
]

aviation_weather_df["WIND_DIRECTION_DEG"] = (
    pd.to_numeric(
        wind_parts["WIND_DIRECTION_RAW"],
        errors="coerce",
    )
    .replace(999, pd.NA)
)

aviation_weather_df["WIND_SPEED_MPS"] = (
    pd.to_numeric(
        wind_parts["WIND_SPEED_RAW"],
        errors="coerce",
    )
    .replace(9999, pd.NA)
    / 10
)

aviation_weather_df["WIND_IS_CALM"] = (
    (wind_parts["WIND_TYPE"] == "C")
    & (aviation_weather_df["WIND_SPEED_MPS"] == 0)
).astype("int8")


visibility_parts = (
    aviation_weather_df["VIS"]
    .str.split(",", expand=True)
)

visibility_parts.columns = [
    "VISIBILITY_RAW",
    "VISIBILITY_QUALITY",
    "VISIBILITY_VARIABILITY",
    "VISIBILITY_VARIABILITY_QUALITY",
]

aviation_weather_df["VISIBILITY_METERS"] = (
    pd.to_numeric(
        visibility_parts["VISIBILITY_RAW"],
        errors="coerce",
    )
    .replace(999999, pd.NA)
)


ceiling_parts = (
    aviation_weather_df["CIG"]
    .str.split(",", expand=True)
)

ceiling_parts.columns = [
    "CEILING_RAW",
    "CEILING_QUALITY",
    "CEILING_METHOD",
    "CAVOK_CODE",
]

aviation_weather_df["CEILING_METERS"] = (
    pd.to_numeric(
        ceiling_parts["CEILING_RAW"],
        errors="coerce",
    )
    .replace(99999, pd.NA)
)

aviation_weather_df["CEILING_UNLIMITED"] = (
    ceiling_parts["CEILING_RAW"] == "22000"
).astype("int8")


temperature_parts = (
    aviation_weather_df["TMP"]
    .str.split(",", expand=True)
)

temperature_parts.columns = [
    "TEMPERATURE_RAW",
    "TEMPERATURE_QUALITY",
]

aviation_weather_df["TEMPERATURE_C"] = (
    pd.to_numeric(
        temperature_parts["TEMPERATURE_RAW"],
        errors="coerce",
    )
    .replace(9999, pd.NA)
    / 10
)


dew_point_parts = (
    aviation_weather_df["DEW"]
    .str.split(",", expand=True)
)

dew_point_parts.columns = [
    "DEW_POINT_RAW",
    "DEW_POINT_QUALITY",
]

aviation_weather_df["DEW_POINT_C"] = (
    pd.to_numeric(
        dew_point_parts["DEW_POINT_RAW"],
        errors="coerce",
    )
    .replace(9999, pd.NA)
    / 10
)


pressure_parts = (
    aviation_weather_df["SLP"]
    .str.split(",", expand=True)
)

pressure_parts.columns = [
    "SEA_LEVEL_PRESSURE_RAW",
    "SEA_LEVEL_PRESSURE_QUALITY",
]

aviation_weather_df["SEA_LEVEL_PRESSURE_HPA"] = (
    pd.to_numeric(
        pressure_parts["SEA_LEVEL_PRESSURE_RAW"],
        errors="coerce",
    )
    .replace(99999, pd.NA)
    / 10
)


precipitation_parts = (
    aviation_weather_df["AA1"]
    .str.split(",", expand=True)
)

precipitation_parts.columns = [
    "PRECIP_PERIOD_RAW",
    "PRECIP_AMOUNT_RAW",
    "PRECIP_CONDITION",
    "PRECIP_QUALITY",
]

aviation_weather_df["PRECIP_PERIOD_HOURS"] = (
    pd.to_numeric(
        precipitation_parts["PRECIP_PERIOD_RAW"],
        errors="coerce",
    )
    .replace(99, pd.NA)
)

aviation_weather_df["PRECIP_AMOUNT_MM"] = (
    pd.to_numeric(
        precipitation_parts["PRECIP_AMOUNT_RAW"],
        errors="coerce",
    )
    .replace(9999, pd.NA)
    / 10
)

aviation_weather_df["PRECIP_1H_MM"] = (
    aviation_weather_df["PRECIP_AMOUNT_MM"]
    .where(
        aviation_weather_df["PRECIP_PERIOD_HOURS"] == 1
    )
)

aviation_weather_df["PRECIP_TRACE"] = (
    precipitation_parts["PRECIP_CONDITION"]
    .eq("2")
    .astype("Int8")
    .mask(
        aviation_weather_df["AA1"].isna(),
        pd.NA,
    )
)


aviation_weather_df["IATA"] = "ATL"

CLEAN_WEATHER_COLUMNS = [
    "STATION",
    "IATA",
    "DATE",
    "REPORT_TYPE",
    "WIND_DIRECTION_DEG",
    "WIND_SPEED_MPS",
    "WIND_IS_CALM",
    "VISIBILITY_METERS",
    "CEILING_METERS",
    "CEILING_UNLIMITED",
    "TEMPERATURE_C",
    "DEW_POINT_C",
    "SEA_LEVEL_PRESSURE_HPA",
    "PRECIP_1H_MM",
    "PRECIP_TRACE",
]

clean_weather_df = (
    aviation_weather_df[CLEAN_WEATHER_COLUMNS]
    .copy()
    .rename(
        columns={
            "DATE": "OBSERVATION_TIME_UTC",
        }
    )
    .sort_values(
        [
            "STATION",
            "OBSERVATION_TIME_UTC",
        ]
    )
    .reset_index(drop=True)
)


duplicate_weather_mask = clean_weather_df.duplicated(
    subset=[
        "STATION",
        "OBSERVATION_TIME_UTC",
    ],
    keep=False,
)

duplicate_count = int(
    duplicate_weather_mask.sum()
)

if duplicate_count > 0:
    raise ValueError(
        f"{duplicate_count} duplicate station-time rows found."
    )


CLEAN_WEATHER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

clean_weather_df.to_csv(
    CLEAN_WEATHER_PATH,
    index=False,
)


print("\nRaw observations:", len(weather_df))
print("Aviation observations:", len(aviation_weather_df))
print("Clean dataset shape:", clean_weather_df.shape)
print("Duplicate station-time rows:", duplicate_count)

print(
    "Observation period:",
    clean_weather_df["OBSERVATION_TIME_UTC"].min(),
    "->",
    clean_weather_df["OBSERVATION_TIME_UTC"].max(),
)

print("\nMissing values:")
print(
    clean_weather_df.isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nClean weather sample:")
print(
    clean_weather_df.head(10)
    .to_string(index=False)
)

print("\nSaved file:", CLEAN_WEATHER_PATH)
print("Saved rows:", len(clean_weather_df))
print("File exists:", CLEAN_WEATHER_PATH.exists())