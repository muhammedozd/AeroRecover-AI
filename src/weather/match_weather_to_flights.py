from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROTATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset_2023.csv"
)

WEATHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather-2023"
    / "ATL_weather_2023_clean.csv"
)

MATCHED_WEATHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather-2023"
    / "ATL_rotation_weather_2023.parquet"
)


rotation_df = pd.read_csv(
    ROTATION_PATH,
    low_memory=False,
)

weather_df = pd.read_csv(
    WEATHER_PATH,
)

print("Rotation dataset shape:", rotation_df.shape)
print("Weather dataset shape:", weather_df.shape)

print("\nRotation time columns:")
print(
    rotation_df[
        [
            "FL_DATE",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nCRS_DEP_TIME data type:")
print(rotation_df["CRS_DEP_TIME"].dtype)

print("\nATL departure flights:")
print((rotation_df["ORIGIN"] == "ATL").sum())

# ATL kalkışlarını seç
atl_flights_df = rotation_df[
    rotation_df["ORIGIN"] == "ATL"
].copy()

atl_flights_df["FL_DATE"] = pd.to_datetime(
    atl_flights_df["FL_DATE"],
    format="%Y-%m-%d",
    errors="raise",
)

atl_flights_df["CRS_DEP_TIME"] = pd.to_numeric(
    atl_flights_df["CRS_DEP_TIME"],
    errors="coerce",
).astype("Int64")

atl_flights_df["CRS_DEP_HOUR"] = (
    atl_flights_df["CRS_DEP_TIME"] // 100
)

atl_flights_df["CRS_DEP_MINUTE"] = (
    atl_flights_df["CRS_DEP_TIME"] % 100
)

valid_departure_time = (
    atl_flights_df["CRS_DEP_HOUR"].between(0, 23)
    & atl_flights_df["CRS_DEP_MINUTE"].between(0, 59)
)

print(
    "\nMissing scheduled departure times:",
    atl_flights_df["CRS_DEP_TIME"].isna().sum(),
)

print(
    "Invalid scheduled departure times:",
    (~valid_departure_time).sum(),
)

print("\nInvalid time sample:")
print(
    atl_flights_df.loc[
        ~valid_departure_time,
        [
            "FL_DATE",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "CRS_DEP_HOUR",
            "CRS_DEP_MINUTE",
        ],
    ]
    .head(10)
    .to_string(index=False)
)

# Tarih, saat ve dakikayı birleştir
atl_flights_df["SCHEDULED_DEPARTURE_LOCAL"] = (
    atl_flights_df["FL_DATE"]
    + pd.to_timedelta(
        atl_flights_df["CRS_DEP_HOUR"],
        unit="h",
    )
    + pd.to_timedelta(
        atl_flights_df["CRS_DEP_MINUTE"],
        unit="m",
    )
)

# Atlanta saat dilimini tanımla
scheduled_departure_atl = (
    atl_flights_df["SCHEDULED_DEPARTURE_LOCAL"]
    .dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="NaT",
    )
)

# Atlanta saatinden UTC'ye çevir
atl_flights_df["SCHEDULED_DEPARTURE_UTC"] = (
    scheduled_departure_atl
    .dt.tz_convert("UTC")
)

print(
    "\nTimezone conversion missing rows:",
    atl_flights_df["SCHEDULED_DEPARTURE_UTC"]
    .isna()
    .sum(),
)

print("\nScheduled departure time sample:")
print(
    atl_flights_df[
        [
            "FL_DATE",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "SCHEDULED_DEPARTURE_LOCAL",
            "SCHEDULED_DEPARTURE_UTC",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nTimezone conversion problem sample:")
print(
    atl_flights_df[
        atl_flights_df[
            "SCHEDULED_DEPARTURE_UTC"
        ].isna()
    ][
        [
            "FL_DATE",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "SCHEDULED_DEPARTURE_LOCAL",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# Hava gözlem zamanını UTC olarak oku
weather_df["OBSERVATION_TIME_UTC"] = pd.to_datetime(
    weather_df["OBSERVATION_TIME_UTC"],
    utc=True,
    errors="raise",
)

weather_columns = [
    "OBSERVATION_TIME_UTC",
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

origin_weather_df = (
    weather_df[weather_columns]
    .copy()
    .rename(
        columns={
            "OBSERVATION_TIME_UTC":
                "ORIGIN_WEATHER_TIME_UTC",
            "REPORT_TYPE":
                "ORIGIN_WEATHER_REPORT_TYPE",
            "WIND_DIRECTION_DEG":
                "ORIGIN_WIND_DIRECTION_DEG",
            "WIND_SPEED_MPS":
                "ORIGIN_WIND_SPEED_MPS",
            "WIND_IS_CALM":
                "ORIGIN_WIND_IS_CALM",
            "VISIBILITY_METERS":
                "ORIGIN_VISIBILITY_METERS",
            "CEILING_METERS":
                "ORIGIN_CEILING_METERS",
            "CEILING_UNLIMITED":
                "ORIGIN_CEILING_UNLIMITED",
            "TEMPERATURE_C":
                "ORIGIN_TEMPERATURE_C",
            "DEW_POINT_C":
                "ORIGIN_DEW_POINT_C",
            "SEA_LEVEL_PRESSURE_HPA":
                "ORIGIN_SEA_LEVEL_PRESSURE_HPA",
            "PRECIP_1H_MM":
                "ORIGIN_PRECIP_1H_MM",
            "PRECIP_TRACE":
                "ORIGIN_PRECIP_TRACE",
        }
    )
)

atl_flights_df["ROTATION_ROW_ID"] = (
    atl_flights_df.index
)

atl_flights_df = atl_flights_df.sort_values(
    "SCHEDULED_DEPARTURE_UTC"
)

origin_weather_df = origin_weather_df.sort_values(
    "ORIGIN_WEATHER_TIME_UTC"
)

matched_flights_df = pd.merge_asof(
    atl_flights_df,
    origin_weather_df,
    left_on="SCHEDULED_DEPARTURE_UTC",
    right_on="ORIGIN_WEATHER_TIME_UTC",
    direction="backward",
    tolerance=pd.Timedelta("2h"),
)

matched_flights_df["ORIGIN_WEATHER_AGE_MIN"] = (
    (
        matched_flights_df["SCHEDULED_DEPARTURE_UTC"]
        - matched_flights_df["ORIGIN_WEATHER_TIME_UTC"]
    )
    .dt.total_seconds()
    / 60
)

matched_count = (
    matched_flights_df["ORIGIN_WEATHER_TIME_UTC"]
    .notna()
    .sum()
)

unmatched_count = (
    matched_flights_df["ORIGIN_WEATHER_TIME_UTC"]
    .isna()
    .sum()
)

match_rate = (
    matched_count
    / len(matched_flights_df)
    * 100
)

print("\nOrigin weather matching results:")
print("ATL flights:", len(matched_flights_df))
print("Matched flights:", matched_count)
print("Unmatched flights:", unmatched_count)
print(f"Match rate: {match_rate:.2f}%")

print("\nWeather age summary in minutes:")
print(
    matched_flights_df[
        "ORIGIN_WEATHER_AGE_MIN"
    ].describe()
)

print("\nMatched flight sample:")
print(
    matched_flights_df[
        [
            "FL_DATE",
            "ORIGIN",
            "DEST",
            "SCHEDULED_DEPARTURE_UTC",
            "ORIGIN_WEATHER_TIME_UTC",
            "ORIGIN_WEATHER_AGE_MIN",
            "ORIGIN_WIND_SPEED_MPS",
            "ORIGIN_VISIBILITY_METERS",
            "ORIGIN_CEILING_METERS",
            "ORIGIN_TEMPERATURE_C",
            "ORIGIN_PRECIP_1H_MM",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

unmatched_flights_df = matched_flights_df[
    matched_flights_df[
        "ORIGIN_WEATHER_TIME_UTC"
    ].isna()
].copy()

matched_flights_df = (
    matched_flights_df
    .sort_values("ROTATION_ROW_ID")
    .reset_index(drop=True)
)

MATCHED_WEATHER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

matched_flights_df.to_parquet(
    MATCHED_WEATHER_PATH,
    index=False,
    compression="snappy",
)

print("\nMatched dataset saved:")
print(MATCHED_WEATHER_PATH)
print("Saved rows:", len(matched_flights_df))
print("File exists:", MATCHED_WEATHER_PATH.exists())