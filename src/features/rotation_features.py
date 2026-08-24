"""Shared feature engineering for rotation model training and inference."""

import numpy as np
import pandas as pd

from src.models.rotation_model_contract import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS


def build_rotation_model_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the ordered 24-feature model matrix from pre-departure data."""
    missing_raw = [column for column in RAW_FEATURE_COLUMNS if column not in rows.columns]
    if missing_raw:
        raise ValueError(f"Cannot build rotation features; missing source columns: {missing_raw}")

    result = rows.copy()
    flight_dates = pd.to_datetime(result["FL_DATE"], errors="coerce")
    crs_dep_time = pd.to_numeric(result["CRS_DEP_TIME"], errors="coerce")
    whole_time = np.floor(crs_dep_time)
    hours = np.floor_divide(whole_time, 100)
    minutes = np.mod(whole_time, 100)
    valid_time = (
        crs_dep_time.notna()
        & np.isclose(crs_dep_time, whole_time)
        & hours.between(0, 23)
        & minutes.between(0, 59)
    )
    minute_of_day = (hours * 60 + minutes).where(valid_time)
    result["CRS_DEP_MIN_SAFE"] = minute_of_day
    angle = 2 * np.pi * minute_of_day / 1440
    result["CRS_DEP_TIME_SIN"] = np.sin(angle)
    result["CRS_DEP_TIME_COS"] = np.cos(angle)
    result["DAY_OF_WEEK"] = flight_dates.dt.dayofweek
    result["MONTH"] = flight_dates.dt.month
    result["IS_WEEKEND"] = flight_dates.dt.dayofweek.isin([5, 6]).astype("Int64")

    positive_delay = pd.to_numeric(result["PREV_ARR_DELAY"], errors="coerce").clip(lower=0)
    planned_turn = pd.to_numeric(result["PLANNED_TURNAROUND"], errors="coerce")
    denominator = planned_turn.mask(planned_turn == 0, pd.NA)
    result["DELAY_EXCESS_OVER_TURN"] = (positive_delay - planned_turn).clip(lower=0)
    result["AVAILABLE_BUFFER_RATIO"] = (
        (planned_turn - positive_delay).clip(lower=0) / denominator
    )
    result["PREV_DELAY_SHORT_TURN_INTERACTION"] = (
        positive_delay * pd.to_numeric(result["IS_SHORT_TURN"], errors="coerce")
    )

    features = result.loc[:, FEATURE_COLUMNS]
    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Rotation feature engineering produced an invalid feature order.")
    numeric = features.select_dtypes(include=["number"])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Rotation feature engineering produced infinite values.")
    return features, flight_dates
