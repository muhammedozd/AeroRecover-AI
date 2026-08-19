"""Validated local XGBoost SHAP explanations for replay flights."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.train_rotation_model import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_propagation_2023_time_split.pkl"
ROTATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv"
MODEL_VERSION = "xgboost_2023_time_split"
FLIGHT_ID_COLUMNS = [
    "FL_DATE", "TAIL_NUM", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM",
    "ORIGIN", "DEST", "CRS_DEP_TIME",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


@dataclass(frozen=True)
class LocalShapExplanation:
    target_flight_id: str
    model_probability: float
    base_value: float
    model_raw_score: float
    shap_raw_score: float
    reconstruction_error: float
    contributions: pd.DataFrame


def load_model_pipeline():
    """Load the exact fitted pipeline used to score validation rotations."""
    return joblib.load(MODEL_PATH)


def _build_target_flight_ids(rows: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(rows["FL_DATE"], format="%Y-%m-%d", errors="raise")
    flight_numbers = rows["OP_CARRIER_FL_NUM"].astype(int).astype(str)
    departure_times = rows["CRS_DEP_TIME"].astype(int).astype(str).str.zfill(4)
    return (
        dates.dt.strftime("%Y%m%d")
        + "_" + rows["OP_UNIQUE_CARRIER"].astype(str)
        + "_" + flight_numbers
        + "_" + rows["ORIGIN"].astype(str)
        + "_" + rows["DEST"].astype(str)
        + "_" + departure_times
        + "_" + rows["TAIL_NUM"].astype(str)
    )


def load_single_validation_rotation(
    target_flight_id: str,
    *,
    chunksize: int = 150_000,
) -> pd.DataFrame:
    """Return exactly one September-October 2023 model row for a target ID."""
    usecols = list(dict.fromkeys([*FLIGHT_ID_COLUMNS, *FEATURE_COLUMNS]))
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        ROTATION_DATA_PATH,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        dates = pd.to_datetime(chunk["FL_DATE"], format="%Y-%m-%d", errors="coerce")
        validation = chunk.loc[
            dates.ge("2023-09-01") & dates.lt("2023-11-01")
        ].copy()
        if validation.empty:
            continue
        validation_ids = _build_target_flight_ids(validation)
        matched = validation.loc[validation_ids.eq(target_flight_id), FEATURE_COLUMNS]
        if not matched.empty:
            matches.append(matched.copy())

    match_count = sum(len(match) for match in matches)
    if match_count != 1:
        raise ValueError(
            f"Expected exactly one validation rotation for TARGET_FLIGHT_ID "
            f"{target_flight_id}, found {match_count}."
        )
    return pd.concat(matches, ignore_index=True)


def _original_feature_mapping(preprocessor) -> list[str]:
    """Map fitted transformed columns to source columns without name guessing."""
    mapping: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue
        output_slice = preprocessor.output_indices_[transformer_name]
        expected_count = output_slice.stop - output_slice.start
        if hasattr(transformer, "categories_"):
            for column, categories in zip(columns, transformer.categories_):
                mapping.extend([str(column)] * len(categories))
        elif expected_count == len(columns):
            mapping.extend(str(column) for column in columns)
        else:
            raise ValueError(
                f"Unsupported fitted transformer for SHAP grouping: {transformer_name}."
            )
        if len(mapping) < output_slice.stop:
            raise ValueError(
                f"Transformer {transformer_name} produced an unexpected feature count."
            )
        if len(mapping[output_slice.start:output_slice.stop]) != expected_count:
            raise ValueError(
                f"Transformer {transformer_name} feature mapping is inconsistent."
            )
    return mapping


def explain_validation_flight(
    target_flight_id: str,
    *,
    pipeline=None,
    expected_probability: float | None = None,
    tolerance: float = 1e-5,
) -> LocalShapExplanation:
    """Explain the exact validation row used by the fitted model pipeline."""
    model_pipeline = pipeline if pipeline is not None else load_model_pipeline()
    model_row = load_single_validation_rotation(target_flight_id)
    if len(model_row) != 1:
        raise ValueError("Local SHAP requires exactly one model input row.")

    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(model_row)
    if transformed.shape[0] != 1:
        raise ValueError("Preprocessing changed the selected row count.")

    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    dmatrix = xgb.DMatrix(transformed, feature_names=transformed_feature_names)
    booster = classifier.get_booster()
    contribution_row = booster.predict(dmatrix, pred_contribs=True)[0]
    model_raw_score = float(booster.predict(dmatrix, output_margin=True)[0])
    model_probability = float(model_pipeline.predict_proba(model_row)[0, 1])
    base_value = float(contribution_row[-1])
    transformed_contributions = contribution_row[:-1]
    shap_raw_score = float(base_value + transformed_contributions.sum())
    reconstruction_error = abs(shap_raw_score - model_raw_score)
    if reconstruction_error >= tolerance:
        raise ValueError(
            "SHAP raw-score reconstruction failed: "
            f"error={reconstruction_error:.8g}, tolerance={tolerance:.8g}."
        )

    probability_from_raw = 1.0 / (1.0 + exp(-model_raw_score))
    probability_error = abs(probability_from_raw - model_probability)
    if probability_error >= tolerance:
        raise ValueError(
            "Raw-score probability validation failed: "
            f"error={probability_error:.8g}, tolerance={tolerance:.8g}."
        )
    if expected_probability is not None:
        expected_error = abs(model_probability - float(expected_probability))
        if expected_error >= tolerance:
            raise ValueError(
                "Selected replay probability does not match its model rotation row: "
                f"error={expected_error:.8g}, tolerance={tolerance:.8g}."
            )

    source_features = _original_feature_mapping(preprocessor)
    if len(source_features) != len(transformed_contributions):
        raise ValueError(
            "Transformed SHAP columns cannot be mapped exactly to source features: "
            f"{len(source_features)} mappings for {len(transformed_contributions)} columns."
        )
    grouped = pd.DataFrame(
        {"feature": source_features, "shap_value": transformed_contributions}
    ).groupby("feature", as_index=False, sort=False)["shap_value"].sum()
    source_values = model_row.iloc[0]
    grouped["feature_value"] = grouped["feature"].map(
        lambda feature: source_values[feature]
    )
    grouped["direction"] = np.where(
        grouped["shap_value"] >= 0, "Increasing", "Decreasing"
    )
    grouped["absolute_importance"] = grouped["shap_value"].abs()
    grouped = grouped.sort_values("absolute_importance", ascending=False).reset_index(drop=True)

    return LocalShapExplanation(
        target_flight_id=target_flight_id,
        model_probability=model_probability,
        base_value=base_value,
        model_raw_score=model_raw_score,
        shap_raw_score=shap_raw_score,
        reconstruction_error=reconstruction_error,
        contributions=grouped,
    )
