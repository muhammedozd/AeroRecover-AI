"""Single source of truth for the active rotation propagation model."""

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_propagation_2023_full_enhanced.pkl"
MODEL_THRESHOLD = 0.47
MODEL_VERSION = "xgboost_2023_full_enhanced"

CATEGORICAL_FEATURES = [
    "PREV_DEST",
    "PREV_DELAY_LEVEL",
    "DEST",
    "OP_UNIQUE_CARRIER",
]
NUMERICAL_FEATURES = [
    "ROTATION_POSITION",
    "PREV_ARR_DELAY",
    "PREV_ARR_MIN",
    "PREV_CRS_ARR_MIN",
    "PLANNED_TURNAROUND",
    "TURN_BUFFER",
    "PREV_DELAY_RATIO",
    "HAS_BUFFER",
    "IS_SHORT_TURN",
    "PREV_DELAYED",
    "DISTANCE",
    "CRS_DEP_MIN_SAFE",
    "CRS_DEP_TIME_SIN",
    "CRS_DEP_TIME_COS",
    "DAY_OF_WEEK",
    "MONTH",
    "IS_WEEKEND",
    "DELAY_EXCESS_OVER_TURN",
    "AVAILABLE_BUFFER_RATIO",
    "PREV_DELAY_SHORT_TURN_INTERACTION",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
RAW_FEATURE_COLUMNS = [
    "FL_DATE",
    "CRS_DEP_TIME",
    "PREV_DEST",
    "PREV_DELAY_LEVEL",
    "DEST",
    "OP_UNIQUE_CARRIER",
    "ROTATION_POSITION",
    "PREV_ARR_DELAY",
    "PREV_ARR_MIN",
    "PREV_CRS_ARR_MIN",
    "PLANNED_TURNAROUND",
    "TURN_BUFFER",
    "PREV_DELAY_RATIO",
    "HAS_BUFFER",
    "IS_SHORT_TURN",
    "PREV_DELAYED",
    "DISTANCE",
]
FORBIDDEN_FEATURES = {
    "DEP_DELAY", "ARR_DELAY", "LATE_AIRCRAFT_DELAY",
    "PROPAGATED_DELAY_MINUTES", "ACTUAL_TURNAROUND", "RECOVERY_MARGIN",
    "DELAYED", "DEP_TIME", "ARR_TIME", "IS_DELAY_PROPAGATED",
}


def selected_model_features(pipeline) -> list[str]:
    """Return source columns selected by a fitted sklearn ColumnTransformer."""
    selected: list[str] = []
    preprocessor = pipeline.named_steps["preprocessor"]
    transformers = getattr(preprocessor, "transformers_", preprocessor.transformers)
    for name, _, columns in transformers:
        if name in {"categorical", "numerical"}:
            selected.extend(str(column) for column in columns)
    return selected


def validate_feature_contract(columns, *, exact: bool = True) -> None:
    """Reject missing, extra, duplicated, or leakage-prone model columns."""
    received = list(columns)
    duplicates = sorted({column for column in received if received.count(column) > 1})
    missing = [column for column in FEATURE_COLUMNS if column not in received]
    extra = [column for column in received if column not in FEATURE_COLUMNS]
    leaked = sorted(set(received) & FORBIDDEN_FEATURES)
    if duplicates or missing or leaked or (exact and extra):
        raise ValueError(
            "Rotation feature contract violation: "
            f"missing={missing}, extra={extra if exact else []}, "
            f"duplicates={duplicates}, forbidden={leaked}."
        )
    if exact and received != FEATURE_COLUMNS:
        raise ValueError(
            "Rotation feature order mismatch: "
            f"expected={FEATURE_COLUMNS}, received={received}."
        )


def validate_pipeline_contract(pipeline) -> None:
    selected = selected_model_features(pipeline)
    validate_feature_contract(selected, exact=True)


def load_model_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Active rotation model not found: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    validate_pipeline_contract(pipeline)
    return pipeline
