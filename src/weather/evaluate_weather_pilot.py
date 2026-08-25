from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather-2023"
    / "ATL_rotation_weather_2023.parquet"
)

BASE_CATEGORICAL_FEATURES = [
    "PREV_DEST",
    "PREV_DELAY_LEVEL",
]

BASE_NUMERICAL_FEATURES = [
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
]

WEATHER_FEATURES = [
    "ORIGIN_WIND_DIRECTION_DEG",
    "ORIGIN_WIND_SPEED_MPS",
    "ORIGIN_WIND_IS_CALM",
    "ORIGIN_VISIBILITY_METERS",
    "ORIGIN_CEILING_METERS",
    "ORIGIN_CEILING_UNLIMITED",
    "ORIGIN_TEMPERATURE_C",
    "ORIGIN_DEW_POINT_C",
    "ORIGIN_SEA_LEVEL_PRESSURE_HPA",
    "ORIGIN_PRECIP_1H_MM",
    "ORIGIN_PRECIP_TRACE",
]
DECISION_THRESHOLD = 0.46

TARGET_COLUMN = "IS_DELAY_PROPAGATED"

DATA_COLUMNS = (
    ["FL_DATE"]
    + BASE_CATEGORICAL_FEATURES
    + BASE_NUMERICAL_FEATURES
    + WEATHER_FEATURES
    + [TARGET_COLUMN]
)


pilot_df = pd.read_parquet(
    DATA_PATH,
    columns=DATA_COLUMNS,
)

pilot_df["FL_DATE"] = pd.to_datetime(
    pilot_df["FL_DATE"],
    errors="raise",
)

def build_model_pipeline(numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                BASE_CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

print("Pilot dataset:", DATA_PATH)
print("Dataset shape:", pilot_df.shape)

print("\nObservation period:")
print(
    pilot_df["FL_DATE"].min(),
    "->",
    pilot_df["FL_DATE"].max(),
)

print("\nTarget counts:")
print(
    pilot_df[TARGET_COLUMN]
    .value_counts(dropna=False)
)

print("\nTarget rates:")
print(
    pilot_df[TARGET_COLUMN]
    .value_counts(normalize=True, dropna=False)
)

print("\nWeather missing values:")
print(
    pilot_df[WEATHER_FEATURES]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

train_mask = (
    pilot_df["FL_DATE"] < "2023-09-01"
)

validation_mask = (
    (pilot_df["FL_DATE"] >= "2023-09-01")
    & (pilot_df["FL_DATE"] < "2023-11-01")
)

test_mask = (
    pilot_df["FL_DATE"] >= "2023-11-01"
)

split_total = (
    train_mask.sum()
    + validation_mask.sum()
    + test_mask.sum()
)

if split_total != len(pilot_df):
    raise ValueError(
        "Time split does not cover the full pilot dataset."
    )

print("\nTime split sizes:")
print("Train:", train_mask.sum())
print("Validation:", validation_mask.sum())
print("Test:", test_mask.sum())
print("Total:", split_total)

print("\nPropagation rate by split:")
print(
    "Train:",
    pilot_df.loc[
        train_mask,
        TARGET_COLUMN,
    ].mean(),
)

print(
    "Validation:",
    pilot_df.loc[
        validation_mask,
        TARGET_COLUMN,
    ].mean(),
)

print(
    "Test:",
    pilot_df.loc[
        test_mask,
        TARGET_COLUMN,
    ].mean(),
)

baseline_numerical_features = (
    BASE_NUMERICAL_FEATURES
)

weather_numerical_features = (
    BASE_NUMERICAL_FEATURES
    + WEATHER_FEATURES
)

baseline_feature_columns = (
    BASE_CATEGORICAL_FEATURES
    + baseline_numerical_features
)

weather_feature_columns = (
    BASE_CATEGORICAL_FEATURES
    + weather_numerical_features
)

X_baseline = pilot_df[
    baseline_feature_columns
]

X_weather = pilot_df[
    weather_feature_columns
]

y = pilot_df[TARGET_COLUMN]

X_baseline_train = X_baseline.loc[
    train_mask
]

X_weather_train = X_weather.loc[
    train_mask
]

y_train = y.loc[
    train_mask
]

baseline_pipeline = build_model_pipeline(
    baseline_numerical_features
)

weather_pipeline = build_model_pipeline(
    weather_numerical_features
)

print("\nTraining ATL baseline model...")

baseline_pipeline.fit(
    X_baseline_train,
    y_train,
)

print("ATL baseline model trained.")

print("\nTraining ATL weather model...")

weather_pipeline.fit(
    X_weather_train,
    y_train,
)

print("ATL weather model trained.")

X_baseline_validation = X_baseline.loc[
    validation_mask
]

X_weather_validation = X_weather.loc[
    validation_mask
]

y_validation = y.loc[
    validation_mask
]

baseline_validation_probability = (
    baseline_pipeline.predict_proba(
        X_baseline_validation
    )[:, 1]
)

weather_validation_probability = (
    weather_pipeline.predict_proba(
        X_weather_validation
    )[:, 1]
)


def calculate_probability_metrics(
    y_true,
    probability,
):
    return {
        "ROC_AUC": roc_auc_score(
            y_true,
            probability,
        ),
        "PR_AUC": average_precision_score(
            y_true,
            probability,
        ),
        "LOG_LOSS": log_loss(
            y_true,
            probability,
        ),
        "BRIER_SCORE": brier_score_loss(
            y_true,
            probability,
        ),
    }


baseline_metrics = calculate_probability_metrics(
    y_validation,
    baseline_validation_probability,
)

weather_metrics = calculate_probability_metrics(
    y_validation,
    weather_validation_probability,
)

comparison_df = pd.DataFrame(
    {
        "ATL_BASELINE": baseline_metrics,
        "ATL_WEATHER": weather_metrics,
    }
)

comparison_df["DIFFERENCE"] = (
    comparison_df["ATL_WEATHER"]
    - comparison_df["ATL_BASELINE"]
)

print("\nValidation probability metrics:")
print(comparison_df)

def calculate_decision_metrics(
    y_true,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "PRECISION": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "RECALL": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "TRUE_NEGATIVE": tn,
        "FALSE_POSITIVE": fp,
        "FALSE_NEGATIVE": fn,
        "TRUE_POSITIVE": tp,
    }


baseline_decision_metrics = (
    calculate_decision_metrics(
        y_validation,
        baseline_validation_probability,
        DECISION_THRESHOLD,
    )
)

weather_decision_metrics = (
    calculate_decision_metrics(
        y_validation,
        weather_validation_probability,
        DECISION_THRESHOLD,
    )
)

decision_comparison_df = pd.DataFrame(
    {
        "ATL_BASELINE": baseline_decision_metrics,
        "ATL_WEATHER": weather_decision_metrics,
    }
)

decision_comparison_df["DIFFERENCE"] = (
    decision_comparison_df["ATL_WEATHER"]
    - decision_comparison_df["ATL_BASELINE"]
)

print(
    f"\nValidation decision metrics "
    f"at threshold {DECISION_THRESHOLD}:"
)

print(decision_comparison_df)

def find_best_f1_threshold(
    y_true,
    probability,
):
    threshold_results = []

    for threshold in np.arange(
        0.05,
        0.951,
        0.01,
    ):
        prediction = (
            probability >= threshold
        ).astype(int)

        threshold_results.append(
            {
                "THRESHOLD": threshold,
                "PRECISION": precision_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
                "RECALL": recall_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
            }
        )

    results_df = pd.DataFrame(
        threshold_results
    )

    best_index = (
        results_df["F1"].idxmax()
    )

    return (
        results_df,
        results_df.loc[best_index],
    )


baseline_threshold_results, baseline_best = (
    find_best_f1_threshold(
        y_validation,
        baseline_validation_probability,
    )
)

weather_threshold_results, weather_best = (
    find_best_f1_threshold(
        y_validation,
        weather_validation_probability,
    )
)

best_threshold_comparison = pd.DataFrame(
    {
        "ATL_BASELINE": baseline_best,
        "ATL_WEATHER": weather_best,
    }
)

print("\nBest validation F1 thresholds:")
print(best_threshold_comparison)