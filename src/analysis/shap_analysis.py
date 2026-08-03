import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb

from src.models.train_rotation_model import (
    MODEL_COLUMNS,
    create_time_masks,
    prepare_features,
)
# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_propagation_2023_time_split.pkl"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset_2023.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------

pipeline = joblib.load(MODEL_PATH)

print("Pipeline basariyla yuklendi.")

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Veri boyutu:", df.shape)
print("Sutun sayisi:", len(df.columns))

X, y, _, _ = prepare_features(df)

_, validation_mask, _ = create_time_masks(df)

X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]

print("Validation X boyutu:", X_validation.shape)
print("Validation y boyutu:", y_validation.shape)

# --------------------------------------------------
# Create SHAP sample
# --------------------------------------------------

sample_size = min(
    5000,
    len(X_validation),
)

X_sample = X_validation.sample(
    n=sample_size,
    random_state=42,
).copy()

print("SHAP ornek boyutu:", X_sample.shape)


# --------------------------------------------------
# Transform sample using trained preprocessor
# --------------------------------------------------

X_sample_transformed = preprocessor.transform(X_sample)

print(
    "Donusturulmus SHAP veri boyutu:",
    X_sample_transformed.shape
)


# --------------------------------------------------
# Convert transformed data to dense format
# --------------------------------------------------

if hasattr(X_sample_transformed, "toarray"):
    X_sample_dense = X_sample_transformed.toarray().astype(float)
else:
    X_sample_dense = X_sample_transformed.astype(float)

print("Dense SHAP veri boyutu:", X_sample_dense.shape)


# --------------------------------------------------
# Get transformed feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

print("Feature ismi sayisi:", len(feature_names))
print("Ilk 10 feature:")
print(feature_names[:10])


# --------------------------------------------------
# Create SHAP explainer
# --------------------------------------------------


booster = classifier.get_booster()
shap_dmatrix = xgb.DMatrix(
    X_sample_transformed,
    feature_names=feature_names.tolist(),
)

shap_contributions = booster.predict(
    shap_dmatrix,
    pred_contribs=True,
)
shap_values = shap.Explanation(
    values=shap_contributions[:, :-1],
    base_values=shap_contributions[:, -1],
    data=X_sample_dense,
    feature_names=feature_names,
)

print("SHAP Explanation olusturuldu.")
print("SHAP boyutu:", shap_values.shape)


# --------------------------------------------------
# Select one flight for local explanation
# --------------------------------------------------

#En yüksek olasılığın bulunduğu konumu verir
sample_probabilities = booster.predict(
    shap_dmatrix
)
single_flight_index = (
    sample_probabilities.argmax()
)

single_flight_probability = (
    sample_probabilities[single_flight_index]
)



single_flight_explanation = shap_values[
    single_flight_index
]


F1_OPTIMAL_THRESHOLD = 0.46

single_flight_prediction = int(
    single_flight_probability
    >= F1_OPTIMAL_THRESHOLD
)
print(
    "Secilen ucusun tahmin sinifi:",
    int(single_flight_prediction)
)

print(
    "Secilen ucusun gecikme yayilma olasiligi:",
    round(single_flight_probability, 4)
)

print("Secilen ucusun orijinal feature degerleri:")
print(X_sample.iloc[single_flight_index])


shap_raw_score = (
    single_flight_explanation.base_values
    + single_flight_explanation.values.sum()
)

model_raw_scores = booster.predict(
    shap_dmatrix,
    output_margin=True,
)

model_raw_score = model_raw_scores[
    single_flight_index
]

print("SHAP raw score :", round(float(shap_raw_score), 4))
print("Model raw score:", round(float(model_raw_score), 4))
# --------------------------------------------------
# Waterfall Plot
# --------------------------------------------------

waterfall_path = (
    FIGURES_DIR
    / "shap_waterfall_2023_validation.png"
)


shap.plots.waterfall(
    single_flight_explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    waterfall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Waterfall Plot kaydedildi:",
    waterfall_path
)


# --------------------------------------------------
# Summary Plot
# --------------------------------------------------

summary_path = (
    FIGURES_DIR
    / "shap_summary_2023_validation.png"
)

plt.figure()

shap.summary_plot(
    shap_values.values,
    X_sample_dense,
    feature_names=feature_names,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Summary Plot kaydedildi:",
    summary_path
)


# --------------------------------------------------
# Dependence Plot
# --------------------------------------------------

dependence_path = (
    FIGURES_DIR
    / "shap_dependence_prev_delay_ratio_2023_validation.png"
)

plt.figure()

shap.dependence_plot(
    "numerical__PREV_DELAY_RATIO",
    shap_values.values,
    X_sample_dense,
    feature_names=feature_names,
    interaction_index="auto",
    show=False
)

plt.tight_layout()

plt.savefig(
    dependence_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Dependence Plot kaydedildi:",
    dependence_path
)


# --------------------------------------------------
# Finish
# --------------------------------------------------

print()
print("Tum SHAP grafikleri basariyla olusturuldu.")
print("Grafik klasoru:", FIGURES_DIR)