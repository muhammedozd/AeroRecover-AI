import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_propagation_classifier.pkl"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset.csv"
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


# --------------------------------------------------
# Define model features
# --------------------------------------------------

categorical_features = [
    "PREV_DEST",
    "PREV_DELAY_LEVEL"
]

numerical_features = [
    "ROTATION_POSITION",
    "PREV_ARR_DELAY",
    "PREV_ARR_MIN",
    "PREV_CRS_ARR_MIN",
    "PLANNED_TURNAROUND",
    "TURN_BUFFER",
    "PREV_DELAY_RATIO",
    "HAS_BUFFER",
    "IS_SHORT_TURN",
    "PREV_DELAYED"
]

features = categorical_features + numerical_features

X = df[features]

print("X boyutu:", X.shape)
print(X.head())


# --------------------------------------------------
# Create SHAP sample
# --------------------------------------------------

sample_size = min(5000, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=42
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

explainer = shap.TreeExplainer(classifier)

print("SHAP Explainer olusturuldu.")


# --------------------------------------------------
# Calculate SHAP values
# --------------------------------------------------

shap_values = explainer(
    X_sample_dense
)

shap_values.feature_names = feature_names

print("SHAP Explanation olusturuldu.")
print("SHAP boyutu:", shap_values.shape)


# --------------------------------------------------
# Select one flight for local explanation
# --------------------------------------------------

single_flight_index = 0

single_flight_explanation = shap_values[
    single_flight_index
]

single_flight_probability = pipeline.predict_proba(
    X_sample.iloc[[single_flight_index]]
)[0, 1]

single_flight_prediction = pipeline.predict(
    X_sample.iloc[[single_flight_index]]
)[0]

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


# --------------------------------------------------
# Waterfall Plot
# --------------------------------------------------

waterfall_path = (
    FIGURES_DIR
    / "shap_waterfall.png"
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
    / "shap_summary.png"
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
    / "shap_dependence_prev_delay_ratio.png"
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