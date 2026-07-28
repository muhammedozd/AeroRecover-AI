from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_propagation_classifier.pkl"

pipeline = joblib.load(MODEL_PATH)

print("Model başarıyla yüklendi.")



preprocessor=pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps["classifier"]

print("Preprocessor başarıyla alındı.")
print("Classifier başarıyla alındı.")

feature_names = preprocessor.get_feature_names_out()

print("Toplam feature sayısı:", len(feature_names))

print("\nİlk 10 feature:")
for feature in feature_names[:10]:
    print(feature)

importance_scores = classifier.feature_importances_

print("Importance sayisi:", len(importance_scores))

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_scores
})

feature_importance_df = feature_importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 features:")
print(feature_importance_df.head(20).to_string(index=False))
