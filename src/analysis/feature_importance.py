import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]
feature_names = preprocessor.get_feature_names_out()
print(feature_names[:20])
print("Toplam feature sayısı:", len(feature_names))