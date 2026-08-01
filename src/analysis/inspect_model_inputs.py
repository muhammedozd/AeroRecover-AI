import joblib

model = joblib.load(
    "models/xgboost_propagation_classifier.pkl"
)

print(model.feature_names_in_)

