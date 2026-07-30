import joblib
from pathlib import Path

import joblib
import streamlit as st

model = joblib.load(
    "models/xgboost_propagation_classifier.pkl"
)

print(model.feature_names_in_)

