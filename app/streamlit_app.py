import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_propagation_classifier.pkl"
)

#model yükleme fonksiyonu
# @st.cache_resource,
# Streamlit her input değiştiğinde modeli diskten tekrar tekrar
# yüklemesin diye modeli bellekte tutar.

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_explainer(_model):
    classifier = _model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    return explainer


def get_previous_delay_level(delay):
    if delay < 0:
        return "Early"
    if delay < 15:
        return "OnTime"
    if delay < 30:
        return "Minor"
    if delay < 60:
        return "Moderate"
    return "Severe"


def time_to_minutes(time_value):
    return (
        time_value.hour * 60
        + time_value.minute
    )



model = load_model()
shap_explainer = load_shap_explainer(model)

from src.decision_support.recommendation_engine import generate_recommendations
from src.optimization.recovery_optimizer import (
    calculate_operational_risk_score,
    determine_risk_level,
    identify_risk_factors,
)


st.set_page_config(
    page_title="AeroRecover AI",
    page_icon="✈️",
)

st.title("✈️ AeroRecover AI")

st.write(
    "Explainable Operational Decision Support System "
    "for Flight Delay Propagation Prediction"
)

st.subheader("Flight Inputs")

previous_arrival_delay = st.number_input(
    "Previous Arrival Delay (minutes)",
    min_value=0,
    value=0,
)

turn_buffer = st.number_input(
    "Turn Buffer (minutes)",
    min_value=0,
    value=0,
)

prev_delay_ratio = st.number_input(
    "Previous Delay Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)


planned_turnaround = st.number_input(
    "Planned Turnaround (minutes)",
    min_value=0,
    value=0,
)

previous_destination = st.text_input(
    "Previous destination airport",
    value="JFK",
    max_chars=3,
)

rotation_position = st.number_input(
    "Rotation position",
    min_value=2,
    value=2,
    step=1,
)

scheduled_arrival_time = st.time_input(
    "Previous scheduled arrival time"
)

actual_arrival_time = st.time_input(
    "Previous actual arrival time"
)

analyze_button = st.button(
    "✈️ Analyze Flight",
    use_container_width=True,
)
if analyze_button:
    st.success(
        "Flight analysis started."
    )

    flight_data = {
        "PREV_ARR_DELAY": previous_arrival_delay,
        "TURN_BUFFER": turn_buffer,
        "PREV_DELAY_RATIO": prev_delay_ratio,
        "PLANNED_TURNAROUND": planned_turnaround,
    }

    flight_data["HAS_BUFFER"] = int(
    flight_data["TURN_BUFFER"] > 0
)

    flight_data["IS_SHORT_TURN"] = int(
    flight_data["PLANNED_TURNAROUND"] < 45
)

    flight_data["PREV_DELAYED"] = int(
    flight_data["PREV_ARR_DELAY"] >= 15
)

    flight_data["PREV_DELAY_LEVEL"] = (
    get_previous_delay_level(
        flight_data["PREV_ARR_DELAY"]
    )
)

    flight_data["PREV_DEST"] = (
    previous_destination.strip().upper()
)

    flight_data["ROTATION_POSITION"] = int(
    rotation_position
)

    flight_data["PREV_CRS_ARR_MIN"] = time_to_minutes(
    scheduled_arrival_time
)

    flight_data["PREV_ARR_MIN"] = time_to_minutes(
    actual_arrival_time
)

    model_input = pd.DataFrame(
    [flight_data]
)

    model_input = model_input[
    model.feature_names_in_
]

    preprocessor = model.named_steps["preprocessor"]

    transformed_input = preprocessor.transform(
    model_input
)

    shap_values = shap_explainer(
    transformed_input
)
    feature_names = preprocessor.get_feature_names_out()

    shap_table = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values.values[0],
})

    shap_table["importance"] = shap_table["shap_value"].abs()

    top_factors = (
    shap_table
    .sort_values(
        by="importance",
        ascending=False
    )
    .head(5)
)


    top_factors["feature"] = (
    top_factors["feature"]
    .str.replace("numerical__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)

    st.dataframe(top_factors)
    
    


    propagation_probability = model.predict_proba(
    model_input
)[0][1]

    propagation_percentage = (
    propagation_probability * 100
)


    score = calculate_operational_risk_score(
        flight_data
    )

    risk_level = determine_risk_level(
    score


)
    prob_column, score_column, level_column = st.columns(3)

    with score_column:
        st.metric(
        label="Operational Risk Score",
        value=f"{score} / 11",
    )

    with level_column:
        st.metric(
        label="Risk Level",
        value=risk_level,
    )
    risk_factors = identify_risk_factors(
        flight_data
    )

    with prob_column:
        st.metric(
        "Propagation Probability",
        f"{propagation_percentage:.1f}%",
    )

    st.subheader("Risk Factors")

    for factor in risk_factors:
        st.write(f"- {factor}")

    
    recommendations = generate_recommendations(
        flight_data
    )

    st.subheader("Recommendations")


    for recommendation in recommendations:
       st.write(
        f"**{recommendation['priority']}** - "
        f"{recommendation['action']}"
    )
       st.caption(
        recommendation["reason"]
    )


 
     
    
    


