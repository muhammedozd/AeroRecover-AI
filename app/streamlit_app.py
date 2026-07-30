import streamlit as st


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

    score = calculate_operational_risk_score(
        flight_data
    )

    risk_level = determine_risk_level(
    score
)

score_column, level_column = st.columns(2)

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


 
     
    
    


