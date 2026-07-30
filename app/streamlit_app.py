import streamlit as st



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