"""AeroRecover AI historical validation overview."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.decision_support.assessment_service import build_decision_report
from src.decision_support.contracts import FlightDecisionInput
from src.models.rotation_model_contract import MODEL_THRESHOLD, MODEL_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "decision_support_replay_validation_full_enhanced.parquet"
)
REPLAY_COLUMNS = [
    "START_FLIGHT_ID", "EDGE_COUNT", "PROPAGATION_PROBABILITY",
    "PREV_ARR_DELAY", "TURN_BUFFER", "PREV_DELAY_RATIO", "PLANNED_TURNAROUND",
    "MODEL_VERSION", "MODEL_THRESHOLD",
]


@st.cache_data
def load_replay_data() -> pd.DataFrame:
    if not REPLAY_DATA_PATH.exists():
        raise FileNotFoundError(f"Replay dataset not found: {REPLAY_DATA_PATH}")
    replay = pd.read_parquet(REPLAY_DATA_PATH, columns=REPLAY_COLUMNS)
    if set(replay["MODEL_VERSION"].dropna().unique()) != {MODEL_VERSION}:
        raise ValueError("Replay data does not match the active model version.")
    if not replay["MODEL_THRESHOLD"].dropna().eq(MODEL_THRESHOLD).all():
        raise ValueError("Replay data does not match the active model threshold.")
    return replay


st.set_page_config(page_title="AeroRecover AI", page_icon="AR", layout="wide")
st.title("AeroRecover AI")
st.write("Explainable historical decision support for flight-delay propagation.")
st.caption(
    f"Model: {MODEL_VERSION} · Frozen threshold: {MODEL_THRESHOLD:.2f} · "
    "Historical validation only — not live operations"
)

try:
    replay_data = load_replay_data()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

# Keep the landing page bounded; the full replay page provides date/airline filters.
options = replay_data["START_FLIGHT_ID"].head(5_000).tolist()
selected_flight_id = st.selectbox(
    "Historical validation flight",
    options=options,
    help="Use Historical Replay for full date, airline, route, map, SHAP, and PDF controls.",
)
selected = replay_data.loc[replay_data["START_FLIGHT_ID"].eq(selected_flight_id)]
if len(selected) != 1:
    st.error("The selected replay flight is not unique.")
    st.stop()
row = selected.iloc[0]
decision_input = FlightDecisionInput(
    propagation_probability=float(row["PROPAGATION_PROBABILITY"]),
    previous_arrival_delay=float(row["PREV_ARR_DELAY"]),
    turn_buffer=float(row["TURN_BUFFER"]),
    previous_delay_ratio=float(row["PREV_DELAY_RATIO"]),
    planned_turnaround=float(row["PLANNED_TURNAROUND"]),
    downstream_edge_count=int(row["EDGE_COUNT"]),
)
decision_report = build_decision_report(decision_input)

probability_column, priority_column = st.columns(2)
probability_column.metric("Propagation Probability", f"{decision_input.propagation_probability:.1%}")
priority_column.metric("Operational Priority", decision_report.assessment.priority.name)
st.subheader("Recommendations")
for recommendation in decision_report.recommendations:
    st.write(f"**{recommendation['priority'].name}** — {recommendation['action']}")
    st.caption(recommendation["reason"])

st.info("Open the Historical Replay page for filters, chain animation, map, SHAP, and PDF reporting.")
