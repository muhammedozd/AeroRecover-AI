"""Historical validation flight replay dashboard."""

from __future__ import annotations

from html import escape
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_support.assessment_service import build_decision_report
from src.decision_support.contracts import FlightDecisionInput
from src.visualization.propagation_map import (
    build_itinerary,
    build_propagation_figure,
    load_airport_reference,
    match_itinerary_coordinates,
)

REPLAY_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "decision_support_replay_validation.parquet"
SCORED_EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "graph" / "scored_tail_edges_2023_validation.parquet"
CHAIN_COLUMNS = [
    "SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID", "CONNECTION_AIRPORT",
    "PROPAGATION_PROBABILITY", "PROPAGATION_ALERT", "PLANNED_CONNECTION_MINUTES",
]

st.set_page_config(page_title="Historical Replay", page_icon="AR", layout="wide")


st.markdown(
    """
    <style>
    :root { --ar-bg:#071827; --ar-secondary:#0B2237; --ar-panel:#0E2A43; --ar-raised:#12344F;
      --ar-border:#245573; --ar-accent:#38BDF8; --ar-cyan:#22D3EE; --ar-text:#F2F7FB;
      --ar-secondary-text:#A5BAC9; --ar-muted:#7892A5; }
    .stApp,[data-testid="stAppViewContainer"]{background:var(--ar-bg)}
    [data-testid="stHeader"]{background:rgba(7,24,39,.9)}
    [data-testid="stSidebar"]{background:var(--ar-secondary);border-right:1px solid var(--ar-border);max-width:292px}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:1.5rem}
    [data-testid="stSidebar"] h2{color:var(--ar-text);font-size:1rem;letter-spacing:.04em;margin-bottom:.65rem}
    [data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:var(--ar-secondary-text)}
    [data-testid="stSidebar"] span[data-baseweb="tag"]{background:#123E5D;border:1px solid #286B8F;color:var(--ar-text)}
    [data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"]{background:var(--ar-accent);border-color:var(--ar-accent)}
    .block-container{max-width:1540px;padding-top:1.5rem;padding-bottom:2.5rem}
    .ar-header{margin:0 0 .8rem}.ar-eyebrow{color:var(--ar-accent);font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
    .ar-title{color:var(--ar-text);font-size:clamp(2rem,3vw,3rem);margin:.2rem 0 .25rem}.ar-subtitle{color:var(--ar-secondary-text);font-size:.98rem;margin:0 0 .65rem}
    .ar-status-badge,.ar-control-status,.ar-count-card,.ar-priority-badge{display:inline-flex;align-items:center;border:1px solid var(--ar-border);border-radius:999px;background:var(--ar-panel);color:var(--ar-secondary-text);font-size:.76rem;font-weight:650;letter-spacing:.035em;padding:.32rem .68rem}
    .ar-count-card{display:flex;border-radius:.55rem;margin-top:.65rem}.ar-section-title{color:var(--ar-text);font-size:1.08rem;margin:1.15rem 0 .55rem}.ar-section-note{color:var(--ar-muted);font-size:.78rem;margin:-.35rem 0 .7rem}
    div[data-testid="stButton"]>button{min-height:2.3rem;background:var(--ar-panel);border:1px solid var(--ar-border);color:var(--ar-text);border-radius:.5rem;box-shadow:none}
    div[data-testid="stButton"]>button:hover{border-color:var(--ar-accent);color:var(--ar-accent)}.st-key-replay_play div[data-testid="stButton"]>button{background:var(--ar-accent);border-color:var(--ar-accent);color:#04131F;font-weight:750}
    .ar-edge-panel{box-sizing:border-box;min-height:610px;background:var(--ar-panel);border:1px solid var(--ar-border);border-radius:.7rem;padding:1.25rem;display:flex;flex-direction:column}
    .ar-edge-route{color:var(--ar-text);font-size:1.08rem;font-weight:680;line-height:1.5}.ar-edge-arrow{color:var(--ar-accent);margin:.25rem 0}.ar-edge-probability{margin:2rem 0}
    .ar-edge-probability span,.ar-card-label{color:var(--ar-muted);font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.ar-edge-probability strong{color:var(--ar-accent);display:block;font-size:3rem;line-height:1.1}
    .ar-edge-detail{border-top:1px solid var(--ar-border);padding:.9rem 0}.ar-edge-detail strong{color:var(--ar-text);display:block;font-size:1.05rem;margin-top:.2rem}
    .ar-timeline-card{min-height:154px;background:#0A2033;border:1px solid #1C425C;border-radius:.6rem;padding:.9rem;margin-bottom:.75rem}.ar-timeline-card.active{background:var(--ar-panel);border-color:var(--ar-accent);box-shadow:0 0 16px rgba(56,189,248,.12)}
    .ar-timeline-card.completed{background:#0B263A;border-color:#31566D;opacity:.84}.ar-timeline-state{color:var(--ar-muted);font-size:.66rem;font-weight:750;letter-spacing:.1em}.ar-timeline-card.active .ar-timeline-state{color:var(--ar-accent)}
    .ar-timeline-route{color:var(--ar-text);font-size:.95rem;font-weight:680;margin:.65rem 0}.ar-timeline-meta{color:var(--ar-secondary-text);font-size:.78rem;line-height:1.55}
    .ar-metric-card{box-sizing:border-box;min-height:112px;background:var(--ar-panel);border:1px solid var(--ar-border);border-radius:.6rem;padding:.9rem 1rem;margin-bottom:.75rem}.ar-card-value{color:var(--ar-text);font-size:1.35rem;font-weight:720;margin-top:.55rem;line-height:1.15}.ar-priority-badge{color:#071827;border:0;margin-top:.5rem}
    .ar-recommendation{background:var(--ar-panel);border:1px solid var(--ar-border);border-left:3px solid var(--ar-accent);border-radius:.6rem;padding:1rem 1.1rem;margin-bottom:.7rem}.ar-rec-code{color:var(--ar-muted);font-size:.72rem;letter-spacing:.08em;margin-left:.55rem}
    .ar-recommendation h4{color:var(--ar-text);margin:.65rem 0 .35rem;font-size:1.02rem}.ar-recommendation p{color:var(--ar-secondary-text);font-size:.86rem;line-height:1.5;margin:.25rem 0}.ar-rec-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:.75rem}
    .ar-rec-detail{background:#0A2033;border-radius:.4rem;padding:.6rem .7rem}.ar-rec-detail strong{color:var(--ar-text);display:block;font-size:.83rem;margin-top:.2rem}.ar-feasibility{border-top:1px solid var(--ar-border);color:var(--ar-muted)!important;margin-top:.7rem!important;padding-top:.65rem}
    .ar-disclaimer{background:var(--ar-secondary);border:1px solid var(--ar-border);border-left:3px solid var(--ar-cyan);border-radius:.45rem;color:var(--ar-secondary-text);font-size:.78rem;line-height:1.6;margin-top:1.3rem;padding:.85rem 1rem}
    @media(max-width:900px){.block-container{padding-left:1rem;padding-right:1rem}.ar-edge-panel{min-height:auto}.ar-rec-grid{grid-template-columns:1fr}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_replay_data() -> pd.DataFrame:
    if not REPLAY_DATA_PATH.exists():
        raise FileNotFoundError(f"Replay dataset not found: {REPLAY_DATA_PATH}")
    replay = pd.read_parquet(REPLAY_DATA_PATH)
    required = {
        "START_FLIGHT_ID", "EDGE_COUNT", "FLIGHT_COUNT", "CUMULATIVE_PROBABILITY",
        "PROPAGATION_PROBABILITY", "PREV_ARR_DELAY", "TURN_BUFFER",
        "PREV_DELAY_RATIO", "PLANNED_TURNAROUND",
    }
    missing = required.difference(replay.columns)
    if missing:
        raise ValueError("Replay dataset is missing columns: " + ", ".join(sorted(missing)))
    id_parts = replay["START_FLIGHT_ID"].astype(str).str.split("_", expand=True)
    if id_parts.shape[1] < 5:
        raise ValueError("Replay dataset contains malformed flight identifiers.")
    replay = replay.copy()
    replay["REPLAY_DATE"] = pd.to_datetime(id_parts[0], format="%Y%m%d", errors="coerce")
    replay["AIRLINE"] = id_parts[1]
    return replay


@st.cache_data
def load_scored_edges() -> pd.DataFrame:
    if not SCORED_EDGES_PATH.exists():
        raise FileNotFoundError(f"Scored graph edges not found: {SCORED_EDGES_PATH}")
    return pd.read_parquet(SCORED_EDGES_PATH, columns=CHAIN_COLUMNS)


@st.cache_data
def load_airports() -> pd.DataFrame:
    return load_airport_reference()


def trace_predicted_chain(scored_edges: pd.DataFrame, start_flight_id: str, max_hops: int = 20) -> pd.DataFrame:
    """Follow active predicted edges from a selected historical start flight."""
    predicted_edges = scored_edges.loc[scored_edges["PROPAGATION_ALERT"] == 1]
    rows: list[dict[str, object]] = []
    current = start_flight_id
    visited: set[str] = set()
    for _ in range(max_hops):
        if current in visited:
            break
        candidates = predicted_edges.loc[predicted_edges["SOURCE_FLIGHT_ID"] == current]
        if candidates.empty:
            break
        visited.add(current)
        edge = candidates.sort_values("PROPAGATION_PROBABILITY", ascending=False).iloc[0]
        rows.append(edge.to_dict())
        current = str(edge["TARGET_FLIGHT_ID"])
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def flight_label(flight_id: str) -> str:
    parts = str(flight_id).split("_")
    if len(parts) < 5:
        return str(flight_id)
    date = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
    date_label = date.strftime("%d %b %Y") if not pd.isna(date) else parts[0]
    return f"{date_label} · {parts[1]} {parts[2]} · {parts[3]} → {parts[4]}"


def flight_route(flight_id: str) -> str:
    parts = str(flight_id).split("_")
    if len(parts) < 5:
        return str(flight_id)
    return f"{parts[1]} {parts[2]} &middot; {parts[3]} &rarr; {parts[4]}"


def metric_card(label: str, value: str, badge_color: str | None = None) -> str:
    safe_label = escape(label)
    safe_value = escape(value)
    if badge_color:
        value_html = (
            f'<span class="ar-priority-badge" style="background:{badge_color}">'
            f"{safe_value}</span>"
        )
    else:
        value_html = f'<div class="ar-card-value">{safe_value}</div>'
    return f'<div class="ar-metric-card"><div class="ar-card-label">{safe_label}</div>{value_html}</div>'


def initialize_replay_state(selected_flight_id: str) -> None:
    if st.session_state.get("replay_flight_id") != selected_flight_id:
        st.session_state.replay_flight_id = selected_flight_id
        st.session_state.replay_active_step = 0
        st.session_state.replay_playing = False


st.markdown(
    """
    <header class="ar-header">
      <div class="ar-eyebrow">AERORECOVER AI &middot; DECISION SUPPORT</div>
      <h1 class="ar-title">Historical Validation Replay</h1>
      <p class="ar-subtitle">Review predicted delay propagation across held-out validation flights.</p>
      <span class="ar-status-badge">Historical data &middot; September&ndash;October 2023 &middot; Not live operations</span>
    </header>
    """,
    unsafe_allow_html=True,
)

try:
    replay_data = load_replay_data()
    scored_edges = load_scored_edges()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("Replay filters")
valid_dates = replay_data["REPLAY_DATE"].dropna()
date_range = st.sidebar.date_input(
    "Departure date", value=(valid_dates.min().date(), valid_dates.max().date()),
    min_value=valid_dates.min().date(), max_value=valid_dates.max().date(),
)
airlines = sorted(replay_data["AIRLINE"].dropna().unique().tolist())
selected_airlines = st.sidebar.multiselect(
    "Airline",
    airlines,
    default=[],
    placeholder="All airlines",
    help="Leave empty to include every airline.",
)
minimum_edges = st.sidebar.slider(
    "Minimum graph edges", int(replay_data["EDGE_COUNT"].min()),
    int(replay_data["EDGE_COUNT"].max()), max(2, int(replay_data["EDGE_COUNT"].min())),
)
minimum_probability = st.sidebar.slider(
    "Minimum start-edge probability", 0.0, 1.0, 0.5, 0.05, format="%.0f%%",
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
filtered_replays = replay_data.loc[
    replay_data["REPLAY_DATE"].dt.date.between(start_date, end_date)
    & (replay_data["AIRLINE"].isin(selected_airlines) if selected_airlines else True)
    & (replay_data["EDGE_COUNT"] >= minimum_edges)
    & (replay_data["PROPAGATION_PROBABILITY"] >= minimum_probability)
].sort_values(["REPLAY_DATE", "START_FLIGHT_ID"])

st.sidebar.markdown(
    f'<div class="ar-count-card">{len(filtered_replays):,} predicted-chain starts match</div>',
    unsafe_allow_html=True,
)
if filtered_replays.empty:
    st.warning("No predicted-chain starts match the selected filters.")
    st.stop()

selected_flight_id = st.selectbox(
    "Selected predicted-chain start", filtered_replays["START_FLIGHT_ID"].tolist(),
    format_func=flight_label,
    help="Each option is a historical flight identified as the start of a predicted chain.",
)
selected_rows = filtered_replays.loc[filtered_replays["START_FLIGHT_ID"] == selected_flight_id]
if len(selected_rows) != 1:
    st.error("The selected flight could not be uniquely identified.")
    st.stop()
selected_flight = selected_rows.iloc[0]

predicted_chain = trace_predicted_chain(scored_edges, selected_flight_id)
missing_chain_columns = set(CHAIN_COLUMNS).difference(predicted_chain.columns)
if predicted_chain.empty:
    st.error("No active predicted edges were found for the selected chain start.")
    st.stop()
if missing_chain_columns:
    st.error("The predicted chain is missing required columns: " + ", ".join(sorted(missing_chain_columns)))
    st.stop()
essential_columns = ["SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID", "PROPAGATION_PROBABILITY"]
if predicted_chain[essential_columns].isna().any().any():
    st.error("The predicted chain contains incomplete flight or probability data.")
    st.stop()

decision_input = FlightDecisionInput(
    propagation_probability=float(selected_flight["PROPAGATION_PROBABILITY"]),
    previous_arrival_delay=float(selected_flight["PREV_ARR_DELAY"]),
    turn_buffer=float(selected_flight["TURN_BUFFER"]),
    previous_delay_ratio=float(selected_flight["PREV_DELAY_RATIO"]),
    planned_turnaround=float(selected_flight["PLANNED_TURNAROUND"]),
    downstream_edge_count=len(predicted_chain),
)
decision_report = build_decision_report(decision_input)
priority_code = decision_report.assessment.priority.name.split("_")[0]

initialize_replay_state(selected_flight_id)
step_count = len(predicted_chain)
st.session_state.replay_active_step = min(st.session_state.replay_active_step, step_count - 1)

st.markdown('<h2 class="ar-section-title">Step navigation</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="ar-section-note">Navigate the active graph edge here. Map animation speed and playback remain inside the map.</p>',
    unsafe_allow_html=True,
)
play_col, pause_col, next_col, reset_col, status_col = st.columns([.85, .85, .85, .85, 3.2])
with play_col:
    if st.button("Play", use_container_width=True, key="replay_play"):
        st.session_state.replay_playing = True
with pause_col:
    if st.button("Pause", use_container_width=True, key="replay_pause"):
        st.session_state.replay_playing = False
with next_col:
    if st.button("Next", use_container_width=True, key="replay_next"):
        st.session_state.replay_playing = False
        st.session_state.replay_active_step = min(st.session_state.replay_active_step + 1, step_count - 1)
with reset_col:
    if st.button("Reset", use_container_width=True, key="replay_reset"):
        st.session_state.replay_playing = False
        st.session_state.replay_active_step = 0
with status_col:
    state_label = "PLAY READY" if st.session_state.replay_playing else "PAUSED"
    st.markdown(
        f'<div style="text-align:right"><span class="ar-control-status">{state_label} &middot; '
        f'Edge {st.session_state.replay_active_step + 1} of {step_count}</span></div>',
        unsafe_allow_html=True,
    )

active_step = st.session_state.replay_active_step
active_edge = predicted_chain.iloc[active_step]
try:
    airports = load_airports()
    itinerary = build_itinerary(predicted_chain)
    coordinates = match_itinerary_coordinates(itinerary, airports)
    figure = build_propagation_figure(
        chain=predicted_chain, coordinates=coordinates, priority=priority_code,
        active_step=active_step,
    )
except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
    st.error(f"The propagation map could not be prepared: {exc}")
    st.stop()

map_col, edge_col = st.columns([7, 3], gap="medium")
with map_col:
    st.markdown('<h2 class="ar-section-title">Predicted propagation map</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ar-section-note">Map animation &middot; choose 0.5x, 1x, or 2x playback inside the map.</p>', unsafe_allow_html=True)
    st.plotly_chart(
        figure, use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )
with edge_col:
    st.markdown('<h2 class="ar-section-title">Active edge</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ar-section-note">Current model-predicted graph transition.</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ar-edge-panel">
          <div class="ar-card-label">Edge {active_step + 1} of {step_count}</div>
          <div class="ar-edge-route" style="margin-top:1.25rem">{flight_route(active_edge['SOURCE_FLIGHT_ID'])}</div>
          <div class="ar-edge-arrow">DOWNSTREAM CONNECTION</div>
          <div class="ar-edge-route">{flight_route(active_edge['TARGET_FLIGHT_ID'])}</div>
          <div class="ar-edge-probability"><span>Propagation probability</span><strong>{float(active_edge['PROPAGATION_PROBABILITY']):.1%}</strong></div>
          <div class="ar-edge-detail"><span class="ar-card-label">Connection airport</span><strong>{escape(str(active_edge['CONNECTION_AIRPORT']))}</strong></div>
          <div class="ar-edge-detail"><span class="ar-card-label">Planned connection</span><strong>{float(active_edge['PLANNED_CONNECTION_MINUTES']):.0f} minutes</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<h2 class="ar-section-title">Domino chain</h2>', unsafe_allow_html=True)
st.markdown('<p class="ar-section-note">Predicted graph sequence; segments are not recorded aircraft trajectories.</p>', unsafe_allow_html=True)
domino_columns = st.columns(min(step_count, 4))
for index, edge in predicted_chain.reset_index(drop=True).iterrows():
    state = "COMPLETED" if index < active_step else "ACTIVE" if index == active_step else "UP NEXT"
    state_class = "completed" if index < active_step else "active" if index == active_step else "future"
    with domino_columns[index % len(domino_columns)]:
        st.markdown(
            f"""
            <div class="ar-timeline-card {state_class}">
              <div class="ar-timeline-state">{state} &middot; EDGE {index + 1}</div>
              <div class="ar-timeline-route">{flight_route(edge['SOURCE_FLIGHT_ID'])}<br>{flight_route(edge['TARGET_FLIGHT_ID'])}</div>
              <div class="ar-timeline-meta">Risk {float(edge['PROPAGATION_PROBABILITY']):.1%}<br>{escape(str(edge['CONNECTION_AIRPORT']))} &middot; {float(edge['PLANNED_CONNECTION_MINUTES']):.0f} min connection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<h2 class="ar-section-title">Decision support assessment</h2>', unsafe_allow_html=True)
risk_colors = {"P1": "#FF5C6C", "P2": "#FFAA4C", "P3": "#FFD166", "P4": "#48D597"}
metric_values = [
    ("Propagation Probability", f"{decision_input.propagation_probability:.1%}", None),
    ("Operational Priority", decision_report.assessment.priority.name.replace("_", " "), risk_colors[priority_code]),
    ("Likelihood", decision_report.assessment.likelihood.value, None),
    ("Network Impact", decision_report.assessment.impact.value.replace("_", " "), None),
    ("Operational Urgency", decision_report.assessment.urgency.value, None),
    ("Edge Count", str(len(predicted_chain)), None),
    ("Flight Count", str(len(predicted_chain) + 1), None),
    ("Cumulative Chain Score", f"{selected_flight['CUMULATIVE_PROBABILITY']:.1%}", None),
]
metric_columns = st.columns(4)
for index, (label, value, badge_color) in enumerate(metric_values):
    metric_columns[index % 4].markdown(metric_card(label, value, badge_color), unsafe_allow_html=True)

st.markdown('<h2 class="ar-section-title">Recommended operational actions</h2>', unsafe_allow_html=True)
for recommendation in decision_report.recommendations:
    priority_name = recommendation["priority"].name.replace("_", " ")
    action_code = recommendation["action_code"].replace("_", " ").title()
    recommendation_priority = recommendation["priority"].name.split("_")[0]
    st.markdown(
        f"""
        <article class="ar-recommendation">
          <span class="ar-priority-badge" style="background:{risk_colors[recommendation_priority]}">{escape(priority_name)}</span>
          <span class="ar-rec-code">{escape(action_code)}</span>
          <h4>{escape(recommendation['action'])}</h4>
          <p>{escape(recommendation['reason'])}</p>
          <div class="ar-rec-grid">
            <div class="ar-rec-detail"><span class="ar-card-label">Owner</span><strong>{escape(recommendation['owner'])}</strong></div>
            <div class="ar-rec-detail"><span class="ar-card-label">Timing</span><strong>{escape(recommendation['timing'])}</strong></div>
          </div>
          <div class="ar-rec-detail" style="margin-top:.65rem"><span class="ar-card-label">Objective</span><strong>{escape(recommendation['objective'])}</strong></div>
          <p class="ar-feasibility"><strong>Feasibility note:</strong> {escape(recommendation['feasibility_note'])}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <aside class="ar-disclaimer"><strong>Historical replay notice.</strong> This is not live aircraft tracking.
    Map movement uses coordinate interpolation and does not represent recorded trajectories. Predictions indicate
    statistical association, not causality, and recommendations are decision-support guidance rather than automatic
    commands. Cumulative Chain Score compounds model probabilities across the displayed chain and should not be
    interpreted as a calibrated end-to-end outcome probability.</aside>
    """,
    unsafe_allow_html=True,
)
