"""Historical validation flight replay dashboard."""

from __future__ import annotations

from datetime import date, datetime
from html import escape
from io import BytesIO
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as PILImage, ImageChops
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_support.assessment_service import build_decision_report
from src.decision_support.contracts import FlightDecisionInput, FlightDecisionReport
from src.explainability.local_shap import (
    MODEL_VERSION,
    LocalShapExplanation,
    explain_validation_flight,
    load_model_pipeline,
)
from src.reporting.decision_report_pdf import build_decision_report_pdf
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
REPLAY_COLUMNS = [
    "START_FLIGHT_ID", "EDGE_COUNT", "FLIGHT_COUNT", "CUMULATIVE_PROBABILITY",
    "PROPAGATION_PROBABILITY", "PREV_ARR_DELAY", "TURN_BUFFER",
    "PREV_DELAY_RATIO", "PLANNED_TURNAROUND",
]
TOPOJSON_DIR = PROJECT_ROOT / "src" / "visualization" / "assets" / "topojson"

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
    replay = pd.read_parquet(REPLAY_DATA_PATH, columns=REPLAY_COLUMNS)
    required = set(REPLAY_COLUMNS)
    missing = required.difference(replay.columns)
    if missing:
        raise ValueError("Replay dataset is missing columns: " + ", ".join(sorted(missing)))
    id_parts = replay["START_FLIGHT_ID"].astype(str).str.split("_", expand=True)
    if id_parts.shape[1] < 5:
        raise ValueError("Replay dataset contains malformed flight identifiers.")
    replay = replay.copy()
    replay["REPLAY_DATE"] = pd.to_datetime(id_parts[0], format="%Y%m%d", errors="coerce")
    replay["REPLAY_DATE_ONLY"] = replay["REPLAY_DATE"].dt.date
    replay["AIRLINE"] = id_parts[1]
    return replay


@st.cache_data
def load_scored_edges() -> pd.DataFrame:
    if not SCORED_EDGES_PATH.exists():
        raise FileNotFoundError(f"Scored graph edges not found: {SCORED_EDGES_PATH}")
    return pd.read_parquet(SCORED_EDGES_PATH, columns=CHAIN_COLUMNS)


@st.cache_resource
def load_active_edge_lookup() -> pd.DataFrame:
    """Build the best active downstream edge lookup once per server process."""
    active_edges = load_scored_edges().loc[
        lambda frame: frame["PROPAGATION_ALERT"].eq(1), CHAIN_COLUMNS
    ]
    if active_edges["SOURCE_FLIGHT_ID"].duplicated().any():
        active_edges = (
            active_edges.sort_values("PROPAGATION_PROBABILITY", ascending=False)
            .drop_duplicates("SOURCE_FLIGHT_ID", keep="first")
        )
    return active_edges.set_index("SOURCE_FLIGHT_ID", drop=False)


@st.cache_data
def load_airports() -> pd.DataFrame:
    return load_airport_reference()


@st.cache_resource
def load_explanation_model():
    return load_model_pipeline()


@st.cache_data(show_spinner=False)
def load_local_explanation(
    target_flight_id: str,
    model_version: str,
    expected_probability: float,
    _pipeline,
) -> LocalShapExplanation:
    if model_version != MODEL_VERSION:
        raise ValueError(f"Unsupported explanation model version: {model_version}.")
    return explain_validation_flight(
        target_flight_id,
        pipeline=_pipeline,
        expected_probability=expected_probability,
    )


def format_feature_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_shap_figure(explanation: LocalShapExplanation) -> go.Figure:
    increasing = explanation.contributions.loc[
        explanation.contributions["shap_value"] > 0
    ].nlargest(5, "absolute_importance")
    decreasing = explanation.contributions.loc[
        explanation.contributions["shap_value"] < 0
    ].nlargest(5, "absolute_importance")
    display = pd.concat([increasing, decreasing]).sort_values("shap_value")
    labels = [
        f"{row.feature}  |  {format_feature_value(row.feature_value)}"
        for row in display.itertuples()
    ]
    colors = ["#2DD4BF" if value < 0 else "#FB7185" for value in display["shap_value"]]
    figure = go.Figure(go.Bar(
        x=display["shap_value"],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{value:+.3f}" for value in display["shap_value"]],
        textposition="outside",
        customdata=display[["feature", "feature_value", "direction"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Value: %{customdata[1]}<br>"
            "Direction: %{customdata[2]}<br>SHAP: %{x:+.5f}<extra></extra>"
        ),
    ))
    figure.add_vline(x=0, line_width=1, line_color="#A5BAC9")
    figure.update_layout(
        height=max(390, 43 * len(display)),
        margin={"l": 20, "r": 65, "t": 15, "b": 45},
        paper_bgcolor="#071827",
        plot_bgcolor="#0E2A43",
        font={"color": "#F2F7FB", "size": 12},
        xaxis={"title": "Contribution to raw model score", "gridcolor": "#245573"},
        yaxis={"title": None, "automargin": True},
        showlegend=False,
    )
    return figure


@st.cache_data(show_spinner=False)
def render_shap_png(contributions: pd.DataFrame) -> bytes:
    display = contributions.head(8).sort_values("shap_value")
    labels = [
        f"{row.feature} = {format_feature_value(row.feature_value)}"
        for row in display.itertuples()
    ]
    colors = ["#159E91" if value < 0 else "#E65D68" for value in display["shap_value"]]
    figure, axis = plt.subplots(figsize=(12, 5.4), facecolor="white")
    axis.barh(labels, display["shap_value"], color=colors)
    axis.axvline(0, color="#60788A", linewidth=1)
    axis.set_xlabel("Contribution to raw model score")
    axis.grid(axis="x", linestyle="--", alpha=0.22)
    axis.spines[["top", "right", "left"]].set_visible(False)
    for index, value in enumerate(display["shap_value"]):
        axis.text(
            value + (0.025 if value >= 0 else -0.025),
            index,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def render_static_map_png(figure_json: str) -> bytes:
    """Render a PDF-safe static copy without changing the interactive figure."""
    static_figure = pio.from_json(figure_json)
    static_figure.frames = []
    static_figure.layout.updatemenus = []
    static_figure.data = tuple(
        trace for trace in static_figure.data
        if "Animated predicted propagation marker"
        not in str(trace.hovertemplate)
    )
    label_positions = {
        "BLI": "top center", "OAK": "top right", "SFO": "top left",
        "SJC": "bottom right", "LAX": "top right", "LAS": "top center",
        "SAN": "bottom left", "PHX": "top center",
    }
    risk_labels = {
        "P1": "Critical", "P2": "High", "P3": "Monitor", "P4": "Normal",
    }
    for trace in static_figure.data:
        if trace.mode == "markers+text" and trace.text is not None:
            trace.textposition = [
                label_positions.get(str(code), "top center")
                for code in trace.text
            ]
        if trace.name and str(trace.name)[:2] in risk_labels:
            priority = str(trace.name)[:2]
            trace.name = f"{priority} - {risk_labels[priority]}"
    static_figure.update_layout(
        width=1400,
        height=700,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#17212B", "size": 16},
        legend={"orientation": "h", "x": 0.02, "y": 0.98},
    )
    static_figure.update_geos(
        bgcolor="#FFFFFF",
        landcolor="#E8F0F5",
        lakecolor="#FFFFFF",
        subunitcolor="#9BB2C2",
        countrycolor="#7892A5",
    )
    previous_topojson = pio.defaults.topojson
    try:
        pio.defaults.topojson = TOPOJSON_DIR.resolve().as_uri() + "/"
        png_bytes = static_figure.to_image(
            format="png",
            width=1400,
            height=700,
            scale=2,
        )
    finally:
        pio.defaults.topojson = previous_topojson

    with PILImage.open(BytesIO(png_bytes)) as image:
        rgb_image = image.convert("RGB")
        white_background = PILImage.new("RGB", rgb_image.size, "white")
        content_box = ImageChops.difference(rgb_image, white_background).getbbox()
        if content_box is None:
            return png_bytes
        padding = 40
        left, top, right, bottom = content_box
        crop_box = (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb_image.width, right + padding),
            min(rgb_image.height, bottom + padding),
        )
        output = BytesIO()
        rgb_image.crop(crop_box).save(output, format="PNG", optimize=True)
        return output.getvalue()


@st.cache_data(show_spinner=False)
def build_cached_decision_report_pdf(
    *,
    flight_id: str,
    decision_input: FlightDecisionInput,
    decision_report: FlightDecisionReport,
    predicted_chain: pd.DataFrame,
    cumulative_chain_score: float,
    map_image_bytes: bytes | None,
    local_explanation: LocalShapExplanation | None,
    shap_image_bytes: bytes | None,
    shap_error_message: str | None,
) -> bytes:
    return build_decision_report_pdf(
        flight_id=flight_id,
        decision_input=decision_input,
        decision_report=decision_report,
        predicted_chain=predicted_chain,
        cumulative_chain_score=cumulative_chain_score,
        map_image_bytes=map_image_bytes,
        local_explanation=local_explanation,
        shap_image_bytes=shap_image_bytes,
        shap_error_message=shap_error_message,
    )


@st.cache_data(show_spinner=False)
def trace_predicted_chain(start_flight_id: str, max_hops: int = 20) -> pd.DataFrame:
    """Follow active predicted edges from a selected historical start flight."""
    edge_lookup = load_active_edge_lookup()
    rows: list[dict[str, object]] = []
    current = start_flight_id
    visited: set[str] = set()
    for _ in range(max_hops):
        if current in visited:
            break
        if current not in edge_lookup.index:
            break
        visited.add(current)
        edge = edge_lookup.loc[current].to_dict()
        rows.append(edge)
        current = str(edge["TARGET_FLIGHT_ID"])
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def normalize_date_range(value: object) -> tuple[date, date]:
    """Normalize every supported Streamlit date-input shape to scalar dates."""
    values = list(value) if isinstance(value, (tuple, list)) else [value]
    if not values:
        raise ValueError("Select at least one departure date.")

    def scalar_date(item: object) -> date:
        if isinstance(item, datetime):
            return item.date()
        if isinstance(item, date):
            return item
        raise ValueError(f"Unsupported departure date value: {item!r}")

    start = scalar_date(values[0])
    end = scalar_date(values[1]) if len(values) > 1 and values[1] is not None else start
    return (start, end) if start <= end else (end, start)


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
        st.session_state.pop("pdf_report_bytes", None)
        st.session_state.pop("pdf_report_flight_id", None)
        st.session_state.pop("pdf_report_filename", None)


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
    load_active_edge_lookup()
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

try:
    start_date, end_date = normalize_date_range(date_range)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()
filtered_replays = replay_data.loc[
    replay_data["REPLAY_DATE_ONLY"].between(start_date, end_date)
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

predicted_chain = trace_predicted_chain(selected_flight_id)
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
figure = None
try:
    airports = load_airports()
    itinerary = build_itinerary(predicted_chain)
    coordinates = match_itinerary_coordinates(itinerary, airports)
    figure = build_propagation_figure(
        chain=predicted_chain, coordinates=coordinates, priority=priority_code,
        active_step=active_step,
    )
except Exception as exc:
    st.warning(f"The propagation map could not be prepared. Details: {exc}")

selected_target_flight_id = str(predicted_chain.iloc[0]["TARGET_FLIGHT_ID"])
selected_target_probability = float(predicted_chain.iloc[0]["PROPAGATION_PROBABILITY"])
local_explanation = None
shap_error_message = None
try:
    local_explanation = load_local_explanation(
        selected_target_flight_id,
        MODEL_VERSION,
        selected_target_probability,
        load_explanation_model(),
    )
except Exception as exc:
    shap_error_message = str(exc)
    st.warning(
        "The local SHAP explanation could not be produced for the selected validation "
        f"rotation. The PDF remains available without a SHAP chart. Details: {exc}"
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

st.markdown('<h2 class="ar-section-title">Why this prediction?</h2>', unsafe_allow_html=True)
if local_explanation is not None:
    st.plotly_chart(
        build_shap_figure(local_explanation),
        width="stretch",
        config={"displayModeBar": False},
    )
    increasing = local_explanation.contributions.loc[
        local_explanation.contributions["shap_value"] > 0
    ].nlargest(5, "absolute_importance")
    decreasing = local_explanation.contributions.loc[
        local_explanation.contributions["shap_value"] < 0
    ].nlargest(5, "absolute_importance")
    increasing = increasing.copy()
    decreasing = decreasing.copy()
    increasing["feature_value"] = increasing["feature_value"].map(format_feature_value)
    decreasing["feature_value"] = decreasing["feature_value"].map(format_feature_value)
    increase_col, decrease_col = st.columns(2)
    with increase_col:
        st.markdown("**Top risk-increasing contributions**")
        st.dataframe(
            increasing[["feature", "feature_value", "shap_value"]],
            hide_index=True,
            width="stretch",
        )
    with decrease_col:
        st.markdown("**Top risk-decreasing contributions**")
        st.dataframe(
            decreasing[["feature", "feature_value", "shap_value"]],
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "SHAP values explain contributions to the model's raw score. They are not causal "
        "effects or direct probability-point changes."
    )
    with st.expander("SHAP validation details", expanded=False):
        validation_columns = st.columns(4)
        validation_columns[0].metric("Model probability", f"{local_explanation.model_probability:.6f}")
        validation_columns[1].metric("Model raw score", f"{local_explanation.model_raw_score:.6f}")
        validation_columns[2].metric("SHAP raw score", f"{local_explanation.shap_raw_score:.6f}")
        validation_columns[3].metric("Reconstruction error", f"{local_explanation.reconstruction_error:.2e}")
else:
    st.info("No local explanation is available for this selected validation rotation.")

generate_col, download_col, empty_col = st.columns([1, 1, 2])
with generate_col:
    generate_pdf = st.button(
        "Generate PDF Report",
        width="stretch",
        key="generate_decision_report_pdf",
    )

if generate_pdf:
    map_image_bytes = None
    shap_image_bytes = None
    if figure is not None:
        try:
            map_image_bytes = render_static_map_png(figure.to_json())
        except Exception as exc:
            st.warning(
                "The static propagation map could not be added to the PDF. "
                f"A map-free report can still be generated. Details: {exc}"
            )
    else:
        st.warning("The PDF will be generated without a map because the interactive map is unavailable.")

    if local_explanation is not None:
        try:
            shap_image_bytes = render_shap_png(local_explanation.contributions)
        except Exception as exc:
            shap_error_message = str(exc)
            st.warning(
                "The SHAP chart could not be added to the PDF. "
                f"The report can still be generated. Details: {exc}"
            )

    try:
        pdf_bytes = build_cached_decision_report_pdf(
            flight_id=selected_flight_id,
            decision_input=decision_input,
            decision_report=decision_report,
            predicted_chain=predicted_chain,
            cumulative_chain_score=float(selected_flight["CUMULATIVE_PROBABILITY"]),
            map_image_bytes=map_image_bytes,
            local_explanation=local_explanation,
            shap_image_bytes=shap_image_bytes,
            shap_error_message=shap_error_message,
        )
        st.session_state.pdf_report_bytes = pdf_bytes
        st.session_state.pdf_report_flight_id = selected_flight_id
        st.session_state.pdf_report_filename = f"aerorecover_{selected_flight_id}_report.pdf"
    except Exception as exc:
        st.warning(f"The PDF report could not be generated. Details: {exc}")

if (
    st.session_state.get("pdf_report_flight_id") == selected_flight_id
    and st.session_state.get("pdf_report_bytes") is not None
):
    with download_col:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=st.session_state.pdf_report_filename,
            mime="application/pdf",
            width="stretch",
            key="download_decision_report_pdf",
        )

map_col, edge_col = st.columns([7, 3], gap="medium")
with map_col:
    st.markdown('<h2 class="ar-section-title">Predicted propagation map</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ar-section-note">Map animation &middot; choose 0.5x, 1x, or 2x playback inside the map.</p>', unsafe_allow_html=True)
    if figure is not None:
        st.plotly_chart(
            figure, use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
        )
    else:
        st.warning("The interactive propagation map is unavailable for this flight.")
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
