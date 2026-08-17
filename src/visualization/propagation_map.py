"""Plotly map helpers for predicted delay-propagation replay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AIRPORT_REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "reference" / "us_airport_coordinates.parquet"
)
REQUIRED_AIRPORT_COLUMNS = [
    "iata_code", "latitude_deg", "longitude_deg", "name", "municipality", "iso_country", "type"
]
EDGE_COLORS = {"completed": "#8493a3", "active": "#36a9ff", "future": "#183b5a"}
RISK_COLORS = {"P1": "#ff5964", "P2": "#ff9f43", "P3": "#ffd166", "P4": "#48d597"}
RISK_LABELS = {"P1": "Critical", "P2": "High", "P3": "Monitor", "P4": "Normal"}


def parse_flight_id(flight_id: str) -> dict[str, str]:
    parts = str(flight_id).split("_")
    if len(parts) < 5:
        raise ValueError(f"Malformed flight identifier: {flight_id}")
    return {"airline": parts[1], "number": parts[2], "origin": parts[3], "destination": parts[4]}


def load_airport_reference(path: Path = AIRPORT_REFERENCE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Airport coordinate reference not found: {path}. Run src/data/build_airport_reference.py."
        )
    airports = pd.read_parquet(path, columns=REQUIRED_AIRPORT_COLUMNS)
    if airports["iata_code"].duplicated().any():
        raise ValueError("Airport coordinate reference contains duplicate IATA codes.")
    if airports[["latitude_deg", "longitude_deg"]].isna().any().any():
        raise ValueError("Airport coordinate reference contains missing coordinates.")
    return airports


def build_itinerary(chain: pd.DataFrame) -> list[str]:
    """Return the ordered airports traversed by all flights in a graph chain."""
    if chain.empty:
        raise ValueError("Cannot build an itinerary from an empty chain.")
    flight_ids = [str(chain.iloc[0]["SOURCE_FLIGHT_ID"]), *chain["TARGET_FLIGHT_ID"].astype(str)]
    flights = [parse_flight_id(value) for value in flight_ids]
    for previous, current in zip(flights, flights[1:]):
        if previous["destination"] != current["origin"]:
            raise ValueError(
                f"Disconnected itinerary: {previous['destination']} does not match {current['origin']}."
            )
    return [flights[0]["origin"], *[flight["destination"] for flight in flights]]


def match_itinerary_coordinates(itinerary: list[str], airports: pd.DataFrame) -> pd.DataFrame:
    lookup = airports.set_index("iata_code")
    missing = sorted(set(itinerary) - set(lookup.index))
    if missing:
        raise ValueError("Missing airport coordinates for IATA code(s): " + ", ".join(missing))
    rows = []
    for sequence, code in enumerate(itinerary):
        row = lookup.loc[code]
        rows.append({"sequence": sequence, "iata_code": code, **row.to_dict()})
    return pd.DataFrame(rows)


def _edge_trace(source: pd.Series, target: pd.Series, probability: float, color: str, width: int) -> go.Scattergeo:
    return go.Scattergeo(
        lon=[source["longitude_deg"], target["longitude_deg"]],
        lat=[source["latitude_deg"], target["latitude_deg"]],
        mode="lines",
        line={"width": width, "color": color},
        hovertemplate=f"{source['iata_code']} → {target['iata_code']}<br>Edge probability: {probability:.1%}<extra></extra>",
        showlegend=False,
    )


def _aircraft_trace(lat: float, lon: float) -> go.Scattergeo:
    """Return a deliberately prominent, Plotly-supported moving marker."""
    return go.Scattergeo(
        lon=[lon],
        lat=[lat],
        mode="markers+text",
        marker={
            "size": 22,
            "color": "#35c5ff",
            "symbol": "triangle-right",
            "line": {"color": "#ffffff", "width": 3},
        },
        text=["✈"],
        textfont={"size": 24, "color": "#ffffff"},
        textposition="middle right",
        hovertemplate="Animated predicted propagation marker<extra></extra>",
        showlegend=False,
    )


def build_animation_frames(
    chain: pd.DataFrame, coordinates: pd.DataFrame, frames_per_edge: int = 24
) -> list[go.Frame]:
    """Interpolate display coordinates; these are not recorded flight trajectories."""
    if not 20 <= frames_per_edge <= 30:
        raise ValueError("frames_per_edge must be between 20 and 30.")
    frames: list[go.Frame] = []
    for edge_index, edge in chain.reset_index(drop=True).iterrows():
        # A graph edge links a source flight to its downstream target flight;
        # animate the target flight leg where the predicted propagation appears.
        source, target = coordinates.iloc[edge_index + 1], coordinates.iloc[edge_index + 2]
        for frame_index in range(frames_per_edge):
            progress = frame_index / (frames_per_edge - 1)
            lat = source["latitude_deg"] + progress * (target["latitude_deg"] - source["latitude_deg"])
            lon = source["longitude_deg"] + progress * (target["longitude_deg"] - source["longitude_deg"])
            traces = []
            for index, row in chain.reset_index(drop=True).iterrows():
                state = "completed" if index < edge_index else "active" if index == edge_index else "future"
                traces.append(_edge_trace(coordinates.iloc[index + 1], coordinates.iloc[index + 2], float(row["PROPAGATION_PROBABILITY"]), EDGE_COLORS[state], 4 if state == "active" else 2))
            traces.append(_aircraft_trace(lat, lon))
            frames.append(go.Frame(name=f"e{edge_index}-f{frame_index}", data=traces, traces=list(range(len(chain) + 1))))
    return frames


def build_propagation_figure(
    chain: pd.DataFrame,
    coordinates: pd.DataFrame,
    priority: str,
    speed: float = 1.0,
    active_step: int = 0,
    frames_per_edge: int = 24,
) -> go.Figure:
    if priority not in RISK_COLORS:
        raise ValueError(f"Unsupported priority: {priority}")
    if speed not in {0.5, 1.0, 2.0}:
        raise ValueError("Animation speed must be 0.5, 1.0, or 2.0.")
    active_step = max(0, min(active_step, len(chain) - 1))
    traces: list[go.BaseTraceType] = []
    for index, edge in chain.reset_index(drop=True).iterrows():
        state = "completed" if index < active_step else "active" if index == active_step else "future"
        traces.append(_edge_trace(coordinates.iloc[index + 1], coordinates.iloc[index + 2], float(edge["PROPAGATION_PROBABILITY"]), EDGE_COLORS[state], 4 if state == "active" else 2))
    traces.append(_aircraft_trace(
        float(coordinates.iloc[active_step + 1]["latitude_deg"]),
        float(coordinates.iloc[active_step + 1]["longitude_deg"]),
    ))
    traces.append(go.Scattergeo(
        lon=coordinates["longitude_deg"], lat=coordinates["latitude_deg"], mode="markers+text",
        marker={"size": 9, "color": "#d9efff", "line": {"color": "#0d4d78", "width": 2}},
        text=coordinates["iata_code"], textposition="top center",
        customdata=coordinates[["name", "municipality"]].fillna("Not available"),
        hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>", showlegend=False,
    ))
    traces.append(go.Scattergeo(lon=[None], lat=[None], mode="markers", marker={"size": 11, "color": RISK_COLORS[priority]}, name=f"{priority} — {RISK_LABELS[priority]}", hoverinfo="skip"))
    figure = go.Figure(data=traces)
    figure.frames = build_animation_frames(chain, coordinates, frames_per_edge)
    # Pass every frame name explicitly. Plotly's implicit ``animate(None)`` can
    # stop after the first logical frame group in multi-edge geo animations.
    frame_names = [frame.name for frame in figure.frames]
    figure.update_layout(
        height=610, margin={"l": 0, "r": 0, "t": 30, "b": 0}, paper_bgcolor="#081a2d", plot_bgcolor="#081a2d",
        font={"color": "#dcefff"}, showlegend=True, legend={"orientation": "h", "y": 1.02, "x": 0.01},
        geo={"scope": "usa", "projection": {"type": "albers usa"}, "showland": True, "landcolor": "#102c45", "showlakes": True, "lakecolor": "#081a2d", "subunitcolor": "#31506a", "countrycolor": "#31506a", "bgcolor": "#081a2d", "fitbounds": "locations"},
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0.01, "y": 0.03, "pad": {"r": 8, "t": 8}, "bgcolor": "#123a59", "bordercolor": "#36a9ff", "font": {"color": "#ffffff"}, "buttons": [
            {"label": "▶ Play 0.5×", "method": "animate", "args": [frame_names, {"frame": {"duration": 180, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": False, "mode": "immediate"}]},
            {"label": "▶ Play 1×", "method": "animate", "args": [frame_names, {"frame": {"duration": 90, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": False, "mode": "immediate"}]},
            {"label": "▶ Play 2×", "method": "animate", "args": [frame_names, {"frame": {"duration": 45, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": False, "mode": "immediate"}]},
            {"label": "Ⅱ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}]},
        ]}],
    )
    return figure
