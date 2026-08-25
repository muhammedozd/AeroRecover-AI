"""Build an airport-aggregated map of alerted flight-level rotation edges.

The model graph uses individual flights as vertices and physical same-aircraft
rotation connections as edges. Airports in this figure are spatial aggregation
units only; this is not the graph used by the prediction or traversal code.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.visualization.paper_figures.paper_style import (
    PAPER_COLORS, PROBABILITY_CMAP, PROJECT_ROOT, apply_paper_style, save_figure,
)

EDGE_PATH = PROJECT_ROOT / "data/processed/graph/scored_tail_edges_2023_validation_full_enhanced.parquet"
AIRPORT_PATH = PROJECT_ROOT / "data/processed/reference/us_airport_coordinates.parquet"
TOPO_PATH = PROJECT_ROOT / "src/visualization/assets/topojson/usa_110m.json"
TOP_N_ROUTES = 150
TOP_N_LABELS = 15
TOP_N_HIGHLIGHTED_ROUTES = 15
CONUS_BOUNDS = (-125.0, -66.5, 24.0, 50.5)  # lon min/max, lat min/max
FLIGHT_RE = re.compile(r"^(\d{8})_([A-Z0-9]+)_([A-Z0-9]+)_([A-Z]{3})_([A-Z]{3})_(\d{4})_(.+)$")


def parse_target_legs(values: pd.Series) -> tuple[pd.DataFrame, int]:
    rows, malformed = [], 0
    for index, value in values.items():
        match = FLIGHT_RE.fullmatch(str(value))
        if not match:
            malformed += 1
            continue
        rows.append((index, match.group(4), match.group(5)))
    return pd.DataFrame(rows, columns=["_INDEX", "ORIGIN", "DESTINATION"]).set_index("_INDEX"), malformed


def decode_topology(path):
    topo = json.loads(path.read_text(encoding="utf-8"))
    scale = topo["transform"]["scale"]; translate = topo["transform"]["translate"]
    decoded = []
    for arc in topo["arcs"]:
        x = y = 0; points = []
        for dx, dy in arc:
            x += dx; y += dy
            points.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
        decoded.append(points)
    def resolve(index):
        return decoded[index] if index >= 0 else list(reversed(decoded[~index]))
    lines = []
    for geometry in topo["objects"]["subunits"]["geometries"]:
        if geometry.get("id") in {"AK", "HI", "PR"}: continue
        polygons = geometry.get("arcs", [])
        if geometry["type"] == "Polygon": polygons = [polygons]
        for polygon in polygons:
            for ring in polygon:
                points = []
                for arc_index in ring:
                    arc = resolve(arc_index)
                    points.extend(arc if not points else arc[1:])
                lines.append(points)
    return lines


def prepare_data():
    columns = ["TARGET_FLIGHT_ID", "PROPAGATION_PROBABILITY", "PROPAGATION_ALERT"]
    alerts = pd.read_parquet(EDGE_PATH, columns=columns)
    alerts = alerts.loc[alerts["PROPAGATION_ALERT"].eq(1)].copy()
    legs, malformed = parse_target_legs(alerts["TARGET_FLIGHT_ID"])
    valid = alerts.join(legs, how="inner")
    airports = pd.read_parquet(AIRPORT_PATH)
    if airports["iata_code"].duplicated().any():
        raise ValueError("Airport reference contains duplicate IATA codes.")
    lookup = airports.set_index("iata_code")
    used_codes = set(valid["ORIGIN"]) | set(valid["DESTINATION"])
    missing = sorted(used_codes - set(lookup.index))
    matched = valid.loc[~valid["ORIGIN"].isin(missing) & ~valid["DESTINATION"].isin(missing)].copy()
    routes = matched.groupby(["ORIGIN", "DESTINATION"], as_index=False).agg(
        ALERT_COUNT=("PROPAGATION_ALERT", "size"),
        MEAN_PROPAGATION_PROBABILITY=("PROPAGATION_PROBABILITY", "mean"),
        MAX_PROPAGATION_PROBABILITY=("PROPAGATION_PROBABILITY", "max"),
    )
    routes = routes.merge(lookup[["latitude_deg", "longitude_deg"]].add_prefix("ORIGIN_") , left_on="ORIGIN", right_index=True)
    routes = routes.merge(lookup[["latitude_deg", "longitude_deg"]].add_prefix("DESTINATION_"), left_on="DESTINATION", right_index=True)
    xmin, xmax, ymin, ymax = CONUS_BOUNDS
    inside = routes["ORIGIN_longitude_deg"].between(xmin, xmax) & routes["DESTINATION_longitude_deg"].between(xmin, xmax) & routes["ORIGIN_latitude_deg"].between(ymin, ymax) & routes["DESTINATION_latitude_deg"].between(ymin, ymax)
    excluded_route_rows = int((~inside).sum())
    conus = routes.loc[inside].sort_values(["ALERT_COUNT", "MAX_PROPAGATION_PROBABILITY"], ascending=False).head(TOP_N_ROUTES).copy()

    touches = []
    for side in ("ORIGIN", "DESTINATION"):
        part = matched[[side, "PROPAGATION_PROBABILITY"]].rename(columns={side:"iata_code"})
        touches.append(part)
    airport_exposure = pd.concat(touches).groupby("iata_code", as_index=False).agg(
        ALERTED_ROTATION_EXPOSURE=("PROPAGATION_PROBABILITY", "size"),
        MEAN_PROPAGATION_PROBABILITY=("PROPAGATION_PROBABILITY", "mean"),
    ).merge(airports, on="iata_code", how="left")
    airport_exposure = airport_exposure.loc[airport_exposure["longitude_deg"].between(xmin,xmax) & airport_exposure["latitude_deg"].between(ymin,ymax)].copy()
    print(f"Alerted flight-level edges: {len(alerts):,}")
    print(f"Malformed TARGET_FLIGHT_ID rows: {malformed:,}")
    print(f"Airports missing coordinates ({len(missing)}): {', '.join(missing) if missing else 'none'}")
    print(f"Aggregated airport pairs before CONUS filter: {len(routes):,}")
    print(f"Airport-pair rows excluded by CONUS filter: {excluded_route_rows:,}")
    print(f"Routes plotted (TOP_N_ROUTES={TOP_N_ROUTES}): {len(conus):,}")
    print(f"CONUS airports exported: {len(airport_exposure):,}")
    return conus, airport_exposure


def main() -> None:
    apply_paper_style()
    routes, airports = prepare_data()
    out = PROJECT_ROOT / "results/paper_figures"; out.mkdir(parents=True, exist_ok=True)
    routes.to_csv(out / "propagation_exposure_routes.csv", index=False)
    airports.sort_values("ALERTED_ROTATION_EXPOSURE", ascending=False).to_csv(out / "propagation_exposure_airports.csv", index=False)
    print(f"Created {out/'propagation_exposure_routes.csv'} ({(out/'propagation_exposure_routes.csv').stat().st_size:,} bytes)")
    print(f"Created {out/'propagation_exposure_airports.csv'} ({(out/'propagation_exposure_airports.csv').stat().st_size:,} bytes)")

    fig, ax = plt.subplots(figsize=(13.8, 7.1), facecolor=PAPER_COLORS["background"])
    ax.set_facecolor("#FAFBFC")
    for line in decode_topology(TOPO_PATH):
        arr = np.asarray(line); ax.plot(arr[:,0], arr[:,1], color=PAPER_COLORS["border"], linewidth=.42, zorder=0, rasterized=False)
    counts = routes["ALERT_COUNT"].astype(float)
    widths = .35 + 1.8 * np.log1p(counts) / np.log1p(counts.max())
    alphas = .15 + .45 * np.log1p(counts) / np.log1p(counts.max())
    segments = [[(r.ORIGIN_longitude_deg, r.ORIGIN_latitude_deg), (r.DESTINATION_longitude_deg, r.DESTINATION_latitude_deg)] for r in routes.itertuples()]
    for rank, (segment, width, alpha) in enumerate(zip(segments, widths, alphas)):
        highlighted = rank < TOP_N_HIGHLIGHTED_ROUTES
        color = PAPER_COLORS["navy"] if highlighted else "#7890A4"
        ax.add_collection(LineCollection([segment], colors=[to_rgba(color, float(alpha))], linewidths=[width], zorder=1, rasterized=False))
    norm = Normalize(vmin=0.47, vmax=0.95)
    sizes = 10 + 95 * np.sqrt(airports.ALERTED_ROTATION_EXPOSURE) / np.sqrt(airports.ALERTED_ROTATION_EXPOSURE.max())
    scatter = ax.scatter(airports.longitude_deg, airports.latitude_deg, s=sizes,
                         c=airports.MEAN_PROPAGATION_PROBABILITY, cmap=PROBABILITY_CMAP, norm=norm,
                         edgecolor="#FFFFFF", linewidth=.6, zorder=2, rasterized=False)
    label_offsets = {
        "SEA": (4, 7), "LAX": (4, 7), "LAS": (4, 7), "PHX": (4, 7), "DEN": (4, 7),
        "DFW": (4, 7), "ORD": (4, 9), "MDW": (4, -12), "ATL": (4, 7), "CLT": (4, 7),
        "DCA": (4, -13), "LGA": (-24, 8), "JFK": (5, -12), "BOS": (5, 8), "MCO": (4, 7),
    }
    for row in airports.nlargest(TOP_N_LABELS, "ALERTED_ROTATION_EXPOSURE").itertuples():
        offset = label_offsets.get(row.iata_code, (4, 6))
        ax.annotate(row.iata_code, (row.longitude_deg, row.latitude_deg), xytext=offset, textcoords="offset points",
                    fontsize=8, fontweight="bold", color=PAPER_COLORS["text"], zorder=3,
                    bbox={"facecolor":"white", "edgecolor":"none", "alpha":.72, "pad":.4})
    cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", fraction=.035, pad=.025, aspect=38)
    cbar.set_label("Mean predicted propagation probability", fontsize=8)
    cbar.set_ticks([.50, .60, .70, .80, .90])
    cbar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    cbar.solids.set_rasterized(False)
    examples = np.quantile(airports.ALERTED_ROTATION_EXPOSURE, [.25,.5,.9]).astype(int)
    handles = [plt.scatter([],[],s=10+95*np.sqrt(v)/np.sqrt(airports.ALERTED_ROTATION_EXPOSURE.max()), c=PAPER_COLORS["steel_blue"], edgecolor="white", label=f"{v:,}") for v in examples]
    leg1 = ax.legend(handles=handles, title="Alerted rotation exposure", loc="lower left", frameon=True, ncol=3, fontsize=8, title_fontsize=8,
                     facecolor="#FFFFFF", edgecolor=PAPER_COLORS["border"], framealpha=.9)
    ax.add_artist(leg1)
    ax.legend(handles=[Line2D([0],[0], color="#7890A4", linewidth=1.5, label="Aggregated alerted target-flight legs")],
              loc="lower right", frameon=True, fontsize=8, facecolor="#FFFFFF", edgecolor=PAPER_COLORS["border"], framealpha=.9)
    ax.set_xlim(CONUS_BOUNDS[:2]); ax.set_ylim(CONUS_BOUNDS[2:]); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_color(PAPER_COLORS["border"]); spine.set_linewidth(.6)
    fig.text(.5, .012, "The airport-level network is an aggregated visualization of alerted flight-level aircraft-rotation edges. Airports are used only for spatial aggregation;\nthe prediction model and multi-hop traversal operate on individual flight vertices.", ha="center", va="bottom", fontsize=8, color=PAPER_COLORS["muted_text"])
    fig.subplots_adjust(bottom=.13)
    save_figure(fig, "propagation_exposure_network_enhanced")
    plt.close(fig)


if __name__ == "__main__":
    main()
