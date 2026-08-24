"""Build the compact, fully vector IEEE-paper propagation exposure map.

Data preparation is delegated to the detailed-map builder so that aggregation,
CONUS filtering, airport exposure, node sizing, and probability calculations
remain identical. Only the requested paper-specific visual subset and styling
are applied here.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from build_propagation_exposure_map import CONUS_BOUNDS, TOPO_PATH, decode_topology, prepare_data
from paper_style import OUTPUT_DIR, PAPER_COLORS, PROBABILITY_CMAP, apply_paper_style

TOP_N_ROUTES_PAPER = 100
TOP_N_HIGHLIGHTED_ROUTES = 15
TOP_N_LABELS_PAPER = 10
PDF_STEM = "propagation_exposure_network_paper"


def main() -> None:
    apply_paper_style()
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    prepared_routes, airports = prepare_data()
    # prepare_data returns the same deterministically ranked route table used by
    # the detailed figure. The paper view changes only the displayed count.
    routes = prepared_routes.head(TOP_N_ROUTES_PAPER).copy()
    print(f"Paper routes plotted: {len(routes):,}")
    print(f"Highlighted highest-volume routes: {TOP_N_HIGHLIGHTED_ROUTES:,}")
    print(f"Airport labels: {TOP_N_LABELS_PAPER:,}")

    fig, ax = plt.subplots(figsize=(12.0, 6.67), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    for line in decode_topology(TOPO_PATH):
        points = np.asarray(line)
        ax.plot(
            points[:, 0], points[:, 1], color=PAPER_COLORS["border"], linewidth=.30,
            zorder=0, rasterized=False,
        )

    counts = routes["ALERT_COUNT"].astype(float)
    widths = .28 + 1.55 * np.log1p(counts) / np.log1p(counts.max())
    base_alphas = .15 + .27 * np.log1p(counts) / np.log1p(counts.max())
    segments = [
        [(row.ORIGIN_longitude_deg, row.ORIGIN_latitude_deg),
         (row.DESTINATION_longitude_deg, row.DESTINATION_latitude_deg)]
        for row in routes.itertuples()
    ]
    for rank, (segment, width, alpha) in enumerate(zip(segments, widths, base_alphas)):
        highlighted = rank < TOP_N_HIGHLIGHTED_ROUTES
        color = PAPER_COLORS["navy"] if highlighted else "#7890A4"
        edge_alpha = min(.60, float(alpha) + .18) if highlighted else float(alpha)
        ax.add_collection(LineCollection(
            [segment], colors=[to_rgba(color, edge_alpha)],
            linewidths=[width + (.22 if highlighted else 0)],
            zorder=1 if not highlighted else 1.2, rasterized=False,
        ))

    norm = Normalize(vmin=.46, vmax=.95)
    sizes = 10 + 95 * np.sqrt(airports["ALERTED_ROTATION_EXPOSURE"]) / np.sqrt(
        airports["ALERTED_ROTATION_EXPOSURE"].max()
    )
    nodes = ax.scatter(
        airports["longitude_deg"], airports["latitude_deg"], s=sizes,
        c=airports["MEAN_PROPAGATION_PROBABILITY"], cmap=PROBABILITY_CMAP, norm=norm,
        edgecolor="#FFFFFF", linewidth=.55, zorder=2, rasterized=False,
    )

    label_offsets = {
        "SEA": (3, 5), "LAX": (3, 5), "LAS": (3, 5), "PHX": (3, 5),
        "DEN": (3, 5), "DFW": (3, 5), "ORD": (3, 6), "MDW": (3, -10),
        "LGA": (-21, 6), "JFK": (4, -10), "BOS": (4, 6), "DCA": (4, -10),
        "ATL": (3, 5), "CLT": (3, 5), "MCO": (3, 5),
    }
    for row in airports.nlargest(TOP_N_LABELS_PAPER, "ALERTED_ROTATION_EXPOSURE").itertuples():
        ax.annotate(
            row.iata_code, (row.longitude_deg, row.latitude_deg),
            xytext=label_offsets.get(row.iata_code, (3, 5)), textcoords="offset points",
            fontsize=7.7, fontweight="bold", color=PAPER_COLORS["text"], zorder=3,
            bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": .76, "pad": .25},
            rasterized=False,
        )

    color_ax = inset_axes(ax, width="1.25%", height="28%", loc="upper right", borderpad=.65)
    colorbar = fig.colorbar(nodes, cax=color_ax, orientation="vertical")
    colorbar.set_ticks([.50, .70, .90])
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.set_label("Mean probability", fontsize=7.5, labelpad=3)
    colorbar.ax.tick_params(labelsize=7, width=.5, length=2)
    colorbar.outline.set_linewidth(.45)
    colorbar.solids.set_rasterized(False)

    ax.set_xlim(CONUS_BOUNDS[:2])
    ax.set_ylim(CONUS_BOUNDS[2:])
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=.002, right=.998, bottom=.002, top=.998)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{PDF_STEM}.pdf"
    png_path = OUTPUT_DIR / f"{PDF_STEM}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=.01, facecolor="#FFFFFF")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=.01, facecolor="#FFFFFF")
    plt.close(fig)

    for path in (pdf_path, png_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Paper figure output was not created: {path}")
        print(f"Created {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
