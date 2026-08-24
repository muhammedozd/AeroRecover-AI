"""Build a real validation-chain aircraft-rotation schematic."""

from __future__ import annotations

import re
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

from paper_style import PAPER_COLORS, PROJECT_ROOT, apply_paper_style, save_figure

EDGE_PATH = PROJECT_ROOT / "data/processed/graph/scored_tail_edges_2023_validation.parquet"
FLIGHT_RE = re.compile(r"^(\d{8})_([A-Z0-9]+)_([A-Z0-9]+)_([A-Z]{3})_([A-Z]{3})_(\d{4})_(.+)$")
MAX_EDGES = 3


def parse_flight(value: str) -> dict[str, str]:
    match = FLIGHT_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"Malformed flight ID in selected chain: {value}")
    date, carrier, number, origin, destination, hhmm, _tail = match.groups()
    return {"date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "carrier": carrier,
            "number": number, "origin": origin, "destination": destination,
            "time": f"{hhmm[:2]}:{hhmm[2:]}"}


def select_chain(edges: pd.DataFrame) -> pd.DataFrame:
    active = edges.loc[edges["PROPAGATION_ALERT"].eq(1)].drop_duplicates("SOURCE_FLIGHT_ID")
    lookup = active.set_index("SOURCE_FLIGHT_ID", drop=False)
    starts = active.loc[~active["SOURCE_FLIGHT_ID"].isin(set(active["TARGET_FLIGHT_ID"])), "SOURCE_FLIGHT_ID"]
    candidates = []
    for start in starts:
        rows, current, seen = [], start, set()
        while current in lookup.index and current not in seen and len(rows) < MAX_EDGES:
            seen.add(current); row = lookup.loc[current]; rows.append(row); current = row["TARGET_FLIGHT_ID"]
        if len(rows) == MAX_EDGES:
            parse_flight(rows[0]["SOURCE_FLIGHT_ID"])
            for row in rows:
                parse_flight(row["TARGET_FLIGHT_ID"])
            candidates.append((sum(float(r["PROPAGATION_PROBABILITY"]) for r in rows), str(start), rows))
    if not candidates:
        raise RuntimeError("No predicted validation chain with at least three edges was found.")
    _, _, rows = max(candidates, key=lambda item: (item[0], item[1]))
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    apply_paper_style()
    columns = ["SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID", "CONNECTION_AIRPORT",
               "PLANNED_CONNECTION_MINUTES", "PROPAGATION_PROBABILITY", "PROPAGATION_ALERT"]
    edges = pd.read_parquet(EDGE_PATH, columns=columns)
    chain = select_chain(edges)
    flight_ids = [chain.iloc[0]["SOURCE_FLIGHT_ID"], *chain["TARGET_FLIGHT_ID"].tolist()]
    flights = [parse_flight(value) for value in flight_ids]
    print("Selected predicted chain:")
    print(chain.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11.2, 3.35)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xs = [.10, .365, .635, .90]; y, w, h = .57, .19, .24
    for index, (x, flight) in enumerate(zip(xs, flights)):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=.012",
                                    facecolor="#F7F9FB", edgecolor=PAPER_COLORS["steel_blue"], linewidth=1.05))
        ax.text(x, y+.045, f"Flight {chr(65+index)}  |  {flight['carrier']} {flight['number']}",
                ha="center", va="center", fontsize=9, fontweight="bold", color=PAPER_COLORS["navy"])
        ax.text(x, y-.015, f"{flight['origin']} -> {flight['destination']}", ha="center", va="center", fontsize=9)
        ax.text(x, y-.075, f"{flight['date']}  {flight['time']}", ha="center", va="center", fontsize=8, color=PAPER_COLORS["muted_text"])
    probs = chain["PROPAGATION_PROBABILITY"].astype(float)
    high_index = int(probs.idxmax())
    for i, row in chain.iterrows():
        color, width = (PAPER_COLORS["navy"], 2.2) if i == high_index else (PAPER_COLORS["steel_blue"], 1.25)
        ax.add_patch(FancyArrowPatch((xs[i]+w/2+.006, y), (xs[i+1]-w/2-.006, y),
                                    arrowstyle="-|>", mutation_scale=12, linewidth=width, color=color))
        ax.text((xs[i]+xs[i+1])/2, y+.145,
                f"p = {float(row['PROPAGATION_PROBABILITY']):.3f}\n{row['CONNECTION_AIRPORT']} | {float(row['PLANNED_CONNECTION_MINUTES']):.0f} min",
                ha="center", va="center", fontsize=8.2, color=PAPER_COLORS["muted_text"],
                bbox={"facecolor":"white", "edgecolor":"none", "pad":1})
    ax.text(.5, .12, "Each edge represents a physical same-aircraft rotation connection. Probabilities are edge-level model estimates\nand are not multiplied into a joint chain probability.",
            ha="center", va="center", fontsize=8, color=PAPER_COLORS["muted_text"], linespacing=1.3)
    save_figure(fig, "aircraft_rotation_propagation_example")
    plt.close(fig)


if __name__ == "__main__":
    main()
