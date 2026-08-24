"""Shared IEEE-friendly Matplotlib style and output helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"

PAPER_COLORS = {
    "navy": "#16324F", "steel_blue": "#456B8C", "pale_blue": "#DCE8F1",
    "teal": "#2F6F73", "pale_teal": "#D9E8E7", "bronze": "#B08D57",
    "text": "#202A35", "muted_text": "#5E6B76", "border": "#CDD5DC",
    "light_border": "#E6EBEF", "background": "#FFFFFF",
}
PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "aerorecover_probability",
    ["#DCE8F1", "#9FB9CB", "#5D829E", "#2F5D73", "#16324F"],
)
NAVY = PAPER_COLORS["navy"]
BLUE = PAPER_COLORS["pale_blue"]
ORANGE = PAPER_COLORS["bronze"]
LIGHT_GRAY = PAPER_COLORS["light_border"]
MID_GRAY = PAPER_COLORS["muted_text"]
DARK = PAPER_COLORS["text"]


def apply_paper_style() -> None:
    """Apply a compact, print-safe style with editable TrueType PDF text."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": PAPER_COLORS["text"],
            "axes.labelcolor": PAPER_COLORS["text"],
            "axes.edgecolor": PAPER_COLORS["border"],
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig, stem: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    png = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor=PAPER_COLORS["background"])
    for path in (pdf, png):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure output was not created: {path}")
        print(f"Created {path} ({path.stat().st_size:,} bytes)")
    return pdf, png
