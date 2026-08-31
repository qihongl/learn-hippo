"""Shared publication style for the exact paper-task figures."""

from pathlib import Path

import matplotlib.pyplot as plt

STYLE = Path("/Users/qlu/.agents/skills/publication-figures/openscience.mplstyle")
if STYLE.exists():
    plt.style.use(str(STYLE))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
INK = "#252a31"
MUTED = "#69717d"
LIGHT = "#eef2f6"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save vector and preview versions of one figure."""

    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_text = svg_path.read_text()
    clean_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
    svg_path.write_text(clean_svg)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
