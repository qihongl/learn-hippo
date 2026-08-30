"""Shared publication style for learned-boundary figures."""

from pathlib import Path

import matplotlib.pyplot as plt

STYLE = Path("/Users/qlu/.agents/skills/publication-figures/openscience.mplstyle")

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
ORANGE = "#eb6834"
INK = "#2a2723"
MUTED = "#6f6a61"
GRID = "#e7e3da"
PALE_BLUE = "#eaf3fd"
PALE_AQUA = "#e7f7f1"
PALE_YELLOW = "#fff5d8"
PALE_RED = "#fdeceb"


def apply_style() -> None:
    """Apply the Open Science style with editable vector text."""

    if STYLE.exists():
        plt.style.use(STYLE)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
        }
    )


def save_bundle(fig: plt.Figure, directory: Path, basename: str) -> None:
    """Save editable and preview exports."""

    fig.savefig(directory / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{basename}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{basename}.png", dpi=300, bbox_inches="tight")
