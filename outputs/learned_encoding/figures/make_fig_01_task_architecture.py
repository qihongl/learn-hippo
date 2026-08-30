"""Generate the task-design and computational-flow schematic."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from plot_style import (
    AQUA,
    BLUE,
    INK,
    MUTED,
    PALE_AQUA,
    PALE_BLUE,
    PALE_RED,
    PALE_YELLOW,
    RED,
    YELLOW,
    apply_style,
    save_bundle,
)

HERE = Path(__file__).resolve().parent


def box(ax, xy, width, height, text, *, face, edge=INK, size=8, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        fontweight=weight,
        linespacing=1.2,
    )
    return patch


def arrow(ax, start, end, *, color=MUTED, dashed=False, curve=0.0, label=None):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.2,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + (0.035 if curve >= 0 else -0.04)
        ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=7, color=MUTED)


def panel_label(ax, label, title):
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.text(0.13, 1.02, title, transform=ax.transAxes, fontsize=10, fontweight="bold")


def draw_representation(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "(a)", "A situation model becomes complete")
    box(ax, (0.03, 0.73), 0.18, 0.15, "event cue\n(context)", face=PALE_BLUE, edge=BLUE)
    ax.text(0.27, 0.86, "feature slots", fontsize=8, color=MUTED)
    headers = ["F1", "F2", "F3", "F4"]
    for index, header in enumerate(headers):
        x = 0.28 + index * 0.16
        ax.text(x + 0.055, 0.79, header, ha="center", fontsize=8, fontweight="bold")

    rows = [
        (
            0.54,
            "mid-event",
            ["+1", "−1", "·", "·"],
            ["1", "1", "0", "0"],
            PALE_YELLOW,
            YELLOW,
        ),
        (
            0.21,
            "boundary\n(complete)",
            ["+1", "−1", "+1", "+1"],
            ["1", "1", "1", "1"],
            PALE_AQUA,
            AQUA,
        ),
    ]
    for y, label, values, mask, face, edge in rows:
        ax.text(0.01, y + 0.075, label, va="center", fontsize=7.5, fontweight="bold")
        for index, value in enumerate(values):
            x = 0.28 + index * 0.16
            value_face = face if value != "·" else "#f2f0ec"
            box(ax, (x, y), 0.11, 0.15, value, face=value_face, edge=edge, size=9)
        ax.text(0.20, y - 0.09, "mask", fontsize=7, color=MUTED)
        for index, bit in enumerate(mask):
            x = 0.28 + index * 0.16
            ax.text(
                x + 0.055,
                y - 0.09,
                bit,
                ha="center",
                fontsize=8,
                color=edge,
                fontweight="bold",
            )
    ax.text(
        0.03,
        0.01,
        "Mask: 1 = observed, 0 = not yet observed. All ones reveals completion.",
        fontsize=7.5,
        color=MUTED,
    )


def draw_trial(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "(b)", "Delayed event reconstruction")
    boxes = [
        (
            0.02,
            "EVENT OBSERVATION\nreveal one feature;\nchoose encode or skip",
            PALE_YELLOW,
            YELLOW,
        ),
        (0.38, "DELAY\ntransient state\ncleared", "#f2f0ec", MUTED),
        (
            0.65,
            "DELAYED QUERY\nretrieve from memory;\npredict missing features",
            PALE_AQUA,
            AQUA,
        ),
    ]
    widths = [0.28, 0.18, 0.32]
    for (x, text, face, edge), width in zip(boxes, widths, strict=True):
        box(
            ax,
            (x, 0.47),
            width,
            0.28,
            text,
            face=face,
            edge=edge,
            size=7.2,
            weight="bold",
        )
    arrow(ax, (0.30, 0.61), (0.38, 0.61), color=YELLOW)
    arrow(ax, (0.56, 0.61), (0.65, 0.61), color=AQUA)
    ax.text(
        0.37,
        0.32,
        "episodic memories persist",
        ha="center",
        fontsize=7.5,
        color=AQUA,
    )
    arrow(ax, (0.38, 0.46), (0.82, 0.46), color=AQUA, curve=0.18)
    box(
        ax,
        (0.76, 0.10),
        0.20,
        0.16,
        "reward = 1 − error",
        face=PALE_RED,
        edge=RED,
        size=8,
    )
    arrow(ax, (0.87, 0.47), (0.87, 0.27), color=RED)
    ax.text(
        0.03,
        0.20,
        "Goal after the delay: reconstruct missing\n"
        "situation features from episodic memory.",
        fontsize=7.4,
        color=MUTED,
    )


def draw_architecture(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "(c)", "Differentiable retrieval and learned encoding")
    box(
        ax,
        (0.02, 0.62),
        0.19,
        0.18,
        "current situation model\ncue + values + mask",
        face=PALE_BLUE,
        edge=BLUE,
    )
    box(
        ax,
        (0.28, 0.70),
        0.16,
        0.14,
        "encoding actor\np(encode)",
        face=PALE_YELLOW,
        edge=YELLOW,
        weight="bold",
    )
    box(
        ax,
        (0.28, 0.48),
        0.16,
        0.14,
        "critic\nV(state)",
        face="#f2f0ec",
        edge=MUTED,
    )
    box(
        ax,
        (0.52, 0.66),
        0.18,
        0.20,
        "episodic memories\nkey: situation model\nvalue: feature values",
        face=PALE_AQUA,
        edge=AQUA,
    )
    arrow(ax, (0.21, 0.72), (0.28, 0.77), color=BLUE)
    arrow(ax, (0.21, 0.69), (0.28, 0.55), color=MUTED)
    arrow(ax, (0.44, 0.77), (0.52, 0.77), color=YELLOW)

    box(
        ax,
        (0.02, 0.18),
        0.19,
        0.16,
        "delayed partial query\n(same event cue)",
        face=PALE_BLUE,
        edge=BLUE,
    )
    box(
        ax,
        (0.29, 0.17),
        0.17,
        0.18,
        "cosine similarity\n+ softmax attention",
        face=PALE_AQUA,
        edge=AQUA,
    )
    box(
        ax,
        (0.54, 0.17),
        0.14,
        0.18,
        "episodic retrieval\nweighted content",
        face=PALE_AQUA,
        edge=AQUA,
    )
    box(
        ax,
        (0.76, 0.17),
        0.20,
        0.18,
        "predict missing features\nthen delayed reward",
        face=PALE_RED,
        edge=RED,
    )
    arrow(ax, (0.21, 0.26), (0.29, 0.26), color=BLUE)
    arrow(ax, (0.61, 0.66), (0.42, 0.35), color=AQUA, curve=0.0)
    arrow(ax, (0.46, 0.26), (0.54, 0.26), color=AQUA)
    arrow(ax, (0.68, 0.26), (0.76, 0.26), color=AQUA)
    arrow(
        ax,
        (0.87, 0.36),
        (0.40, 0.48),
        color=RED,
        dashed=True,
        curve=0.24,
        label="delayed advantage",
    )
    ax.text(
        0.74,
        0.77,
        "An incomplete memory can dominate\nretrieval and block the endpoint memory.",
        fontsize=7.5,
        color=MUTED,
        va="center",
    )


def main():
    apply_style()
    fig = plt.figure(figsize=(7.2, 5.3), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.25],
        width_ratios=[1.0, 1.08],
        left=0.04,
        right=0.98,
        bottom=0.05,
        top=0.90,
        hspace=0.38,
        wspace=0.20,
    )
    draw_representation(fig.add_subplot(grid[0, 0]))
    draw_trial(fig.add_subplot(grid[0, 1]))
    draw_architecture(fig.add_subplot(grid[1, :]))
    fig.suptitle(
        "A controlled test of learned boundary-selective episodic encoding",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    save_bundle(fig, HERE, "fig_01_task_architecture")
    plt.close(fig)


if __name__ == "__main__":
    main()
