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
    ax.text(0.09, 1.02, title, transform=ax.transAxes, fontsize=10, fontweight="bold")


def draw_representation(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "(a)", "Event representation becomes complete")
    box(ax, (0.03, 0.73), 0.18, 0.15, "random\nevent cue", face=PALE_BLUE, edge=BLUE)
    ax.text(0.27, 0.86, "feature slots", fontsize=8, color=MUTED)
    headers = ["F1", "F2", "F3", "F4"]
    for index, header in enumerate(headers):
        x = 0.28 + index * 0.16
        ax.text(x + 0.055, 0.79, header, ha="center", fontsize=8, fontweight="bold")

    rows = [
        (0.56, "mid-event", ["+1", "−1", "?", "?"], PALE_YELLOW, YELLOW),
        (0.24, "boundary", ["+1", "−1", "+1", "+1"], PALE_AQUA, AQUA),
    ]
    for y, label, values, face, edge in rows:
        ax.text(0.03, y + 0.075, label, va="center", fontsize=8, fontweight="bold")
        for index, value in enumerate(values):
            x = 0.28 + index * 0.16
            value_face = face if value != "?" else "#f2f0ec"
            box(ax, (x, y), 0.11, 0.15, value, face=value_face, edge=edge, size=9)
    ax.text(0.28, 0.10, "mask", fontsize=7, color=MUTED)
    ax.text(0.40, 0.10, "1  1  1  1", fontsize=8, color=AQUA, fontweight="bold")
    ax.text(
        0.03,
        0.01,
        "The gate sees cue + accumulated values + mask, but no boundary bit.",
        fontsize=7.5,
        color=MUTED,
    )


def draw_trial(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "(b)", "One study–test episode")
    boxes = [
        (
            0.02,
            "STUDY + WRITE\nreveal one feature;\nsample gate after each state",
            PALE_YELLOW,
            YELLOW,
        ),
        (0.38, "RESET\ncontroller\ncleared", "#f2f0ec", MUTED),
        (
            0.65,
            "DELAYED TEST\npartial query and read\npredict missing features",
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
        "episodic memory persists",
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
        "reward = 1 − MSE",
        face=PALE_RED,
        edge=RED,
        size=8,
    )
    arrow(ax, (0.87, 0.47), (0.87, 0.27), color=RED)


def draw_architecture(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "(c)", "Differentiable read, discrete learned write")
    box(
        ax,
        (0.02, 0.62),
        0.19,
        0.18,
        "partial event state\ncue + values + mask",
        face=PALE_BLUE,
        edge=BLUE,
    )
    box(
        ax,
        (0.28, 0.70),
        0.16,
        0.14,
        "write actor\np(write)",
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
        "episodic slots\nkey: partial state\nvalue: feature state",
        face=PALE_AQUA,
        edge=AQUA,
    )
    arrow(ax, (0.21, 0.72), (0.28, 0.77), color=BLUE)
    arrow(ax, (0.21, 0.69), (0.28, 0.55), color=MUTED)
    arrow(ax, (0.44, 0.77), (0.52, 0.77), color=YELLOW, label="sample 0/1")

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
        "memory read\nweighted value",
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
        "Incomplete traces can win\nattention and block the endpoint.",
        fontsize=7.5,
        color=MUTED,
        va="center",
    )


def main():
    apply_style()
    fig = plt.figure(figsize=(7.2, 5.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.22, wspace=0.18)
    draw_representation(fig.add_subplot(grid[0, 0]))
    draw_trial(fig.add_subplot(grid[0, 1]))
    draw_architecture(fig.add_subplot(grid[1, :]))
    fig.suptitle(
        "A controlled test of learned boundary-selective episodic writing",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    save_bundle(fig, HERE, "fig_01_task_architecture")
    plt.close(fig)


if __name__ == "__main__":
    main()
