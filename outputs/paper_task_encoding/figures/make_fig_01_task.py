"""Generate the exact task representation and trial-sequence schematic."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from plot_style import AQUA, BLUE, INK, LIGHT, MUTED, ORANGE, RED, VIOLET, save_figure


def box(ax, x, y, width, height, text, *, face=LIGHT, edge=MUTED, size=7):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size
    )


def arrow(ax, start, end, *, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
        )
    )


fig = plt.figure(figsize=(7.2, 4.5))
grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.35)

ax = fig.add_subplot(grid[0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(-0.02, 1.03, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
ax.set_title(
    "Stimulus and the original 37-dimensional input", loc="left", fontweight="bold"
)

example_values = [2, 0, 3, 1, 1, 2, 0, 3, 2, 1, 3, 0, 1, 3, 2, 0]
colors = [BLUE, AQUA, ORANGE, VIOLET]
start_x = 0.02
tile_width = 0.034
for index, value in enumerate(example_values):
    x = start_x + index * (tile_width + 0.004)
    ax.add_patch(
        Rectangle(
            (x, 0.60), tile_width, 0.19, facecolor=colors[value], edgecolor="white"
        )
    )
    ax.text(
        x + tile_width / 2,
        0.695,
        str(value),
        color="white",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )
    ax.text(
        x + tile_width / 2,
        0.55,
        f"F{index + 1}",
        ha="center",
        va="top",
        fontsize=5.8,
        color=MUTED,
    )
ax.text(
    0.02,
    0.88,
    "Situation = 16 features; each feature has one of four values",
    fontsize=7.5,
    color=INK,
)

segments = [
    ("observed feature\n16 one-hot units", 0.37, BLUE),
    ("observed value\n4 one-hot units", 0.14, AQUA),
    ("queried feature\n16 one-hot units", 0.37, ORANGE),
    ("penalty\n1 unit", 0.08, VIOLET),
]
x = 0.02
for label, display_width, color in segments:
    ax.add_patch(
        Rectangle(
            (x, 0.12),
            display_width,
            0.24,
            facecolor=color,
            alpha=0.88,
            edgecolor="white",
        )
    )
    ax.text(
        x + display_width / 2,
        0.24,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=6.6,
        fontweight="bold",
    )
    x += display_width
ax.text(
    0.02,
    0.04,
    "At every step the model predicts four possible values or ‘don’t know.’ "
    "No boundary flag or completion mask is added.",
    fontsize=7,
    color=INK,
)

ax = fig.add_subplot(grid[1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(-0.02, 1.03, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
ax.set_title("One trial contains three events", loc="left", fontweight="bold")

box(
    ax,
    0.02,
    0.54,
    0.19,
    0.28,
    "a1: distracting event\nnew random situation\n16 + delay steps",
    face="#e8f1fb",
    edge=BLUE,
)
box(
    ax,
    0.29,
    0.54,
    0.19,
    0.28,
    "b1: first target event\nnew random situation\n16 + delay steps",
    face="#e7f6f0",
    edge=AQUA,
)
box(
    ax,
    0.57,
    0.54,
    0.14,
    0.28,
    "memory condition\napplied before b2",
    face="#fff5df",
    edge=ORANGE,
)
box(
    ax,
    0.79,
    0.54,
    0.19,
    0.28,
    "b2: prediction event\nrelated or unrelated\n16 + delay steps",
    face="#f0edf9",
    edge=VIOLET,
)
arrow(ax, (0.21, 0.68), (0.29, 0.68))
arrow(ax, (0.48, 0.68), (0.57, 0.68))
arrow(ax, (0.71, 0.68), (0.79, 0.68))

box(
    ax,
    0.06,
    0.14,
    0.23,
    0.20,
    "RM — recent memory\nb1 state remains active\nb2 repeats b1",
    face="#edf7ec",
    edge=AQUA,
    size=6.5,
)
box(
    ax,
    0.385,
    0.14,
    0.23,
    0.20,
    "DM — distant memory\nb1 state is reset\nb2 repeats b1",
    face="#fff5df",
    edge=ORANGE,
    size=6.5,
)
box(
    ax,
    0.71,
    0.14,
    0.23,
    0.20,
    "NM — no relevant memory\nstate is reset\nb2 is unrelated",
    face="#fdecec",
    edge=RED,
    size=6.5,
)
for target_x in (0.175, 0.50, 0.825):
    arrow(ax, (0.64, 0.54), (target_x, 0.34), color=MUTED)
ax.text(
    0.02,
    0.01,
    "Observation order is random. Query order is fixed, begins after a delay "
    "of 0–4 steps in meta-training, and has delay 0 in held-out evaluation.",
    fontsize=7,
    color=INK,
)

save_figure(fig, Path(__file__).with_name("fig_01_task"))
plt.close(fig)
