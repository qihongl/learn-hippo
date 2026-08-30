"""Generate the encoding, retrieval, and optimization schematic."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, RED, VIOLET, save_figure


def box(ax, x, y, width, height, title, body, *, color=BLUE):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.015,
        y + height - 0.055,
        title,
        fontsize=7.5,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        x + 0.015,
        y + height - 0.12,
        body,
        fontsize=6.5,
        color=MUTED,
        va="top",
        linespacing=1.3,
    )


def arrow(ax, start, end, *, color=MUTED, dashed=False, label=None, label_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=6.2, color=color)


fig, axes = plt.subplots(
    2, 1, figsize=(7.2, 4.8), gridspec_kw={"height_ratios": [1.25, 0.9]}
)
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

ax = axes[0]
ax.text(-0.02, 1.03, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
ax.set_title(
    "Encoding and retrieval use only information available in the original task",
    loc="left",
    fontweight="bold",
)
box(
    ax,
    0.02,
    0.56,
    0.18,
    0.32,
    "Original input",
    "feature key (16)\nvalue (4)\nquery key (16)\npenalty (1)",
    color=BLUE,
)
box(
    ax,
    0.27,
    0.56,
    0.19,
    0.32,
    "Situation model",
    "feature–value table\n+ query history\n80 numbers",
    color=AQUA,
)
box(
    ax,
    0.53,
    0.56,
    0.18,
    0.32,
    "Encoding actor",
    "shared for a1 and b1\n$p_t$ = P(encode)\nencode or skip",
    color=ORANGE,
)
box(
    ax,
    0.78,
    0.56,
    0.18,
    0.32,
    "Episodic store",
    "state snapshots\n≤ 40 traces\nno eviction",
    color=VIOLET,
)
ax.text(
    0.50,
    0.49,
    "Actor input excludes boundary, countdown, and RM/DM/NM condition labels",
    ha="center",
    va="center",
    fontsize=6.8,
    color=RED,
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fdecec", "edgecolor": RED},
)
arrow(ax, (0.20, 0.725), (0.27, 0.725))
arrow(ax, (0.46, 0.725), (0.53, 0.725))
arrow(ax, (0.71, 0.725), (0.78, 0.725))

box(
    ax,
    0.12,
    0.05,
    0.20,
    0.29,
    "b2 partial situation",
    "current observations form\na retrieval cue",
    color=BLUE,
)
box(
    ax,
    0.40,
    0.05,
    0.20,
    0.29,
    "Competitive retrieval",
    "content-match gate ×\ncosine-softmax attention",
    color=VIOLET,
)
box(
    ax,
    0.68,
    0.05,
    0.20,
    0.29,
    "Prediction",
    "four value responses\n+ ‘don’t know’\nexpected task reward",
    color=AQUA,
)
arrow(ax, (0.32, 0.195), (0.40, 0.195))
arrow(ax, (0.60, 0.195), (0.68, 0.195))
arrow(
    ax,
    (0.87, 0.56),
    (0.55, 0.34),
    color=VIOLET,
    label="stored traces",
    label_xy=(0.72, 0.41),
)

ax = axes[1]
ax.text(-0.02, 1.03, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
ax.set_title(
    "Staged optimization and the delayed learning signal", loc="left", fontweight="bold"
)
stages = [
    (
        0.015,
        "1  Task + oracle",
        "validate task\nand fixed schedules",
        BLUE,
    ),
    (
        0.265,
        "2  Prediction + retrieval",
        "verify endpoint memory\nimproves prediction",
        AQUA,
    ),
    (
        0.515,
        "3  Forced exploration",
        "random actions train\nthe value estimator",
        ORANGE,
    ),
    (
        0.765,
        "4  Free selection",
        "actor chooses encoding\nlater b2 reward gives credit",
        VIOLET,
    ),
]
for x, title, body, color in stages:
    box(ax, x, 0.32, 0.205, 0.38, title, body, color=color)
for left in (0.22, 0.47, 0.72):
    arrow(ax, (left, 0.51), (left + 0.045, 0.51))
arrow(
    ax,
    (0.88, 0.32),
    (0.61, 0.21),
    color=RED,
    dashed=True,
    label="delayed reward",
    label_xy=(0.76, 0.19),
)
ax.text(
    0.02,
    0.02,
    "During held-out evaluation all model weights are frozen; only within-trial "
    "episodic memories change.",
    fontsize=7,
    color=INK,
)

save_figure(fig, Path(__file__).with_name("fig_02_model"))
plt.close(fig)
