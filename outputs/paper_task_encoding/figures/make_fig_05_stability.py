"""Plot every seed in the predeclared optimizer-stability screen."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import AQUA, BLUE, INK, MUTED, VIOLET, YELLOW, save_figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "outputs/paper_task_encoding"

CELLS = [
    (
        "constant, batch 16",
        "sampled_hazard_stability_constant_b16_summary.json",
        BLUE,
    ),
    (
        "constant, batch 32",
        "sampled_hazard_stability_constant_b32_summary.json",
        AQUA,
    ),
    (
        "cosine decay, batch 16",
        "sampled_hazard_stability_cosine_b16_summary.json",
        YELLOW,
    ),
    (
        "cosine decay, batch 32",
        "sampled_hazard_stability_cosine_b32_summary.json",
        VIOLET,
    ),
]


def load_summary(name: str) -> dict:
    """Load one compact, versioned experiment summary."""

    return json.loads((OUTPUT / name).read_text())


summaries = [(label, load_summary(name), color) for label, name, color in CELLS]
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)

ax = axes[0]
ax.text(-0.13, 1.04, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
for label, summary, color in summaries:
    for run in summary["runs"]:
        trajectory = run["checkpoint_trajectory"]
        ax.plot(
            [point["epoch"] for point in trajectory],
            [point["endpoint_probability"] for point in trajectory],
            color=color,
            alpha=0.22,
            linewidth=0.75,
        )
    curve = summary["learning_curves"]
    ax.plot(
        [point["epoch"] for point in curve],
        [point["endpoint_probability"]["mean"] for point in curve],
        color=color,
        linewidth=1.7,
        label=label,
    )
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_xlim(0, 405)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("Policy-training epoch (256 sequences per epoch)")
ax.set_ylabel("Held-out endpoint probability")
ax.set_title("Full-mixture learning trajectories")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.01, 0.52))

ax = axes[1]
ax.text(-0.13, 1.04, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
for index, (_, summary, color) in enumerate(summaries):
    values = np.asarray([run["endpoint_probability"] for run in summary["runs"]])
    offsets = np.linspace(-0.10, 0.10, len(values))
    ax.scatter(
        np.full(len(values), index) + offsets,
        values,
        color=color,
        edgecolor="white",
        linewidth=0.4,
        s=28,
        zorder=3,
    )
    ax.plot(
        [index - 0.18, index + 0.18],
        [values.mean()] * 2,
        color=INK,
        linewidth=1.2,
    )
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_xticks(
    range(4),
    [
        "constant\nbatch 16",
        "constant\nbatch 32",
        "cosine\nbatch 16",
        "cosine\nbatch 32",
    ],
)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.03, 1.05)
ax.set_ylabel("Final endpoint probability")
ax.set_title("Final checkpoints for all paired seeds")

save_figure(fig, HERE / "fig_05_stability")
plt.close(fig)
