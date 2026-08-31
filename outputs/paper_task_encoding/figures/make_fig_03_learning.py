"""Plot multiseed held-out learning curves and final seed distributions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, save_figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "outputs/paper_task_encoding"


def load_summary(name: str) -> dict:
    """Load one compact, versioned experiment summary."""

    return json.loads((OUTPUT / name).read_text())


fixed_dm = load_summary("sampled_hazard_dm_fixed_replication_summary.json")
variable_dm = load_summary("sampled_hazard_dm_variable_replication_summary.json")
selected_full = load_summary("sampled_hazard_credit_selected_400_summary.json")

regimes = [
    ("fixed-duration DM (10 seeds)", fixed_dm, AQUA),
    ("variable-duration DM (10 seeds)", variable_dm, BLUE),
    ("full mixture, selected method (3 seeds)", selected_full, ORANGE),
]

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

ax = axes[0]
ax.text(-0.13, 1.04, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
for label, summary, color in regimes:
    curve = summary["learning_curves"]
    epochs = np.asarray([point["epoch"] for point in curve])
    means = np.asarray([point["endpoint_probability"]["mean"] for point in curve])
    standard_deviations = np.asarray(
        [point["endpoint_probability"]["std"] for point in curve]
    )
    ax.fill_between(
        epochs,
        np.clip(means - standard_deviations, 0, 1),
        np.clip(means + standard_deviations, 0, 1),
        color=color,
        alpha=0.13,
        linewidth=0,
    )
    ax.plot(epochs, means, color=color, linewidth=1.5, label=label)
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.text(397, 0.815, "declared criterion", ha="right", color=MUTED, fontsize=6.5)
ax.set_xlim(0, 405)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("Policy-training epoch (256 sequences per epoch)")
ax.set_ylabel("Held-out endpoint probability")
ax.set_title("Boundary learning is reliable only in fixed-duration DM")
ax.legend(frameon=False, loc="lower right")

ax = axes[1]
ax.text(-0.13, 1.04, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
for index, (_, summary, color) in enumerate(regimes):
    values = np.asarray([run["endpoint_probability"] for run in summary["runs"]])
    offsets = np.linspace(-0.12, 0.12, len(values))
    ax.scatter(
        np.full(len(values), index) + offsets,
        values,
        color=color,
        edgecolor="white",
        linewidth=0.4,
        s=25,
        zorder=3,
    )
    ax.plot([index - 0.18, index + 0.18], [values.mean()] * 2, color=INK, linewidth=1.2)
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_xticks(
    range(3),
    ["fixed DM\n10 seeds", "variable DM\n10 seeds", "full mixture\n3 seeds"],
)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.03, 1.05)
ax.set_ylabel("Final endpoint probability")
ax.set_title("Means conceal discrete optimization failures")

save_figure(fig, HERE / "fig_03_learning")
plt.close(fig)
