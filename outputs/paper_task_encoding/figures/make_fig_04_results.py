"""Plot the mixed-objective audit and bounded credit interventions."""

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


temporal = load_summary("temporal_hazard_full_mixture_summary.json")
factorial = load_summary("sampled_hazard_credit_factorial_summary.json")
selected = load_summary("sampled_hazard_credit_selected_400_summary.json")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0), constrained_layout=True)

ax = axes[0]
ax.text(-0.13, 1.03, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
time_probabilities = np.asarray(
    [run["encoding_time_probabilities"][:16] for run in temporal["runs"]]
)
times = np.arange(1, 17)
for probabilities in time_probabilities:
    ax.plot(times, probabilities, color=BLUE, alpha=0.24, linewidth=0.9)
ax.plot(
    times,
    time_probabilities.mean(axis=0),
    color=BLUE,
    marker="o",
    markersize=2.8,
    linewidth=1.6,
    label="learned mean (5 seeds)",
)
ax.axvline(16, color=ORANGE, linestyle="--", linewidth=1.0, label="event endpoint")
ax.set_xticks([1, 4, 8, 12, 16])
ax.set_xlim(0.6, 16.4)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Observation position")
ax.set_ylabel("Probability of first encoding")
ax.set_title(
    "Exact mixed reward has a better endpoint solution\n"
    "but gradients stop one or two steps early"
)
ax.legend(frameon=False, loc="upper left")

ax = axes[1]
ax.text(-0.16, 1.03, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
factor_labels = {
    "sampled_hazard_credit_low_centered_annealed": "low / centered / annealed",
    "sampled_hazard_credit_low_centered_fixed": "low / centered / direct",
    "sampled_hazard_credit_low_critic_annealed": "low / critic / annealed",
    "sampled_hazard_credit_low_critic_fixed": "low / critic / direct",
    "sampled_hazard_credit_neutral_centered_annealed": "neutral / centered / annealed",
    "sampled_hazard_credit_neutral_centered_fixed": "neutral / centered / direct",
    "sampled_hazard_credit_neutral_critic_annealed": "neutral / critic / annealed",
    "sampled_hazard_credit_neutral_critic_fixed": "neutral / critic / direct",
}
cells = factorial["cells"]
labels = [factor_labels[cell["name"]] for cell in cells] + [
    "selected method / 400 epochs"
]
rows = [[run["endpoint_probability"] for run in cell["runs"]] for cell in cells]
rows.append([run["endpoint_probability"] for run in selected["runs"]])
y = np.arange(len(rows))
for index, values in enumerate(rows):
    color = ORANGE if index in (0, len(rows) - 1) else AQUA
    offsets = np.linspace(-0.09, 0.09, len(values))
    ax.scatter(
        values,
        np.full(len(values), index) + offsets,
        color=color,
        s=22,
        zorder=3,
    )
    ax.plot(
        [np.mean(values)] * 2,
        [index - 0.16, index + 0.16],
        color=INK,
        linewidth=1.1,
    )
ax.axvline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.text(0.79, -0.55, "criterion", ha="right", color=MUTED, fontsize=6.5)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(-0.04, 1.04)
ax.set_xlabel("Final endpoint probability")
ax.set_title(
    "A bounded credit screen finds a promising\n"
    "but unstable full-mixture method"
)
ax.text(
    0.02,
    -0.17,
    "label order: initial policy / reward baseline / training schedule",
    transform=ax.transAxes,
    fontsize=6.4,
    color=MUTED,
)

save_figure(fig, HERE / "fig_04_results")
plt.close(fig)
