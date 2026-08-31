"""Summarize boundary selectivity and reward across follow-up regimes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import AQUA, BLUE, INK, MUTED, ORANGE, RED, save_figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "outputs/paper_task_encoding"


def load_summary(name: str) -> dict:
    """Load one compact, versioned experiment summary."""

    return json.loads((OUTPUT / name).read_text())


temporal = load_summary("temporal_hazard_summary.json")
exact_dm = load_summary("neural_counterfactual_summary.json")
sampled_dm = load_summary("sampled_hazard_summary.json")
factorial = load_summary("sampled_hazard_factorial_summary.json")
sampled_full = load_summary("sampled_hazard_full_mixture_summary.json")
exact_full = load_summary("neural_counterfactual_full_mixture_summary.json")
curriculum = load_summary("neural_counterfactual_curriculum_summary.json")

labels = [
    "temporal audit\nDM, exact credit",
    "neural policy\nDM, exact credit",
    "neural policy\nDM, sampled reward",
    "neural policy\nDM, variable timing",
    "neural policy\nfull mix, sampled",
    "neural policy\nfull mix, exact",
    "DM curriculum\nthen full mix",
]
endpoint_gaps = np.asarray(
    [
        temporal["mean"]["endpoint_probability_gap"],
        exact_dm["final_evaluation"]["endpoint_probability_gap"],
        sampled_dm["final_evaluation"]["endpoint_probability_gap"],
        factorial["dm_variable_delay_and_removal"]["endpoint_probability"]
        - factorial["dm_variable_delay_and_removal"]["nonendpoint_probability"],
        sampled_full["final_evaluation"]["DM"]["endpoint_probability_gap"],
        exact_full["final_evaluation"]["DM"]["endpoint_probability_gap"],
        curriculum["final_evaluation"]["DM"]["endpoint_probability"]
        - curriculum["final_evaluation"]["DM"]["nonendpoint_probability"],
    ]
)

reward_rows = [
    (
        exact_dm["final_evaluation"]["learned_expected_reward"],
        exact_dm["final_evaluation"]["matched_random_one_reward"],
        exact_dm["final_evaluation"]["never_pair_reward"],
        exact_dm["final_evaluation"]["endpoint_pair_reward"],
    ),
    (
        sampled_dm["final_evaluation"]["learned_expected_reward"],
        sampled_dm["final_evaluation"]["matched_random_one_reward"],
        sampled_dm["final_evaluation"]["never_pair_reward"],
        sampled_dm["final_evaluation"]["endpoint_pair_reward"],
    ),
    (
        factorial["dm_variable_delay_and_removal"]["learned_reward"],
        factorial["dm_variable_delay_and_removal"]["matched_random_one_reward"],
        factorial["dm_variable_delay_and_removal"]["never_reward"],
        factorial["dm_variable_delay_and_removal"]["forced_endpoint_reward"],
    ),
    (
        sampled_full["final_evaluation"]["DM"]["learned_reward"],
        sampled_full["final_evaluation"]["DM"]["matched_random_one_reward"],
        sampled_full["final_evaluation"]["DM"]["never_pair_reward"],
        sampled_full["final_evaluation"]["DM"]["endpoint_pair_reward"],
    ),
    (
        exact_full["final_evaluation"]["DM"]["learned_reward"],
        exact_full["final_evaluation"]["DM"]["matched_random_one_reward"],
        exact_full["final_evaluation"]["DM"]["never_pair_reward"],
        exact_full["final_evaluation"]["DM"]["endpoint_pair_reward"],
    ),
    (
        curriculum["final_evaluation"]["DM"]["learned_reward"],
        curriculum["final_evaluation"]["DM"]["matched_random_one_reward"],
        curriculum["final_evaluation"]["DM"]["never_reward"],
        curriculum["final_evaluation"]["DM"]["forced_endpoint_reward"],
    ),
]
reward_labels = labels[1:]
learned_fraction = np.asarray(
    [
        (learned - never) / (endpoint - never)
        for learned, _, never, endpoint in reward_rows
    ]
)
random_fraction = np.asarray(
    [
        (random - never) / (endpoint - never)
        for _, random, never, endpoint in reward_rows
    ]
)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.15), constrained_layout=True)

ax = axes[0]
ax.text(-0.15, 1.03, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
y = np.arange(len(labels))
colors = [BLUE, BLUE, AQUA, AQUA, RED, RED, ORANGE]
ax.scatter(endpoint_gaps, y, color=colors, s=34, zorder=3)
for index, value in enumerate(endpoint_gaps):
    ax.plot([0, value], [index, index], color=colors[index], linewidth=1.1, alpha=0.7)
ax.axvline(0, color=MUTED, linewidth=0.8)
ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(-0.11, 1.06)
ax.set_xlabel("Endpoint probability minus nonendpoint mean")
ax.set_title("Final held-out encoding policy")

ax = axes[1]
ax.text(-0.15, 1.03, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
y = np.arange(len(reward_labels))
ax.axvline(0, color=MUTED, linestyle=":", linewidth=0.8)
ax.axvline(1, color=MUTED, linestyle="--", linewidth=0.8)
for index in y:
    low, high = sorted([learned_fraction[index], random_fraction[index]])
    ax.plot([low, high], [index, index], color="#c5cbd2", linewidth=1.2, zorder=1)
ax.scatter(
    learned_fraction,
    y,
    color=AQUA,
    s=32,
    label="learned policy",
    zorder=3,
)
ax.scatter(
    random_fraction,
    y,
    color=INK,
    marker="x",
    s=30,
    label="matched random one",
    zorder=3,
)
ax.set_yticks(y, reward_labels)
ax.invert_yaxis()
ax.set_xlim(-0.10, 1.10)
ax.set_xlabel("Fraction of forced-endpoint DM reward gain")
ax.set_title("Prediction benefit does not imply boundary encoding")
ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.19),
    ncol=2,
)

save_figure(fig, HERE / "fig_04_results")
plt.close(fig)
