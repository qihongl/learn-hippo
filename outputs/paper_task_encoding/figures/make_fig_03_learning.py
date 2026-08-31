"""Plot held-out learning curves for the follow-up encoding experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import AQUA, BLUE, MUTED, ORANGE, RED, VIOLET, save_figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "outputs/paper_task_encoding"


def load_summary(name: str) -> dict:
    """Load one compact, versioned experiment summary."""

    return json.loads((OUTPUT / name).read_text())


sampled_dm = load_summary("sampled_hazard_summary.json")
sampled_full = load_summary("sampled_hazard_full_mixture_summary.json")
exact_dm = load_summary("neural_counterfactual_summary.json")
exact_full = load_summary("neural_counterfactual_full_mixture_summary.json")

curves = [
    (
        "sampled reward, DM only",
        sampled_dm["held_out_curve"],
        AQUA,
        sampled_dm["final_evaluation"]["never_pair_reward"],
        sampled_dm["final_evaluation"]["endpoint_pair_reward"],
    ),
    (
        "exact credit, DM only",
        exact_dm["held_out_curve"],
        BLUE,
        exact_dm["final_evaluation"]["never_pair_reward"],
        exact_dm["final_evaluation"]["endpoint_pair_reward"],
    ),
    (
        "sampled reward, full mixture",
        sampled_full["dm_held_out_curve"],
        RED,
        sampled_full["final_evaluation"]["DM"]["never_pair_reward"],
        sampled_full["final_evaluation"]["DM"]["endpoint_pair_reward"],
    ),
    (
        "exact credit, full mixture",
        exact_full["dm_curve"],
        ORANGE,
        exact_full["final_evaluation"]["DM"]["never_pair_reward"],
        exact_full["final_evaluation"]["DM"]["endpoint_pair_reward"],
    ),
]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85), constrained_layout=True)

ax = axes[0]
ax.text(-0.13, 1.04, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
for label, curve, color, _, _ in curves:
    epochs = np.asarray([point["epoch"] for point in curve])
    endpoint = np.asarray([point["endpoint_probability"] for point in curve])
    ax.plot(epochs, endpoint, marker="o", markersize=2.6, color=color, label=label)
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.text(
    397,
    0.815,
    "declared endpoint criterion",
    ha="right",
    color=MUTED,
    fontsize=6.5,
)
ax.set_xlim(0, 405)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("Policy-training epoch (256 sequences per epoch)")
ax.set_ylabel("Held-out endpoint probability")
ax.set_title("Boundary preference emerges only in DM-only training")
ax.legend(frameon=False, loc="center right")

ax = axes[1]
ax.text(-0.13, 1.04, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
for label, curve, color, never_reward, endpoint_reward in curves:
    epochs = np.asarray([point["epoch"] for point in curve])
    rewards = np.asarray([point["reward"] for point in curve])
    recovered = (rewards - never_reward) / (endpoint_reward - never_reward)
    ax.plot(epochs, recovered, marker="o", markersize=2.6, color=color, label=label)
ax.axhline(1.0, color=VIOLET, linestyle="--", linewidth=0.8)
ax.axhline(0.0, color=MUTED, linestyle=":", linewidth=0.8)
ax.text(397, 1.025, "forced endpoint = 1", ha="right", color=VIOLET, fontsize=6.5)
ax.text(397, 0.025, "never encode = 0", ha="right", color=MUTED, fontsize=6.5)
ax.set_xlim(0, 405)
ax.set_ylim(-0.12, 1.12)
ax.set_xlabel("Policy-training epoch (256 sequences per epoch)")
ax.set_ylabel("Fraction of forced-endpoint reward gain")
ax.set_title("Prediction reward can improve without a boundary policy")

save_figure(fig, HERE / "fig_03_learning")
plt.close(fig)
