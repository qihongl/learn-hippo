"""Generate learned encoding dynamics and held-out selectivity evidence."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import BLUE, INK, MUTED, ORANGE, RED, save_figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
files = sorted((REPOSITORY / "outputs/paper_task_encoding/reported").glob("*.json"))
summary = json.loads(
    (REPOSITORY / "outputs/paper_task_encoding/reported_summary.json").read_text()
)
seeds = [json.loads(path.read_text()) for path in files]

endpoint = np.array(
    [
        [
            point["endpoint_probability"]
            for point in run["training"]["free_policy_history"]
        ]
        for run in seeds
    ]
)
nonendpoint = np.array(
    [
        [
            point["nonendpoint_probability"]
            for point in run["training"]["free_policy_history"]
        ]
        for run in seeds
    ]
)
steps = np.arange(1, endpoint.shape[1] + 1)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.6, 2.6),
    gridspec_kw={"width_ratios": [1.25, 1.05, 0.78], "wspace": 0.38},
)

ax = axes[0]
ax.text(-0.18, 1.06, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
for values, color, label in (
    (endpoint, ORANGE, "endpoint"),
    (nonendpoint, BLUE, "nonendpoint mean"),
):
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    ax.plot(steps, mean, color=color, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
ax.set_xlabel("Free-selection update")
ax.set_ylabel("Encoding probability")
ax.set_title("Training probabilities")
ax.set_ylim(0, max(0.055, float(np.max([endpoint, nonendpoint])) * 1.08))
ax.legend(frameon=False, loc="upper right")

ax = axes[1]
ax.text(-0.18, 1.06, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
time = summary["conditions"]["DM"]["time_probabilities"]
means = np.array([cell["mean"] for cell in time])
low = np.array([cell["ci95"][0] for cell in time])
high = np.array([cell["ci95"][1] for cell in time])
x = np.arange(1, 17)
ax.plot(x, means, color=BLUE, marker="o", markersize=3)
ax.fill_between(x, low, high, color=BLUE, alpha=0.16, linewidth=0)
ax.axvline(16, color=RED, linestyle="--", linewidth=1)
ax.text(
    15.7,
    max(high) * 0.96,
    "event\nboundary",
    ha="right",
    va="top",
    color=RED,
    fontsize=6.5,
)
ax.set_xlabel("Step in held-out event")
ax.set_ylabel("P(encode)")
ax.set_title("Held-out time profile")
ax.set_xticks([1, 4, 8, 12, 16])
ax.set_ylim(0, max(0.04, high.max() * 1.12))

ax = axes[2]
ax.text(-0.24, 1.06, "c", fontweight="bold", fontsize=10, transform=ax.transAxes)
gaps = np.array(summary["audit_metrics"]["endpoint_probability_gap"]["values_by_seed"])
rng = np.random.default_rng(20260830)
ax.scatter(
    rng.normal(0, 0.035, len(gaps)),
    gaps,
    s=22,
    color=BLUE,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.4,
)
cell = summary["audit_metrics"]["endpoint_probability_gap"]
ax.errorbar(
    0.20,
    cell["mean"],
    yerr=[[cell["mean"] - cell["ci95"][0]], [cell["ci95"][1] - cell["mean"]]],
    fmt="o",
    color=RED,
    capsize=3,
    markersize=4,
)
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.set_xlim(-0.15, 0.35)
ax.set_xticks([0, 0.20], ["seeds", "mean"])
ax.set_ylabel("Endpoint gap")
ax.set_title("Endpoint selectivity")
ax.text(0.03, 0.04, "4/10 > 0", transform=ax.transAxes, color=INK, fontsize=7)

save_figure(fig, HERE / "fig_03_learning")
plt.close(fig)
