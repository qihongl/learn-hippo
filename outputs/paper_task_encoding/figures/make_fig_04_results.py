"""Generate forced-schedule and causal-ablation result panels."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import AQUA, BLUE, INK, LIGHT, MUTED, ORANGE, RED, VIOLET, save_figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
summary = json.loads(
    (REPOSITORY / "outputs/paper_task_encoding/reported_summary.json").read_text()
)
seed_files = sorted(
    (REPOSITORY / "outputs/paper_task_encoding/reported").glob("*.json")
)
seeds = [json.loads(path.read_text()) for path in seed_files]

fig, axes = plt.subplots(
    1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.05, 1.0]}
)

ax = axes[0]
ax.text(-0.18, 1.04, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
schedule_order = [
    "endpoint_only",
    "midpoint_only",
    "midpoint_plus_endpoint",
    "dense",
    "never",
    "matched_random_one",
]
labels = ["endpoint", "midpoint", "mid + end", "dense", "never", "random one"]
values = [
    summary["conditions"]["DM"]["forced_reward"][name]["mean"]
    for name in schedule_order
]
colors = [ORANGE, BLUE, VIOLET, MUTED, LIGHT, AQUA]
bars = ax.barh(
    np.arange(len(labels)), values, color=colors, edgecolor="white", height=0.68
)
ax.set_yticks(np.arange(len(labels)), labels)
ax.invert_yaxis()
ax.set_xlim(0.45, 0.69)
ax.set_xlabel("Expected prediction reward")
ax.set_title("Forced schedules: endpoint is best in DM")
for bar, value in zip(bars, values, strict=True):
    ax.text(
        value + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=6.5,
        color=INK,
    )

ax = axes[1]
ax.text(-0.18, 1.04, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
methods = [
    "learned",
    "retrieval off",
    "target removed",
    "lure removed",
    "content gate off",
]
per_seed = []
for run in seeds:
    dm = run["evaluation"]["DM"]
    per_seed.append(
        [
            dm["learned"]["reward"]["mean"],
            dm["retrieval_off_same_actions"]["reward"]["mean"],
            dm["ablations"]["target_memory_removed"]["reward"]["mean"],
            dm["ablations"]["lure_memory_removed"]["reward"]["mean"],
            dm["ablations"]["content_gate_off"]["reward"]["mean"],
        ]
    )
per_seed = np.asarray(per_seed)
for row in per_seed:
    ax.plot(
        np.arange(len(methods)),
        row,
        color="#b7bec7",
        linewidth=0.7,
        alpha=0.65,
        zorder=1,
    )
ax.scatter(
    np.tile(np.arange(len(methods)), len(seeds)),
    per_seed.T.flatten(order="F"),
    color=BLUE,
    s=10,
    alpha=0.6,
    zorder=2,
)
means = per_seed.mean(axis=0)
ax.plot(
    np.arange(len(methods)),
    means,
    color=RED,
    marker="o",
    linewidth=1.4,
    markersize=4,
    zorder=3,
)
ax.set_xticks(
    np.arange(len(methods)),
    [
        "learned",
        "retrieval\noff",
        "target\nremoved",
        "lure\nremoved",
        "content gate\noff",
    ],
)
ax.set_ylabel("Expected prediction reward")
ax.set_title("Paired DM causal ablations")
ax.set_ylim(0.35, 0.52)
ax.text(
    0.03,
    0.06,
    "gray = model seeds\nred = across-seed mean",
    transform=ax.transAxes,
    fontsize=6.5,
    color=INK,
)

save_figure(fig, HERE / "fig_04_results")
plt.close(fig)
