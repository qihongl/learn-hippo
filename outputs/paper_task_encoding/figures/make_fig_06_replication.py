"""Plot every seed in the locked batch-32 optimizer replication."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import BLUE, INK, MUTED, RED, save_figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "outputs/paper_task_encoding"

summary = json.loads(
    (OUTPUT / "sampled_hazard_stability_replication_b32_summary.json").read_text()
)
audit = json.loads((OUTPUT / "optimizer_replication_audit.json").read_text())
passed_by_seed = {item["seed"]: item["passed"] for item in audit["seed_passes"]}

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)

ax = axes[0]
ax.text(-0.13, 1.04, "a", fontweight="bold", fontsize=10, transform=ax.transAxes)
for run in summary["runs"]:
    trajectory = run["checkpoint_trajectory"]
    ax.plot(
        [point["epoch"] for point in trajectory],
        [point["endpoint_probability"] for point in trajectory],
        color=BLUE,
        alpha=0.20,
        linewidth=0.8,
    )
curve = summary["learning_curves"]
ax.plot(
    [point["epoch"] for point in curve],
    [point["endpoint_probability"]["mean"] for point in curve],
    color=INK,
    linewidth=1.8,
    label="ten-seed mean",
)
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_xlim(0, 405)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("Policy-training epoch (256 sequences per epoch)")
ax.set_ylabel("Held-out endpoint probability")
ax.set_title("Locked replication trajectories")
ax.legend(frameon=False, loc="lower right")

ax = axes[1]
ax.text(-0.13, 1.04, "b", fontweight="bold", fontsize=10, transform=ax.transAxes)
seeds = np.asarray([run["seed"] for run in summary["runs"]])
values = np.asarray([run["endpoint_probability"] for run in summary["runs"]])
passed = np.asarray([passed_by_seed[int(seed)] for seed in seeds])
ax.scatter(
    seeds[passed],
    values[passed],
    color=BLUE,
    edgecolor="white",
    linewidth=0.4,
    s=34,
    label="individual pass",
)
ax.scatter(
    seeds[~passed],
    values[~passed],
    color=RED,
    edgecolor="white",
    linewidth=0.4,
    s=34,
    label="individual fail",
)
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.8)
ax.axhline(values.mean(), color=INK, linewidth=1.2)
ax.text(
    seeds[-1] + 0.35,
    values.mean(),
    f"mean {values.mean():.3f}",
    va="center",
    fontsize=7,
)
ax.set_xticks(seeds, [str(seed) for seed in seeds], rotation=45)
ax.set_xlim(seeds[0] - 0.6, seeds[-1] + 1.7)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("Fresh model seed")
ax.set_ylabel("Final endpoint probability")
ax.set_title("Six of ten seeds pass the individual rule")
ax.legend(frameon=False, loc="center right")

save_figure(fig, HERE / "fig_06_replication")
plt.close(fig)
