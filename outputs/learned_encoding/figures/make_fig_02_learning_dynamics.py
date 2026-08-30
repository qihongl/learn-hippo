"""Generate confirmatory learning dynamics and learned write timing."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import (
    AQUA,
    BLUE,
    GRID,
    MUTED,
    PALE_AQUA,
    YELLOW,
    apply_style,
    save_bundle,
)

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent


def load_results() -> tuple[np.ndarray, dict]:
    """Load all preregistered training curves and the mechanism analysis."""

    curves = []
    for path in sorted((OUTPUT_ROOT / "reported").glob("*.json")):
        result = json.loads(path.read_text())
        curves.append(result["training_curves"]["mean_reward"])
    if len(curves) != 15:
        raise ValueError(f"Expected 15 confirmatory curves, found {len(curves)}")
    mechanism = json.loads((OUTPUT_ROOT / "mechanism_results.json").read_text())
    return np.asarray(curves, dtype=float), mechanism


def draw_training(ax, curves: np.ndarray) -> None:
    updates = np.arange(1, curves.shape[1] + 1)
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1)
    ax.fill_between(updates, mean - sd, mean + sd, color=PALE_AQUA, linewidth=0)
    ax.plot(updates, mean, color=AQUA, linewidth=1.7, label="learned gate")
    ax.axhline(1.0, color=BLUE, linewidth=1.1, linestyle="--", label="endpoint oracle")
    ax.axhline(
        0.371,
        color=YELLOW,
        linewidth=1.1,
        linestyle=":",
        label="random one-write",
    )
    ax.set(
        xlabel="Training update",
        ylabel="Mean reward",
        xlim=(1, 300),
        ylim=(0.18, 1.04),
    )
    ax.set_title("(a) Optimization discovers the high-reward regime", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.95,
        "mean ± 1 SD across seeds\n(raw, unsmoothed minibatches)",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        color=MUTED,
    )


def draw_write_timing(ax, mechanism: dict) -> None:
    summaries = mechanism["write_probability_by_progress"]
    progress = np.arange(1, 5)
    per_seed = np.asarray(
        [list(summaries[str(step)]["per_seed"].values()) for step in progress]
    ).T
    means = np.asarray([summaries[str(step)]["mean"] for step in progress])
    intervals = np.asarray([summaries[str(step)]["ci95"] for step in progress])

    for seed_values in per_seed:
        ax.plot(
            progress,
            seed_values,
            color="#b8b4ad",
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
        ax.scatter(progress, seed_values, color="#b8b4ad", s=8, alpha=0.55, zorder=2)
    errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
    ax.errorbar(
        progress,
        means,
        yerr=errors,
        color=AQUA,
        marker="o",
        markersize=5,
        linewidth=1.8,
        capsize=3,
        zorder=3,
        label="mean and bootstrap 95% CI",
    )
    ax.axvspan(3.72, 4.28, color=PALE_AQUA, zorder=0)
    ax.text(4, 0.82, "complete\nevent", ha="center", color=AQUA, fontsize=8)
    ax.annotate(
        "99.4%",
        xy=(4, means[-1]),
        xytext=(3.82, 1.01),
        arrowprops={"arrowstyle": "-", "color": AQUA, "linewidth": 0.8},
        ha="right",
        color=AQUA,
        fontsize=8,
        fontweight="bold",
    )
    ax.annotate(
        "2.8%",
        xy=(3, means[-2]),
        xytext=(2.55, 0.17),
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
        color=MUTED,
        fontsize=7.5,
    )
    ax.set(
        xlabel="Observed features / semantic progress",
        ylabel="Probability of writing",
        xlim=(0.72, 4.28),
        ylim=(-0.03, 1.05),
        xticks=progress,
    )
    ax.set_title("(b) The learned gate writes at completion", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.text(
        0.02,
        0.47,
        "gray = individual model seeds",
        transform=ax.transAxes,
        fontsize=7,
        color=MUTED,
    )


def main() -> None:
    curves, mechanism = load_results()
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    draw_training(axes[0], curves)
    draw_write_timing(axes[1], mechanism)
    fig.suptitle(
        "Learning dynamics reveal a boundary-selective write policy",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    save_bundle(fig, HERE, "fig_02_learning_dynamics")
    plt.close(fig)


if __name__ == "__main__":
    main()
