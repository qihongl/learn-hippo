"""Generate duration robustness and mechanism-boundary figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import (
    AQUA,
    BLUE,
    GRID,
    MUTED,
    ORANGE,
    RED,
    VIOLET,
    YELLOW,
    apply_style,
    save_bundle,
)

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent


def load_results() -> tuple[dict, dict]:
    """Load confirmatory and post-confirmatory mechanism results."""

    confirmatory = json.loads((OUTPUT_ROOT / "reported_summary.json").read_text())
    mechanism = json.loads((OUTPUT_ROOT / "mechanism_results.json").read_text())
    return confirmatory, mechanism


def values(summary: dict, method: str) -> np.ndarray:
    """Return held-out reward ordered by preregistered seed."""

    per_seed = summary["runs"][method]["reward"]
    return np.asarray([per_seed[str(seed)] for seed in summary["seeds"]], dtype=float)


def draw_duration(ax, confirmatory: dict) -> None:
    original = values(confirmatory, "learned_policy")
    ood = values(confirmatory, "learned_policy_ood_duration")
    for left, right in zip(original, ood, strict=True):
        ax.plot([0, 1], [left, right], color="#b8b4ad", linewidth=0.7, alpha=0.7)
    ax.scatter(np.zeros_like(original), original, color=AQUA, s=17, zorder=3)
    ax.scatter(np.ones_like(ood), ood, color=BLUE, s=17, zorder=3)
    ax.set(
        ylabel="Held-out stochastic reward",
        ylim=(0.95, 1.001),
        xlim=(-0.35, 1.35),
        xticks=[0, 1],
        xticklabels=["Training\nduration", "+3 null\nstates"],
    )
    ax.set_title("(a) Duration generalization", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.text(
        0.5,
        0.04,
        "same frozen weights",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color=MUTED,
    )
    ax.text(
        0.04,
        0.91,
        "mean: 0.987",
        transform=ax.transAxes,
        fontsize=7,
        color=AQUA,
    )
    ax.text(
        0.60,
        0.55,
        "mean: 0.982",
        transform=ax.transAxes,
        fontsize=7,
        color=BLUE,
    )


def draw_input_ablation(ax, mechanism: dict) -> None:
    methods = [
        ("input_full", "Full\ninput", AQUA),
        ("input_mask_only", "Mask\nonly", YELLOW),
        ("input_state_only", "Values\nonly", ORANGE),
        ("input_cue_only", "Cue\nonly", VIOLET),
    ]
    for index, (method, _label, color) in enumerate(methods):
        stat = mechanism["summary"][method]["reward"]
        mean = stat["mean"]
        low, high = stat["ci95"]
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=color,
            markersize=6,
            capsize=4,
            linewidth=1.7,
        )
    ax.annotate(
        "mask is sufficient",
        xy=(1, mechanism["summary"]["input_mask_only"]["reward"]["mean"]),
        xytext=(1.32, 0.83),
        ha="center",
        arrowprops={"arrowstyle": "-", "color": YELLOW, "linewidth": 0.9},
        color="#9a6800",
        fontsize=7.5,
        fontweight="bold",
    )
    ax.set(
        ylabel="Reward after input ablation",
        ylim=(-0.04, 1.06),
        xticks=np.arange(len(methods)),
        xticklabels=[method[1] for method in methods],
    )
    ax.set_title("(b) Gate-input ablation", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.text(
        0.02,
        0.50,
        "mean and bootstrap 95% CI\n15 frozen models",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
    )


def draw_retrieval(ax, mechanism: dict) -> None:
    sweep = mechanism["retrieval_ablation"]["temperature_sweep"]
    temperature = np.asarray([row["temperature"] for row in sweep])
    endpoint = np.asarray([row["endpoint_only_reward"] for row in sweep])
    always = np.asarray([row["always_write_reward"] for row in sweep])
    midpoint = np.asarray([row["midpoint_plus_endpoint_reward"] for row in sweep])
    ax.plot(
        temperature,
        endpoint,
        color=BLUE,
        marker="o",
        label="endpoint only (forced)",
    )
    ax.plot(
        temperature,
        always,
        color=ORANGE,
        marker="o",
        label="always encode (forced)",
    )
    ax.plot(
        temperature,
        midpoint,
        color=VIOLET,
        marker="o",
        label="midpoint + end (forced)",
    )
    ax.axvline(0.1, color=MUTED, linestyle=":", linewidth=1.0)
    ax.text(0.105, 0.04, "reported\ntemperature", fontsize=6.8, color=MUTED)
    ax.text(
        0.97,
        0.88,
        "Latest-memory retrieval:\nalways encode (forced) = 1.00",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=RED,
        fontsize=7.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": RED},
    )
    ax.set_xscale("log")
    ax.set(
        xlabel="Softmax retrieval temperature",
        ylabel="Reward under forced policy",
        ylim=(-0.04, 1.06),
    )
    ax.set_title("(c) Retrieval boundary condition", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(
        frameon=False,
        loc="center right",
        bbox_to_anchor=(1.0, 0.49),
        fontsize=6.8,
    )


def main() -> None:
    confirmatory, mechanism = load_results()
    apply_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.25),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.0, 1.25]},
    )
    draw_duration(axes[0], confirmatory)
    draw_input_ablation(axes[1], mechanism)
    draw_retrieval(axes[2], mechanism)
    fig.suptitle(
        "Robustness and mechanism checks define the scope of the result",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    save_bundle(fig, HERE, "fig_04_robustness_mechanism")
    plt.close(fig)


if __name__ == "__main__":
    main()
