"""Generate confirmatory baselines, displacement, and gap-closure figure."""

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


def load_summary() -> dict:
    """Load the preregistered confirmatory aggregate."""

    return json.loads((OUTPUT_ROOT / "reported_summary.json").read_text())


def values(summary: dict, method: str, metric: str = "reward") -> np.ndarray:
    """Return one value per confirmatory model seed."""

    per_seed = summary["runs"][method][metric]
    return np.asarray([per_seed[str(seed)] for seed in summary["seeds"]], dtype=float)


def draw_baselines(ax, summary: dict) -> None:
    methods = [
        ("intervention_endpoint_only", "Endpoint\noracle", BLUE),
        ("learned_policy", "Learned\ngate", AQUA),
        ("intervention_matched_random_one_write", "Random\none-write", YELLOW),
        ("intervention_always_write", "Always\nwrite", ORANGE),
        ("intervention_midpoint_plus_endpoint", "Midpoint\n+ endpoint", VIOLET),
    ]
    rng = np.random.default_rng(20260830)
    for index, (method, _label, color) in enumerate(methods):
        y = values(summary, method)
        jitter = rng.uniform(-0.11, 0.11, size=y.size)
        ax.scatter(
            np.full_like(y, index, dtype=float) + jitter,
            y,
            s=12,
            facecolor="white",
            edgecolor=color,
            linewidth=0.65,
            alpha=0.8,
            zorder=2,
        )
        stat = summary["summary"][method]["reward"]
        mean = stat["mean"]
        low, high = stat["ci95"]
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=color,
            markersize=5.5,
            capsize=3,
            linewidth=1.5,
            zorder=3,
        )
    ax.set(
        ylabel="Held-out reward",
        ylim=(-0.04, 1.07),
        xticks=np.arange(len(methods)),
        xticklabels=[method[1] for method in methods],
    )
    ax.set_title("(a) Learned timing approaches the oracle", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.text(
        0.02,
        0.63,
        "open circles = model seeds\nsolid = mean and bootstrap 95% CI",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
    )


def draw_displacement(ax, summary: dict) -> None:
    original = values(summary, "learned_policy_deterministic")
    displaced = values(summary, "intervention_displaced_learned")
    for left, right in zip(original, displaced, strict=True):
        ax.plot([0, 1], [left, right], color="#b8b4ad", linewidth=0.75, alpha=0.65)
    ax.scatter(np.zeros_like(original), original, color=AQUA, s=22, zorder=3)
    ax.scatter(np.ones_like(displaced), displaced, color=RED, s=22, zorder=3)
    ax.annotate(
        "loss = 1.00\nfor every seed",
        xy=(0.5, 0.5),
        ha="center",
        va="center",
        color=RED,
        fontsize=8,
        fontweight="bold",
    )
    ax.set(
        ylabel="Deterministic reward",
        ylim=(-0.06, 1.06),
        xlim=(-0.35, 1.35),
        xticks=[0, 1],
        xticklabels=["Learned\ntiming", "Same write\ndisplaced"],
    )
    ax.set_title("(b) Timing is causal", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.text(
        0.5,
        0.03,
        "15 identical pairs overlap",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.8,
        color=MUTED,
    )


def draw_gap_closure(ax, summary: dict) -> None:
    learned = values(summary, "learned_policy")
    endpoint = values(summary, "intervention_endpoint_only")
    random = values(summary, "intervention_matched_random_one_write")
    closure = (learned - random) / (endpoint - random)
    rng = np.random.default_rng(20260830)
    x = rng.uniform(-0.08, 0.08, size=closure.size)
    ax.scatter(x, closure, color=AQUA, s=18, alpha=0.8, zorder=2)
    audit = summary["success_audit"]
    mean = audit["mean_gap_closure"]
    low, high = audit["gap_closure_ci95"]
    ax.errorbar(
        0,
        mean,
        yerr=[[mean - low], [high - mean]],
        fmt="o",
        color="#006c49",
        markersize=6,
        capsize=4,
        linewidth=1.8,
        zorder=3,
    )
    ax.axhline(0.8, color=RED, linewidth=1.1, linestyle="--")
    ax.text(0.10, 0.802, "preregistered threshold", color=RED, va="bottom", fontsize=7)
    ax.annotate(
        "mean 0.980",
        xy=(0, mean),
        xytext=(0.1, 0.955),
        color="#006c49",
        fontsize=8,
        fontweight="bold",
    )
    ax.set(
        ylabel="Oracle gap closed",
        ylim=(0.76, 1.015),
        xlim=(-0.35, 0.55),
        xticks=[0],
        xticklabels=["15 model seeds"],
    )
    ax.set_title("(c) Confirmatory criterion passes", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def main() -> None:
    summary = load_summary()
    apply_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.45, 0.9, 0.95]},
    )
    draw_baselines(axes[0], summary)
    draw_displacement(axes[1], summary)
    draw_gap_closure(axes[2], summary)
    fig.suptitle(
        "Confirmatory tests support learned boundary-selective encoding",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    save_bundle(fig, HERE, "fig_03_confirmatory_evidence")
    plt.close(fig)


if __name__ == "__main__":
    main()
