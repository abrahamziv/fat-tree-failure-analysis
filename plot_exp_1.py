"""Experiment 1 plots vs. link failure probability, mean over trials with a 95% CI band.

Usage: uv run plot_exp_1.py results/Exp_1/results.csv
Writes next to the CSV:
    exp1_mean.png   average pair's surviving shortest paths, analytic curves dashed
    exp1_worst.png  worst pair's surviving shortest paths, average-pair analytic curves dashed gray for reference
    exp1_ports.png  worst switch's live ports per layer, analytic E[min] dashed, mean switch dashed gray
At most 4 k values per CSV (one color each).
"""

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fattree.metrics import expected_worst_ports, port_layers

K_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp: small k light, large k dark
INK, MUTED, GRID, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#8c8b87"
CLASSES = [("intra", "Same pod"), ("inter", "Different pods")]
# Expected surviving fraction of shortest paths under independent link failures (README §4.4).
ANALYTIC = {"intra": ("(1-p)²", lambda p: (1 - p) ** 2), "inter": ("(1-p)⁴", lambda p: (1 - p) ** 4)}
# One figure per statistic over pairs: column suffix -> (file name, y-axis label).
FIGURES = {
    "mean": ("exp1_mean.png", "fraction of shortest paths surviving (mean pair)"),
    "min": ("exp1_worst.png", "fraction of shortest paths surviving (worst pair)"),
}
LAYERS = [("edge", "Edge switches: k/2 uplinks"), ("agg", "Aggregation switches: k ports"),
          ("core", "Core switches: k ports")]


def normalize(df):
    """Divide path counts by their maximum: k/2 intra-pod, (k/2)^2 inter-pod."""
    half = df.k // 2
    return df.assign(intra_mean=df.intra_mean / half, intra_min=df.intra_min / half,
                     inter_mean=df.inter_mean / half**2, inter_min=df.inter_min / half**2)


def summarize(df, col, x="fail_rate"):
    """Mean and 95% CI half-width over trials, per (k, x). A single trial has no CI (drawn as 0)."""
    g = df.groupby(["k", x])[col]
    return pd.DataFrame({"mean": g.mean(), "ci": (stats.t.ppf(0.975, g.count() - 1) * g.sem()).fillna(0)})


def style(ax, title):
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel("link failure probability p", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot(df, stat, out):
    name, ylabel = FIGURES[stat]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    p_fine = np.linspace(0, df.fail_rate.max(), 100)
    k_colors = dict(zip(sorted(df.k.unique()), K_COLORS))  # legend order follows plotting order: k ascending
    for ax, (cls, title) in zip(axes, CLASSES):
        formula, expected = ANALYTIC[cls]
        for k, color in k_colors.items():
            s = summarize(df[df.k == k], f"{cls}_{stat}").loc[k]
            ax.plot(s.index, s["mean"], "o-", color=color, lw=2, ms=4, label=f"k = {k}")
            ax.fill_between(s.index, s["mean"] - s["ci"], s["mean"] + s["ci"], color=color, alpha=0.2, lw=0)
        if stat == "mean":  # on top of the k curves it overlaps, with a white ring to stay visible
            ax.plot(p_fine, expected(p_fine), "--", color=INK, lw=1.2, zorder=3, label="analytic",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            title += f": {formula} expected"
        else:  # reference only: below the k curves (lines are zorder 2), above the CI bands (zorder 1)
            ax.plot(p_fine, expected(p_fine), "--", color=REFERENCE, lw=1, zorder=1.5,
                    label="average pair (analytic)")
        style(ax, title)
    axes[0].set_ylabel(ylabel, color=MUTED)
    axes[0].set_ylim(0, 1.02)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside right center", frameon=False)
    fig.savefig(out / name, dpi=150)


def plot_ports(df, out):
    """Worst switch's live switch-to-switch ports / ports per switch, one panel per layer."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, layout="constrained")
    p_fine = np.linspace(0, df.fail_rate.max(), 100)
    k_colors = dict(zip(sorted(df.k.unique()), K_COLORS))
    for ax, (layer, title) in zip(axes, LAYERS):
        for k, color in k_colors.items():
            n = port_layers(k)[layer][1]
            d = df[df.k == k]
            s = summarize(d.assign(frac=d[f"{layer}_ports_min"] / n), "frac").loc[k]
            ax.plot(s.index, s["mean"], "o-", color=color, lw=2, ms=4, label=f"k = {k}")
            ax.fill_between(s.index, s["mean"] - s["ci"], s["mean"] + s["ci"], color=color, alpha=0.2, lw=0)
            ax.plot(p_fine, expected_worst_ports(k, p_fine)[layer] / n, "--", color=color, lw=1)
        ax.plot([], [], "--", color=INK, lw=1, label="worst switch (analytic)")  # legend proxy for the k lines
        ax.plot(p_fine, 1 - p_fine, "--", color=REFERENCE, lw=1, zorder=1.5, label="average switch: 1-p")
        style(ax, title)
    axes[0].set_ylabel("worst switch's live-port fraction", color=MUTED)
    axes[0].set_ylim(0, 1.02)
    fig.legend(*axes[-1].get_legend_handles_labels(), loc="outside right center", frameon=False)
    fig.savefig(out / "exp1_ports.png", dpi=150)


def main(csv):
    df = pd.read_csv(csv)
    assert df.k.nunique() <= len(K_COLORS), f"at most {len(K_COLORS)} k values per plot"
    df = normalize(df)
    for stat in FIGURES:
        plot(df, stat, Path(csv).parent)
    plot_ports(df, Path(csv).parent)


if __name__ == "__main__":
    main(sys.argv[1])
