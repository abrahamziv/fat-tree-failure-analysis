"""Experiment 1 plots: surviving shortest paths vs. link failure probability, mean over trials with a 95% CI band.

Usage: uv run plot_exp_1.py results/Exp_1/results.csv
Writes next to the CSV:
    exp1_mean.png   average pair, analytic curves dashed
    exp1_worst.png  worst pair, average-pair analytic curves dashed gray for reference
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

K_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp: small k light, large k dark
INK, MUTED, GRID, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#8c8b87"
CLASSES = [("intra", "Same pod"), ("inter", "Different pods")]
# Expected surviving fraction of shortest paths under independent link failures (README §4.4.1).
ANALYTIC = {"intra": ("(1-p)²", lambda p: (1 - p) ** 2), "inter": ("(1-p)⁴", lambda p: (1 - p) ** 4)}
# One figure per statistic over pairs: column suffix -> (file name, y-axis label).
FIGURES = {
    "mean": ("exp1_mean.png", "mean surviving shortest paths / maximum"),
    "min": ("exp1_worst.png", "worst pair's surviving shortest paths / maximum"),
}


def normalize(df):
    """Divide path counts by their maximum: k/2 intra-pod, (k/2)^2 inter-pod."""
    half = df.k // 2
    return df.assign(intra_mean=df.intra_mean / half, intra_min=df.intra_min / half,
                     inter_mean=df.inter_mean / half**2, inter_min=df.inter_min / half**2)


def summarize(df, col):
    """Mean and 95% CI half-width over trials, per (k, p)."""
    g = df.groupby(["k", "fail_rate"])[col]
    return pd.DataFrame({"mean": g.mean(), "ci": stats.t.ppf(0.975, g.count() - 1) * g.sem()})


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
        if stat == "mean":
            ax.plot(p_fine, expected(p_fine), "--", color=INK, lw=1, label="analytic")
            title += f": {formula} expected"
        else:  # reference only: below the k curves (lines are zorder 2), above the CI bands (zorder 1)
            ax.plot(p_fine, expected(p_fine), "--", color=REFERENCE, lw=1, zorder=1.5,
                    label="average pair (analytic)")
        style(ax, title)
    axes[0].set_ylabel(ylabel, color=MUTED)
    axes[0].set_ylim(0, 1.02)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside right center", frameon=False)
    fig.savefig(out / name, dpi=150)


def main(csv):
    df = pd.read_csv(csv)
    assert (df.fail_switches == 0).all(), "Experiment 1 is link failures only (fail_switches = 0)"
    assert df.k.nunique() <= len(K_COLORS), f"at most {len(K_COLORS)} k values per plot"
    df = normalize(df)
    for stat in FIGURES:
        plot(df, stat, Path(csv).parent)


if __name__ == "__main__":
    main(sys.argv[1])
