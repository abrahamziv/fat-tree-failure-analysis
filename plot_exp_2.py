"""Experiment 2 plots: link vs. switch failures on a common axis, the expected fraction of dead switch-to-switch links.

Usage: uv run plot_exp_2.py results/Exp_1/results.csv results/Exp_2/results.csv [--k 48]
Writes next to the switch-failure CSV (file names get _k{k} for k other than 48):
    exp2_mean.png    average pair, mean ± 95% CI over trials, analytic curves dashed
    exp2_worst.png   worst pair, mean ± 95% CI over trials
    exp2_spread.png  trial-to-trial standard deviation of the average pair's fraction
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fattree.failures import link_equivalent, switch_rate_for
from plot_exp_1 import ANALYTIC, CLASSES, MUTED, normalize, style, summarize

DEFAULT_K = 48
X_LABEL = "expected fraction of dead switch-to-switch links, p_eq"
# Failure type -> (color, expected surviving fraction per class as a function of p_eq; README §4.4.1–4.4.2).
SERIES = {
    "link failures": ("#1c5cab", {cls: f for cls, (_, f) in ANALYTIC.items()}),
    "switch failures": ("#d9730d", {"intra": lambda p: 1 - switch_rate_for(p),
                                    "inter": lambda p: (1 - switch_rate_for(p)) ** 3}),
}
# One figure per statistic: column suffix -> (file name, y-axis label).
FIGURES = {
    "mean": ("exp2_mean.png", "mean surviving shortest paths / maximum"),
    "min": ("exp2_worst.png", "worst pair's surviving shortest paths / maximum"),
    "spread": ("exp2_spread.png", "trial-to-trial std of mean surviving fraction"),
}


def plot(runs, stat, k, out):
    name, ylabel = FIGURES[stat]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    p_fine = np.linspace(0, max(df.p_eq.max() for df in runs.values()), 100)
    for ax, (cls, title) in zip(axes, CLASSES):
        for label, df in runs.items():
            color, expected = SERIES[label]
            if stat == "spread":
                s = df.groupby("p_eq")[f"{cls}_mean"].std()
                ax.plot(s.index, s, "o-", color=color, lw=2, ms=4, label=label)
                continue
            # summarize groups by fail_rate; here that column holds p_eq.
            s = summarize(df.assign(fail_rate=df.p_eq), f"{cls}_{stat}").loc[k]
            ax.plot(s.index, s["mean"], "o-", color=color, lw=2, ms=4, label=label)
            ax.fill_between(s.index, s["mean"] - s["ci"], s["mean"] + s["ci"], color=color, alpha=0.2, lw=0)
            if stat == "mean":
                ax.plot(p_fine, expected[cls](p_fine), "--", color=color, lw=1, label=f"{label} (analytic)")
        style(ax, title)
        ax.set_xlabel(X_LABEL, color=MUTED)
    axes[0].set_ylabel(ylabel, color=MUTED)
    axes[0].set_ylim(0, None if stat == "spread" else 1.02)
    fig.suptitle(f"k = {k}", x=0.01, ha="left", fontsize=12)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside right center", frameon=False)
    suffix = "" if k == DEFAULT_K else f"_k{k}"
    fig.savefig(out / name.replace(".png", f"{suffix}.png"), dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("link_csv", help="Experiment 1 results (link failures only)")
    p.add_argument("switch_csv", help="Experiment 2 results (switch failures only)")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    args = p.parse_args(argv)

    links, switches = pd.read_csv(args.link_csv), pd.read_csv(args.switch_csv)
    assert (links.fail_switches == 0).all(), "link CSV must have fail_switches = 0 everywhere"
    assert (switches.fail_rate == 0).all(), "switch CSV must have fail_rate = 0 everywhere"
    for df, csv in [(links, args.link_csv), (switches, args.switch_csv)]:
        assert args.k in set(df.k), f"k = {args.k} not in {csv}"
    runs = {
        "link failures": links.assign(p_eq=links.fail_rate),
        "switch failures": switches.assign(p_eq=link_equivalent(switches.fail_switches)),
    }
    runs = {label: normalize(df[df.k == args.k]) for label, df in runs.items()}
    for stat in FIGURES:
        plot(runs, stat, args.k, Path(args.switch_csv).parent)


if __name__ == "__main__":
    main()
