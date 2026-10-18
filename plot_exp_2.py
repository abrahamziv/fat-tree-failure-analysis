"""Experiment 2 plots: delivery ratio of new flows, static routing vs. the §3.8 fault-tolerance scheme vs. ideal.

Usage: uv run plot_exp_2.py results/Exp_2/results.csv results/Exp_2_switch/results.csv
Writes, next to the CSV each figure plots:
    exp2_links.png     link failures: delivery ratio vs. p, mean ± 95% CI over trials, analytic curves dashed
    exp2_switches.png  switch failures: the same vs. q
    exp2_worst.png     link failures: the worst flow's §3.8 delivery probability vs. p, inter-pod
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from fattree.metrics import expected_delivery
from plot_exp_1 import CLASSES, INK, MUTED, style, summarize

# Scheme -> (legend label, color): categorical slots 1-3, validated as a set.
SCHEMES = {
    "static": ("static (§3.5)", "#2a78d6"),
    "ft": ("fault tolerance (§3.8)", "#eb6834"),
    "ideal": ("ideal shortest-path", "#1baf7a"),
}
K_STYLES = [((0, (1, 1.5)), "o"), ((0, (5, 1.5, 1, 1.5)), "s"), ((0, (8, 2)), "^"), ("-", "D")]
ANALYTIC_STYLE = dict(color=INK, lw=0.9, ls=(0, (2, 2)), zorder=1.5)  # thin, below the measured lines
X = {"fail_rate": "link failure probability p", "fail_switches": "switch failure probability q"}


def k_styles(df):
    return dict(zip(sorted(df.k.unique()), K_STYLES))


def plot_line(ax, df, col, x, k, scheme, ls, marker):
    color = SCHEMES[scheme][1]
    s = summarize(df[df.k == k], col, x).loc[k]
    ax.plot(s.index, s["mean"], color=color, ls=ls, marker=marker, lw=1.8, ms=4)
    ax.fill_between(s.index, s["mean"] - s["ci"], s["mean"] + s["ci"], color=color, alpha=0.15, lw=0)


def legend(fig, df, schemes, analytic):
    handles = [Line2D([], [], color=SCHEMES[s][1], lw=2, label=SCHEMES[s][0]) for s in schemes]
    handles += [Line2D([], [], color=MUTED, ls=ls, marker=m, ms=4, lw=1.5, label=f"k = {k}")
                for k, (ls, m) in k_styles(df).items()]
    if analytic:
        handles.append(Line2D([], [], label="analytic (§4.4.5)", **{**ANALYTIC_STYLE, "zorder": None}))
    fig.legend(handles=handles, loc="outside right center", frameon=False)


def plot_delivery(df, x, out):
    """Figures A and B: one panel per pair class, color = scheme, line style = k."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    fine = np.linspace(0, df[x].max(), 100)
    for ax, (cls, title) in zip(axes, CLASSES):
        for k, (ls, marker) in k_styles(df).items():
            expected = expected_delivery(k, **{"p" if x == "fail_rate" else "q": fine})
            ax.plot(fine, expected[f"{cls}_ideal"], **ANALYTIC_STYLE)
            for scheme in SCHEMES:
                plot_line(ax, df, f"{cls}_{scheme}_mean", x, k, scheme, ls, marker)
        ax.plot(fine, expected[f"{cls}_static"], **ANALYTIC_STYLE)  # independent of k
        if cls == "intra":  # identical per flow (README §4.4.5), so the ideal line hides the §3.8 one
            ax.annotate("§3.8 = ideal (identical per flow)", (0.22, 0.95), xycoords="axes fraction",
                        ha="left", va="top", color=MUTED, fontsize=9)
        style(ax, title)
        ax.set_xlabel(X[x], color=MUTED)
    axes[0].set_ylabel("delivery ratio of new flows", color=MUTED)
    axes[0].set_ylim(0, 1.02)
    legend(fig, df, SCHEMES, analytic=True)
    fig.savefig(out, dpi=150)


def plot_worst(df, out):
    """Figure C: the worst inter-pod flow's §3.8 delivery probability, per k."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    for k, (ls, marker) in k_styles(df).items():
        plot_line(ax, df, "inter_ft_min", "fail_rate", k, "ft", ls, marker)
    style(ax, "Different pods, worst flow")
    ax.set_ylabel("worst flow's delivery probability (§3.8)", color=MUTED)
    ax.set_ylim(0, 1.02)
    legend(fig, df, ["ft"], analytic=False)
    fig.savefig(out, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("link_csv", help="link failures only (configs/exp_2.yaml)")
    p.add_argument("switch_csv", help="switch failures only (configs/exp_2_switch.yaml)")
    args = p.parse_args(argv)

    links, switches = pd.read_csv(args.link_csv), pd.read_csv(args.switch_csv)
    assert (links.fail_switches == 0).all(), "link CSV must have fail_switches = 0 everywhere"
    assert (switches.fail_rate == 0).all(), "switch CSV must have fail_rate = 0 everywhere"
    for df, csv in [(links, args.link_csv), (switches, args.switch_csv)]:
        assert "inter_ft_mean" in df, f"{csv} has no delivery columns (run with --fail-scope switch-links)"
        assert df.k.nunique() <= len(K_STYLES), f"at most {len(K_STYLES)} k values per plot"
    plot_delivery(links, "fail_rate", Path(args.link_csv).parent / "exp2_links.png")
    plot_delivery(switches, "fail_switches", Path(args.switch_csv).parent / "exp2_switches.png")
    plot_worst(links, Path(args.link_csv).parent / "exp2_worst.png")


if __name__ == "__main__":
    main()
