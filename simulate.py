"""Fat-tree failure simulator: sweeps the parameters in SWEEP, runs N trials per point, writes results.csv."""

import argparse
import functools
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fattree.failures import inject_failures
from fattree.metrics import live_ports, surviving_paths
from fattree.topology import build_fat_tree

# Swept parameters, each given as a list of values. To add one: add its name here and its argument below.
SWEEP = ["k", "fail_rate"]


def run_trial(G, cfg, rng) -> dict:
    """One failure sample and its metrics. To add a metric: merge its dict into the result."""
    H = inject_failures(G, cfg.fail_rate, cfg.fail_scope, rng)
    return {**surviving_paths(H, cfg.fail_scope, cfg.pairs, rng), **live_ports(H)}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="YAML file; keys are the flag names with underscores. CLI flags override it.")
    p.add_argument("--k", type=int, nargs="+", help="ports per switch (even, 4..254)")
    p.add_argument("--fail-rate", type=float, nargs="+", default=[0.0],
                   help="link failure probability p (each link fails independently)")
    p.add_argument("--fail-scope", choices=["switch-links", "all-links"], default="switch-links")
    p.add_argument("--pairs", type=lambda s: s if s == "all" else int(s), default="all",
                   help="'all' or the number of uniformly sampled pairs")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="trial t uses seed + t")
    p.add_argument("--out", default="results")

    # First pass only reads --config, whose values become defaults so that CLI flags win.
    config = p.parse_known_args(argv)[0].config
    if config:
        p.set_defaults(**yaml.safe_load(Path(config).read_text()))
    args = p.parse_args(argv)
    if not args.k:
        p.error("--k is required, on the command line or in the config file")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    graph = functools.cache(build_fat_tree)
    rows = []
    for values in itertools.product(*(getattr(args, name) for name in SWEEP)):
        point = dict(zip(SWEEP, values))
        print(point)
        cfg = argparse.Namespace(**{**vars(args), **point})
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + t)
            rows.append({**point, "trial": t, **run_trial(graph(cfg.k), cfg, rng)})
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    main()
