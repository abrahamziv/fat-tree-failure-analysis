"""Draw a fat-tree (optionally with failures): uv run draw_topology.py --k 4 [--fail-rate 0.15 --seed 3]"""
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fattree.topology import build_fat_tree
from fattree.failures import inject_failures

COLORS = {"core": "#c0504d", "agg": "#4f81bd", "edge": "#9bbb59", "host": "#7f7f7f"}
Y = {"core": 3, "agg": 2, "edge": 1, "host": 0}

def layout(G):
    h = G.graph["k"] // 2
    stride = h * h + 1                          # k^2/4 hosts per pod, plus one empty slot between pods
    width = G.graph["k"] * stride - 1
    pos = {}
    for n, d in G.nodes(data=True):
        f = list(map(int, n.split("_")[1:]))
        if d["layer"] == "host":                # h_{pod}_{edge}_{id}
            x = f[0] * stride + f[1] * h + (f[2] - 2)
        elif d["layer"] in ("edge", "agg"):     # e_/a_{pod}_{i}
            x = f[0] * stride + f[1] * h + (h - 1) / 2
        else:                                   # c_{j}_{i}, 1-based
            idx = (f[0] - 1) * h + (f[1] - 1)
            x = (idx + 0.5) * width / (h * h) - 0.5
        pos[n] = (x, Y[d["layer"]])
    return pos

def label_placement(G, pos, n):
    """Offset (points) and alignment that put an IP label where the node has no links."""
    layer = G.nodes[n]["layer"]
    if layer == "core":                         # core links only go down
        return (0, 10), "center", "bottom"
    if layer == "host":                         # host links only go up
        return (0, -6), "center", "top"
    # Pod switches: in-pod diagonal links run toward the pod center, so use the outer side;
    # edge labels above the node, aggregation labels below (between the two layers).
    left = G.nodes[n]["index"] < (G.graph["k"] // 2 - 1) / 2   # left half of the pod
    dy, va = (8, "bottom") if layer == "edge" else (-8, "top")
    return (-6 if left else 6, dy), "right" if left else "left", va

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: Resources/topology_k{k}.png")
    a = ap.parse_args()

    G = build_fat_tree(a.k)
    H = inject_failures(G, a.fail_rate, "switch-links", np.random.default_rng(a.seed))
    pos = layout(G)
    alive = [e for e in G.edges if H.has_edge(*e)]
    dead = [e for e in G.edges if not H.has_edge(*e)]

    slots = a.k * (a.k ** 2 // 4 + 1)            # host positions incl. gaps between pods
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * slots), 4.5))
    nx.draw_networkx_edges(G, pos, edgelist=alive, edge_color="#999999", width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=dead, edge_color="red", style="dashed", width=1.5, ax=ax)
    for layer, c in COLORS.items():
        nodes = [n for n, d in G.nodes(data=True) if d["layer"] == layer]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=c,
                               node_size=60 if layer == "host" else 260, ax=ax, label=layer)
    if a.k <= 4:  # IP labels only when readable
        for n, ip in G.nodes(data="ip"):
            offset, ha, va = label_placement(G, pos, n)
            ax.annotate(ip, pos[n], xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=6)
    ax.set_title(f"k={a.k} fat-tree"
                 + (f" (p={a.fail_rate}, seed={a.seed}; failures in red)" if dead else ""))
    ax.legend(loc="upper left", fontsize=8, frameon=False, markerscale=0.5, labelspacing=1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out or f"Resources/topology_k{a.k}.png", dpi=200)

if __name__ == "__main__":
    main()
