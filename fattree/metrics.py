"""Metrics under failures, computed from the fat-tree structure (no path enumeration).

Between two edge switches in the same pod, each aggregation switch gives one shortest path
(2 links). Between pods, each core switch gives one (4 links); core c_{a+1}_{c} is reached
only through aggregation switch a of each pod.

surviving_paths: per pair class (intra-pod, inter-pod), the mean and the minimum number of
surviving shortest paths over the pairs.
live_ports: per switch layer, the mean and the minimum number of live switch-to-switch ports
over the switches; expected_worst_ports: the expected minimum (README §4.5).
"""

import networkx as nx
import numpy as np
from scipy.stats import binom


def _pairs(n: int, g: int, n_pairs, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Unordered endpoint pairs (i, j) that are not under the same edge switch.

    Endpoints are ordered by (pod, edge, id), so each edge switch owns a block of g of them.
    """
    if n_pairs == "all":
        i, j = np.triu_indices(n, 1)
        keep = i // g != j // g
        return i[keep], j[keep]
    i = rng.integers(n, size=n_pairs)
    j = (i // g * g + g + rng.integers(n - g, size=n_pairs)) % n  # uniform outside i's block
    return i, j


def _liveness(H: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """up[x, e, a]: edge-agg link alive. core[x, a, c]: agg-core link alive.
    shared[x, y, a]: cores in row a+1 reachable from aggregation switch a of both pods x and y.
    """
    k = H.graph["k"]
    half = k // 2
    up = np.array([[[H.has_edge(f"e_{x}_{e}", f"a_{x}_{a}") for a in range(half)]
                    for e in range(half)] for x in range(k)], dtype=np.int32)
    core = np.array([[[H.has_edge(f"a_{x}_{a}", f"c_{a + 1}_{c + 1}") for c in range(half)]
                      for a in range(half)] for x in range(k)], dtype=np.int32)
    shared = np.einsum("xac,yac->xya", core, core)
    return up, core, shared


def _summarize(values: dict[str, np.ndarray], same_pod: np.ndarray) -> dict[str, float]:
    """{name: per-pair values} -> {class}_{name}_mean and {class}_{name}_min ("" name: {class}_mean)."""
    out = {}
    for cls, mask in (("intra", same_pod), ("inter", ~same_pod)):
        if mask.any():  # a small sample can miss one class
            for name, v in values.items():
                prefix = f"{cls}_{name}_" if name else f"{cls}_"
                out[prefix + "mean"] = v[mask].mean()
                out[prefix + "min"] = v[mask].min()
    return out


def surviving_paths(H: nx.Graph, scope: str, n_pairs, rng: np.random.Generator) -> dict[str, float]:
    k = H.graph["k"]
    half = k // 2
    up, _, shared = _liveness(H)

    # Endpoints: edge switches, or hosts when host links can fail (then a pair needs both host links).
    if scope == "all-links":
        g = half
        endpoint_alive = np.array([H.has_edge(f"h_{x}_{e}_{h}", f"e_{x}_{e}") for x in range(k)
                                   for e in range(half) for h in range(2, half + 2)], dtype=np.int32)
    else:
        g = 1
        endpoint_alive = np.ones(k * half, dtype=np.int32)

    i, j = _pairs(len(endpoint_alive), g, n_pairs, rng)
    x1, e1 = np.divmod(i // g, half)
    x2, e2 = np.divmod(j // g, half)
    same_pod = x1 == x2
    via = np.where(same_pod[:, None], 1, shared[x1, x2])
    paths = (up[x1, e1] * up[x2, e2] * via).sum(axis=1) * endpoint_alive[i] * endpoint_alive[j]
    return _summarize({"": paths}, same_pod)



def port_layers(k: int) -> dict[str, tuple[int, int]]:
    """{layer: (switches N, switch-to-switch ports per switch n)}."""
    half = k // 2
    return {"edge": (k * half, half), "agg": (k * half, k), "core": (half**2, k)}


def live_ports(H: nx.Graph) -> dict[str, float]:
    """{layer}_ports_mean and {layer}_ports_min: live switch-to-switch ports over the switches.

    Edge: uplinks (host ports never count). Agg: uplinks + downlinks. Core: downlinks.
    """
    up, core, _ = _liveness(H)
    ports = {
        "edge": up.sum(axis=2),
        "agg": up.sum(axis=1) + core.sum(axis=2),
        "core": core.sum(axis=0),
    }
    out = {}
    for layer, v in ports.items():
        out[f"{layer}_ports_mean"] = v.mean()
        out[f"{layer}_ports_min"] = v.min()
    return out


def expected_worst_ports(k: int, p: float) -> dict[str, float]:
    """E[min live ports] per layer under independent link failures only (README §4.5); p may be an array.

    A switch keeps L ~ Bin(n, 1-p) ports, independently of the other switches of its layer (they share
    no switch-to-switch link), so P(min >= m) = P(L >= m)^N and E[min] = sum_{m=1..n} P(L >= m)^N.
    """
    return {layer: sum(binom.sf(m - 1, n, 1 - p) ** N for m in range(1, n + 1))
            for layer, (N, n) in port_layers(k).items()}
