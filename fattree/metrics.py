"""Shortest-path diversity under failures, computed from the fat-tree structure (no path enumeration).

Between two edge switches in the same pod, each aggregation switch gives one shortest path
(2 links). Between pods, each core switch gives one (4 links); core c_{a+1}_{c} is reached
only through aggregation switch a of each pod.

For each pair class (intra-pod, inter-pod) it reports the mean and the minimum number of
surviving shortest paths over the pairs.
"""

import networkx as nx
import numpy as np


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


def surviving_paths(H: nx.Graph, scope: str, n_pairs, rng: np.random.Generator) -> dict[str, float]:
    k = H.graph["k"]
    half = k // 2

    # Link liveness. A failed switch is absent from H, so its links read as dead.
    up = np.array([[[H.has_edge(f"e_{x}_{e}", f"a_{x}_{a}") for a in range(half)]
                    for e in range(half)] for x in range(k)], dtype=np.int32)
    core = np.array([[[H.has_edge(f"a_{x}_{a}", f"c_{a + 1}_{c + 1}") for c in range(half)]
                      for a in range(half)] for x in range(k)], dtype=np.int32)
    # shared[x, y, a]: cores in row a+1 reachable from aggregation switch a of both pods x and y.
    shared = np.einsum("xac,yac->xya", core, core)

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

    out = {}
    for name, mask in (("intra", same_pod), ("inter", ~same_pod)):
        if mask.any():  # a small sample can miss one class
            out[f"{name}_mean"] = paths[mask].mean()
            out[f"{name}_min"] = paths[mask].min()
    return out
