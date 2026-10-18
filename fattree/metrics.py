"""Metrics under failures, computed from the fat-tree structure (no path enumeration).

Between two edge switches in the same pod, each aggregation switch gives one shortest path
(2 links). Between pods, each core switch gives one (4 links); core c_{a+1}_{c} is reached
only through aggregation switch a of each pod.

surviving_paths: per pair class (intra-pod, inter-pod), the mean and the minimum number of
surviving shortest paths over the pairs.
delivery: per pair class and routing scheme, the mean and the minimum delivery probability
of a new flow over the flows (README §4.4.5).
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


def _liveness(H: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """up[x, e, a]: edge-agg link alive. core[x, a, c]: agg-core link alive.
    shared[x, y, a]: cores in row a+1 reachable from aggregation switch a of both pods x and y.

    A failed switch is absent from H, so its links read as dead.
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


def static_route(k: int, e1, h):
    """Algorithm 1 of [1]: the static path from edge switch e1 to destination host ID h (2..k/2+1).

    Returns (a, c): aggregation switch a_{x1}_{a}, and for inter-pod traffic core c_{a+1}_{c+1}.
    Edge switches use the aggregation switches' suffix table ([1, §3.5]: they only omit the /24
    prefixes), so position z = e1 goes up to port (h-2+z) mod k/2 + k/2, i.e. agg a. Agg a sits at
    position z = k/2 + a and goes up to (h-2+k/2+a) mod k/2 = (h-2+a) mod k/2.
    """
    half = k // 2
    a = (h - 2 + e1) % half
    return a, (h - 2 + a) % half


def _flow_delivery(H: nx.Graph, n_pairs, rng: np.random.Generator):
    """Per-flow delivery probability of each scheme, between edge switches (switch-links scope).

    Returns (src, dst, same_pod, {scheme: per-flow probability}); src and dst index edge switches
    as pod * k/2 + edge. Both directions of every pair are separate flows.
    """
    k = H.graph["k"]
    half = k // 2
    up, core, shared = _liveness(H)

    i, j = _pairs(k * half, 1, n_pairs, rng)
    src, dst = np.concatenate([i, j]), np.concatenate([j, i])
    x1, e1 = np.divmod(src, half)
    x2, e2 = np.divmod(dst, half)
    same_pod = x1 == x2
    up1, up2 = up[x1, e1], up[x2, e2]
    # good[:, a]: once at aggregation switch a, the flow can still reach e2 on a shortest path.
    good = up2 * np.where(same_pod[:, None], 1, shared[x1, x2] > 0)

    # Static (§3.5): one path per destination host ID; delivered iff all its links are alive.
    static = np.zeros(len(src))
    for h in range(2, half + 2):
        a, c = static_route(k, e1, h)
        hop = np.where(same_pod, 1, core[x1, a, c] * core[x2, a, c])
        static += up[x1, e1, a] * hop * up[x2, e2, a]
    static /= half

    # Ideal: at least one shortest path survives.
    ideal = (up1 * good).any(axis=1).astype(float)

    # §3.8: intra-pod, the edge switch sees every tag of its pod, so it delivers iff ideal does.
    # Inter-pod, it knows only its own uplinks: a uniform live uplink, then success iff that agg is good.
    n_up = up1.sum(axis=1)
    ft = np.where(same_pod, ideal, (up1 * good).sum(axis=1) / np.maximum(n_up, 1))
    return src, dst, same_pod, {"static": static, "ft": ft, "ideal": ideal}


def delivery(H: nx.Graph, scope: str, n_pairs, rng: np.random.Generator) -> dict[str, float]:
    """Delivery ratio of new flows per scheme. Only defined between edge switches: empty under all-links."""
    if scope != "switch-links":
        return {}
    _, _, same_pod, schemes = _flow_delivery(H, n_pairs, rng)
    return _summarize(schemes, same_pod)


def expected_delivery(k: int, p=0.0, q=0.0) -> dict:
    """Expected per-flow delivery probability under independent failures (README §4.4.5)."""
    half = k // 2
    s, r = 1 - p, 1 - q
    row = 1 - (1 - r * s**2) ** half  # at least one core of a row with both its links alive
    return {
        "intra_static": r * s**2,
        "inter_static": r**3 * s**4,
        "intra_ideal": 1 - (1 - r * s**2) ** half,
        "inter_ideal": 1 - (1 - r**2 * s**2 * row) ** half,
        "intra_ft": 1 - (1 - r * s**2) ** half,
        "inter_ft": r * s * row * (1 - (1 - r * s) ** half),
    }
