"""Failure injection: returns a failed copy of the healthy fat-tree.

Every eligible element fails independently (bernoulli). Each element gets one uniform draw,
and it fails when its draw is below the rate. The draws do not depend on the rate, so for a
fixed seed and switch rate the failed sets are nested across p (common random numbers):
everything that fails at p = 0.05 also fails at p = 0.1.
"""

import networkx as nx
import numpy as np

SCOPES = {
    "switch-links": {"edge-agg", "agg-core"},
    "all-links": {"edge-agg", "agg-core", "host-edge"},
}


def _choose(items: list, rate: float, rng: np.random.Generator) -> list:
    return [items[i] for i in np.flatnonzero(rng.random(len(items)) < rate)]


def inject_failures(G: nx.Graph, rate: float, switch_rate: float, scope: str,
                    rng: np.random.Generator) -> nx.Graph:
    H = G.copy()
    # Switch failures first: a failed agg/core switch is removed together with its links.
    switches = [n for n, layer in G.nodes(data="layer") if layer in ("agg", "core")]
    H.remove_nodes_from(_choose(switches, switch_rate, rng))
    # Then link failures among the surviving links in scope.
    links = [(u, v) for u, v, cls in H.edges(data="cls") if cls in SCOPES[scope]]
    H.remove_edges_from(_choose(links, rate, rng))
    return H


def link_equivalent(q: float) -> float:
    """Expected fraction of dead switch-to-switch links when every agg/core switch fails with probability q.

    An edge-agg link dies with probability q (its agg switch fails), an agg-core link with
    1 - (1-q)^2 (either end fails). The two classes have k^3/4 links each, so the mean is (3q - q^2) / 2.
    """
    return (3 * q - q**2) / 2


def switch_rate_for(p_eq: float) -> float:
    """Inverse of link_equivalent: the switch failure probability q that kills a fraction p_eq of links."""
    return (3 - np.sqrt(9 - 8 * p_eq)) / 2
