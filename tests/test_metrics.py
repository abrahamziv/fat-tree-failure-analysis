import functools

import numpy as np
import pytest

from fattree.failures import inject_failures
from fattree.metrics import _pairs, expected_worst_ports, live_ports, port_layers, surviving_paths
from fattree.topology import build_fat_tree


def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("scope", ["switch-links", "all-links"])
@pytest.mark.parametrize("k", [4, 8])
def test_healthy_network_has_all_paths(k, scope):
    m = surviving_paths(build_fat_tree(k), scope, "all", rng())
    assert m["intra_mean"] == m["intra_min"] == k // 2
    assert m["inter_mean"] == m["inter_min"] == (k // 2) ** 2


def test_single_edge_agg_failure():
    G = build_fat_tree(4)
    G.remove_edge("e_0_0", "a_0_0")
    m = surviving_paths(G, "switch-links", "all", rng())
    # 4 intra pairs: (e_0_0, e_0_1) keeps 1 path, the other 3 keep 2.
    assert (m["intra_min"], m["intra_mean"]) == (1, 7 / 4)
    # 24 inter pairs: the 6 from e_0_0 keep 2 (cores of row 2 only), the other 18 keep 4.
    assert (m["inter_min"], m["inter_mean"]) == (2, (6 * 2 + 18 * 4) / 24)


def test_host_link_failure_disconnects_host_pairs():
    G = build_fat_tree(4)
    G.remove_edge("h_0_0_2", "e_0_0")
    m = surviving_paths(G, "all-links", "all", rng())
    assert m["intra_min"] == m["inter_min"] == 0


@pytest.mark.parametrize("g", [1, 2])
def test_pairs_skip_same_edge_switch(g):
    n = 16
    i, j = _pairs(n, g, "all", rng())
    assert len(i) == n * (n - 1) // 2 - (n // g) * g * (g - 1) // 2
    i, j = _pairs(n, g, 1000, rng())
    assert np.all(i // g != j // g) and j.min() >= 0 and j.max() < n


def test_matches_analytic_expectation():
    """Bernoulli link failures: E[paths] = (k/2)(1-p)^2 intra, (k/2)^2 (1-p)^4 inter."""
    k, p, trials = 8, 0.3, 300
    G = build_fat_tree(k)
    runs = []
    for t in range(trials):
        r = np.random.default_rng(t)
        runs.append(surviving_paths(inject_failures(G, p, "switch-links", r), "switch-links", "all", r))

    h, s = k // 2, (1 - p) ** 2  # s: both links of a 2-link segment alive
    expected = {"intra_mean": h * s, "inter_mean": h**2 * s**2}
    for key, value in expected.items():
        x = np.array([run[key] for run in runs])
        assert abs(x.mean() - value) < 4 * x.std(ddof=1) / np.sqrt(trials), key


# --- Live ports per switch (README §4.5) ---


@pytest.mark.parametrize("k", [4, 8])
def test_healthy_network_has_all_ports(k):
    m = live_ports(build_fat_tree(k))
    for layer, (_, n) in port_layers(k).items():
        assert m[f"{layer}_ports_mean"] == m[f"{layer}_ports_min"] == n


def test_link_failure_lowers_its_two_switches():
    """k = 4: 8 edge, 8 agg and 4 core switches, with 2, 4 and 4 ports."""
    G = build_fat_tree(4)
    G.remove_edge("e_0_0", "a_0_0")
    m = live_ports(G)
    assert (m["edge_ports_min"], m["edge_ports_mean"]) == (1, (1 + 7 * 2) / 8)
    assert (m["agg_ports_min"], m["agg_ports_mean"]) == (3, (3 + 7 * 4) / 8)
    assert (m["core_ports_min"], m["core_ports_mean"]) == (4, 4)

    G = build_fat_tree(4)
    G.remove_edge("a_0_0", "c_1_1")
    m = live_ports(G)
    assert (m["edge_ports_min"], m["edge_ports_mean"]) == (2, 2)
    assert (m["agg_ports_min"], m["agg_ports_mean"]) == (3, (3 + 7 * 4) / 8)
    assert (m["core_ports_min"], m["core_ports_mean"]) == (3, (3 + 3 * 4) / 4)


@functools.cache
def port_runs(k, p, trials):
    G = build_fat_tree(k)
    return [live_ports(inject_failures(G, p, "switch-links", np.random.default_rng(t))) for t in range(trials)]


def assert_matches(runs, expected):
    for key, value in expected.items():
        x = np.array([run[key] for run in runs])
        assert abs(x.mean() - value) < 4 * x.std(ddof=1) / np.sqrt(len(runs)), key


def test_mean_ports_match_analytic_expectation():
    """Each port survives with probability s = 1-p, so E[mean live ports] = s n."""
    k, p = 8, 0.3
    assert_matches(port_runs(k, p, 300), {f"{layer}_ports_mean": (1 - p) * n for layer, (_, n) in port_layers(k).items()})


def test_worst_ports_match_analytic_expectation():
    k, p = 8, 0.3
    assert_matches(port_runs(k, p, 300), {f"{layer}_ports_min": v for layer, v in expected_worst_ports(k, p).items()})
