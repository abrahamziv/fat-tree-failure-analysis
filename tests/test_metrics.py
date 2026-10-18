import numpy as np
import pytest

from fattree.failures import inject_failures
from fattree.metrics import _flow_delivery, _pairs, delivery, expected_delivery, static_route, surviving_paths
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
        runs.append(surviving_paths(inject_failures(G, p, 0, "switch-links", r), "switch-links", "all", r))

    h, s = k // 2, (1 - p) ** 2  # s: both links of a 2-link segment alive
    expected = {"intra_mean": h * s, "inter_mean": h**2 * s**2}
    for key, value in expected.items():
        x = np.array([run[key] for run in runs])
        assert abs(x.mean() - value) < 4 * x.std(ddof=1) / np.sqrt(trials), key


# --- Delivery ratio (README §4.4.5) ---

SCHEMES = ["static", "ft", "ideal"]


def flows(H):
    """{(src, dst): {scheme: delivery probability}}, edge switches indexed as pod * k/2 + edge."""
    src, dst, _, d = _flow_delivery(H, "all", rng())
    return {(s, t): {name: v[n] for name, v in d.items()} for n, (s, t) in enumerate(zip(src, dst))}


@pytest.mark.parametrize("k", [4, 8])
def test_healthy_network_delivers_everything(k):
    m = delivery(build_fat_tree(k), "switch-links", "all", rng())
    assert len(m) == 2 * 3 * 2 and all(v == 1 for v in m.values())


def test_delivery_undefined_under_all_links():
    assert delivery(build_fat_tree(4), "all-links", "all", rng()) == {}


def test_both_directions_are_flows():
    k = 4
    src, dst, _, _ = _flow_delivery(build_fat_tree(k), "all", rng())
    n = k * k // 2  # edge switches
    assert len(src) == n * (n - 1) and set(zip(src, dst)) == {(i, j) for i in range(n) for j in range(n) if i != j}


def test_destination_side_link_black_holes_ft():
    """e_1_0 - a_1_0 dead. From e_0_0 the edge switch doesn't know: half its flows go to a_0_0, whose cores
    are all tagged for e_1_0's subnet (§3.8.1, class 3), and are lost. Ideal still has row 2."""
    G = build_fat_tree(4)
    G.remove_edge("e_1_0", "a_1_0")
    f = flows(G)[(0, 2)]  # e_0_0 -> e_1_0
    assert (f["static"], f["ft"], f["ideal"]) == (0.5, 0.5, 1)


def test_source_side_link_is_avoided_by_ft():
    """Same failure, reverse direction: e_1_0 sees its own dead uplink (§3.8.1, class 1) and avoids it."""
    G = build_fat_tree(4)
    G.remove_edge("e_1_0", "a_1_0")
    f = flows(G)[(2, 0)]  # e_1_0 -> e_0_0
    assert (f["static"], f["ft"], f["ideal"]) == (0.5, 1, 1)


@pytest.mark.parametrize("p, q", [(0.1, 0), (0.3, 0), (0, 0.2), (0.1, 0.1)])
def test_schemes_are_ordered_per_flow(p, q):
    G = build_fat_tree(8)
    for seed in range(5):
        r = np.random.default_rng(seed)
        src, dst, same_pod, d = _flow_delivery(inject_failures(G, p, q, "switch-links", r), "all", r)
        assert np.all(d["static"] <= d["ft"]) and np.all(d["ft"] <= d["ideal"])
        assert np.all(d["ft"][same_pod] == d["ideal"][same_pod])


def test_static_route_reproduces_paper_example():
    """[1, §3.5] Routing Example, k = 4: 10.0.1.2 -> 10.2.0.3 (host ID 3).

    The source edge switch 10.0.1.1 forwards on port 2 to 10.0.2.1, which forwards on port 3. The text says
    port 3 connects to core 10.4.1.1, but Algorithm 1 and the paper's own Figure 4 (port 3 -> 10.4.1.2)
    give 10.4.1.2. We follow Algorithm 1.
    """
    G = build_fat_tree(4)
    assert G.nodes["e_0_1"]["ip"] == "10.0.1.1"
    a, c = static_route(4, e1=1, h=3)
    agg, core = f"a_0_{a}", f"c_{a + 1}_{c + 1}"
    assert G.nodes[agg]["ip"] == "10.0.2.1" and G.edges["e_0_1", agg]["port"]["e_0_1"] == 2
    assert G.nodes[core]["ip"] == "10.4.1.2" and G.edges[agg, core]["port"][agg] == 3
    # Down: the core's only port to pod 2 reaches 10.2.2.1, which delivers to the subnet's switch 10.2.0.1.
    assert G.nodes[f"a_2_{a}"]["ip"] == "10.2.2.1" and G.has_edge(core, f"a_2_{a}")


@pytest.mark.parametrize("p, q", [(0.2, 0), (0.3, 0), (0, 0.3), (0.1, 0.1)])
def test_delivery_matches_analytic_expectation(p, q):
    k, trials = 8, 200
    G = build_fat_tree(k)
    runs = []
    for t in range(trials):
        r = np.random.default_rng(t)
        runs.append(delivery(inject_failures(G, p, q, "switch-links", r), "switch-links", "all", r))
    for key, value in expected_delivery(k, p, q).items():
        x = np.array([run[f"{key}_mean"] for run in runs])
        assert abs(x.mean() - value) < 4 * x.std(ddof=1) / np.sqrt(trials), key
