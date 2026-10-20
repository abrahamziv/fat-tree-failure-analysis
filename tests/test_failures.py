from collections import Counter

import numpy as np

from fattree.failures import inject_failures
from fattree.topology import build_fat_tree

K = 8
G = build_fat_tree(K)


def fail(p=0.0, scope="switch-links", seed=0):
    return inject_failures(G, p, scope, np.random.default_rng(seed))


def failed_links(H):
    return {frozenset(e) for e in G.edges} - {frozenset(e) for e in H.edges}


def test_zero_rate_is_identity():
    H = fail()
    assert set(H.nodes) == set(G.nodes) and not failed_links(H)


def test_full_rate_keeps_only_host_links():
    H = fail(1.0)
    assert Counter(cls for *_, cls in H.edges(data="cls")) == {"host-edge": K**3 // 4}


def test_full_rate_all_links_scope_removes_every_link():
    assert fail(1.0, scope="all-links").number_of_edges() == 0


def test_same_seed_same_result():
    assert failed_links(fail(0.1, seed=7)) == failed_links(fail(0.1, seed=7))
    assert failed_links(fail(0.1, seed=7)) != failed_links(fail(0.1, seed=8))


def test_failures_nested_across_rates():
    low, high = failed_links(fail(0.05)), failed_links(fail(0.1))
    assert low < high

