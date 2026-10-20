import pytest

from fattree.topology import build_fat_tree


@pytest.mark.parametrize("k", [4, 6, 8])
def test_counts_ports_and_ips(k):
    G = build_fat_tree(k)
    layers = [layer for _, layer in G.nodes(data="layer")]
    assert layers.count("host") == k**3 // 4
    assert len(layers) - layers.count("host") == 5 * k**2 // 4
    assert G.number_of_edges() == 3 * k**3 // 4

    ips = [ip for _, ip in G.nodes(data="ip")]
    assert len(set(ips)) == len(ips)

    # Every switch uses exactly ports 0..k-1, hence has degree k.
    for s, layer in G.nodes(data="layer"):
        if layer != "host":
            assert sorted(G.edges[s, n]["port"][s] for n in G.neighbors(s)) == list(range(k))


def test_port_convention():
    G = build_fat_tree(4)
    assert G.edges["h_1_0_3", "e_1_0"]["port"] == {"h_1_0_3": 0, "e_1_0": 1}  # edge port id-2 -> host
    assert G.edges["e_1_0", "a_1_1"]["port"] == {"e_1_0": 3, "a_1_1": 0}      # edge port k/2+a -> agg a
    assert G.edges["a_3_1", "c_2_1"]["port"] == {"a_3_1": 2, "c_2_1": 3}      # agg port k/2+c -> core (a+1, c+1); core port x -> pod x
    assert G.nodes["a_0_0"]["ip"] == "10.0.2.1"
    assert G.nodes["c_1_2"]["ip"] == "10.4.1.2"


@pytest.mark.parametrize("k", [2, 5, 256])
def test_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        build_fat_tree(k)
