"""Canonical k-ary fat-tree of Al-Fares et al. (section 2.2), as a NetworkX graph.

Port convention (every switch has ports 0..k-1; 0..k/2-1 point down, k/2..k-1 point up):
    edge i, pod x : port id-2  -> host h_x_i_id        port k/2+a -> agg a_x_a
    agg a,  pod x : port i     -> edge e_x_i           port k/2+j -> core c_{a+1}_{j+1}
    core (j, i)   : port x     -> agg a_x_{j-1} (one port per pod)
"""

import networkx as nx


def build_fat_tree(k: int) -> nx.Graph:
    # One-byte address fields (10.pod.switch.ID, 10.k.j.i) cap k at 254 (section 3.2).
    if k % 2 or not 4 <= k <= 254:
        raise ValueError(f"k must be even and in [4, 254], got {k}")
    half = k // 2
    G = nx.Graph(k=k)

    for j in range(1, half + 1):
        for i in range(1, half + 1):
            G.add_node(f"c_{j}_{i}", layer="core", pod=None, index=(j, i), ip=f"10.{k}.{j}.{i}")

    for x in range(k):
        for a in range(half):
            agg = f"a_{x}_{a}"
            G.add_node(agg, layer="agg", pod=x, index=a, ip=f"10.{x}.{half + a}.1")
            for j in range(half):
                G.add_edge(agg, f"c_{a + 1}_{j + 1}", cls="agg-core",
                           port={agg: half + j, f"c_{a + 1}_{j + 1}": x})

        for e in range(half):
            edge = f"e_{x}_{e}"
            G.add_node(edge, layer="edge", pod=x, index=e, ip=f"10.{x}.{e}.1")
            for a in range(half):
                G.add_edge(edge, f"a_{x}_{a}", cls="edge-agg",
                           port={edge: half + a, f"a_{x}_{a}": e})
            for hid in range(2, half + 2):
                host = f"h_{x}_{e}_{hid}"
                G.add_node(host, layer="host", pod=x, index=hid, ip=f"10.{x}.{e}.{hid}")
                G.add_edge(host, edge, cls="host-edge", port={host: 0, edge: hid - 2})

    return G
