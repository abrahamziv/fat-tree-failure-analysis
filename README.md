# Fat-Tree Topology Generator & Failure Analysis

## 1. Quick start

**Requirements:** Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd <repo>
uv sync                      # install dependencies into .venv
uv run pytest                # run all tests (topology, failures, metrics, CLI)
```

**Reproduce all figures:**
```bash
uv run draw_topology.py --k 4                     # Figure 1 -> Resources/topology_k4.png
uv run simulate.py --config configs/exp_1.yaml    # Experiment 1 (§5) -> results/Exp_1
uv run plot_exp_1.py results/Exp_1/results.csv    # writes exp1_mean.png, exp1_worst.png next to the CSV
uv run simulate.py --config configs/exp_2.yaml          # Experiment 2 (§5.2), link failures -> results/Exp_2
uv run simulate.py --config configs/exp_2_switch.yaml   # Experiment 2, switch failures -> results/Exp_2_switch
uv run plot_exp_2.py results/Exp_2/results.csv results/Exp_2_switch/results.csv   # exp2_*.png next to each CSV
```

**Single run example:** k=8, 5% link failures, 50 trials.
```bash
uv run simulate.py --k 8 --fail-rate 0.05 --trials 50 --out results/demo
```

### Inputs

Inputs can be given on the command line or in a YAML config file (`--config`). Keys are the flag names with underscores. Command-line values override the config file. Flags marked *sweep* accept several values, and the program runs every combination.

| Flag | Default | Meaning |
|---|---|---|
| `--k` *(sweep)* | required | Ports per switch. Even, 4 ≤ k ≤ 254 (the cap comes from the one-byte addressing of [1, §3.2]). Builds the canonical fat-tree of [1, §2.2]: k pods, (k/2)² core switches, k³/4 hosts. |
| `--fail-rate` *(sweep)* | `0` | Link failure rate p ∈ [0, 1] |
| `--fail-switches` *(sweep)* | `0` | Switch failure rate q ∈ [0, 1]. A failed switch is removed together with all its links ([1, §3.8]). Only aggregation and core switches are eligible, because a failed edge switch simply disconnects its hosts. Applied before link failures. |
| `--fail-scope` | `switch-links` | `switch-links`: only edge–aggregation and aggregation–core links can fail (k³/2 links); these are the failures [1, §3.8] can survive. `all-links`: host–edge links can fail too (3k³/4 links); a failed host link always disconnects that host. |
| `--pairs` | `all` | Which (source, destination) pairs the metrics use. `all`: every pair. `N`: N pairs sampled uniformly (for large k). Pairs are edge switches under `switch-links` (exact, since all hosts under one edge switch share the same paths) and hosts under `all-links`. Pairs under the same edge switch are excluded, because their traffic never enters the fabric. The delivery metrics (§4.4.5) use both directions of every pair as separate flows. |
| `--trials` | `30` | Independent failure samples for each parameter combination. Plots show the mean ± 95% CI over trials. |
| `--seed` | `0` | Base random seed. Trial t uses seed + t, so every trial is reproducible and the same draws are reused across failure rates. |
| `--config` | none | YAML file with any of the flags above. |
| `--out` | results | Output directory. |

### Outputs (`--out DIR`)

- `config.json`: the exact parameters used, after combining the command line and the config file.
- `results.csv`: one row per (k, p, q, trial). Columns:
  - `intra_*` and `inter_*`: pairs in the same pod and in different pods.
  - `*_mean`: mean number of surviving shortest paths per pair (maximum k/2 intra-pod, (k/2)² inter-pod).
  - `*_min`: the worst pair's number of surviving paths.
  - `{class}_{scheme}_mean` and `{class}_{scheme}_min`, for class ∈ {`intra`, `inter`} and scheme ∈ {`static`, `ft`, `ideal`}: the mean and the worst flow's delivery probability of a new flow under static routing ([1, §3.5]), the fault-tolerance scheme of [1, §3.8], and ideal shortest-path routing (§4.4.5). Only written under `--fail-scope switch-links`.
- `exp1_mean.png`, `exp1_worst.png`: Experiment 1 figures (§5), written next to the CSV by `plot_exp_1.py`.
- `exp2_links.png`, `exp2_worst.png` (next to the link-failure CSV) and `exp2_switches.png` (next to the switch-failure CSV): Experiment 2 figures (§5.2), written by `plot_exp_2.py`.

## 2. Background: the k-ary fat-tree ([1, §2.2])
A k-ary fat-tree is built entirely from identical k-port switches arranged in three layers. It has k **pods**. Each pod contains k/2 **edge** switches and k/2 **aggregation** switches, fully connected to each other. Each edge switch serves k/2 hosts. On top sit (k/2)² **core** switches, arranged in a (k/2)×(k/2) grid. Core row i connects to aggregation switch i in every pod, so each core switch has exactly one port to each pod. Every switch uses all k ports.

<table><tr><td valign="top">

| Quantity | Formula |
|---|---|
| Pods | k |
| Edge switches | k²/2 |
| Aggregation switches | k²/2 |
| Core switches | (k/2)² |
| **Total switches** | 5k²/4 |
| **Hosts** | k³/4 |
| Shortest paths, same edge switch (switched locally, 0 hops) | 1 |
| Shortest paths, same pod (2 edge-to-edge hops) | k/2 |
| Shortest paths, different pods (4 edge-to-edge hops) | (k/2)² |

</td><td valign="top">

| Links | Formula |
|---|---|
| Host–edge | k³/4 |
| Edge–aggregation | k³/4 |
| Aggregation–core | k³/4 |
| **Total** | 3k³/4 |
| Switch–switch (failure candidates) | k³/2 |

</td></tr></table>

![k=4 fat-tree with addresses](Resources/topology_k4.png)

*Figure 1: The k=4 fat-tree built by our generator, with the addresses of [1, §3.2]. Each edge switch and its k/2 hosts form one /24 subnet (10.pod.switch.0/24). Generated by `uv run draw_topology.py --k 4`.*

## 3. Implementation

| File | Role |
|---|---|
| `fattree/topology.py` | `build_fat_tree(k)`: builds the fat-tree of §2 as a NetworkX graph. |
| `fattree/failures.py` | `inject_failures(...)`: returns a failed copy of the graph (§4.1–4.2). |
| `fattree/metrics.py` | `surviving_paths(...)`: surviving shortest paths per edge-switch pair (§4.3). `delivery(...)`: delivery ratio of new flows under static, §3.8 and ideal routing, and `expected_delivery(...)`, its analytic expectation (§4.4.5). |
| `simulate.py` | Experiment runner: parameter sweeps, trials, seeds; writes `results.csv` and `config.json`. |
| `plot_exp_1.py` | Experiment 1 figures: mean ± 95% CI over trials for the average and the worst pair, with the analytic curves of §4.4.1. |
| `plot_exp_2.py` | Experiment 2 figures (§5.2): delivery ratio per scheme vs. link and switch failure rate, with the analytic curves of §4.4.5, and the worst §3.8 flow. |
| `draw_topology.py` | Draws the topology (Figure 1), optionally with failures marked. |

**Topology.** Nodes are named `e_{pod}_{i}`, `a_{pod}_{i}`, `c_{j}_{i}` and `h_{pod}_{edge}_{id}`, and carry their layer, pod and IP address from [1, §3.2]. Each link records its class (host–edge, edge–aggregation, aggregation–core) and the port number at both ends. Ports 0…k/2−1 face down and k/2…k−1 face up. Aggregation switch i, port k/2+j connects to core c_{i,j}.

**Failures.** The healthy graph is built once per k. Each trial copies it, removes failed aggregation and core switches (together with their links), then removes failed in-scope links. Every element gets exactly one uniform draw, which is what makes the failure sets nested across rates (§4.2).

**Path counting.** Paths are never enumerated. For each failed graph we build two boolean arrays: `up[pod, edge, i]` (edge–aggregation links alive) and `core[pod, i, j]` (aggregation–core links alive). From these, `shared[x, y, i] = Σ_j core[x,i,j]·core[y,i,j]` counts the cores in row i that reach both pods x and y. The number of surviving paths of a pair is then

- intra-pod: Σᵢ up[e₁,i]·up[e₂,i]
- inter-pod: Σᵢ up[e₁,i]·up[e₂,i]·shared[x₁,x₂,i]

This is vectorized over all pairs with NumPy. The delivery metric (§4.4.5) reads the same three arrays, plus the core links of the one static path that Algorithm 1 of [1] picks for each destination host ID.

## 4. Failure Modeling
### 4.1 What can fail
- **Links** (`--fail-rate p`). By default (`--fail-scope switch-links`), only switch-to-switch links can fail: edge–aggregation and aggregation–core, k³/2 links in total. These are the two failure classes the paper's fault-tolerance scheme can route around ([1, §3.8]). Host–edge links are excluded by default: a host has a single uplink, so losing it disconnects the host whatever the fabric does. `--fail-scope all-links` adds them, and that only adds loss no routing can recover.
- **Switches** (`--fail-switches q`). A failed switch is removed together with all of its links, matching [1, §3.8] ("switch failures trigger the same BFD alerts and elicit the same responses"). Only aggregation and core switches are eligible. A failed edge switch disconnects its hosts, which is the same argument as for host links.
- **Order:** switch failures are applied first, then link failures on the in-scope links that remain.
### 4.2 How failures are sampled
Failures are independent Bernoulli trials: every eligible link fails with probability p, and every eligible switch with probability q, independently of all other elements. The failure percentage from the assignment is therefore read as a per-element failure probability. The expected failed fraction is p, with standard deviation √(p(1−p)/L) for L eligible links. That is about one percentage point or less for k ≥ 16 (k = 16, L = 2048: ±0.66 pp at p = 0.1, ±1.0 pp at p = 0.3). Independence is the standard failure model, and it makes the expectations in §4.4 exactly computable.

**Seeds and nesting.** Trial t uses seed `seed + t`. Each element gets one uniform draw u and fails if u < p (or q). The draws don't depend on the rate, so for a fixed seed the failure sets are nested: everything that fails at p = 0.05 also fails at p = 0.1. All rates are compared on the same random draws (common random numbers), which keeps the curves smooth and monotone. Switch draws come before link draws, so nesting in p holds for each fixed q.
### 4.3 What counts as a surviving path
We count **shortest (equal-cost) paths** between pairs of edge switches. Under `switch-links` scope, every host under an edge switch sees the same paths, so edge switches are the exact endpoints. Pairs under the same edge switch are excluded, because their traffic never enters the fabric. Because of the fat-tree structure, every shortest path is fixed by a small choice:
- **Same pod:** e₁ → aggregation switch aᵢ → e₂. There are k/2 paths in total, one per i ∈ {1,…,k/2}.
- **Different pods:** e₁ → aᵢ (pod 1) → core c_{i,j} → aᵢ (pod 2) → e₂. Core row i connects only to aggregation switch i in every pod, so both pods use the same i. There are (k/2)² paths in total, one per (i, j).

A path **survives** if all of its links and switches are alive.
### 4.4 Analytical results
We derive the expected number of surviving shortest paths between a pair of edge switches. All failures are independent (§4.2). Let $s = 1-p$ be the probability that a link survives, and $r = 1-q$ the probability that a switch survives.

#### 4.4.1 Link failures ($q = 0$)
Every switch-to-switch link fails with probability $p$, independently; no switch fails.
##### Intra-pod pairs
Let $e_1$ and $e_2$ be two different edge switches in the same pod. There are $k/2$ shortest paths (2 links each) between them, one through each aggregation switch of the pod:

$$e_1 \leftrightarrow a_i \leftrightarrow e_2, \qquad i \in \{1,\dots,k/2\}$$

A path survives only if both of its links survive, so $P(e_1 \leftrightarrow a_i \leftrightarrow e_2 \text{ survives}) = s^2$.\
We define $B_{i} \sim \mathrm{Ber}(s^2)$ as the indicator that equals 1 if $e_1 \leftrightarrow a_i \leftrightarrow e_2$ survived. Different paths use disjoint links, so the $B_i$ are **independent**.\
We define $X = \sum_{i=1}^{k/2} B_{i}$, the number of surviving paths between $e_1$ and $e_2$. Then $X \sim \mathrm{Bin}(k/2,\, s^2)$ and

$$\mathbb{E}[X_{\text{intra}}] = \frac{k}{2}\, s^2$$

##### Inter-pod pairs
Let $e_1$ and $e_2$ be edge switches in different pods. Denote by $a_i^{(1)}$ and $a_i^{(2)}$ the $i$-th aggregation switch of each pod, and by $c_{i,j}$ the $j$-th core switch in core row $i$. Core row $i$ connects only to aggregation switch $i$ of every pod, so a shortest path (4 links) must use the same index $i$ on both sides:

$$e_1 \leftrightarrow a_i^{(1)} \leftrightarrow c_{i,j} \leftrightarrow a_i^{(2)} \leftrightarrow e_2, \qquad i, j \in \{1,\dots,k/2\}$$

This gives $(k/2)^2$ shortest paths, one per core switch. A path survives only if all four of its links survive, so $P(\text{path } (i,j) \text{ survives}) = s^4$.\
We define $B_{i,j} \sim \mathrm{Ber}(s^4)$ as the indicator that equals 1 if path $(i,j)$ survived, and $X = \sum_{i=1}^{k/2}\sum_{j=1}^{k/2} B_{i,j}$, the number of surviving paths between $e_1$ and $e_2$.\
Unlike the intra-pod case, the $B_{i,j}$ are **not independent**: all $k/2$ paths with the same $i$ share the links $e_1 \leftrightarrow a_i^{(1)}$ and $a_i^{(2)} \leftrightarrow e_2$, so if one of those fails, all $k/2$ paths of row $i$ fail together. $X$ is therefore not binomial. Linearity of expectation does not require independence, however, so

$$\mathbb{E}[X_{\text{inter}}] = \sum_{i=1}^{k/2}\sum_{j=1}^{k/2} \mathbb{E}[B_{i,j}] = \left(\frac{k}{2}\right)^2 s^4$$

Normalized by the number of paths in the healthy network ($k/2$ and $(k/2)^2$), the expected fraction of surviving paths is $s^2$ intra-pod and $s^4$ inter-pod, independent of $k$.

#### 4.4.2 Switch failures ($p = 0$)
We now fail switches instead of links: every aggregation and core switch fails with probability $q$, independently, and no link fails on its own ($p = 0$). Edge switches never fail, so every pair still has both of its endpoints.

**Intra-pod pairs.** Path $e_1 \leftrightarrow a_i \leftrightarrow e_2$ survives exactly when $a_i$ survives. Different paths use different aggregation switches, so $B_i \sim \mathrm{Ber}(r)$ are independent, $X \sim \mathrm{Bin}(k/2,\, r)$ and

$$\mathbb{E}[X_{\text{intra}}] = \frac{k}{2}\, r$$

**Inter-pod pairs.** Path $(i,j)$ survives when $a_i^{(1)}$, $c_{i,j}$ and $a_i^{(2)}$ all survive, so $B_{i,j} \sim \mathrm{Ber}(r^3)$. As in §4.4.1, the paths of row $i$ share $a_i^{(1)}$ and $a_i^{(2)}$, so they are dependent. By linearity:

$$\mathbb{E}[X_{\text{inter}}] = \left(\frac{k}{2}\right)^2 r^3$$

Normalized by the number of paths in the healthy network, the expected fraction of surviving paths is $r$ intra-pod and $r^3$ inter-pod, independent of $k$.

#### 4.4.3 Link and switch failures together
Now both failure types are active: every aggregation and core switch fails with probability $q$, and every switch-to-switch link fails with probability $p$, all independently. A link is usable only if the link itself survives **and** both of its endpoint switches survive. (The simulator applies link failures only to links that survived the switch failures. This gives the same result, since a link attached to a dead switch is dead either way.)

**Intra-pod pairs.** Path $e_1 \leftrightarrow a_i \leftrightarrow e_2$ needs one switch ($a_i$) and two links to survive. These are independent events, so $B_i \sim \mathrm{Ber}(r s^2)$. Different paths still use disjoint switches and links, so the $B_i$ are independent, $X \sim \mathrm{Bin}(k/2,\, r s^2)$ and

$$\mathbb{E}[X_{\text{intra}}] = \frac{k}{2}\, r s^2$$

**Inter-pod pairs.** Path $(i,j)$ needs three switches ($a_i^{(1)}$, $c_{i,j}$, $a_i^{(2)}$) and four links to survive, so $B_{i,j} \sim \mathrm{Ber}(r^3 s^4)$. Paths of the same row $i$ share both aggregation switches and both edge–aggregation links, so they are dependent. By linearity:

$$\mathbb{E}[X_{\text{inter}}] = \left(\frac{k}{2}\right)^2 r^3 s^4$$

§4.4.1 and §4.4.2 are special cases: $q = 0$ ($r = 1$) gives §4.4.1 and $p = 0$ ($s = 1$) gives §4.4.2. Normalized, the expected surviving fractions are $r s^2$ intra-pod and $r^3 s^4$ inter-pod.

#### 4.4.4 What the expectations do not tell
These are per-pair expectations. By linearity of expectation they hold however failures are correlated across pairs, but they say nothing about how the damage is distributed. One failed link removes paths for every pair that uses it, and one failed switch removes paths for many pairs at once. Quantities such as the worst pair's number of surviving paths (`*_min`), or how much the damage varies from one failure sample to the next, depend on the joint failure pattern and have no simple closed form. The simulation measures them, and uses the expectations above to validate itself.

#### 4.4.5 Delivery ratio of new flows
Experiment 2 (§5.2) asks whether a new flow from edge switch $e_1$ (pod $x_1$) reaches a destination host with host ID $h \in \{2,\dots,k/2+1\}$ under edge switch $e_2$ (pod $x_2$). We compare three routing schemes. A flow's **delivery probability** is taken over the scheme's own choices, and the **delivery ratio** is its mean over flows. Delivery is directional, so $e_1 \to e_2$ and $e_2 \to e_1$ are separate flows. We assume:
- **Steady state:** every failure has been detected by BFD, and every tag of [1, §3.8] has reached the switches it is sent to. Only new flows count. Flows that were already on a broken path are restarted end to end ([1, §3.8, footnote 5]).
- **Global tag propagation:** a tag reaches every switch that the paper says it is broadcast or mirrored to, with no loss or delay.
- **Shortest paths only:** no scheme detours. A packet that goes down never goes up again, so an aggregation switch that has no usable core drops the flow.
- **Delivery, not throughput:** flows don't interact and there is no congestion. Each flow's delivery probability is computed exactly from the failed graph, with no flow sampling.

As in §4.4.3, $s = 1-p$, $r = 1-q$, and paths use the same index $i$ in both pods.

**Static routing ([1, §3.5]).** The two-level tables of Algorithm 1 give one path per destination host ID. The source edge switch uses the aggregation switches' suffix table (lower pod switches omit only the /24 prefixes) and goes up to aggregation switch $i = (h-2+e_1) \bmod k/2$. For inter-pod traffic, that switch sits at position $k/2+i$ and goes up to core column $(h-2+i) \bmod k/2$. A flow is delivered iff that one path is alive. Each path is one of the paths of §4.4.3, so

$$\mathbb{E}[D_{\text{intra}}^{\text{static}}] = r s^2, \qquad \mathbb{E}[D_{\text{inter}}^{\text{static}}] = r^3 s^4$$

($s^2$ and $s^4$ when only links fail). The host IDs map one-to-one onto the aggregation switches, so a pair's static delivery probability (the mean over its $k/2$ host IDs) has the same expectation.

**Ideal shortest-path routing (oracle).** A flow is delivered iff at least one shortest path survives, i.e. iff the pair's surviving path count of §4.3 is positive.
- *Intra-pod:* the $k/2$ paths are independent, each surviving with probability $r s^2$ (§4.4.3):
$$\mathbb{E}[D_{\text{intra}}^{\text{ideal}}] = 1 - (1 - r s^2)^{k/2}$$
- *Inter-pod:* row $i$ consists of $a_i^{(1)}$, $a_i^{(2)}$, the two edge–aggregation links, and the $k/2$ cores $c_{i,j}$ with their links. Different rows use disjoint switches and links, so rows are independent. Row $i$ carries a path iff both aggregation switches and both edge links are alive ($r^2 s^2$) and at least one of its $k/2$ cores is alive together with both of that core's links ($1 - (1 - r s^2)^{k/2}$):
$$\mathbb{E}[D_{\text{inter}}^{\text{ideal}}] = 1 - \left(1 - r^2 s^2 \left[1 - (1 - r s^2)^{k/2}\right]\right)^{k/2}$$

**Fault tolerance ([1, §3.8]).** A failed switch needs no special case: it triggers the same BFD alerts as its links, which therefore read as dead.
- *Intra-pod:* $e_1$ avoids its own dead uplinks (§3.8.1, class 1). An aggregation switch that loses its link to $e_2$ tags it to every edge switch of the pod (class 2). $e_1$ therefore uses only aggregation switches with both links alive, and delivers iff such a switch exists. This is exactly the ideal: $D_{\text{intra}}^{\text{ft}} = D_{\text{intra}}^{\text{ideal}}$ for every flow.
- *Inter-pod:* $e_1$ knows only its own uplinks. Let $U_i$ indicate that the link $e_1 \leftrightarrow a_i^{(1)}$ is alive (so $a_i^{(1)}$ is alive). $e_1$ picks uniformly among the $N = \sum_i U_i$ live uplinks. The tags about the destination side, class 3 of §3.8.1 for $a_i^{(2)} \leftrightarrow e_2$ and §3.8.2 for core–$a_i^{(2)}$ links, stop at the source aggregation switches. Once at $a_i^{(1)}$, the flow therefore succeeds iff row $i$ is *good*: $G_i$ = ($a_i^{(2)}$ and its link to $e_2$ are alive, and at least one core of row $i$ has both links alive). Otherwise $a_i^{(1)}$ has no untagged core for that subnet and the flow is black-holed. Per flow,
$$D_{\text{inter}}^{\text{ft}} = \frac{\sum_i U_i G_i}{\sum_i U_i} \quad (0 \text{ if } N = 0)$$

  The mean still has a closed form. The pairs $(U_i, G_i)$ are independent across rows. Given $U_i = 1$, $a_i^{(1)}$ is alive, so $G_i \sim \mathrm{Ber}(g)$ with $g = r s \left[1 - (1 - r s^2)^{k/2}\right]$ ($a_i^{(2)}$, its link to $e_2$, and one good core). This is independent of the other rows. Hence $\mathbb{E}[D \mid U_1,\dots,U_{k/2}] = g$ whenever $N > 0$, and with $P(N > 0) = 1 - (1 - r s)^{k/2}$,
$$\mathbb{E}[D_{\text{inter}}^{\text{ft}}] = r s \left[1 - (1 - r s^2)^{k/2}\right]\left[1 - (1 - r s)^{k/2}\right]$$

  For large k both brackets tend to 1 and $\mathbb{E}[D_{\text{inter}}^{\text{ft}}] \to r s$.

Per flow, static ≤ §3.8 ≤ ideal. The worst flow's delivery probability (`*_min`) depends on the joint failure pattern and has no closed form (§4.4.4), so the simulation measures it.

## 5. Experiments

### 5.1 Experiment 1: surviving shortest paths vs. link failure probability

#### Question
How much shortest-path diversity does a fat-tree keep when switch-to-switch links fail at random, and how does that depend on the switch size k? Concretely:
1. **Average pair:** does the mean fraction of surviving shortest paths per edge-switch pair follow the expectations of §4.4.1, $s^2$ intra-pod and $s^4$ inter-pod, for every k?
2. **Worst pair:** how far below the average does the worst-off pair of edge switches fall, and how does that gap change with k? §4.4.4 explains why this has no simple closed form.

#### Setup
| Parameter | Value |
|---|---|
| Switch size k | 8, 16, 32, 48 (128 to 27,648 hosts; k = 48 is the example of [1, §2.2]) |
| Link failure probability p | 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3 |
| Switch failure probability q | 0 |
| Failure scope | `switch-links` (edge–aggregation and aggregation–core links) |
| Pairs | all edge-switch pairs, split into intra-pod and inter-pod (table below) |
| Trials | 30 independent failure samples per (k, p), seeds 0–29 |

| k | Edge switches | Intra-pod pairs | Inter-pod pairs | Max paths (intra / inter) |
|---|---|---|---|---|
| 8 | 32 | 48 | 448 | 4 / 16 |
| 16 | 128 | 448 | 7,680 | 8 / 64 |
| 32 | 512 | 3,840 | 126,976 | 16 / 256 |
| 48 | 1,152 | 13,248 | 649,728 | 24 / 576 |

**Measured per trial**, separately for intra-pod and inter-pod pairs:
- **mean surviving fraction:** the average over pairs of (surviving shortest paths) / (maximum: k/2 or (k/2)²);
- **worst-pair fraction:** the minimum over pairs of the same ratio.

Each plotted point is the mean over the 30 trials, with a 95% confidence interval (Student-t).

```bash
uv run simulate.py --config configs/exp_1.yaml
uv run plot_exp_1.py results/Exp_1/results.csv
```

#### Results

![Experiment 1: mean surviving fraction](results/Exp_1/exp1_mean.png)

*Figure 2: Mean fraction of surviving shortest paths per edge-switch pair vs. link failure probability p. Dashed: the analytic expectations $(1-p)^2$ and $(1-p)^4$ of §4.4.1.*

![Experiment 1: worst-pair fraction](results/Exp_1/exp1_worst.png)

*Figure 3: The worst-off pair's fraction of surviving shortest paths vs. p. Dashed gray: the average pair (analytic), for reference.*

| p = 0.1 | k = 8 | k = 16 | k = 32 | k = 48 | analytic |
|---|---|---|---|---|---|
| mean, intra-pod | 0.824 | 0.807 | 0.811 | 0.809 | 0.810 |
| mean, inter-pod | 0.666 | 0.655 | 0.655 | 0.655 | 0.656 |
| worst pair, intra-pod | 0.400 | 0.346 | 0.417 | 0.465 | — |
| worst pair, inter-pod | 0.148 | 0.162 | 0.262 | 0.313 | — |

#### Discussion
**The average pair matches the analysis (Figure 2).** For k ≥ 16, every point is within 0.003 of $(1-p)^2$ and $(1-p)^4$, which validates the simulator. The average surviving fraction does not depend on k. Larger fabrics only reduce the trial-to-trial noise, because they average over more links.

**The worst pair is far from the average (Figure 3).** At p = 0.01 the average pair loses 2–4% of its paths, while the worst pair loses 20–30%. At p = 0.1, even in the k = 48 fabric (27,648 hosts), the worst inter-pod pair keeps only 31% of its paths, against an average of 66%. The cause is the edge layer. An edge switch has only k/2 uplinks, and among k²/2 edge switches a few always lose several of them. Every pair touching such a switch loses a large share of its paths at once. Larger k helps (worst inter-pod at p = 0.1: 0.16 → 0.26 → 0.31 for k = 16 → 32 → 48), but the gap to the average stays large.

**Why this matters for AI factories.** Training jobs synchronize through collective operations (e.g. all-reduce), which complete only when the *slowest* participating flow does. A flow between a badly hit pair of racks has fewer live uplinks to spread its traffic over, so it becomes that straggler. The average path diversity, which is what a per-link analysis gives, looks comfortable at any scale. But a collective spanning thousands of hosts almost certainly includes the worst pair, so its speed is set by the minimum of Figure 3, not the mean of Figure 2. For such workloads, fat-tree resilience should be judged by its worst-case path diversity, and the edge uplinks are where it is lost.


### 5.2 Experiment 2: delivery under the fault-tolerance scheme of [1, §3.8]

#### Question
[1] routes with static two-level tables (§3.5): each flow follows one hashed shortest path, so a single failed link on that path breaks it. §3.8 adds a fault-tolerance scheme: switches detect failed neighbours with BFD and broadcast tags so that new flows avoid the failure. How much does this scheme improve delivery under random link and switch failures, and how does the improvement depend on the failure rate and on k?

#### Setup
| Parameter | Value |
|---|---|
| Switch size k | 8, 16, 32, 48 |
| Link failure probability p | 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, with q = 0 (`configs/exp_2.yaml`) |
| Switch failure probability q | 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, with p = 0 (`configs/exp_2_switch.yaml`) |
| Failure scope | `switch-links` (edge–aggregation and aggregation–core links; aggregation and core switches) |
| Flows | both directions of every edge-switch pair: 96 / 896 (k = 8) to 26,496 / 1,299,456 (k = 48) intra-pod / inter-pod flows |
| Trials | 30 independent failure samples per point, seeds 0–29 |

| Scheme | Source edge switch | Source aggregation switch (inter-pod) | Delivered iff |
|---|---|---|---|
| static ([1, §3.5]) | Algorithm 1: aggregation switch $(h-2+e_1) \bmod k/2$ for destination host ID $h$ | Algorithm 1: core column $(h-2+i) \bmod k/2$ | that one path is alive |
| §3.8 ([1, §3.8]) | uniform over its live uplinks; intra-pod also skips aggregation switches tagged for $e_2$ | any live core not tagged for $e_2$'s subnet or pod | the chosen aggregation switch can still reach $e_2$ |
| ideal (oracle) | any aggregation switch that is on a surviving shortest path | any core on a surviving shortest path | some shortest path survives |

**Measured per trial**, separately for intra-pod and inter-pod flows and for each scheme (§4.4.5): the **delivery ratio** (mean delivery probability over flows, `{class}_{scheme}_mean`) and the **worst flow's** delivery probability (`{class}_{scheme}_min`). Each plotted point is the mean over the 30 trials, with a 95% confidence interval (Student-t).

```bash
uv run simulate.py --config configs/exp_2.yaml
uv run simulate.py --config configs/exp_2_switch.yaml
uv run plot_exp_2.py results/Exp_2/results.csv results/Exp_2_switch/results.csv
```

#### Results

![Experiment 2: delivery ratio under link failures](results/Exp_2/exp2_links.png)

*Figure 4: Delivery ratio of new flows vs. link failure probability p (q = 0), per scheme (color) and k (line style). Dashed: the analytic expectations of §4.4.5 for static ($s^2$, $s^4$, the same for every k) and ideal routing (one curve per k).*

![Experiment 2: delivery ratio under switch failures](results/Exp_2_switch/exp2_switches.png)

*Figure 5: The same vs. switch failure probability q (p = 0). Dashed: analytic static ($r$, $r^3$) and ideal curves.*

![Experiment 2: worst flow under §3.8](results/Exp_2/exp2_worst.png)

*Figure 6: The worst inter-pod flow's delivery probability under §3.8 vs. link failure probability p, per k.*

Mean over 30 trials at p = 0.1 (q = 0). In parentheses: the analytic expectation of §4.4.5.

| p = 0.1 | k = 8 | k = 16 | k = 32 | k = 48 |
|---|---|---|---|---|
| static, intra-pod | 0.8236 (0.8100) | 0.8074 (0.8100) | 0.8109 (0.8100) | 0.8089 (0.8100) |
| §3.8, intra-pod | 0.9993 (0.9987) | 1.0000 (1.0000) | 1.0000 (1.0000) | 1.0000 (1.0000) |
| ideal, intra-pod | 0.9993 (0.9987) | 1.0000 (1.0000) | 1.0000 (1.0000) | 1.0000 (1.0000) |
| static, inter-pod | 0.6663 (0.6561) | 0.6544 (0.6561) | 0.6553 (0.6561) | 0.6553 (0.6561) |
| §3.8, inter-pod | 0.9070 (0.8987) | 0.8982 (0.9000) | 0.9005 (0.9000) | 0.8994 (0.9000) |
| ideal, inter-pod | 0.9998 (0.9987) | 1.0000 (1.0000) | 1.0000 (1.0000) | 1.0000 (1.0000) |
| §3.8 worst flow, inter-pod | 0.347 | 0.333 | 0.470 | 0.547 |
| gap closed by §3.8, intra-pod | 100.0% | 100.0% | 100.0% | 100.0% |
| gap closed by §3.8, inter-pod | 72.2% | 70.5% | 71.1% | 70.8% |

The last two rows give the share of the static-to-ideal gap that §3.8 closes, $(\text{ft} - \text{static}) / (\text{ideal} - \text{static})$, computed from the delivery ratios above.

#### Discussion
TODO

## References
[1] M. Al-Fares, A. Loukissas, and A. Vahdat, "A scalable, commodity data center network architecture," *SIGCOMM Comput. Commun. Rev.*, vol. 38, no. 4, pp. 63–74, Oct. 2008. doi: [10.1145/1402946.1402967](https://doi.org/10.1145/1402946.1402967)

## AI usage
This project was developed with the help of Claude (Anthropic). Claude was used as a reading companion for [1], for discussing the simulator design and the analytical derivations, for generating and refactoring code via Claude Code, and for drafting parts of this README.
