
import networkx as nx
from collections import defaultdict


def refine_connectivity(G, partition):

    communities = defaultdict(list)

    for u, c in partition.items():
        communities[c].append(u)

    refined = {}
    next_label = 0

    for nodes in communities.values():

        components = nx.connected_components(
            G.subgraph(nodes)
        )

        for component in components:

            for u in component:
                refined[u] = next_label

            next_label += 1

    return refined


def predicted_connectivity_gain(G, partition):

    m = G.size(weight="weight")

    if m == 0:
        return 0.0

    communities = defaultdict(list)

    for u, c in partition.items():
        communities[c].append(u)

    degrees = dict(G.degree(weight="weight"))

    gain = 0.0

    for nodes in communities.values():

        totals = [
            sum(degrees[u] for u in component)
            for component in nx.connected_components(
                G.subgraph(nodes)
            )
        ]

        for i in range(len(totals)):

            for j in range(i + 1, len(totals)):

                gain += (
                    totals[i] * totals[j]
                    / (2 * m * m)
                )

    return gain
