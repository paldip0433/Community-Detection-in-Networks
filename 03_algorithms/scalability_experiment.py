
import random
import networkx as nx


def generate_sparse_sbm(
    n,
    communities=3,
    internal_degree=18,
    external_degree=3,
    seed=42
):

    if n % communities != 0:
        raise ValueError("Unequal community sizes.")

    size = n // communities

    p_in = internal_degree / (size - 1)
    p_out = external_degree / (n - size)

    if not 0 <= p_out < p_in <= 1:
        raise ValueError("Invalid SBM probabilities.")

    probabilities = [
        [
            p_in if i == j else p_out
            for j in range(communities)
        ]
        for i in range(communities)
    ]

    G = nx.stochastic_block_model(
        [size] * communities,
        probabilities,
        seed=seed,
        sparse=True
    )

    labels = {
        u: G.nodes[u]["block"]
        for u in G
    }

    nx.set_edge_attributes(G, 1.0, "weight")

    return G, labels, p_in, p_out


def update_sparse_sbm(
    G,
    labels,
    mechanism,
    batch_size,
    seed=42
):

    rng = random.Random(seed)

    H = G.copy()

    updated_labels = labels.copy()

    blocks = sorted(set(labels.values()))

    members = {
        c: [u for u in G if labels[u] == c]
        for c in blocks
    }

    n = len(G)

    if mechanism == "community_focused":

        for u in range(n, n + batch_size):

            target = rng.choice(blocks)

            H.add_node(u)

            updated_labels[u] = target

            internal = rng.sample(
                members[target], 18
            )

            external_pool = [
                v
                for c in blocks
                if c != target
                for v in members[c]
            ]

            external = rng.sample(
                external_pool, 3
            )

            for v in internal + external:
                H.add_edge(u, v, weight=1.0)

    elif mechanism == "bridge":

        for u in range(n, n + batch_size):

            H.add_node(u)

            for c in blocks:

                neighbors = rng.sample(
                    members[c], 7
                )

                for v in neighbors:
                    H.add_edge(u, v, weight=1.0)

    elif mechanism == "existing_edge":

        required = 21 * batch_size

        added = 0

        while added < required:

            c1, c2 = rng.sample(blocks, 2)

            u = rng.choice(members[c1])
            v = rng.choice(members[c2])

            if not H.has_edge(u, v):

                H.add_edge(u, v, weight=1.0)

                added += 1

    else:

        raise ValueError("Unknown update mechanism.")

    return H, updated_labels
