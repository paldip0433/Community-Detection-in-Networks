
import sys
import random
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "03_algorithms"))

from legacy_louvain import (
    modularity,
    aggregate_graph,
    local_move,
    louvain,
    matched_change_fraction
)

TOL = 1e-10

results = []


# Helper functions


def partition_to_sets(partition):

    groups = {}

    for node, community in partition.items():

        groups.setdefault(community, set()).add(node)

    return list(groups.values())


def random_partition(G, seed, K=4):

    rng = random.Random(seed)

    return {
        u: rng.randrange(K)
        for u in G.nodes()
    }


def singleton_partition(G):

    return {
        u: i
        for i, u in enumerate(G.nodes())
    }


def record(test, graph, seed, error, passed):

    results.append({

        "test": test,

        "graph": graph,

        "seed": seed,

        "error": float(error),

        "passed": bool(passed)

    })



# Construct test graphs


graphs = {}

# Unweighted graph
graphs["unweighted"] = nx.erdos_renyi_graph(
    30, 0.2, seed=42
)

#  Weighted graph
G_weighted = nx.erdos_renyi_graph(
    30, 0.2, seed=43
)

rng = random.Random(42)

for u, v in G_weighted.edges():

    G_weighted[u][v]["weight"] = rng.uniform(
        0.1, 5.0
    )

graphs["weighted"] = G_weighted

# Weighted graph with self-loops
G_loops = G_weighted.copy()

for u in range(10):

    G_loops.add_edge(
        u, u,
        weight=rng.uniform(0.1, 5.0)
    )

graphs["weighted_self_loops"] = G_loops

# Disconnected graph
graphs["disconnected"] = nx.disjoint_union(
    nx.cycle_graph(10),
    nx.complete_graph(8)
)

#  Graph with isolated nodes
G_isolated = nx.path_graph(15)

G_isolated.add_nodes_from(range(15, 25))

graphs["isolated_nodes"] = G_isolated

# Karate Club (unweighted)
G_karate = nx.karate_club_graph()

for u, v in G_karate.edges():

    G_karate[u][v]["weight"] = 1.0

graphs["karate_club"] = G_karate

# Stochastic Block Model
graphs["SBM"] = nx.stochastic_block_model(
    [20, 20, 20],
    [
        [0.30, 0.03, 0.03],
        [0.03, 0.30, 0.03],
        [0.03, 0.03, 0.30]
    ],
    seed=42
)



# TEST 1: Modularity agreement with NetworkX


for name, G in graphs.items():

    for seed in range(10):

        part = random_partition(G, seed)

        Q_ours = modularity(G, part)

        Q_nx = nx.community.modularity(
            G,
            partition_to_sets(part),
            weight="weight"
        )

        error = abs(Q_ours - Q_nx)

        record(
            "modularity_agreement",
            name, seed,
            error,
            error < TOL
        )



# TEST 2: Analytical modularity gain


for name, G in graphs.items():

    m = G.size(weight="weight")

    degrees = dict(G.degree(weight="weight"))

    for seed in range(10):

        part = random_partition(G, seed)

        Q_before = modularity(G, part)

        max_error = 0.0

        for u in G.nodes():

            old = part[u]

            ku = degrees[u]

            # Include a possible new singleton
            candidates = set(part.values())
            candidates.add(max(candidates) + 1)

            def insertion_score(c):

                k_u_c = sum(
                    data.get("weight", 1.0)
                    for v, data in G[u].items()
                    if v != u and part[v] == c
                )

                K_c = sum(
                    degrees[v]
                    for v in G.nodes()
                    if v != u and part[v] == c
                )

                return (
                    k_u_c / m
                    - ku * K_c / (2 * m * m)
                )

            for target in candidates:

                predicted_gain = (
                    insertion_score(target)
                    - insertion_score(old)
                )

                new_part = part.copy()

                new_part[u] = target

                actual_gain = (
                    modularity(G, new_part)
                    - Q_before
                )

                error = abs(
                    predicted_gain - actual_gain
                )

                max_error = max(max_error, error)

        record(
            "modularity_gain",
            name, seed,
            max_error,
            max_error < TOL
        )



# TEST 3: Aggregation invariance


for name, G in graphs.items():

    for seed in range(10):

        part = random_partition(G, seed)

        H = aggregate_graph(G, part)

        Q_original = modularity(G, part)

        Q_aggregated = modularity(
            H,
            singleton_partition(H)
        )

        error_Q = abs(
            Q_original - Q_aggregated
        )

        error_weight = abs(
            G.size(weight="weight")
            - H.size(weight="weight")
        )

        error_degree = max(

            abs(
                H.degree(c, weight="weight")
                - sum(
                    G.degree(u, weight="weight")
                    for u in G.nodes()
                    if part[u] == c
                )
            )

            for c in H.nodes()
        )

        record(
            "aggregation_modularity",
            name, seed,
            error_Q,
            error_Q < TOL
        )

        record(
            "aggregation_weight",
            name, seed,
            error_weight,
            error_weight < TOL
        )

        record(
            "aggregation_degree",
            name, seed,
            error_degree,
            error_degree < TOL
        )



# TEST 4: Local moving monotonicity


for name, G in graphs.items():

    for seed in range(10):

        initial = random_partition(G, seed)

        Q_before = modularity(G, initial)

        final, moved, passes = local_move(
            G,
            initial_partition=initial,
            seed=seed
        )

        Q_after = modularity(G, final)

        decrease = max(0.0, Q_before - Q_after)

        record(
            "local_move_monotonicity",
            name, seed,
            decrease,
            decrease < TOL
        )



# TEST 5: Multilevel Louvain validation


for name, G in graphs.items():

    for seed in range(10):

        initial = random_partition(G, seed)

        Q_before = modularity(G, initial)

        final, diagnostics = louvain(
            G,
            initial_partition=initial,
            seed=seed
        )

        valid_partition = (
            set(final.keys()) == set(G.nodes())
        )

        record(
            "partition_validity",
            name, seed,
            0 if valid_partition else 1,
            valid_partition
        )

        if valid_partition:

            Q_after = modularity(G, final)

            decrease = max(
                0.0, Q_before - Q_after
            )

            record(
                "multilevel_monotonicity",
                name, seed,
                decrease,
                decrease < TOL
            )



# Additional test: Label invariance


part_a = {
    0: 0, 1: 0,
    2: 1, 3: 1
}

part_b = {
    0: 8, 1: 8,
    2: 5, 3: 5
}

change = matched_change_fraction(
    part_a, part_b, list(part_a)
)

record(
    "label_invariance",
    "manual_example",
    0,
    abs(change),
    abs(change) < TOL
)


# Save and display test results


df = pd.DataFrame(results)

summary = df.groupby("test").agg(

    total_tests=("passed", "size"),

    passed_tests=("passed", "sum"),

    maximum_error=("error", "max")

)

summary["failed_tests"] = (
    summary["total_tests"]
    - summary["passed_tests"]
)

df.to_csv(
    ROOT / "10_validation/legacy_test_results.csv",
    index=False
)

summary.to_csv(
    ROOT / "10_validation/legacy_test_summary.csv"
)

print("\nLEGACY LOUVAIN VALIDATION")
print("=" * 75)

print(summary.to_string())

print("\nTotal tests:", len(df))
print("Passed:", int(df["passed"].sum()))
print("Failed:", int((~df["passed"]).sum()))

if df["passed"].all():

    print("\nALL TESTS PASSED.")

else:

    print("\nFAILED TEST DETAILS")
    print(df[~df["passed"]].to_string(index=False))
