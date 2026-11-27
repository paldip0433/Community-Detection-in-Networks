
import random
import time
import numpy as np
import networkx as nx

from louvain_strategies import (
    static_louvain,
    warm_louvain,
    adaptive_louvain
)

from evaluation_metrics import (
    evaluate_partition,
    efficiency_metrics
)


def participation_coefficients(G, labels):

    participation = {}

    for u in G.nodes():

        degree = G.degree(u)

        if degree == 0:
            participation[u] = 0.0
            continue

        counts = {}

        for v in G.neighbors(u):

            c = labels[v]

            counts[c] = counts.get(c, 0) + 1

        participation[u] = 1 - sum(
            (count / degree) ** 2
            for count in counts.values()
        )

    return participation


def select_arrival_nodes(
    G,
    participation,
    batch_size,
    mechanism,
    seed=42,
    pool_fraction=0.4
):

    rng = random.Random(seed)

    nodes = list(G.nodes())

    if not 0 < batch_size < len(nodes):
        raise ValueError("Invalid batch size.")

    if not 0 < pool_fraction <= 1:
        raise ValueError("Invalid pool fraction.")

    if mechanism == "random":

        pool = nodes

    elif mechanism in {
        "bridge", "community_focused"
    }:

        rng.shuffle(nodes)

        ranked = sorted(
            nodes,
            key=lambda u: participation[u],
            reverse=(mechanism == "bridge")
        )

        pool_size = max(
            batch_size,
            int(np.ceil(pool_fraction * len(nodes)))
        )

        pool = ranked[:pool_size]

    else:

        raise ValueError("Unknown arrival mechanism.")

    return rng.sample(pool, batch_size)


def construct_arrival_graphs(G, arrivals):

    arrivals = set(arrivals)

    if not arrivals.issubset(G.nodes()):
        raise ValueError("Unknown arrival nodes.")

    old_nodes = [
        u for u in G.nodes()
        if u not in arrivals
    ]

    G_previous = G.subgraph(old_nodes).copy()

    G_current = G.copy()

    return G_previous, G_current


def run_comparison(
    G_previous,
    G_current,
    previous_partition=None,
    reference_labels=None,
    seed=42,
    repeats=3
):

    if repeats < 1:
        raise ValueError("repeats must be positive.")

    if previous_partition is None:

        previous_partition, _ = static_louvain(
            G_previous,
            seed=seed
        )

    strategies = {

        "static": lambda: static_louvain(
            G_current,
            seed=seed
        ),

        "warm": lambda: warm_louvain(
            G_previous,
            G_current,
            previous_partition,
            seed=seed
        ),

        "adaptive": lambda: adaptive_louvain(
            G_previous,
            G_current,
            previous_partition,
            seed=seed
        )
    }

    rng = random.Random(seed)

    timings = {
        name: [] for name in strategies
    }

    partitions = {}

    diagnostics = {}

    timing_records = []

    for repetition in range(repeats):

        names = list(strategies)

        rng.shuffle(names)

        for name in names:

            start = time.perf_counter()

            partition, diag = strategies[name]()

            elapsed = time.perf_counter() - start

            if name in partitions:

                assert partition == partitions[name]

                assert (
                    diag["work"]
                    == diagnostics[name]["work"]
                )

            else:

                partitions[name] = partition

                diagnostics[name] = diag

            timings[name].append({
                "runtime_seconds": elapsed,
                "initialization_seconds":
                    diag["initialization_seconds"],
                "optimization_seconds":
                    diag["optimization_seconds"]
            })

            timing_records.append({
                "strategy": name,
                "repetition": repetition,
                "runtime_seconds": elapsed,
                "initialization_seconds":
                    diag["initialization_seconds"],
                "optimization_seconds":
                    diag["optimization_seconds"]
            })

    results = {}

    for name in strategies:

        diag = diagnostics[name].copy()

        for key in [
            "runtime_seconds",
            "initialization_seconds",
            "optimization_seconds"
        ]:

            diag[key] = float(np.median([
                row[key] for row in timings[name]
            ]))

        result = evaluate_partition(
            G_current,
            partitions[name],
            diag,
            partitions["static"],
            previous_partition=previous_partition,
            reference_labels=reference_labels
        )

        values = [
            row["runtime_seconds"]
            for row in timings[name]
        ]

        result["runtime_IQR"] = float(
            np.quantile(values, 0.75)
            - np.quantile(values, 0.25)
        )

        result["runtime_repeats"] = repeats

        results[name] = result

    for name in results:

        results[name].update(
            efficiency_metrics(
                results[name],
                results["static"],
                results["warm"]
            )
        )

        results[name]["strategy"] = name

    return (
        results,
        partitions,
        previous_partition,
        timing_records
    )


def arrival_statistics(G, arrivals, participation):

    arrivals = set(arrivals)

    degrees = [
        G.degree(u) for u in arrivals
    ]

    participation_values = [
        participation[u] for u in arrivals
    ]

    new_old_edges = sum(
        (u in arrivals) != (v in arrivals)
        for u, v in G.edges()
    )

    new_new_edges = sum(
        u in arrivals and v in arrivals
        for u, v in G.edges()
    )

    return {
        "batch_size": len(arrivals),
        "mean_arrival_degree": float(np.mean(degrees)),
        "mean_arrival_participation":
            float(np.mean(participation_values)),
        "new_old_edges": new_old_edges,
        "new_new_edges": new_new_edges,
        "total_new_edges": new_old_edges + new_new_edges
    }
