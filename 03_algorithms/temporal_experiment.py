
import random
import time

import numpy as np
import pandas as pd
import networkx as nx

from louvain_strategies import (
    static_louvain,
    warm_louvain,
    adaptive_louvain
)

from evaluation_metrics import (
    evaluate_partition,
    efficiency_metrics,
    partition_ari
)

from louvain_core import initial_affected_region


def build_temporal_snapshots(
    interactions,
    interval_seconds
):

    if interval_seconds <= 0:
        raise ValueError("Invalid snapshot interval.")

    data = interactions[
        ["source", "target", "timestamp"]
    ].copy()

    data = data.sort_values(
        "timestamp",
        kind="stable"
    )

    start = int(data["timestamp"].min())

    data["window"] = (
        (data["timestamp"] - start)
        // interval_seconds
    ).astype(int)

    grouped = {
        int(w): group
        for w, group in data.groupby("window")
    }

    G = nx.Graph()

    snapshots = []
    records = []

    total_windows = int(data["window"].max()) + 1

    for window in range(total_windows):

        old_nodes = set(G.nodes())

        previous_edges = G.number_of_edges()

        group = grouped.get(window)

        interactions_count = 0

        new_edges = []

        if group is not None:

            interactions_count = len(group)

            for u, v in group[
                ["source", "target"]
            ].itertuples(index=False, name=None):

                G.add_nodes_from([u, v])

                if u == v:
                    continue

                if not G.has_edge(u, v):

                    G.add_edge(u, v, weight=1.0)

                    new_edges.append((u, v))

        new_nodes = set(G.nodes()) - old_nodes

        old_old_edges = sum(
            u in old_nodes and v in old_nodes
            for u, v in new_edges
        )

        affected = set(new_nodes)

        for u, v in new_edges:

            affected.add(u)
            affected.add(v)

        changed = bool(new_nodes or new_edges)

        metadata = {
            "window": window,
            "window_start_seconds":
                window * interval_seconds,
            "elapsed_seconds":
                (window + 1) * interval_seconds,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "interactions": interactions_count,
            "new_nodes": len(new_nodes),
            "new_edges": len(new_edges),
            "old_old_edges": old_old_edges,
            "affected_nodes": len(affected),
            "affected_fraction":
                len(affected) / len(G) if len(G) else 0,
            "topology_changed": changed
        }

        assert (
            G.number_of_edges() - previous_edges
            == len(new_edges)
        )

        records.append(metadata)

        if changed:

            snapshots.append({
                "graph": G.copy(),
                "metadata": metadata.copy()
            })

    return snapshots, pd.DataFrame(records)


def timed_strategies(
    G_previous,
    G_current,
    histories,
    seed,
    repeats
):

    strategies = {
        "static": lambda: static_louvain(
            G_current,
            seed=seed
        )
    }

    for protocol, history in histories.items():

        warm_previous = history["warm"]
        adaptive_previous = history["adaptive"]

        strategies[(protocol, "warm")] = (
            lambda p=warm_previous: warm_louvain(
                G_previous,
                G_current,
                p,
                seed=seed
            )
        )

        strategies[(protocol, "adaptive")] = (
            lambda p=adaptive_previous: adaptive_louvain(
                G_previous,
                G_current,
                p,
                seed=seed
            )
        )

    rng = random.Random(seed)

    partitions = {}
    diagnostics = {}

    timings = {
        key: [] for key in strategies
    }

    timing_records = []

    for repetition in range(repeats):

        order = list(strategies)

        rng.shuffle(order)

        for key in order:

            start = time.perf_counter()

            partition, diag = strategies[key]()

            elapsed = time.perf_counter() - start

            if key in partitions:

                assert partition == partitions[key]

                assert (
                    diag["work"]
                    == diagnostics[key]["work"]
                )

            else:

                partitions[key] = partition

                diagnostics[key] = diag

            timing = {
                "runtime_seconds": elapsed,
                "initialization_seconds":
                    diag["initialization_seconds"],
                "optimization_seconds":
                    diag["optimization_seconds"]
            }

            timings[key].append(timing)

            if key == "static":

                protocol = "shared_static"
                strategy = "static"

            else:

                protocol, strategy = key

            timing_records.append({
                "protocol": protocol,
                "strategy": strategy,
                "repetition": repetition,
                **timing
            })

    for key in strategies:

        diag = diagnostics[key].copy()

        for metric in [
            "runtime_seconds",
            "initialization_seconds",
            "optimization_seconds"
        ]:

            diag[metric] = float(np.median([
                row[metric]
                for row in timings[key]
            ]))

        diagnostics[key] = diag

    return partitions, diagnostics, timing_records


def run_temporal_seed(
    snapshots,
    dataset,
    reference_labels,
    seed,
    repeats=2
):

    if not snapshots:
        raise ValueError("No temporal snapshots.")

    first = snapshots[0]

    G_previous = first["graph"]

    initial_partition, _ = static_louvain(
        G_previous,
        seed=seed
    )

    previous_static = initial_partition.copy()

    previous_warm = initial_partition.copy()

    previous_adaptive = initial_partition.copy()

    records = []
    timing_records = []
    partition_records = []

    for strategy in ["static", "warm", "adaptive"]:

        for u, c in initial_partition.items():

            partition_records.append({
                "dataset": dataset,
                "seed": seed,
                "protocol": "initial",
                "strategy": strategy,
                "window": first["metadata"]["window"],
                "node": u,
                "community": c
            })

    for index, snapshot in enumerate(
        snapshots[1:],
        start=1
    ):

        G_current = snapshot["graph"]

        metadata = snapshot["metadata"]

        affected = initial_affected_region(
            G_previous,
            G_current
        )

        assert len(affected) == metadata["affected_nodes"]

        histories = {
            "common_history": {
                "warm": previous_static,
                "adaptive": previous_static
            },

            "independent": {
                "warm": previous_warm,
                "adaptive": previous_adaptive
            }
        }

        partitions, diagnostics, timings = (
            timed_strategies(
                G_previous,
                G_current,
                histories,
                seed=seed + index,
                repeats=repeats
            )
        )

        static_partition = partitions["static"]

        base_metadata = {
            "dataset": dataset,
            "seed": seed,
            "optimization_seed": seed + index,
            "snapshot_index": index,
            "previous_nodes": len(G_previous),
            "previous_edges":
                G_previous.number_of_edges(),
            **metadata
        }

        for protocol, history in histories.items():

            results = {}

            results["static"] = evaluate_partition(
                G_current,
                static_partition,
                diagnostics["static"],
                static_partition,
                previous_partition=previous_static,
                reference_labels=reference_labels
            )

            for strategy in ["warm", "adaptive"]:

                key = (protocol, strategy)

                results[strategy] = evaluate_partition(
                    G_current,
                    partitions[key],
                    diagnostics[key],
                    static_partition,
                    previous_partition=history[strategy],
                    reference_labels=reference_labels
                )

            for strategy, result in results.items():

                result.update(
                    efficiency_metrics(
                        result,
                        results["static"],
                        results["warm"]
                    )
                )

                if strategy == "static":

                    partition = static_partition

                else:

                    partition = partitions[
                        (protocol, strategy)
                    ]

                result["ARI_vs_common_previous"] = (
                    partition_ari(
                        partition,
                        previous_static
                    )
                )

                if strategy == "static":

                    trajectory_ari = 1.0

                else:

                    trajectory_ari = partition_ari(
                        partitions[
                            ("common_history", strategy)
                        ],
                        partitions[
                            ("independent", strategy)
                        ]
                    )

                result["ARI_between_protocols"] = (
                    trajectory_ari
                )

                assert result["surviving_nodes"] == len(
                    G_previous
                )

                records.append({
                    **base_metadata,
                    "protocol": protocol,
                    "strategy": strategy,
                    **result
                })

                for u, c in partition.items():

                    partition_records.append({
                        "dataset": dataset,
                        "seed": seed,
                        "protocol": protocol,
                        "strategy": strategy,
                        "window": metadata["window"],
                        "node": u,
                        "community": c
                    })

        for row in timings:

            timing_records.append({
                **base_metadata,
                **row
            })

        previous_static = static_partition.copy()

        previous_warm = partitions[
            ("independent", "warm")
        ].copy()

        previous_adaptive = partitions[
            ("independent", "adaptive")
        ].copy()

        G_previous = G_current

        if index % 10 == 0:

            print(
                f"{dataset}, seed={seed}, "
                f"transition={index}/{len(snapshots)-1}",
                flush=True
            )

    return records, timing_records, partition_records
