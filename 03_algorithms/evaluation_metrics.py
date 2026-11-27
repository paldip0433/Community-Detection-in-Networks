
"""
Evaluation metrics for dynamic community detection.

MSc Project: Community Detection in Networks.
"""

import numpy as np

from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

from louvain_core import modularity



#  Adjusted Rand Index


def partition_ari(part_a, part_b, nodes=None):

    if nodes is None:

        nodes = [
            u for u in part_a
            if u in part_b
        ]

    else:

        nodes = list(nodes)

    if not nodes:
        return np.nan

    labels_a = [part_a[u] for u in nodes]

    labels_b = [part_b[u] for u in nodes]

    return adjusted_rand_score(
        labels_a, labels_b
    )


#  Matched perturbation fraction


def matched_change_fraction(
    part_a,
    part_b,
    nodes=None
):

    if nodes is None:

        nodes = [
            u for u in part_a
            if u in part_b
        ]

    else:

        nodes = list(nodes)

    if not nodes:
        return np.nan

    labels_a = list(dict.fromkeys(
        part_a[u] for u in nodes
    ))

    labels_b = list(dict.fromkeys(
        part_b[u] for u in nodes
    ))

    index_a = {
        c: i for i, c in enumerate(labels_a)
    }

    index_b = {
        c: j for j, c in enumerate(labels_b)
    }

    contingency = np.zeros(
        (len(labels_a), len(labels_b)),
        dtype=np.int64
    )

    for u in nodes:

        i = index_a[part_a[u]]

        j = index_b[part_b[u]]

        contingency[i, j] += 1

    rows, columns = linear_sum_assignment(
        -contingency
    )

    matched = contingency[
        rows, columns
    ].sum()

    return 1.0 - matched / len(nodes)


# Safe division


def safe_ratio(numerator, denominator):

    if denominator == 0:
        return np.nan

    return numerator / denominator



#  Reference community agreement


def reference_agreement(
    partition,
    reference_labels
):

    if reference_labels is None:
        return np.nan

    common = [
        u for u in partition
        if u in reference_labels
    ]

    return partition_ari(
        partition,
        reference_labels,
        common
    )


#  Community connectivity


def community_connectivity(G, partition):

    import networkx as nx

    communities = {}

    for u, c in partition.items():

        communities.setdefault(c, set()).add(u)

    disconnected = 0

    disconnected_nodes = 0

    total_components = 0

    for nodes in communities.values():

        H = G.subgraph(nodes)

        components = nx.number_connected_components(H)

        total_components += components

        if components > 1:

            disconnected += 1

            disconnected_nodes += len(nodes)

    K = len(communities)

    return {

        "disconnected_communities":
            disconnected,

        "disconnected_community_fraction":
            safe_ratio(disconnected, K),

        "nodes_in_disconnected_communities":
            disconnected_nodes,

        "disconnected_node_fraction":
            safe_ratio(disconnected_nodes, len(G)),

        "total_induced_components":
            total_components

    }



#  Evaluate one detected partition


def evaluate_partition(
    G,
    partition,
    diagnostics,
    static_partition,
    previous_partition=None,
    reference_labels=None
):

    nodes = list(G.nodes())

    Q = modularity(G, partition)

    Q_static = modularity(
        G, static_partition
    )

    if previous_partition is None:

        stability = np.nan

        perturbation = np.nan

        surviving_nodes = 0

    else:

        common = [
            u for u in nodes
            if u in previous_partition
        ]

        surviving_nodes = len(common)

        stability = partition_ari(
            previous_partition,
            partition,
            common
        )

        perturbation = matched_change_fraction(
            previous_partition,
            partition,
            common
        )

    result = {

        "nodes": len(G),

        "edges": G.number_of_edges(),

        "communities":
            len(set(partition.values())),

        "modularity": Q,

        "modularity_gap": Q - Q_static,

        "ARI_vs_static":
            partition_ari(
                static_partition,
                partition,
                nodes
            ),

        "ARI_vs_reference":
            reference_agreement(
                partition,
                reference_labels
            ),

        "reference_nodes":
            sum(
                u in reference_labels
                for u in nodes
            ) if reference_labels is not None else 0,

        "old_node_stability": stability,

        "matched_perturbation": perturbation,

        "surviving_nodes": surviving_nodes,

        "runtime_seconds":
            diagnostics["runtime_seconds"],

        "initialization_seconds":
            diagnostics["initialization_seconds"],

        "optimization_seconds":
            diagnostics["optimization_seconds"],

        "levels":
            diagnostics["levels"],

        "initial_active_fraction":
            diagnostics["initial_active_fraction"],

        "level0_visited_fraction":
            diagnostics["level0_visited_fraction"],

        "represented_visited_fraction":
            diagnostics["represented_visited_fraction"],

        "total_measured_scans":
            diagnostics["total_measured_scans"],

        "stop_reason":
            diagnostics["stop_reason"],

        "local_pass_limit_reached":
            diagnostics["local_pass_limit_reached"],

        **diagnostics["work"]

    }

    result.update(
        community_connectivity(G, partition)
    )

    return result



# Compare computational efficiency


def efficiency_metrics(
    result,
    static_result,
    warm_result
):

    metrics = {}

    baselines = {

        "static": static_result,

        "warm": warm_result

    }

    work_keys = {

        "node_visit": "node_visits",

        "candidate": "candidate_evaluations",

        "scan": "total_measured_scans"

    }

    for name, baseline in baselines.items():

        metrics[f"speedup_vs_{name}"] = safe_ratio(

            baseline["runtime_seconds"],

            result["runtime_seconds"]

        )

        for metric, key in work_keys.items():

            metrics[f"{metric}_ratio_vs_{name}"] = (
                safe_ratio(
                    result[key],
                    baseline[key]
                )
            )

    return metrics
