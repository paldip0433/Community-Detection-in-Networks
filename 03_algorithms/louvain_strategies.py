
"""
Static, warm-start and adaptive Louvain interfaces.

MSc Project: Community Detection in Networks.
"""

from time import perf_counter

from louvain_core import (
    louvain,
    validate_graph,
    validate_partition,
    warm_start_partition,
    initial_affected_region
)



# Additional experimental diagnostics


def add_experiment_diagnostics(G, diagnostics):

    records = diagnostics["level_records"]

    N = len(G)

    level0 = records[0] if records else {}

    diagnostics["level0_visited_fraction"] = (

        level0.get("visited_nodes", 0) / N

        if N else 0.0

    )

    diagnostics["level0_node_visits"] = (

        level0.get("node_visits", 0)

    )

    diagnostics["represented_visited_fraction"] = (

        diagnostics["visited_fraction"]

    )

    work = diagnostics["work"]

    scan_keys = [

        "preprocessing_scans",

        "local_neighbor_scans",

        "activation_scans",

        "aggregation_edge_scans",

        "coarse_activation_scans"

    ]

    diagnostics["total_measured_scans"] = sum(

        work.get(key, 0)

        for key in scan_keys

    )

    return diagnostics



#  Static Louvain


def static_louvain(
    G,
    seed=42,
    **kwargs
):

    start = perf_counter()

    partition, diagnostics = louvain(

        G,

        initial_partition=None,

        active_nodes=None,

        mode="global",

        seed=seed,

        **kwargs

    )

    elapsed = perf_counter() - start

    diagnostics["strategy"] = "static"

    diagnostics["runtime_seconds"] = elapsed

    diagnostics["initialization_seconds"] = 0.0

    diagnostics["optimization_seconds"] = elapsed

    return partition, add_experiment_diagnostics(

        G, diagnostics

    )


#  Global warm-start Louvain


def warm_louvain(
    G_previous,
    G_current,
    previous_partition,
    seed=42,
    **kwargs
):

    start = perf_counter()

    validate_partition(
        G_previous, previous_partition
    )

    initial = warm_start_partition(

        G_current, previous_partition

    )

    initialization_end = perf_counter()

    partition, diagnostics = louvain(

        G_current,

        initial_partition=initial,

        active_nodes=None,

        mode="global",

        seed=seed,

        **kwargs

    )

    end = perf_counter()

    diagnostics["strategy"] = "warm"

    diagnostics["runtime_seconds"] = end - start

    diagnostics["initialization_seconds"] = (

        initialization_end - start

    )

    diagnostics["optimization_seconds"] = (

        end - initialization_end

    )

    return partition, add_experiment_diagnostics(

        G_current, diagnostics

    )


# Adaptive incremental Louvain


def adaptive_louvain(
    G_previous,
    G_current,
    previous_partition,
    seed=42,
    weight_tolerance=0.0,
    **kwargs
):

    start = perf_counter()

    validate_graph(G_previous)

    validate_partition(

        G_previous, previous_partition

    )

    initial = warm_start_partition(

        G_current, previous_partition

    )

    affected = initial_affected_region(

        G_previous,

        G_current,

        weight_tolerance=weight_tolerance

    )

    initialization_end = perf_counter()

    partition, diagnostics = louvain(

        G_current,

        initial_partition=initial,

        active_nodes=affected,

        mode="adaptive",

        seed=seed,

        **kwargs

    )

    end = perf_counter()

    diagnostics["strategy"] = "adaptive"

    diagnostics["runtime_seconds"] = end - start

    diagnostics["initialization_seconds"] = (

        initialization_end - start

    )

    diagnostics["optimization_seconds"] = (

        end - initialization_end

    )

    return partition, add_experiment_diagnostics(

        G_current, diagnostics

    )
