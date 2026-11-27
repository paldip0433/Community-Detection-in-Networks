
from time import perf_counter

from louvain_strategies import (
    static_louvain,
    adaptive_louvain
)

from connectivity_refinement import refine_connectivity


def hybrid_louvain(
    G_previous,
    G_current,
    previous_partition,
    transition_index,
    reset_interval=None,
    refine=False,
    seed=42
):

    if reset_interval is not None and reset_interval < 1:
        raise ValueError("Invalid reset interval.")

    start = perf_counter()

    reset = (
        reset_interval is not None
        and transition_index % reset_interval == 0
    )

    if reset:

        partition, diagnostics = static_louvain(
            G_current,
            seed=seed
        )

    else:

        partition, diagnostics = adaptive_louvain(
            G_previous,
            G_current,
            previous_partition,
            seed=seed
        )

    refinement_runtime = 0.0

    if refine:

        refinement_start = perf_counter()

        partition = refine_connectivity(
            G_current,
            partition
        )

        refinement_runtime = (
            perf_counter() - refinement_start
        )

    diagnostics["runtime_seconds"] = (
        perf_counter() - start
    )

    diagnostics["refinement_runtime"] = refinement_runtime

    diagnostics["reset_performed"] = reset

    diagnostics["refinement_enabled"] = refine

    return partition, diagnostics
