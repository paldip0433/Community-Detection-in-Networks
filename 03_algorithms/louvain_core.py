
"""
Unified Louvain engine for static and dynamic networks.

MSc Project: Community Detection in Networks

Part 1: Mathematical foundation and initialization.
"""

import math
import random
import networkx as nx

from collections import defaultdict

from legacy_louvain import (
    modularity,
    relabel,
    aggregate_graph
)



# Graph validation


def validate_graph(G):

    if G.is_directed():
        raise ValueError("Graph must be undirected.")

    if G.is_multigraph():
        raise ValueError("Multigraphs are not supported.")

    for u, v, data in G.edges(data=True):

        w = data.get("weight", 1.0)

        if not math.isfinite(w) or w < 0:

            raise ValueError(
                f"Invalid edge weight on ({u}, {v}): {w}"
            )

    return True



#  Partition validation


def validate_partition(G, partition):

    if set(partition) != set(G.nodes()):

        missing = set(G.nodes()) - set(partition)

        extra = set(partition) - set(G.nodes())

        raise ValueError(
            f"Invalid partition: "
            f"{len(missing)} missing nodes, "
            f"{len(extra)} extra nodes."
        )

    return True


# Singleton initialization


def singleton_partition(G):

    return {
        u: i
        for i, u in enumerate(G.nodes())
    }



#  Warm-start initialization


def warm_start_partition(G, previous_partition):

    partition = {}

    label_map = {}

    next_label = 0

    # Preserve the previous community memberships

    for u in G.nodes():

        if u in previous_partition:

            old_label = previous_partition[u]

            if old_label not in label_map:

                label_map[old_label] = next_label

                next_label += 1

            partition[u] = label_map[old_label]

    # Assign each new node a distinct singleton

    for u in G.nodes():

        if u not in partition:

            partition[u] = next_label

            next_label += 1

    return partition



# Community degree totals


def community_degree_totals(G, partition):

    totals = defaultdict(float)

    for u, degree in G.degree(weight="weight"):

        totals[partition[u]] += degree

    return totals



#  Neighbor-community weights


def neighbor_community_weights(G, u, partition):

    weights = defaultdict(float)

    for v, data in G[u].items():

        # Self-loops cancel in move comparisons

        if v == u:
            continue

        w = data.get("weight", 1.0)

        weights[partition[v]] += w

    return weights


#  Analytical modularity gain


def modularity_gain(G, partition, u, target):

    old = partition[u]

    if old == target:
        return 0.0

    m = G.size(weight="weight")

    if m == 0:
        return 0.0

    ku = G.degree(u, weight="weight")

    totals = community_degree_totals(
        G, partition
    )

    weights = neighbor_community_weights(
        G, u, partition
    )

    # Temporarily remove u from its community

    totals[old] -= ku

    old_score = (
        weights.get(old, 0.0) / m
        - ku * totals[old] / (2 * m * m)
    )

    target_score = (
        weights.get(target, 0.0) / m
        - ku * totals.get(target, 0.0)
        / (2 * m * m)
    )

    return target_score - old_score


# Initial affected region


def initial_affected_region(
    G_previous,
    G_current,
    weight_tolerance=0.0
):

    if weight_tolerance < 0:
        raise ValueError("Tolerance must be nonnegative.")

    previous_nodes = set(G_previous.nodes())

    current_nodes = set(G_current.nodes())

    new_nodes = current_nodes - previous_nodes

    affected = set(new_nodes)

    # Newly added edges and changed edge weights

    for u, v, data in G_current.edges(data=True):

        if not G_previous.has_edge(u, v):

            affected.update([u, v])

        else:

            old_weight = G_previous[u][v].get(
                "weight", 1.0
            )

            new_weight = data.get("weight", 1.0)

            if abs(new_weight - old_weight) > weight_tolerance:

                affected.update([u, v])

    # Deleted edges: activate surviving endpoints

    for u, v in G_previous.edges():

        if not G_current.has_edge(u, v):

            if u in current_nodes:
                affected.add(u)

            if v in current_nodes:
                affected.add(v)

    return affected


# Improved local moving algorithm


def local_move(
    G,
    initial_partition=None,
    active_nodes=None,
    mode="global",
    seed=42,
    max_passes=100,
    tolerance=1e-12,
    allow_singletons=True,
    verify_moves=False
):

   
    #  Input validation


    validate_graph(G)

    if mode not in {"global", "adaptive"}:

        raise ValueError(
            "Mode must be global or adaptive."
        )

    if max_passes < 1:
        raise ValueError("max_passes must be positive.")

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid tolerance.")

    rng = random.Random(seed)

    nodes = list(G.nodes())
    node_rank = {u: i for i, u in enumerate(nodes)}

    node_set = set(nodes)

   
    # Partition initialization
    

    if initial_partition is None:

        part = singleton_partition(G)

    else:

        validate_partition(G, initial_partition)

        part = initial_partition.copy()

    # Convert community labels to consecutive integers

    label_map = {}

    for u in nodes:

        c = part[u]

        if c not in label_map:

            label_map[c] = len(label_map)

        part[u] = label_map[c]

    next_label = len(label_map)

   
    # Initial active region
   

    if mode == "global":

        active = set(nodes)

    else:

        active = (
            set(nodes)
            if active_nodes is None
            else set(active_nodes)
        )

        if not active.issubset(node_set):

            raise ValueError(
                "Active region contains unknown nodes."
            )

    initial_active = active.copy()

    visited_nodes = set()

    activated_nodes = active.copy()

    moved_nodes = set()

   
    # Computational diagnostics
   

    work = {

        "preprocessing_scans": 0,

        "local_neighbor_scans": 0,

        "activation_scans": 0,

        "node_visits": 0,

        "candidate_evaluations": 0,

        "accepted_moves": 0,

        "singleton_creations": 0,

        "passes": 0

    }

    move_history = []

    pass_history = []

    m = G.size(weight="weight")

    # Handle graphs without positive edge weight

    if m == 0:

        return part, {

            "work": work,

            "initial_active": initial_active,

            "visited_nodes": visited_nodes,

            "activated_nodes": activated_nodes,

            "moved_nodes": moved_nodes,

            "move_history": move_history,

            "pass_history": pass_history,

            "stop_reason": "zero_total_weight"

        }

    # Precompute weighted degrees and adjacency
   

    degrees = dict(
        G.degree(weight="weight")
    )

    community_degree = defaultdict(float)

    community_size = defaultdict(int)

    for u in nodes:

        community_degree[part[u]] += degrees[u]

        community_size[part[u]] += 1

    adjacency = {}

    for u in nodes:

        adjacency[u] = []

        for v, data in G[u].items():

            work["preprocessing_scans"] += 1

            # Self-loops cancel in move comparisons

            if u == v:
                continue

            w = data.get("weight", 1.0)

            adjacency[u].append((v, w))

    # Local optimization passes
   

    stop_reason = "max_passes"

    for pass_number in range(max_passes):

        if not active:

            stop_reason = "empty_active_region"

            break

        work["passes"] += 1

        # Use graph node order before random shuffling
        # for reproducibility

        if mode == "global":
            order = nodes.copy()
        elif len(active) * max(1, math.log2(max(1, len(active)))) < len(nodes):
            order = sorted(active, key=node_rank.__getitem__)
        else:
            order = [u for u in nodes if u in active]

        rng.shuffle(order)

        next_active = set()

        moves_this_pass = 0

        gain_this_pass = 0.0

     
        # Examine each currently active node
      

        for u in order:

            visited_nodes.add(u)

            work["node_visits"] += 1

            old = part[u]

            ku = degrees[u]

            if ku == 0:
                continue

            # Calculate edge weights from u to
            # neighboring communities

            weights = defaultdict(float)

            for v, w in adjacency[u]:

                work["local_neighbor_scans"] += 1

                weights[part[v]] += w

            # Temporarily remove u

            community_degree[old] -= ku

            community_size[old] -= 1

            if community_size[old] == 0:

                community_degree[old] = 0.0

            # Score of returning to old community

            old_score = (

                weights.get(old, 0.0) / m

                - ku * community_degree[old]
                / (2 * m * m)

            )

            best = old

            best_score = old_score

            work["candidate_evaluations"] += 1

        
            # Evaluate neighboring communities
            

            candidates = sorted(weights)

            for c in candidates:

                if c == old:
                    continue

                score = (

                    weights[c] / m

                    - ku * community_degree[c]
                    / (2 * m * m)

                )

                work["candidate_evaluations"] += 1

                if score > best_score:

                    best = c

                    best_score = score

            # Evaluate a new singleton community
          

            create_singleton = False

            if allow_singletons:

                if community_size[old] > 0:

                    work["candidate_evaluations"] += 1

                    singleton_score = 0.0

                    if singleton_score > best_score:

                        best = next_label

                        best_score = singleton_score

                        create_singleton = True

           
            # Accept only a strictly improving move
           

            gain = best_score - old_score

            accepted = (

                best != old

                and gain > tolerance

            )

            if not accepted:

                best = old

                create_singleton = False

            # Optional expensive verification

            if accepted and verify_moves:

                Q_before = modularity(G, part)

            # Reinsert u into selected community

            part[u] = best

            community_degree[best] += ku

            community_size[best] += 1

            if accepted:

                if create_singleton:

                    next_label += 1

                    work["singleton_creations"] += 1

                moved_nodes.add(u)

                moves_this_pass += 1

                gain_this_pass += gain

                work["accepted_moves"] += 1

                # Verify analytical gain against
                # direct modularity recomputation

                if verify_moves:

                    Q_after = modularity(G, part)

                    actual_gain = Q_after - Q_before

                    error = abs(actual_gain - gain)

                    assert error < 1e-10, (

                        f"Gain mismatch at node {u}: "
                        f"{gain} versus {actual_gain}"

                    )

                    assert actual_gain >= -1e-10

                    move_history.append({

                        "pass": pass_number + 1,

                        "node": u,

                        "old_community": old,

                        "new_community": best,

                        "predicted_gain": gain,

                        "actual_gain": actual_gain,

                        "error": error

                    })

                # Adaptive region expansion

                if mode == "adaptive":

                    next_active.add(u)

                    for v, _ in adjacency[u]:

                        work["activation_scans"] += 1

                        next_active.add(v)

        
        # End of one optimization pass
       

        pass_history.append({

            "pass": pass_number + 1,

            "active_nodes": len(active),

            "moves": moves_this_pass,

            "modularity_gain": gain_this_pass

        })

        if moves_this_pass == 0:

            stop_reason = "no_improving_moves"

            break

        if mode == "global":

            active = node_set.copy()

        else:

            active = next_active

            activated_nodes.update(next_active)

    # Return partition and diagnostics
   

    diagnostics = {

        "work": work,

        "initial_active": initial_active,

        "visited_nodes": visited_nodes,

        "activated_nodes": activated_nodes,

        "moved_nodes": moved_nodes,

        "move_history": move_history,

        "pass_history": pass_history,

        "stop_reason": stop_reason

    }

    return part, diagnostics



#  Improved multilevel Louvain algorithm


def louvain(
    G,
    initial_partition=None,
    active_nodes=None,
    mode="global",
    seed=42,
    max_levels=20,
    max_passes=100,
    tolerance=1e-12,
    allow_singletons=True,
    verify_moves=False
):

    validate_graph(G)

    if mode not in {"global", "adaptive"}:
        raise ValueError("Invalid optimization mode.")

    if max_levels < 1:
        raise ValueError("max_levels must be positive.")

    if max_passes < 1:
        raise ValueError("max_passes must be positive.")

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid tolerance.")

   
    # Initialization
   

    current = G.copy()

    original_nodes = set(G.nodes())

    members = {
        u: {u}
        for u in current.nodes()
    }

    if initial_partition is None:

        partition = singleton_partition(current)

    else:

        validate_partition(current, initial_partition)

        partition = initial_partition.copy()

    if mode == "global" or active_nodes is None:

        active = set(current.nodes())

    else:

        active = set(active_nodes)

        if not active.issubset(original_nodes):

            raise ValueError(
                "Active region contains unknown nodes."
            )

    initial_active = active.copy()

    initial_Q = modularity(G, partition)

   
    # Multilevel diagnostics
    

    total_work = {

        "preprocessing_scans": 0,

        "local_neighbor_scans": 0,

        "activation_scans": 0,

        "aggregation_edge_scans": 0,

        "coarse_activation_scans": 0,

        "node_visits": 0,

        "candidate_evaluations": 0,

        "accepted_moves": 0,

        "singleton_creations": 0,

        "passes": 0

    }

    visited_original = set()

    activated_original = set()

    moved_original = set()

    level_records = []

    modularity_history = [initial_Q]

    stop_reason = "max_levels"

   
    # Multilevel optimization
    

    for level in range(max_levels):

        Q_before = modularity(
            current, partition
        )

        # Phase I: Local moving

        partition, diag = local_move(

            current,

            initial_partition=partition,

            active_nodes=active,

            mode=mode,

            seed=seed + level,

            max_passes=max_passes,

            tolerance=tolerance,

            allow_singletons=allow_singletons,

            verify_moves=verify_moves

        )

        Q_after = modularity(
            current, partition
        )

        if verify_moves:

            assert Q_after >= Q_before - 1e-10

        modularity_history.append(Q_after)

        # Collect work counters

        for key, value in diag["work"].items():

            total_work[key] += value

        # Translate diagnostic sets to original nodes

        for u in diag["visited_nodes"]:

            visited_original.update(members[u])

        for u in diag["activated_nodes"]:

            activated_original.update(members[u])

        for u in diag["moved_nodes"]:

            moved_original.update(members[u])

        # Record level-specific information

        n = current.number_of_nodes()

        k = len(set(partition.values()))

        level_records.append({

            "level": level,

            "graph_nodes": n,

            "graph_edges": current.number_of_edges(),

            "communities": k,

            "initial_active_nodes":

                len(diag["initial_active"]),

            "visited_nodes":

                len(diag["visited_nodes"]),

            "activated_nodes":

                len(diag["activated_nodes"]),

            "represented_original_nodes_visited":

                sum(
                    len(members[u])
                    for u in diag["visited_nodes"]
                ),

            "modularity_before": Q_before,

            "modularity_after": Q_after,

            "modularity_gain": Q_after - Q_before,

            "local_stop_reason": diag["stop_reason"],

            **diag["work"]

        })

    
        # Termination conditions
      

        if current.size(weight="weight") == 0:

            stop_reason = "zero_total_weight"

            break

        if mode == "adaptive" and not active:

            stop_reason = "empty_active_region"

            break

        if k <= 1:

            stop_reason = "single_community"

            break

        if k == n:

            stop_reason = "no_further_contraction"

            break

        if level + 1 == max_levels:

            stop_reason = "max_levels"

            break

       
        # Phase II: Graph aggregation
      

        total_work["aggregation_edge_scans"] += (
            current.number_of_edges()
        )

        new_graph = aggregate_graph(
            current, partition
        )

        # Construct original-node membership mapping

        new_members = {
            c: set()
            for c in partition.values()
        }

        for u, c in partition.items():

            new_members[c].update(members[u])

        
        #  Propagate the active region
       

        if mode == "global":

            new_active = set(new_graph.nodes())

        else:

            active_supernodes = {

                partition[u]

                for u in diag["activated_nodes"]

            }

            new_active = active_supernodes.copy()

            # One-hop expansion on contracted graph

            for c in active_supernodes:

                for neighbor in new_graph.neighbors(c):

                    total_work[
                        "coarse_activation_scans"
                    ] += 1

                    new_active.add(neighbor)

       
        #  Initialize the next level
     

        current = new_graph

        members = new_members

        active = new_active

        partition = singleton_partition(current)

        if verify_moves:

            Q_contracted = modularity(
                current, partition
            )

            assert abs(
                Q_contracted - Q_after
            ) < 1e-10

   
    # Reconstruct original-node partition
   

    final_partition = {}

    for u, c in partition.items():

        for original in members[u]:

            final_partition[original] = c

    final_partition = relabel(final_partition)

    validate_partition(G, final_partition)

    final_Q = modularity(
        G, final_partition
    )

    if verify_moves:

        assert final_Q >= initial_Q - 1e-10

        assert abs(
            final_Q - modularity_history[-1]
        ) < 1e-10

    
    # Final diagnostics
  

    N = len(G)

    diagnostics = {

        "mode": mode,

        "levels": len(level_records),

        "initial_modularity": initial_Q,

        "final_modularity": final_Q,

        "modularity_history": modularity_history,

        "communities":

            len(set(final_partition.values())),

        "initial_active_original": initial_active,

        "visited_original": visited_original,

        "activated_original": activated_original,

        "moved_original": moved_original,

        "initial_active_fraction":

            len(initial_active) / N if N else 0.0,

        "visited_fraction":

            len(visited_original) / N if N else 0.0,

        "activated_fraction":

            len(activated_original) / N if N else 0.0,

        "stop_reason": stop_reason,

        "local_pass_limit_reached": any(

            record["local_stop_reason"] == "max_passes"

            for record in level_records

        ),

        "level_records": level_records,

        "work": total_work

    }

    return final_partition, diagnostics
