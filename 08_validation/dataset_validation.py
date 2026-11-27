
import numpy as np
import pandas as pd
import networkx as nx


# Construct a simple undirected aggregate graph
def aggregate_temporal_graph(interactions):

    G = nx.Graph()

    G.add_edges_from(
        interactions[["source", "target"]].itertuples(
            index=False, name=None
        )
    )

    G.remove_edges_from(list(nx.selfloop_edges(G)))

    nx.set_edge_attributes(G, 1.0, "weight")

    return G


# Calculate structural network statistics
def graph_statistics(G, name, labels=None):

    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = np.array(
        [degree for node, degree in G.degree()]
    )

    components = list(nx.connected_components(G))

    largest_component = max(
        (len(c) for c in components),
        default=0
    )

    stats = {
        "dataset": name,
        "nodes": n,
        "edges": m,
        "average_degree": degrees.mean(),
        "degree_std": degrees.std(),
        "minimum_degree": degrees.min(),
        "maximum_degree": degrees.max(),
        "density": nx.density(G),
        "connected_components": len(components),
        "largest_component": largest_component,
        "largest_component_fraction": largest_component / n,
        "isolated_nodes": nx.number_of_isolates(G),
        "self_loops": nx.number_of_selfloops(G),
        "average_clustering": nx.average_clustering(G)
    }

    if labels is not None:

        observed_labels = {
            labels[node]
            for node in G.nodes()
            if node in labels
        }

        stats["reference_communities"] = len(observed_labels)

        stats["missing_labels"] = sum(
            node not in labels for node in G.nodes()
        )

        stats["unused_labels"] = len(
            set(labels) - set(G.nodes())
        )

    else:

        stats["reference_communities"] = np.nan
        stats["missing_labels"] = np.nan
        stats["unused_labels"] = np.nan

    return stats


# Calculate temporal dataset statistics
def temporal_statistics(interactions, name):

    nodes = set(interactions["source"]).union(
        interactions["target"]
    )

    start = interactions["timestamp"].min()
    end = interactions["timestamp"].max()

    return {
        "dataset": name,
        "observed_nodes": len(nodes),
        "total_interactions": len(interactions),
        "unique_timestamps": interactions["timestamp"].nunique(),
        "first_timestamp": start,
        "last_timestamp": end,
        "duration_seconds": end - start,
        "duration_hours": (end - start) / 3600,
        "duration_days": (end - start) / 86400,
        "missing_values": interactions.isna().sum().sum(),
        "duplicate_records": interactions.duplicated().sum(),
        "self_interactions": (
            interactions["source"] == interactions["target"]
        ).sum()
    }
