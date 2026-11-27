
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter


def analyze_community_structure(G, labels, name, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Verify reference label coverage
    assert all(node in labels for node in G.nodes())

    communities = sorted({
        labels[node] for node in G.nodes()
    })

    K = len(communities)

    community_index = {
        c: i for i, c in enumerate(communities)
    }

    groups = {
        c: set() for c in communities
    }

    for node in G.nodes():
        groups[labels[node]].add(node)

    sizes = np.array([
        len(groups[c]) for c in communities
    ])

   
    #Reference partition modularity
   

    Q_reference = nx.community.modularity(
        G,
        list(groups.values()),
        weight=None
    )

    
    #Community connectivity matrix
   

    edge_counts = np.zeros((K, K), dtype=int)

    for u, v in G.edges():

        a = community_index[labels[u]]
        b = community_index[labels[v]]

        edge_counts[a, b] += 1

        if a != b:
            edge_counts[b, a] += 1

    probability_matrix = np.full(
        (K, K), np.nan
    )

    for a in range(K):

        for b in range(K):

            if a == b:

                possible_edges = (
                    sizes[a] * (sizes[a] - 1) / 2
                )

            else:

                possible_edges = sizes[a] * sizes[b]

            if possible_edges > 0:

                probability_matrix[a, b] = (
                    edge_counts[a, b] / possible_edges
                )

    # Pooled within-community edge density
    internal_edges = np.trace(edge_counts)

    possible_internal = sum(
        n * (n - 1) / 2 for n in sizes
    )

    p_in = internal_edges / possible_internal

    # Pooled between-community edge density
    external_edges = G.number_of_edges() - internal_edges

    possible_external = (
        G.number_of_nodes() *
        (G.number_of_nodes() - 1) / 2
        - possible_internal
    )

    p_out = external_edges / possible_external

    
    # Participation coefficients
   

    node_results = []

    for node in G.nodes():

        degree = G.degree(node)

        neighbor_counts = Counter(
            labels[v] for v in G.neighbors(node)
        )

        if degree > 0:

            participation = 1 - sum(
                (count / degree)**2
                for count in neighbor_counts.values()
            )

        else:

            participation = 0.0

        node_results.append({

            "node": node,

            "reference_community": labels[node],

            "degree": degree,

            "participation_coefficient":
                participation,

            "neighbor_communities":
                len(neighbor_counts)

        })

    node_stats = pd.DataFrame(node_results)

    node_stats = node_stats.sort_values(
        ["participation_coefficient", "degree"],
        ascending=False
    )

   
    # Save numerical results
   

    filename = name.lower().replace(" ", "_")

    matrix_df = pd.DataFrame(
        probability_matrix,
        index=communities,
        columns=communities
    )

    matrix_df.to_csv(
        save_dir / f"{filename}_connectivity.csv"
    )

    node_stats.to_csv(
        save_dir / f"{filename}_participation.csv",
        index=False
    )

    pd.DataFrame({
        "community": communities,
        "size": sizes
    }).to_csv(
        save_dir / f"{filename}_community_sizes.csv",
        index=False
    )

    
    # Plot connectivity matrix
   

    fig, ax = plt.subplots(
        figsize=(12, 10) if K > 15 else (8, 7)
    )

    image = ax.imshow(
        probability_matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))

    ax.set_xticklabels(
        communities,
        rotation=90,
        fontsize=7 if K > 15 else 10
    )

    ax.set_yticklabels(
        communities,
        fontsize=7 if K > 15 else 10
    )

    if K <= 10:

        for a in range(K):

            for b in range(K):

                value = probability_matrix[a, b]

                if np.isfinite(value):

                    ax.text(
                        b, a,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white"
                        if value > 0.5 else "black"
                    )

    fig.colorbar(
        image,
        ax=ax,
        label="Empirical edge density"
    )

    ax.set_title(
        f"{name}: Reference Community Connectivity"
    )

    ax.set_xlabel("Reference community")
    ax.set_ylabel("Reference community")

    fig.tight_layout()

    fig.savefig(
        save_dir / f"{filename}_connectivity.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        save_dir / f"{filename}_connectivity.pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # Return summary
  

    summary = {

        "dataset": name,

        "reference_communities": K,

        "reference_modularity": Q_reference,

        "internal_edges": internal_edges,

        "external_edges": external_edges,

        "internal_edge_fraction":
            internal_edges / G.number_of_edges(),

        "empirical_p_in": p_in,

        "empirical_p_out": p_out,

        "p_in_p_out_ratio":
            p_in / p_out if p_out > 0 else np.inf,

        "mean_participation":
            node_stats["participation_coefficient"].mean(),

        "maximum_participation":
            node_stats["participation_coefficient"].max()

    }

    return summary, node_stats
