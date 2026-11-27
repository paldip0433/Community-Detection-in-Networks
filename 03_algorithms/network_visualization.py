
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pathlib import Path


def plot_network(G, name, save_path, labels=None, seed=42):

    fig, ax = plt.subplots(figsize=(12, 10))

    n = G.number_of_nodes()

    # Reproducible force-directed layout
    pos = nx.spring_layout(
        G,
        seed=seed,
        iterations=100,
        weight=None
    )

    # Node size according to degree
    degrees = dict(G.degree())

    node_sizes = [
        20 + 8 * np.sqrt(degrees[node])
        for node in G.nodes()
    ]

    if n <= 50:
        node_sizes = [
            100 + 30 * np.sqrt(degrees[node])
            for node in G.nodes()
        ]

    # Assign colors using reference labels
    if labels is not None:

        communities = sorted(set(
            labels[node]
            for node in G.nodes()
            if node in labels
        ))

        community_index = {
            c: i for i, c in enumerate(communities)
        }

        cmap = plt.get_cmap(
            "turbo",
            len(communities)
        )

        node_colors = [
            cmap(community_index[labels[node]])
            if node in labels else "gray"
            for node in G.nodes()
        ]

    else:

        node_colors = "steelblue"

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        alpha=0.35 if n <= 50 else 0.08,
        width=0.8 if n <= 50 else 0.4
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=0.4
    )

    # Display node IDs only for small networks
    if n <= 50:

        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=8
        )

    ax.set_title(
        f"{name}\n"
        f"Nodes = {G.number_of_nodes():,} | "
        f"Edges = {G.number_of_edges():,}",
        fontsize=16,
        fontweight="bold"
    )

    ax.axis("off")

    fig.tight_layout()

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        save_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        save_path.with_suffix(".pdf"),
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(f"Saved: {save_path.name}.png and .pdf")
