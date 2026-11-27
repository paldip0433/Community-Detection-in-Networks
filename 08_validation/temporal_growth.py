
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def calculate_temporal_growth(interactions, interval_seconds):

    data = interactions.sort_values("timestamp").copy()

    start = data["timestamp"].min()

    # Time intervals measured from first observation
    data["window"] = (
        (data["timestamp"] - start) // interval_seconds
    ).astype(int)

    grouped = {
        window: group
        for window, group in data.groupby("window")
    }

    observed_nodes = set()
    observed_edges = set()

    cumulative_interactions = 0

    results = []

    for window in range(data["window"].max() + 1):

        batch = grouped.get(window)

        batch_nodes = set()
        batch_edges = set()

        if batch is not None:

            batch_nodes = set(batch["source"]).union(
                batch["target"]
            )

            for u, v in batch[
                ["source", "target"]
            ].itertuples(index=False, name=None):

                if u != v:

                    edge = (min(u, v), max(u, v))
                    batch_edges.add(edge)

            interaction_count = len(batch)

        else:

            interaction_count = 0

        # Identify changes relative to previous snapshot
        new_nodes = batch_nodes - observed_nodes

        new_edges = batch_edges - observed_edges

        # New edges between previously observed nodes
        old_old_edges = sum(
            u in observed_nodes and v in observed_nodes
            for u, v in new_edges
        )

        # Initial affected region for unweighted updates
        affected_nodes = set(new_nodes)

        for u, v in new_edges:

            affected_nodes.add(u)
            affected_nodes.add(v)

        # Update cumulative network
        observed_nodes.update(batch_nodes)
        observed_edges.update(batch_edges)

        cumulative_interactions += interaction_count

        n = len(observed_nodes)
        m = len(observed_edges)

        results.append({

            "window": window + 1,

            "elapsed_hours":
                (window + 1) * interval_seconds / 3600,

            "elapsed_days":
                (window + 1) * interval_seconds / 86400,

            "interactions": interaction_count,

            "cumulative_interactions":
                cumulative_interactions,

            "new_nodes": len(new_nodes),

            "new_edges": len(new_edges),

            "new_old_old_edges": old_old_edges,

            "cumulative_nodes": n,

            "cumulative_edges": m,

            "initial_affected_nodes":
                len(affected_nodes),

            "initial_affected_fraction":
                len(affected_nodes) / n if n > 0 else 0,

            "density":
                2 * m / (n * (n - 1)) if n > 1 else 0

        })

    return pd.DataFrame(results)


def plot_temporal_growth(stats, name, save_dir, time_unit):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    time_column = f"elapsed_{time_unit}"

    t = stats[time_column]

    plots = [

        ("cumulative_nodes", "Cumulative nodes"),

        ("cumulative_edges", "Cumulative unique edges"),

        ("new_nodes", "Newly observed nodes"),

        ("new_edges", "Newly observed edges"),

        ("interactions", "Interactions per interval"),

        ("initial_affected_fraction",
         "Initial affected fraction")

    ]

    fig, axes = plt.subplots(
        3, 2, figsize=(15, 13)
    )

    colors = [
        "royalblue",
        "seagreen",
        "darkorange",
        "purple",
        "teal",
        "crimson"
    ]

    for ax, (column, label), color in zip(
        axes.flat, plots, colors
    ):

        if column in [
            "new_nodes", "new_edges", "interactions"
        ]:

            ax.bar(
                t,
                stats[column],
                width=0.8 * (t.iloc[1] - t.iloc[0])
                if len(t) > 1 else 0.8,
                color=color,
                alpha=0.8
            )

        else:

            ax.plot(
                t,
                stats[column],
                color=color,
                linewidth=2
            )

        ax.set_xlabel(f"Elapsed time ({time_unit})")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.2)

        if column == "initial_affected_fraction":
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        f"Temporal Network Growth: {name}",
        fontsize=18,
        fontweight="bold"
    )

    fig.tight_layout()

    filename = name.lower().replace(" ", "_")

    fig.savefig(
        save_dir / f"{filename}_growth.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        save_dir / f"{filename}_growth.pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)
