
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    "static": "#3569B0",
    "warm": "#E68A00",
    "adaptive": "#15976D",
    "bridge": "#D1495B",
    "random": "#3569B0",
    "community_focused": "#15976D"
}

LABELS = {
    "static": "Static",
    "warm": "Global warm-start",
    "adaptive": "Adaptive",
    "bridge": "Bridge",
    "random": "Random",
    "community_focused": "Community-focused"
}

MARKERS = {
    "static": "o",
    "warm": "s",
    "adaptive": "^",
    "bridge": "o",
    "random": "s",
    "community_focused": "^"
}


def save_figure(fig, folder, filename):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()

    fig.savefig(
        folder / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        folder / f"{filename}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_strategy_comparison(
    raw,
    metric,
    ylabel,
    dataset,
    folder,
    reference_line=None
):

    mechanisms = [
        "random",
        "bridge",
        "community_focused"
    ]

    strategies = [
        "static",
        "warm",
        "adaptive"
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 4.8),
        sharey=True
    )

    for ax, mechanism in zip(axes, mechanisms):

        subset = raw[
            raw["mechanism"] == mechanism
        ]

        for strategy in strategies:

            data = subset[
                subset["strategy"] == strategy
            ]

            stats = data.groupby(
                "batch_size"
            )[metric].agg(["mean", "std"])

            x = stats.index.to_numpy(dtype=float)
            y = stats["mean"].to_numpy(dtype=float)
            sd = stats["std"].fillna(0).to_numpy(dtype=float)

            ax.plot(
                x, y,
                color=COLORS[strategy],
                marker=MARKERS[strategy],
                linewidth=2,
                label=LABELS[strategy]
            )

            ax.fill_between(
                x,
                y - sd,
                y + sd,
                color=COLORS[strategy],
                alpha=0.10
            )

        if reference_line is not None:

            ax.axhline(
                reference_line,
                color="gray",
                linestyle="--",
                linewidth=1
            )

        ax.set_title(LABELS[mechanism])
        ax.set_xlabel("Number of arriving nodes")
        ax.grid(alpha=0.2)
        ax.set_xticks(sorted(subset["batch_size"].unique()))

    axes[0].set_ylabel(ylabel)

    axes[0].legend(fontsize=9)

    fig.suptitle(
        f"{dataset}: {ylabel}",
        fontsize=15,
        fontweight="bold"
    )

    save_figure(
        fig,
        folder,
        f"strategy_{metric}"
    )


def plot_adaptive_mechanisms(
    raw,
    metric,
    ylabel,
    dataset,
    folder,
    statistic="mean",
    reference_line=None
):

    adaptive = raw[
        raw["strategy"] == "adaptive"
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for mechanism in [
        "random",
        "bridge",
        "community_focused"
    ]:

        subset = adaptive[
            adaptive["mechanism"] == mechanism
        ]

        grouped = subset.groupby(
            "batch_size"
        )[metric]

        if statistic == "median":

            center = grouped.median()
            lower = grouped.quantile(0.25)
            upper = grouped.quantile(0.75)

        else:

            center = grouped.mean()
            sd = grouped.std().fillna(0)

            lower = center - sd
            upper = center + sd

        x = center.index.to_numpy(dtype=float)

        ax.plot(
            x,
            center.to_numpy(dtype=float),
            color=COLORS[mechanism],
            marker=MARKERS[mechanism],
            linewidth=2,
            label=LABELS[mechanism]
        )

        ax.fill_between(
            x,
            lower.to_numpy(dtype=float),
            upper.to_numpy(dtype=float),
            color=COLORS[mechanism],
            alpha=0.12
        )

    if reference_line is not None:

        ax.axhline(
            reference_line,
            color="gray",
            linestyle="--",
            linewidth=1
        )

    ax.set_xlabel("Number of arriving nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset}: adaptive Louvain")
    ax.set_xticks(sorted(adaptive["batch_size"].unique()))
    ax.grid(alpha=0.2)
    ax.legend()

    save_figure(
        fig,
        folder,
        f"adaptive_{metric}"
    )


def plot_arrival_characteristics(
    arrivals,
    dataset,
    folder
):

    metrics = [
        ("mean_arrival_participation", "Mean participation"),
        ("mean_arrival_degree", "Mean arrival degree"),
        ("total_new_edges", "Number of new edges")
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4.5)
    )

    for ax, (metric, ylabel) in zip(axes, metrics):

        for mechanism in [
            "random",
            "bridge",
            "community_focused"
        ]:

            subset = arrivals[
                arrivals["mechanism"] == mechanism
            ]

            stats = subset.groupby(
                "batch_size"
            )[metric].mean()

            ax.plot(
                stats.index,
                stats.values,
                marker=MARKERS[mechanism],
                color=COLORS[mechanism],
                linewidth=2,
                label=LABELS[mechanism]
            )

        ax.set_xlabel("Number of arriving nodes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(arrivals["batch_size"].unique()))
        ax.grid(alpha=0.2)

    axes[0].legend()

    fig.suptitle(
        f"{dataset}: arrival characteristics",
        fontsize=15
    )

    save_figure(
        fig,
        folder,
        "arrival_characteristics"
    )


def generate_arrival_figures(
    raw,
    arrivals,
    dataset,
    folder
):

    strategy_metrics = [
        ("modularity", "Modularity", None),
        ("modularity_gap", "Modularity gap", 0),
        ("ARI_vs_static", "ARI vs static", None),
        ("ARI_vs_reference", "ARI vs reference", None),
        ("old_node_stability", "Old-node stability", None),
        ("matched_perturbation", "Matched perturbation", None),
        ("node_visits", "Node visits", None),
        ("candidate_evaluations", "Candidate evaluations", None),
        ("total_measured_scans", "Measured scans", None),
        ("runtime_seconds", "Runtime (seconds)", None),
        ("disconnected_communities", "Disconnected communities", None)
    ]

    for metric, ylabel, reference in strategy_metrics:

        plot_strategy_comparison(
            raw,
            metric,
            ylabel,
            dataset,
            folder,
            reference_line=reference
        )

    adaptive_metrics = [
        ("old_node_stability", "Old-node stability", "mean", None),
        ("matched_perturbation", "Matched perturbation", "mean", None),
        ("ARI_vs_static", "ARI vs static", "mean", None),
        ("initial_active_fraction", "Initial affected fraction", "mean", None),
        ("level0_visited_fraction", "Level-0 visited fraction", "mean", None),
        ("node_visit_ratio_vs_static", "Node visit ratio", "mean", 1),
        ("candidate_ratio_vs_static", "Candidate evaluation ratio", "mean", 1),
        ("scan_ratio_vs_static", "Measured scan ratio", "mean", 1),
        ("speedup_vs_static", "Speedup vs static", "median", 1),
        ("speedup_vs_warm", "Speedup vs warm-start", "median", 1)
    ]

    for metric, ylabel, statistic, reference in adaptive_metrics:

        plot_adaptive_mechanisms(
            raw,
            metric,
            ylabel,
            dataset,
            folder,
            statistic=statistic,
            reference_line=reference
        )

    plot_arrival_characteristics(
        arrivals,
        dataset,
        folder
    )
