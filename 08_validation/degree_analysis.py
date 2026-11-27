
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def analyze_degrees(graphs, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    colors = [
        "royalblue",
        "darkorange",
        "seagreen",
        "purple"
    ]

    
    # Individual degree distributions
    

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10)
    )

    for ax, (name, G), color in zip(
        axes.flat, graphs.items(), colors
    ):

        degrees = np.array(
            [d for _, d in G.degree()]
        )

        k, counts = np.unique(
            degrees, return_counts=True
        )

        probabilities = counts / len(degrees)

        ax.bar(
            k,
            probabilities,
            color=color,
            alpha=0.8
        )

        ax.axvline(
            degrees.mean(),
            color="red",
            linestyle="--",
            label=f"Mean = {degrees.mean():.2f}"
        )

        ax.set_title(name)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Empirical probability")
        ax.legend()
        ax.grid(alpha=0.2)

        summary.append({
            "dataset": name,
            "mean_degree": degrees.mean(),
            "median_degree": np.median(degrees),
            "degree_std": degrees.std(),
            "coefficient_of_variation":
                degrees.std() / degrees.mean(),
            "degree_90_percentile":
                np.percentile(degrees, 90),
            "degree_95_percentile":
                np.percentile(degrees, 95),
            "degree_99_percentile":
                np.percentile(degrees, 99),
            "maximum_degree": degrees.max()
        })

    fig.suptitle(
        "Degree Distributions of Real-World Networks",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(
        save_dir / "degree_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        save_dir / "degree_distributions.pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

  
    # Comparative CCDF
    

    fig, ax = plt.subplots(figsize=(10, 7))

    for (name, G), color in zip(
        graphs.items(), colors
    ):

        degrees = np.array(
            [d for _, d in G.degree()]
        )

        k, counts = np.unique(
            degrees, return_counts=True
        )

        ccdf = np.cumsum(
            counts[::-1]
        )[::-1] / len(degrees)

        # Logarithmic x-axis cannot display zero
        positive = k > 0

        ax.step(
            k[positive],
            ccdf[positive],
            where="pre",
            linewidth=2,
            color=color,
            label=name
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Degree k")
    ax.set_ylabel("P(K >= k)")
    ax.set_title("Complementary Cumulative Degree Distribution")

    ax.legend()
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        save_dir / "degree_ccdf.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        save_dir / "degree_ccdf.pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    return pd.DataFrame(summary)
