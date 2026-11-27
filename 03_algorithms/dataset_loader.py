
from pathlib import Path

import pandas as pd
import networkx as nx


ROOT = Path("/content/drive/MyDrive/MSc_Community_Detection")
RAW = ROOT / "01_datasets/raw"


# Zachary's Karate Club


def load_karate_club():

    G = nx.karate_club_graph()

    labels = {
        node: G.nodes[node]["club"]
        for node in G.nodes()
    }

    # Convert to an unweighted graph
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    return G, labels



#  Email-Eu-core


def load_email_eu_core():

    folder = RAW / "email_eu_core"

    edges = pd.read_csv(
        folder / "email-Eu-core.txt.gz",
        sep=r"\s+",
        header=None,
        names=["source", "target"]
    )

    metadata = pd.read_csv(
        folder / "email-Eu-core-department-labels.txt.gz",
        sep=r"\s+",
        header=None,
        names=["node", "department"]
    )

    labels = dict(
        zip(metadata["node"], metadata["department"])
    )

    # Convert directed edges into undirected edges
    G = nx.Graph()

    G.add_nodes_from(labels.keys())

    G.add_edges_from(
        edges[["source", "target"]].itertuples(
            index=False, name=None
        )
    )

    G.remove_edges_from(list(nx.selfloop_edges(G)))

    nx.set_edge_attributes(G, 1.0, "weight")

    return G, labels



# French High School 2013


def load_french_high_school():

    folder = RAW / "french_high_school"

    contacts = pd.read_csv(
        folder / "HighSchool2013_proximity_net.csv.gz",
        sep=r"\s+",
        header=None,
        names=[
            "timestamp",
            "source",
            "target",
            "source_class",
            "target_class"
        ]
    )

    metadata = pd.read_csv(
        folder / "HighSchool2013_metadata.csv"
    )

    contacts = contacts.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    labels = dict(
        zip(metadata["student_id"], metadata["class"])
    )

    return contacts, labels, metadata


# Email-Eu-core-temporal


def load_email_eu_core_temporal():

    folder = RAW / "email_eu_core_temporal"

    interactions = pd.read_csv(
        folder / "email-Eu-core-temporal.txt.gz",
        sep=r"\s+",
        header=None,
        names=["source", "target", "timestamp"]
    )

    interactions = interactions.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return interactions
