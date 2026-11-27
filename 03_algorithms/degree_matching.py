
import random
from collections import defaultdict


def create_degree_matched_pairs(
    G,
    participation,
    min_gap=0.10,
    seed=42
):

    rng = random.Random(seed)

    degree_groups = defaultdict(list)

    for u, degree in G.degree():

        if degree > 0:
            degree_groups[degree].append(u)

    pairs = []

    for degree, nodes in sorted(degree_groups.items()):

        rng.shuffle(nodes)

        ranked = sorted(
            nodes,
            key=lambda u: participation[u]
        )

        left = 0
        right = len(ranked) - 1

        while left < right:

            low = ranked[left]
            high = ranked[right]

            gap = (
                participation[high]
                - participation[low]
            )

            if gap < min_gap:
                break

            pairs.append({
                "high_node": high,
                "low_node": low,
                "degree": degree,
                "high_participation": participation[high],
                "low_participation": participation[low],
                "participation_gap": gap
            })

            left += 1
            right -= 1

    return pairs


def sample_matched_arrivals(
    pairs,
    batch_size,
    seed=42
):

    if not 0 < batch_size <= len(pairs):
        raise ValueError("Invalid matched batch size.")

    rng = random.Random(seed)

    selected = rng.sample(pairs, batch_size)

    high = [
        pair["high_node"]
        for pair in selected
    ]

    low = [
        pair["low_node"]
        for pair in selected
    ]

    return high, low, selected
