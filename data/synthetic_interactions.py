"""
Synthetic user-item interaction dataset.
Simulates implicit feedback (views, clicks, purchases) suitable for ALS training.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def generate_interactions(
    n_users: int = 5_000,
    n_items: int = 2_000,
    interactions_per_user: tuple = (5, 50),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates implicit feedback interactions.
    Returns a DataFrame of (user_id, item_id, rating, interaction_type).
    rating is implicit: view=1, click=2, purchase=3.
    """
    rng = np.random.default_rng(seed)

    # Latent user and item factors (simulate structure)
    n_factors = 20
    user_factors = rng.standard_normal((n_users, n_factors))
    item_factors = rng.standard_normal((n_items, n_factors))

    records = []
    for u in range(n_users):
        n_interactions = rng.integers(*interactions_per_user)
        # Users prefer items similar to their latent factor
        scores = user_factors[u] @ item_factors.T
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        items = rng.choice(n_items, size=n_interactions, replace=False, p=probs)
        for item in items:
            # Interaction type depends on affinity strength
            affinity = scores[item]
            if affinity > 1.0:
                itype, rating = "purchase", 3
            elif affinity > 0.0:
                itype, rating = "click", 2
            else:
                itype, rating = "view", 1
            records.append({"user_id": u, "item_id": int(item),
                            "rating": rating, "interaction_type": itype})

    df = pd.DataFrame(records)
    print(f"Generated {len(df):,} interactions | {n_users:,} users | {n_items:,} items")
    print(df["interaction_type"].value_counts().to_string())
    return df


def to_sparse_matrix(df: pd.DataFrame) -> csr_matrix:
    return csr_matrix(
        (df["rating"].values, (df["user_id"].values, df["item_id"].values))
    )