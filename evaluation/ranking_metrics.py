"""
Ranking evaluation metrics: NDCG@K, Precision@K, MAP, coverage, novelty.
All functions accept lists/arrays and are Spark-free for easy unit testing.
"""
import numpy as np
from collections import Counter


def dcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    relevances = [1 if item in relevant else 0 for item in recommended[:k]]
    ideal = sorted(relevances, reverse=True)
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def average_precision(recommended: list, relevant: set) -> float:
    hits, running_precision = 0, 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            running_precision += hits / (i + 1)
    return running_precision / len(relevant) if relevant else 0.0


def evaluate_recommendations(
    recommendations: dict[int, list],   # user_id → [item_ids]
    ground_truth: dict[int, set],        # user_id → {item_ids}
    k_values: list[int] = [5, 10, 20],
) -> dict:
    """Evaluate a full recommendation set across all users."""
    all_items = [item for recs in recommendations.values() for item in recs]
    item_counts = Counter(all_items)

    results = {}
    for k in k_values:
        ndcgs, precisions, aps = [], [], []
        for user, recs in recommendations.items():
            relevant = ground_truth.get(user, set())
            if not relevant:
                continue
            ndcgs.append(ndcg_at_k(recs, relevant, k))
            precisions.append(precision_at_k(recs, relevant, k))
            aps.append(average_precision(recs[:k], relevant))
        results[f"ndcg@{k}"] = np.mean(ndcgs)
        results[f"precision@{k}"] = np.mean(precisions)
        results[f"map@{k}"] = np.mean(aps)

    n_items = len(item_counts)
    results["catalog_coverage"] = n_items / max(all_items) if all_items else 0
    return results