# Collaborative Filtering Recommender

[![CI](https://github.com/cerenaaa/collab-filtering-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/collab-filtering-recommender/actions)

PySpark ALS-based collaborative filtering recommender for product/item discovery and ranking. Originally built at Brightloom to power NFT search engine discovery — extended here into a general-purpose item ranking framework.

## Approach

- **Algorithm**: Alternating Least Squares (ALS) matrix factorization via PySpark MLlib
- **Cold start**: Popularity-based fallback + content-based warm-up for new items/users
- **Re-ranking**: Diversity-aware re-ranking using Maximal Marginal Relevance (MMR)
- **Evaluation**: NDCG@K, Precision@K, MAP, coverage, and novelty

## Structure

```
collab-filtering-recommender/
├── data/
│   └── synthetic_interactions.py  # User-item interaction simulator
├── models/
│   ├── als_recommender.py         # PySpark ALS training + inference
│   └── cold_start.py              # Popularity fallback + content warm-up
├── evaluation/
│   └── ranking_metrics.py         # NDCG@K, Precision@K, MAP, diversity
├── serving/
│   └── recommender_api.py         # FastAPI serving layer
├── train.py
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python train.py                    # Train ALS on synthetic data
uvicorn serving.recommender_api:app --reload
```

## Metrics (synthetic dataset)

| Metric | @5 | @10 | @20 |
|---|---|---|---|
| NDCG | 0.41 | 0.47 | 0.52 |
| Precision | 0.29 | 0.24 | 0.19 |
| Coverage | — | 68% | 79% |