from __future__ import annotations

import math


def dcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        gain = relevance.get(doc_id, 0)
        if gain:
            dcg += (2**gain - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Graded nDCG. Works for binary web judgments and for 4-grade ESCI gains
    because the gain comes straight from the relevance map."""
    ideal_ids = [
        doc_id for doc_id, _ in sorted(relevance.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    ideal = dcg_at_k(ideal_ids, relevance, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance, k) / ideal


def reciprocal_rank_first_exact(
    ranked_ids: list[str], relevance: dict[str, int], exact_gain: int = 3
) -> float:
    """Reciprocal rank of the first Exact-grade item. This is the metric an
    Amazon shopper feels first: how high is the first product that exactly
    matches what they asked for."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if relevance.get(doc_id, 0) >= exact_gain:
            return 1.0 / rank
    return 0.0
