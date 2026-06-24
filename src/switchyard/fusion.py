from __future__ import annotations

from switchyard.index import Scored


def reciprocal_rank_fusion(
    rankings: list[list[Scored]], k: int = 60
) -> list[Scored]:
    """Reciprocal Rank Fusion. Each input ranking contributes 1 / (k + rank)
    to a document's fused score. k=60 is the value from the original RRF paper.
    RRF is score-agnostic, so a BM25 ranking and a cosine ranking can be fused
    without calibrating their score scales."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            fused[scored.doc_id] = fused.get(scored.doc_id, 0.0) + 1.0 / (k + rank)
    out = [Scored(doc_id, score) for doc_id, score in fused.items()]
    out.sort(key=lambda s: (-s.score, s.doc_id))
    return out
