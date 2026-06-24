from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from switchyard.dense import tfidf_cosine_rank
from switchyard.esci import Candidate, EsciQuery
from switchyard.fusion import reciprocal_rank_fusion
from switchyard.index import BM25Scorer, Document, InvertedIndex, Scored
from switchyard.text import tokenize


class Route(str, Enum):
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANK = "rerank"


# Modeled relative cost per route, in milliseconds. These are placeholders that
# encode the ordering lexical < dense < hybrid < rerank. Phase 3 replaces them
# with latencies measured under k6 load against the OpenSearch serving path.
ROUTE_COST_MS: dict[Route, float] = {
    Route.LEXICAL: 2.0,
    Route.DENSE: 8.0,
    Route.HYBRID: 11.0,
    Route.RERANK: 35.0,
}


@dataclass(frozen=True)
class RouteResult:
    route: Route
    ranked_ids: list[str]
    latency_ms: float


def _lexical_rank(query: str, candidates: list[Candidate]) -> list[Scored]:
    index = InvertedIndex()
    index.add_many(Document(c.product_id, c.text) for c in candidates)
    return BM25Scorer(index).score_all(query)


def _dense_rank(query: str, candidates: list[Candidate]) -> list[Scored]:
    return tfidf_cosine_rank(query, [(c.product_id, c.text) for c in candidates])


def _exact_match_boost(query: str, candidate: Candidate) -> float:
    """Field-aware boost that the cross-encoder route approximates. An exact
    model-number or brand-plus-color hit is a strong product-search signal that
    pure lexical BM25 underweights. Amazon has documented that exact-match and
    negation handling are real product-search weaknesses, so this boost and the
    negation filter target a documented failure mode, not an invented one."""
    q_tokens = set(tokenize(query))
    boost = 0.0
    title_tokens = set(tokenize(candidate.title))
    # full multi-token query phrase contained in the title
    if q_tokens and q_tokens.issubset(title_tokens):
        boost += 2.0
    if candidate.brand and candidate.brand.lower() in query.lower():
        boost += 0.5
    if candidate.color and candidate.color.lower() in query.lower():
        boost += 0.3
    return boost


def run_route(route: Route, esci_query: EsciQuery) -> RouteResult:
    """Run one retrieval route over a query's candidate set and return the
    ranked product ids plus the modeled latency for that route."""
    query, candidates = esci_query.query, esci_query.candidates
    if route is Route.LEXICAL:
        ranked = [s.doc_id for s in _lexical_rank(query, candidates)]
    elif route is Route.DENSE:
        ranked = [s.doc_id for s in _dense_rank(query, candidates)]
    elif route is Route.HYBRID:
        ranked = [
            s.doc_id
            for s in reciprocal_rank_fusion(
                [_lexical_rank(query, candidates), _dense_rank(query, candidates)]
            )
        ]
    elif route is Route.RERANK:
        fused = reciprocal_rank_fusion(
            [_lexical_rank(query, candidates), _dense_rank(query, candidates)]
        )
        by_id = {c.product_id: c for c in candidates}
        n = len(fused) or 1
        boosted: list[Scored] = []
        for rank, s in enumerate(fused):
            base = (n - rank) / n  # fused rank turned into a [0,1] base score
            boost = _exact_match_boost(query, by_id[s.doc_id])
            boosted.append(Scored(s.doc_id, base + boost))
        boosted.sort(key=lambda s: (-s.score, s.doc_id))
        ranked = [s.doc_id for s in boosted]
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"unknown route: {route}")
    return RouteResult(route=route, ranked_ids=ranked, latency_ms=ROUTE_COST_MS[route])


ALL_ROUTES: tuple[Route, ...] = (Route.LEXICAL, Route.DENSE, Route.HYBRID, Route.RERANK)
