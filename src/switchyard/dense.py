from __future__ import annotations

from collections import Counter
import math

from switchyard.index import Scored
from switchyard.text import tokenize

"""From-scratch TF-IDF cosine retrieval.

This is the semantic-proxy route for Phase 0: it runs anywhere with zero
external dependencies and lets the router compare a lexical route against a
vector route. In Phase 3 the vectorizer is swapped for sentence-transformer
embeddings served from OpenSearch HNSW. The route interface does not change,
so the router and the evaluation harness stay identical.
"""


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {term: count / total for term, count in counts.items()}


def _idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    n = len(docs_tokens)
    df: Counter[str] = Counter()
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] += 1
    return {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}


def _vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _tf(tokens)
    return {term: weight * idf.get(term, 0.0) for term, weight in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def tfidf_cosine_rank(query: str, candidates: list[tuple[str, str]]) -> list[Scored]:
    """candidates is a list of (doc_id, text). Ranks them against the query
    by TF-IDF cosine. The idf is fit over the candidate set, which mirrors a
    per-query reranking setup."""
    docs_tokens = [tokenize(text) for _, text in candidates]
    idf = _idf(docs_tokens + [tokenize(query)])
    query_vec = _vector(tokenize(query), idf)
    scored = [
        Scored(doc_id, _cosine(query_vec, _vector(tokens, idf)))
        for (doc_id, _), tokens in zip(candidates, docs_tokens)
    ]
    scored.sort(key=lambda s: (-s.score, s.doc_id))
    return scored
