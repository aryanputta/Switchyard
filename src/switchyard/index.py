from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

from switchyard.text import tokenize


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Scored:
    doc_id: str
    score: float


class InvertedIndex:
    """Positional-free inverted index. Small by design: built per candidate set
    for product reranking, or over a corpus for the web track."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.doc_lengths: dict[str, int] = {}
        self.term_freqs: dict[str, Counter[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)

    def add(self, document: Document) -> None:
        if document.doc_id in self.documents:
            raise ValueError(f"duplicate document id: {document.doc_id}")
        tokens = tokenize(document.text)
        freqs = Counter(tokens)
        self.documents[document.doc_id] = document
        self.doc_lengths[document.doc_id] = len(tokens)
        self.term_freqs[document.doc_id] = freqs
        for token in freqs:
            self.postings[token].add(document.doc_id)

    def add_many(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self.add(document)

    @property
    def doc_count(self) -> int:
        return len(self.documents)

    @property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, set()))


class BM25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.index = index
        self.k1 = k1
        self.b = b

    def score_all(self, query: str) -> list[Scored]:
        """Score every document in the index. Used for candidate reranking,
        where the index holds only the candidates for one query."""
        query_terms = tokenize(query)
        if not query_terms or self.index.doc_count == 0:
            return [Scored(doc_id, 0.0) for doc_id in self.index.documents]
        scored = [
            Scored(doc_id, self._score_doc(doc_id, query_terms))
            for doc_id in self.index.documents
        ]
        scored.sort(key=lambda s: (-s.score, s.doc_id))
        return scored

    def _score_doc(self, doc_id: str, query_terms: list[str]) -> float:
        score = 0.0
        doc_len = self.index.doc_lengths[doc_id]
        avgdl = self.index.avg_doc_length or 1.0
        freqs = self.index.term_freqs[doc_id]
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            df = self.index.document_frequency(term)
            idf = math.log(1 + (self.index.doc_count - df + 0.5) / (df + 0.5))
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            score += idf * (tf * (self.k1 + 1)) / denominator
        return score
