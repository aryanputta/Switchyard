from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

# Official ESCI grade to graded gain. Exact is most relevant, Irrelevant is zero.
ESCI_GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}


@dataclass(frozen=True)
class Candidate:
    product_id: str
    title: str
    brand: str = ""
    color: str = ""
    description: str = ""
    bullets: str = ""
    price: float | None = None
    label: str = "I"

    @property
    def gain(self) -> int:
        return ESCI_GAIN.get(self.label, 0)

    @property
    def text(self) -> str:
        # Field-ordered text. Title and brand carry the most reranking signal,
        # so they lead. The product reranker boosts exact id/model matches on
        # top of this lexical signal (see routes.product_rerank).
        parts = [self.title, self.brand, self.color, self.bullets, self.description]
        return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class EsciQuery:
    query_id: str
    query: str
    locale: str
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def relevance(self) -> dict[str, int]:
        return {c.product_id: c.gain for c in self.candidates}


def load_esci_jsonl(path: Path) -> list[EsciQuery]:
    """Load ESCI in the candidate-reranking shape: one query per line with its
    own provided candidate list and graded labels. This is the correct ESCI
    task. We never score a query against products outside its candidate set."""
    queries: list[EsciQuery] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            for required in ("query_id", "query", "candidates"):
                if required not in payload:
                    raise ValueError(f"line {line_no}: missing field '{required}'")
            by_id: dict[str, Candidate] = {}
            for c in payload["candidates"]:
                candidate = Candidate(
                    product_id=str(c["product_id"]),
                    title=str(c.get("title", "")),
                    brand=str(c.get("brand", "")),
                    color=str(c.get("color", "")),
                    description=str(c.get("description", "")),
                    bullets=str(c.get("bullets", "")),
                    price=(float(c["price"]) if c.get("price") is not None else None),
                    label=str(c.get("label", "I")).upper()[:1],
                )
                # ESCI can repeat a (query, product) pair. Keep the highest grade
                # so a single product is judged once at its best label.
                existing = by_id.get(candidate.product_id)
                if existing is None or candidate.gain > existing.gain:
                    by_id[candidate.product_id] = candidate
            candidates = list(by_id.values())
            queries.append(
                EsciQuery(
                    query_id=str(payload["query_id"]),
                    query=str(payload["query"]),
                    locale=str(payload.get("locale", "us")),
                    candidates=candidates,
                )
            )
    return queries
