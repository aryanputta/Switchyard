"""Index an ESCI product slice into OpenSearch for the BM25 + HNSW serving path.

Creates the index from opensearch/product-index.json and bulk-loads the
products that appear in a prepared candidate-reranking file. Embeddings are
optional: with the ml extra installed, each product title+bullets is encoded
with a sentence-transformer for the knn_vector field; without it, only the
lexical fields are indexed and the dense route falls back at serving time.

    pip install switchyard[real]            # opensearch client
    pip install switchyard[ml]              # optional, for embeddings
    python scripts/index_opensearch.py --data data/esci_real.jsonl \
        --url http://localhost:9200 --index switchyard-products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from switchyard.esci import load_esci_jsonl

MAPPING = Path(__file__).resolve().parents[1] / "opensearch" / "product-index.json"


def _encoder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index ESCI products into OpenSearch")
    parser.add_argument("--data", type=Path, default=Path("data/esci_real.jsonl"))
    parser.add_argument("--url", default="http://localhost:9200")
    parser.add_argument("--index", default="switchyard-products")
    args = parser.parse_args(argv)

    from opensearchpy import OpenSearch, helpers  # lazy: only needed for indexing

    client = OpenSearch(hosts=[args.url])
    if client.indices.exists(args.index):
        client.indices.delete(args.index)
    client.indices.create(args.index, body=json.loads(MAPPING.read_text()))

    encoder = _encoder()
    seen: set[str] = set()
    products = []
    for q in load_esci_jsonl(args.data):
        for c in q.candidates:
            if c.product_id in seen:
                continue
            seen.add(c.product_id)
            doc = {
                "_index": args.index,
                "_id": c.product_id,
                "product_id": c.product_id,
                "title": c.title,
                "brand": c.brand,
                "color": c.color,
                "bullets": c.bullets,
                "description": c.description,
                "locale": q.locale,
            }
            if encoder is not None:
                doc["embedding"] = encoder.encode(c.text).tolist()
            products.append(doc)

    helpers.bulk(client, products)
    client.indices.refresh(args.index)
    print(f"indexed {len(products)} products into {args.index} "
          f"(embeddings: {'yes' if encoder else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
