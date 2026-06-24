"""Build a real ESCI candidate-reranking slice from the official dataset.

The Amazon Shopping Queries Dataset (ESCI) is published as product and example
parquet tables. This script joins them, filters to one locale and the small
version, groups by query into candidate sets with graded labels, and writes the
exact JSONL shape Switchyard evaluates.

It is a candidate-reranking export: every output line is one query with only its
own judged products. No unjudged catalog product is ever introduced as a
negative.

Usage:
    pip install switchyard[real]   # pandas + pyarrow
    python scripts/prepare_esci.py --locale us --max-queries 2000 \
        --split test --out data/esci_real.jsonl

Source: amazon-science/esci-data (Apache-2.0). Mirrored on the Hugging Face hub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HF_REPO = "tasksource/esci"

LABEL_MAP = {"exact": "E", "substitute": "S", "complement": "C", "irrelevant": "I"}


def build(locale: str, split: str, max_queries: int, out: Path) -> int:
    import os

    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    from datasets import load_dataset  # lazy: core library stays dependency-free

    # tasksource/esci ships the products already joined onto the examples and
    # split into train/test. Filter to one locale and the small version.
    ds = load_dataset(HF_REPO, split=split)
    ds = ds.filter(
        lambda r: r["product_locale"] == locale and r["small_version"] == 1,
        num_proc=1,
    )
    df = ds.to_pandas()

    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for query_id, group in df.groupby("query_id"):
            candidates = []
            for _, row in group.iterrows():
                label = LABEL_MAP.get(str(row.get("esci_label", "")).lower(), "I")
                candidates.append(
                    {
                        "product_id": str(row["product_id"]),
                        "title": _clean(row.get("product_title")),
                        "brand": _clean(row.get("product_brand")),
                        "color": _clean(row.get("product_color")),
                        "description": _clean(row.get("product_description"))[:1000],
                        "bullets": _clean(row.get("product_bullet_point"))[:1000],
                        "label": label,
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "query_id": str(query_id),
                        "query": _clean(group.iloc[0]["query"]),
                        "locale": locale,
                        "candidates": candidates,
                    }
                )
                + "\n"
            )
            written += 1
            if written >= max_queries:
                break
    return written


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "nan":
        return ""
    return " ".join(text.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a real ESCI reranking slice")
    parser.add_argument("--locale", default="us")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("data/esci_real.jsonl"))
    args = parser.parse_args(argv)
    n = build(args.locale, args.split, args.max_queries, args.out)
    print(f"wrote {n} queries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
