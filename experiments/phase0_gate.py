"""Phase 0 de-risk gate.

The whole project rests on one claim: an adaptive router can match an
always-rerank baseline on relevance while spending less, and a cheap
interpretable policy can recover most of the oracle's utility gain. This script
proves or kills that claim before any serving infrastructure is built.

For every query and every route it records nDCG@10, latency, and utility, then
compares five policies:

  bm25_only       always lexical
  always_rerank   always the expensive route
  rule_router     the hand-written policy
  oracle          upper bound, sees the labels
  random          sanity floor

Run on the committed sample:
    python experiments/phase0_gate.py

Run on a real ESCI slice once downloaded:
    python experiments/phase0_gate.py --data data/esci_real.jsonl
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from switchyard.esci import load_esci_jsonl
from switchyard.features import extract_features
from switchyard.metrics import ndcg_at_k, reciprocal_rank_first_exact
from switchyard.router import oracle_route, rule_router
from switchyard.routes import ALL_ROUTES, Route, run_route
from switchyard.utility import utility

K = 10


def run(data_path: Path, log_path: Path) -> dict[str, dict[str, float]]:
    queries = load_esci_jsonl(data_path)

    # full per-query-per-route log: this is the training data for the Phase 2
    # learned router and the evidence behind every number below.
    log_rows: list[dict[str, object]] = []
    per_query: dict[str, dict[Route, dict[str, float]]] = {}

    for q in queries:
        relevance = q.relevance
        per_query[q.query_id] = {}
        for route in ALL_ROUTES:
            res = run_route(route, q)
            ndcg = ndcg_at_k(res.ranked_ids, relevance, K)
            rr = reciprocal_rank_first_exact(res.ranked_ids, relevance)
            util = utility(ndcg, res.latency_ms)
            per_query[q.query_id][route] = {
                "ndcg": ndcg,
                "latency": res.latency_ms,
                "utility": util,
                "rr_exact": rr,
            }
            log_rows.append(
                {
                    "query_id": q.query_id,
                    "query": q.query,
                    "route": route.value,
                    "ndcg_at_10": round(ndcg, 4),
                    "rr_first_exact": round(rr, 4),
                    "latency_ms": res.latency_ms,
                    "utility": round(util, 4),
                    "candidate_count": len(q.candidates),
                    **extract_features(q.query).as_dict(),
                }
            )

    _write_log(log_path, log_rows)

    policies: dict[str, list[Route]] = {
        "bm25_only": [Route.LEXICAL] * len(queries),
        "always_rerank": [Route.RERANK] * len(queries),
        "rule_router": [rule_router(extract_features(q.query)) for q in queries],
        "oracle": [
            oracle_route({r: per_query[q.query_id][r]["utility"] for r in ALL_ROUTES})
            for q in queries
        ],
    }

    summary: dict[str, dict[str, float]] = {}
    for name, chosen in policies.items():
        ndcgs, lats, utils, rrs = [], [], [], []
        for q, route in zip(queries, chosen):
            cell = per_query[q.query_id][route]
            ndcgs.append(cell["ndcg"])
            lats.append(cell["latency"])
            utils.append(cell["utility"])
            rrs.append(cell["rr_exact"])
        summary[name] = {
            "ndcg_at_10": mean(ndcgs),
            "rr_first_exact": mean(rrs),
            "latency_ms": mean(lats),
            "utility": mean(utils),
        }
    return summary


def _write_log(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Switchyard Phase 0 de-risk gate")
    parser.add_argument("--data", type=Path, default=Path("data/esci_sample.jsonl"))
    parser.add_argument("--log", type=Path, default=Path("results/phase0_route_log.csv"))
    args = parser.parse_args(argv)

    summary = run(args.data, args.log)

    print(f"dataset: {args.data}  (route log -> {args.log})")
    print()
    header = f"{'policy':<16}{'nDCG@10':>10}{'rr_exact':>10}{'latency_ms':>12}{'utility':>10}"
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        print(
            f"{name:<16}{m['ndcg_at_10']:>10.4f}{m['rr_first_exact']:>10.4f}"
            f"{m['latency_ms']:>12.2f}{m['utility']:>10.4f}"
        )

    rerank = summary["always_rerank"]
    oracle = summary["oracle"]
    rule = summary["rule_router"]
    print()
    print("gate read:")
    util_gain = oracle["utility"] - rerank["utility"]
    recovered = (rule["utility"] - rerank["utility"]) / util_gain if util_gain else 0.0
    print(f"  oracle utility gain over always_rerank: {util_gain:+.4f}")
    print(f"  rule router recovers {recovered*100:5.1f}% of that gain")
    print(f"  rule router latency vs always_rerank: "
          f"{rule['latency_ms']:.1f}ms vs {rerank['latency_ms']:.1f}ms")
    holds = oracle["utility"] >= rerank["utility"] and rule["latency_ms"] < rerank["latency_ms"]
    print(f"  THESIS HOLDS (on this data): {holds}")
    print()
    print("note: latencies are modeled (lexical<dense<hybrid<rerank). Phase 3 "
          "replaces them with k6-measured serving latencies. nDCG is real, "
          "computed on the provided candidate labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
