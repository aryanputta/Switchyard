"""Phase 2: train the learned router and test it with significance.

Splits ESCI by query id, trains the multinomial-logistic router to imitate the
per-query oracle from query features alone, then evaluates on the held-out test
queries against every baseline. Differences are reported with paired-bootstrap
confidence intervals, the query as the unit.

Usage:
    PYTHONPATH=src python experiments/phase2_learned_router.py --data data/esci_real.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
import zlib

from switchyard.esci import EsciQuery, load_esci_jsonl
from switchyard.features import extract_features
from switchyard.learned_router import LearnedRouter, train
from switchyard.metrics import ndcg_at_k, reciprocal_rank_first_exact
from switchyard.router import oracle_route, rule_router
from switchyard.routes import ALL_ROUTES, Route, run_route
from switchyard.stats import paired_bootstrap
from switchyard.utility import utility

K = 10


def _split(queries: list[EsciQuery], test_frac: float = 0.3) -> tuple[list, list]:
    # deterministic split by hashing the query id, so a query never straddles
    # train and test and the split is reproducible without storing an index
    train_q, test_q = [], []
    for q in queries:
        bucket = (zlib.crc32(q.query_id.encode()) % 100) / 100.0
        (test_q if bucket < test_frac else train_q).append(q)
    return train_q, test_q


def _route_cells(queries: list[EsciQuery]) -> dict[str, dict[Route, dict[str, float]]]:
    cells: dict[str, dict[Route, dict[str, float]]] = {}
    for q in queries:
        relevance = q.relevance
        cells[q.query_id] = {}
        for route in ALL_ROUTES:
            res = run_route(route, q)
            ndcg = ndcg_at_k(res.ranked_ids, relevance, K)
            cells[q.query_id][route] = {
                "ndcg": ndcg,
                "latency": res.latency_ms,
                "utility": utility(ndcg, res.latency_ms),
                "rr": reciprocal_rank_first_exact(res.ranked_ids, relevance),
            }
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Switchyard Phase 2 learned router")
    parser.add_argument("--data", type=Path, default=Path("data/esci_real.jsonl"))
    parser.add_argument("--model-out", type=Path, default=Path("results/router_model.json"))
    args = parser.parse_args(argv)

    queries = load_esci_jsonl(args.data)
    train_q, test_q = _split(queries)
    print(f"queries: {len(queries)}  train: {len(train_q)}  test: {len(test_q)}")

    train_cells = _route_cells(train_q)
    samples = [
        (
            extract_features(q.query),
            oracle_route({r: train_cells[q.query_id][r]["utility"] for r in ALL_ROUTES}),
        )
        for q in train_q
    ]
    model = train(samples)
    model.save(args.model_out)
    print(f"trained router on {len(samples)} queries -> {args.model_out}")

    test_cells = _route_cells(test_q)

    def policy_routes(name: str) -> list[Route]:
        if name == "bm25_only":
            return [Route.LEXICAL] * len(test_q)
        if name == "always_rerank":
            return [Route.RERANK] * len(test_q)
        if name == "rule_router":
            return [rule_router(extract_features(q.query)) for q in test_q]
        if name == "learned_router":
            return [model.predict(extract_features(q.query)) for q in test_q]
        if name == "oracle":
            return [
                oracle_route({r: test_cells[q.query_id][r]["utility"] for r in ALL_ROUTES})
                for q in test_q
            ]
        raise ValueError(name)

    names = ["bm25_only", "always_rerank", "rule_router", "learned_router", "oracle"]
    series: dict[str, dict[str, list[float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        routes = policy_routes(name)
        ndcgs, lats, utils, rrs = [], [], [], []
        for q, route in zip(test_q, routes):
            cell = test_cells[q.query_id][route]
            ndcgs.append(cell["ndcg"])
            lats.append(cell["latency"])
            utils.append(cell["utility"])
            rrs.append(cell["rr"])
        series[name] = {"ndcg": ndcgs, "utility": utils}
        summary[name] = {
            "ndcg": mean(ndcgs),
            "rr": mean(rrs),
            "latency": mean(lats),
            "utility": mean(utils),
        }

    header = f"{'policy':<16}{'nDCG@10':>10}{'rr_exact':>10}{'latency_ms':>12}{'utility':>10}"
    print("\n" + header)
    print("-" * len(header))
    for name in names:
        m = summary[name]
        print(f"{name:<16}{m['ndcg']:>10.4f}{m['rr']:>10.4f}{m['latency']:>12.2f}{m['utility']:>10.4f}")

    print("\npaired bootstrap, query as unit (test set):")
    for metric in ("utility", "ndcg"):
        print(f"  metric = {metric}")
        for a, b in [
            ("learned_router", "always_rerank"),
            ("learned_router", "rule_router"),
            ("learned_router", "bm25_only"),
        ]:
            res = paired_bootstrap(series[a][metric], series[b][metric])
            star = "significant" if res.significant else "not significant"
            print(
                f"    {a} - {b}: {res.mean_diff:+.4f} "
                f"95% CI [{res.ci_low:+.4f}, {res.ci_high:+.4f}] "
                f"W/T/L {res.wins}/{res.ties}/{res.losses}  ({star})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
