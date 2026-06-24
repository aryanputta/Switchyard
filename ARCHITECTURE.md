# Switchyard Architecture

## The single idea

Every query and every crawl decision should not pay the same fixed cost. A controller predicts the value of each available action, prices it against a budget, and selects the action that maximizes relevance per unit cost.

Two instances of the same controller:

| | Product search (Amazon) | Web search (Google) |
|---|---|---|
| Scarce budget | query latency / compute | crawl + index budget |
| Actions | lexical, dense, hybrid, rerank routes | fetch / skip / refresh each URL |
| Value signal | predicted nDCG per route | value-of-crawl model |
| Decision | route per query | budgeted document selection |
| Truth | ESCI graded judgments | LLM-as-judge quality labels |

## Components

```
                         +-------------------------+
   query --------------> |   query understanding   |  features + (Phase 4) LLM agent
                         +-----------+-------------+
                                     |
                                     v
                         +-------------------------+
                         |     route controller    |  rule router -> learned router
                         +-----------+-------------+
                                     |
        +-------------+--------------+--------------+-------------+
        v             v                             v             v
   lexical(BM25)   dense(ANN)                 hybrid(RRF)     rerank(cross-encoder)
        |             |                             |             |
        +-------------+--------------+--------------+-------------+
                                     v
                         +-------------------------+
                         |   evaluation harness    |  nDCG@10, rr@Exact, paired bootstrap
                         +-------------------------+

   web track:  crawl frontier -> value-of-crawl model -> budgeted selector -> index
               quality labels supplied by LLM-as-judge
```

## Phases

Each phase ships a working baseline.

### Phase 0 (done): de-risk gate
Pure-Python core. All four routes over ESCI candidates, cost-aware utility, oracle and rule routers, the de-risk experiment. 28 tests. Proves route selection has headroom before any infra is built.

### Phase 1: real data and rigor
- Official ESCI loader (`esci-data` parquet via the `real` extra) and MS MARCO / TREC DL judgments for the web track.
- Split strictly by query id, never by query-document row.
- Frozen experiment manifest: dataset hashes, model names, seeds, index settings.
- Paired bootstrap confidence intervals with the query as the statistical unit. Per-slice reporting: model numbers, negations, long questions, lexical-vs-dense disagreement.
- Never tune on final test judgments.

### Phase 2: learned router and constraints
- Embeddings (sentence-transformers) for the dense route, real cross-encoder for rerank.
- LightGBM router trained on the Phase 0 per-query-per-route log to predict the route. Compared against bm25-only, always-rerank, the rule router, and the oracle upper bound.
- Constraint-aware path: parse and enforce price, brand, color, compatibility, negation. Keep a separate golden constraint suite because ESCI does not carry every structured field. Negation handling targets a documented Amazon product-search weakness.

### Phase 3: serving path
- Go online search API. Python owns research and offline router training and is the single source of truth for the trained model; Go loads the frozen model artifact and never reimplements routing logic.
- OpenSearch for BM25 and HNSW dense retrieval (its hybrid-search pipeline is the public platform analog).
- Redis cache, PostgreSQL for query and experiment logs.
- Prometheus metrics, Grafana dashboards.
- k6 load tests, mixed web and product traffic. Report p50/p95/p99, throughput, timeout rate, fallback rate, cache hit rate. A tight request deadline forces a cheaper route. If embedding or rerank fails, return the strongest completed lexical or hybrid ranking rather than failing the request. Test neural-service failure, Redis failure, cold cache, malformed records, OpenSearch delay.

### Phase 4: web acquisition + agent toolkit
- Crawl frontier with a polite crawler: robots.txt, seed-domain only, rate limited.
- Value-of-crawl model trained on features with quality labels from an LLM-as-judge over a fetched sample. Budgeted greedy selection (`acquisition.py`) decides what to acquire. This is the Fan Bu (Google) angle: ML-driven web data acquisition.
- Agent toolkit: an LLM agent that drives query understanding (rewrite, constraint extraction, intent) and crawl decisions (which seeds, when to refresh) through typed tools. Each tool wraps an existing deterministic component so the agent adds judgment without becoming the system.

### Phase 5: deployment
- Docker Compose for local bring-up.
- k3s manifests for the serving path: API Deployment, OpenSearch StatefulSet, Redis, Postgres, Prometheus and Grafana, HorizontalPodAutoscaler on the API driven by request latency. k3s keeps it lightweight enough to run on a single node or a small cluster and demonstrates real orchestration, not just a single container.
- One-command demo and a live endpoint for the blog post.

## Experiment discipline (applies from Phase 1 on)

- Freeze dataset versions, hashes, model names, query ids, index settings, seeds.
- Bootstrap CIs, query as the unit. Report mean differences and per-query win/tie/loss.
- Never fabricate performance numbers. Modeled values (such as Phase 0 latencies) are labeled as modeled until measured.

## Deliverables

Reproducible repo, one-command demo, this architecture, frozen experiment manifest, TREC and ESCI results, ANN recall-vs-latency curve, Grafana dashboard, and a failure-fallback demo.
