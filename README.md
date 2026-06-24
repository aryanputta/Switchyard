# Switchyard

An SLO-aware retrieval router for web and product search.

One thesis: spend the scarce budget where it buys the most relevance.

- **Product search (Amazon track):** the scarce resource is query latency. A controller picks the cheapest retrieval route (lexical, dense, hybrid, rerank) per query over the official Amazon Shopping Queries (ESCI) candidate set.
- **Web search (Google track):** the scarce resource is the crawl budget. A value-of-crawl model, trained on labels from an LLM-as-judge, decides which documents are worth acquiring under a budget.

Both tracks share one controller pattern: predict the value of each action, price it against a budget, pick the action that maximizes relevance per unit cost.

## Why this is not a search-engine clone

Most student search projects optimize relevance at unlimited compute. Switchyard optimizes relevance per millisecond and relevance per crawl. That is the constraint real search systems at Google and Amazon actually live under. The project does not claim to reproduce either company's proprietary infrastructure. It rebuilds the public engineering problems they publish about, on real public datasets, with honest measured numbers.

## Current status: Phases 0 to 2 on real ESCI, plus the serving path

The whole project rests on one claim: an adaptive router can match an always-rerank baseline on relevance while spending far less, and a learned policy can recover a real share of the gain. That is now tested on real data with significance.

```
python -m pytest                                              # 36 tests
python scripts/prepare_esci.py --split test --max-queries 1500 --out data/esci_real.jsonl
PYTHONPATH=src python experiments/phase0_gate.py --data data/esci_real.jsonl
PYTHONPATH=src python experiments/phase2_learned_router.py --data data/esci_real.jsonl
```

### Real ESCI, 1500 US test queries (~38 graded candidates each)

| policy        | nDCG@10 | rr_exact | latency_ms | utility |
|---------------|--------:|---------:|-----------:|--------:|
| bm25_only     |  0.7465 |   0.7193 |       2.00 |  0.7385 |
| always_rerank |  0.7510 |   0.7361 |      35.00 |  0.6110 |
| rule_router   |  0.7479 |   0.7280 |      18.93 |  0.6722 |
| oracle        |  0.7769 |   0.7783 |       4.26 |  0.7599 |

Always-rerank barely beats BM25 on quality (0.7510 vs 0.7465) yet costs 17x. The oracle reaches 0.7769 by routing per query. Spending the same compute everywhere is the waste.

### Learned router, held-out test split, paired bootstrap (query as unit)

The multinomial-logistic router is trained on train-split queries to imitate the per-query oracle from features alone, then tested on held-out queries:

- learned_router vs always_rerank, utility: **+0.133, 95% CI [+0.127, +0.139], W/T/L 449/0/12, significant**
- learned_router vs rule_router, utility: **+0.071, 95% CI [+0.064, +0.078], W/T/L 397/16/48, significant**
- learned_router vs bm25_only, nDCG: +0.001, not significant

Honest reading: on this ESCI slice the Phase 0 routes (TF-IDF cosine, heuristic rerank) barely beat BM25 on relevance, so the router correctly learns to stay cheap and wins big on latency-aware utility. That is the concrete argument for Phase 3: swap in real embeddings and a cross-encoder so the expensive routes earn their cost and the routing decision carries more relevance, not just latency.

## Algorithms implemented

- BM25 lexical retrieval (`index.py`)
- TF-IDF cosine retrieval from scratch, the Phase 0 semantic proxy (`dense.py`)
- Reciprocal Rank Fusion (`fusion.py`)
- Field-aware product reranker with exact-match and brand/color boosting (`routes.py`)
- Query feature extraction: model numbers, money, negation, compatibility, color (`features.py`)
- Graded nDCG and reciprocal-rank-of-first-Exact (`metrics.py`)
- Cost-aware utility function (`utility.py`)
- Oracle router and interpretable rule router (`router.py`)
- Value-of-crawl model and budgeted greedy selection for web acquisition (`acquisition.py`)

## Layout

```
src/switchyard/      core library, zero required dependencies
  index.py dense.py fusion.py routes.py     retrieval algorithms
  features.py router.py learned_router.py   routing
  metrics.py utility.py stats.py            evaluation + significance
  esci.py acquisition.py                    product + web data
experiments/         phase0_gate.py, phase2_learned_router.py
scripts/             prepare_esci.py (real data), index_opensearch.py, load_test.js (k6)
serving/             Go online API (loads the frozen router model), Dockerfile
deploy/k3s/          Kubernetes manifests: API + HPA, OpenSearch, Redis
opensearch/          BM25 + HNSW index mapping
monitoring/          Prometheus config
docker-compose.yml   full local stack
data/ results/ docs/blog/ tests/
```

## Serving the model

Python trains the router and writes `results/router_model.json`. The Go service loads that exact artifact and applies the same feature extraction and decision rule, so the deployed router equals the evaluated one.

```
PYTHONPATH=src python experiments/phase2_learned_router.py    # writes router_model.json
go -C serving build -o switchyard .
SWITCHYARD_MODEL=results/router_model.json ./serving/switchyard
curl -s -X POST localhost:8080/search -d '{"query":"sony wh-1000xm5"}'
# {"route":"lexical", ...}
curl -s localhost:8080/metrics        # switchyard_route_selected_total per route

docker compose up --build             # full stack: API + OpenSearch + Redis + Postgres + Prometheus + Grafana
```

## Dataset

ESCI is the official Amazon Shopping Queries Dataset. It is a candidate-reranking task: each query ships with up to ~40 judged candidate products graded Exact, Substitute, Complement, Irrelevant. Switchyard ranks only those candidates and never treats unjudged catalog products as negatives. Gains: E=3, S=2, C=1, I=0.

The committed `data/esci_sample.jsonl` is a small fixture in the exact ESCI shape so the gate and tests run anywhere. The official data loader and download path land in Phase 1.

## Limitations (read before trusting any number)

- The dense route is from-scratch TF-IDF cosine and the reranker is a field-aware boost, not yet a neural embedding model or a trained cross-encoder. On ESCI neither reliably beats BM25 on nDCG, which is why the learned router wins on utility (latency) but ties BM25 on relevance. Phase 3 fixes this.
- Route latencies are modeled (lexical < dense < hybrid < rerank), not measured under load. They become real numbers when the k6 + OpenSearch path runs.
- Results are on the ESCI small-version US test slice (1500 queries). Not the full multilingual set.
- The web-acquisition track (value-of-crawl, LLM-as-judge labels, crawler) is implemented as the budgeted-selection algorithm and design; the live crawler and labeling pipeline are Phase 4.
- OpenSearch, Redis, Postgres, Prometheus, Grafana, and k3s are wired in compose and manifests; the Go service currently makes the routing decision and exposes metrics, and the live retrieval calls attach at the marked points.

No performance number in this repo is fabricated. Modeled values are labeled as modeled.

## Roadmap

See `ARCHITECTURE.md` for the full plan: real embeddings + cross-encoder (Phase 3), MS MARCO web track, the LLM agent layer for crawl and query understanding, the live crawler with LLM-as-judge labels, and k6 load and failure testing.
