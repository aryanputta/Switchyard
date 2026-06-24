# Switchyard Master Prompt

Use this to resume the build in any session.

You are the staff-level search engineer completing this repository. Build a
production-style, reproducible search platform that demonstrates the public
engineering problems relevant to Google Search and Amazon Search. Do not claim
to reproduce either company's proprietary infrastructure.

## Thesis
Every query and every crawl decision should not pay the same fixed cost. Build
an SLO-aware controller that selects the cheapest action expected to satisfy
relevance and hard constraints. Product search spends a latency budget across
retrieval routes; web search spends a crawl budget across documents.

## Routes
lexical (BM25), dense (ANN), hybrid (RRF), constraint-aware hybrid, cross-encoder rerank.

## Domains
- Web: MS MARCO, TREC Deep Learning, optional Natural Questions / BEIR transfer.
- Product: official Amazon Shopping Queries (ESCI), evaluated as candidate
  reranking. Rank only each query's provided candidates. Gains E=3, S=2, C=1, I=0.
  Report nDCG@10 and reciprocal rank of the first Exact result.

## Stack (Phase 3+)
Go online API, OpenSearch (BM25 + HNSW), Python research + ML, sentence-transformers
embeddings and cross-encoder, Redis cache, PostgreSQL logs, Prometheus + Grafana,
k6 load tests, Docker Compose, k3s deployment. Python owns router training and is
the single source of truth; Go loads the frozen model and never reimplements routing.

## Adaptive routing
Run every training query through every route. Log per query-route: query id, text,
route, nDCG@10, latency, compute cost, timeout, candidate count, query features.
utility = nDCG@10 - latency_penalty - compute_penalty - violation_penalty - timeout_penalty.
Derive oracle routes from train/val. Train a LightGBM classifier to predict the
route. Compare bm25, dense, static hybrid, always-rerank, rule router, learned
router, oracle.

## Web acquisition + agent (Phase 4)
Polite crawler (robots.txt, seed-domain only, rate limited). Value-of-crawl model
with quality labels from an LLM-as-judge; budgeted greedy selection. LLM agent
toolkit drives query understanding and crawl decisions through typed tools wrapping
deterministic components.

## Experiment rules
Split by query id, never by row. Freeze dataset versions, hashes, model names,
seeds, index settings. Never tune on final test judgments. Paired bootstrap CIs
with queries as the unit; report mean differences and per-query win/tie/loss.
Per-slice reporting: short queries, long questions, model numbers, brands,
negations, compatibility, lexical-vs-dense disagreement. Never fabricate numbers.

## Reliability
Measure p50/p95/p99, throughput, timeout rate, error rate, fallback rate, cache
hit rate, dense calls, reranker calls. Test neural-service failure, Redis failure,
cold cache, tight deadlines, malformed records, OpenSearch delay. Verify fallback
results stay useful.

## Rule
Preserve a working baseline after every phase. Do not replace retrieval research
with a generic LLM chatbot. Author is Aryan Putta only.

See ARCHITECTURE.md for the phase plan. Phase 0 is complete.
