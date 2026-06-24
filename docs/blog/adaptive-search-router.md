# Every query pays the same price, and that is the bug

I was reading how Amazon evaluates product search and noticed something I had been getting wrong in my own retrieval code. The Shopping Queries dataset does not ask you to search a catalog of millions. It hands you a query and up to forty candidate products that humans already judged Exact, Substitute, Complement, or Irrelevant, and asks you to order those. The first time I built an evaluator for it I scored each query against the whole product set and counted everything unjudged as irrelevant. That inflates the numbers and measures the wrong task. The real task is reranking a fixed candidate list. Once I fixed that, a second thing bothered me more.

I was running the same retrieval pipeline on every query. `sony wh-1000xm5` is an exact model number. BM25 nails it in two milliseconds. `comfortable shoes for nurses working long shifts` is a descriptive need where lexical matching falls apart and you want semantic retrieval and a reranker. `black headphones under $100 without microphone` has a price ceiling and a negation that no relevance score will respect on its own. Three queries, three different right answers, and I was paying the most expensive pipeline on all of them. At Google and Amazon scale that is not a style problem. That is the difference between a search system that holds its latency budget and one that does not.

So I built Switchyard around one rule: spend the scarce budget where it buys the most relevance.

## The controller

For product search the scarce budget is query latency. Switchyard has four routes that cost increasingly more: lexical BM25, dense TF-IDF cosine (a stand-in for embeddings until the serving path is built), hybrid fusion of the two with Reciprocal Rank Fusion, and a field-aware reranker that boosts exact title, brand, and color matches on top of the fused order. A controller picks one route per query.

To pick, you need a number that trades quality against cost. That is the utility function:

```
utility = nDCG@10 - latency_penalty * latency_ms - violation_penalty * constraints - timeout_penalty
```

With the latency penalty at zero, utility is just nDCG, and the controller always picks the most expensive route. That is the always-rerank baseline. Turn the penalty up and cheaper routes start winning the queries where they tie on relevance. That single knob is the whole idea.

## Does the idea even hold

Before writing a Go API or standing up OpenSearch, I wanted proof the routing has headroom. The experiment computes, for every query and every route, the nDCG and the utility, then compares policies against an oracle that is allowed to see the labels and pick the best route per query. I ran it on 1500 real ESCI US test queries, each with around 38 graded candidates.

```
policy             nDCG@10  rr_exact  latency_ms   utility
bm25_only           0.7465    0.7193        2.00    0.7385
always_rerank       0.7510    0.7361       35.00    0.6110
rule_router         0.7479    0.7280       18.93    0.6722
oracle              0.7769    0.7783        4.26    0.7599
```

Read the always-rerank row against bm25. Running the expensive cross-encoder route on every query buys you 0.0045 nDCG over plain BM25 and costs seventeen times the latency. That is the waste the whole project is about. Now read the oracle row: 0.7769 nDCG, higher than always-rerank, at 4.26 milliseconds. The oracle is not doing one thing to every query. It is sending model-number queries to lexical and saving the cross-encoder for the queries that actually move when you rerank them. Different routes win different queries, and picking the right one per query beats doing the expensive thing everywhere, on both axes at once.

The latencies are modeled, lexical cheaper than dense cheaper than hybrid cheaper than rerank, and they get replaced with numbers measured under load once the serving path exists. The nDCG is real, computed on the human ESCI judgments. I would rather show a modeled-latency gate that is honest about what is modeled than a polished chart with invented percentages.

## The learned router, and an uncomfortable result

The oracle cheats. It sees the labels. The deployable router has to pick a route from the query text alone. I trained a multinomial logistic classifier on the training-split queries to imitate the oracle's route choice from eight query features, then tested it on held-out queries with a paired bootstrap, the query as the unit:

```
learned_router - always_rerank   utility +0.133  95% CI [+0.127, +0.139]  W/T/L 449/0/12   significant
learned_router - rule_router     utility +0.071  95% CI [+0.064, +0.078]  W/T/L 397/16/48  significant
learned_router - bm25_only       nDCG    +0.001                            not significant
```

The router significantly beats always-rerank and the hand-written rules on latency-aware utility. But look at the last line. On relevance, the learned router is statistically tied with plain BM25. It learned the honest truth in my current setup: my dense route is a from-scratch TF-IDF cosine and my reranker is a field-aware boost, and on ESCI neither reliably beats BM25, so the smartest policy is to stay on the cheap route and pocket the latency. The router is correct. My routes are the weak part.

That is not a failure to hide, it is the finding that tells me exactly what to build next. The routing machinery works and is significant. The expensive routes do not yet earn their cost because they are not strong enough. Phase 3 swaps the TF-IDF proxy for real sentence-transformer embeddings over OpenSearch HNSW and the heuristic boost for a real cross-encoder. The same router, the same evaluation, stronger routes, and then the oracle gap becomes relevance the router can actually capture.

## The other budget

The same controller runs the web side, where the scarce resource is not latency but the crawl budget. You cannot fetch the whole web, so a value-of-crawl model scores each candidate URL by expected relevance contribution, and a budgeted selector picks the set that maximizes acquired value per fetch cost. The quality labels that train that model come from an LLM acting as a judge over a sample of fetched pages. Predict value, price it against a budget, pick the best set. It is the product-search router with the axes relabeled.

## The serving path

The router model is trained in Python and written to a small JSON. The online API is a Go service that loads that exact file, runs the same eight-feature extraction, and applies the same linear decision. The deployed router and the evaluated router are identical by construction, which is the boundary I wanted: Python owns research and training, Go owns serving and never reimplements the policy. The service also handles the request deadline. If a query routes to rerank but the caller passed a tight deadline, it downgrades to the most capable route that fits, and if a downstream neural service is down it returns the strongest ranking it already has rather than failing the request. It exposes Prometheus counters per route, sits behind a k3s deployment with a CPU-driven autoscaler, and OpenSearch holds the BM25 and HNSW indexes.

```
curl -s -X POST localhost:8080/search -d '{"query":"sony wh-1000xm5"}'
{"query":"sony wh-1000xm5","route":"lexical","route_budget_ms":2, ...}
```

A model number goes to lexical, in two milliseconds, because that is the cheapest route that gets it right.

## What is next

Phase 3 is the real-embedding upgrade the tie above is asking for: sentence-transformer vectors over OpenSearch HNSW and a real cross-encoder, then rerun the same experiment and watch whether the expensive routes finally earn their latency. After that, MS MARCO for the web track and the value-of-crawl model with LLM-as-judge labels for the acquisition side. Every phase keeps a working baseline, and no number gets reported that I did not measure.

The repo, the de-risk experiment, and the learned-router experiment reproduce every table here. Run `make demo`.
