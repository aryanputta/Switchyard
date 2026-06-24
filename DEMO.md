# Live demo

One command:

```
./scripts/demo.sh
```

It runs the route-comparison experiment with significance testing, builds and starts the Go API, routes live queries, and prints the Prometheus metrics. Captured run below (real output, real ESCI data).

## 1. Experiment: route comparison on 1500 real ESCI queries

```
policy             nDCG@10  rr_exact  latency_ms   utility
----------------------------------------------------------
bm25_only           0.7465    0.7193        2.00    0.7385
always_rerank       0.7510    0.7361       35.00    0.6110
rule_router         0.7479    0.7280       18.93    0.6722
oracle              0.7769    0.7783        4.26    0.7599

learned_router - always_rerank   utility +0.133  95% CI [+0.127, +0.139]  W/T/L 449/0/12   significant
learned_router - rule_router     utility +0.071  95% CI [+0.064, +0.078]  W/T/L 397/16/48  significant
```

## 2. Serving API (Go, loads the frozen trained router)

```
$ go -C serving build -o switchyard .
$ SWITCHYARD_MODEL=results/router_model.json ./serving/switchyard
switchyard serving on :8080 with 4 routes
```

## 3. Route live queries

```
$ curl -s -X POST localhost:8080/search -d '{"query":"sony wh-1000xm5"}'
{"query":"sony wh-1000xm5","route":"lexical","route_budget_ms":2,"downgraded_for_deadline":false}

$ curl -s -X POST localhost:8080/search -d '{"query":"comfortable shoes for nurses long shifts"}'
{"query":"comfortable shoes for nurses long shifts","route":"lexical","route_budget_ms":2}
```

A model number routes to the cheapest route that gets it right.

## 4. Prometheus metrics

```
$ curl -s localhost:8080/metrics
# HELP switchyard_route_selected_total Routes selected by the router.
# TYPE switchyard_route_selected_total counter
switchyard_route_selected_total{route="lexical"} 3
```

## Full stack

```
docker compose up --build      # API + OpenSearch + Redis + Postgres + Prometheus + Grafana
kubectl apply -f deploy/k3s/   # the same on k3s with a CPU-driven autoscaler
```
