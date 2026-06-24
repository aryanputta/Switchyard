#!/usr/bin/env bash
# One-command live demo: run the experiment, start the API, and query it.
#   ./scripts/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=data/esci_sample.jsonl
[ -f data/esci_real.jsonl ] && DATA=data/esci_real.jsonl

echo "== 1. experiment (route comparison + significance) =="
PYTHONPATH=src python3 experiments/phase2_learned_router.py --data "$DATA" | tail -20

echo
echo "== 2. build and start the Go serving API =="
go -C serving build -o switchyard .
SWITCHYARD_MODEL=results/router_model.json ./serving/switchyard &
API_PID=$!
trap 'kill $API_PID 2>/dev/null; rm -f serving/switchyard' EXIT
sleep 1

echo
echo "== 3. route some queries =="
for q in "sony wh-1000xm5" "comfortable shoes for nurses long shifts" "stainless steel water bottle 32 oz"; do
  printf '  %-45s -> ' "$q"
  curl -s -X POST localhost:8080/search -d "{\"query\":\"$q\"}" | python3 -c "import sys,json; print(json.load(sys.stdin)['route'])"
done

echo
echo "== 4. prometheus metrics =="
curl -s localhost:8080/metrics | grep switchyard_route_selected_total
