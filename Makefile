.PHONY: test gate data phase2 serve demo clean

test:
	python3 -m pytest -q

data:
	python3 scripts/prepare_esci.py --split test --max-queries 1500 --out data/esci_real.jsonl

gate:
	PYTHONPATH=src python3 experiments/phase0_gate.py --data data/esci_real.jsonl

phase2:
	PYTHONPATH=src python3 experiments/phase2_learned_router.py --data data/esci_real.jsonl

serve:
	go -C serving build -o switchyard .
	SWITCHYARD_MODEL=results/router_model.json ./serving/switchyard

# One command: tests, then the gate and learned-router experiment on the bundled
# sample (no network). Run `make data` first for the real 1500-query ESCI slice.
demo: test
	PYTHONPATH=src python3 experiments/phase0_gate.py
	PYTHONPATH=src python3 experiments/phase2_learned_router.py --data data/esci_sample.jsonl

clean:
	rm -rf src/switchyard/__pycache__ tests/__pycache__ .pytest_cache serving/switchyard
