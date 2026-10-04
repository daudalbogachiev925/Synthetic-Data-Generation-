.PHONY: install generate validate train test lint format clean

install:
	pip install -r requirements.txt

generate:
	python scripts/generate.py --config configs/config.yaml

validate:
	python scripts/validate.py --input data/synthetic.jsonl

train:
	python scripts/train.py --data data/synthetic.jsonl

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

clean:
	rm -rf outputs/* data/*.jsonl __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
