.PHONY: install test lint format demo clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

demo:
	./scripts/run_real_agent.sh

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache
	rm -rf .pytest-tmp
	find runs -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
