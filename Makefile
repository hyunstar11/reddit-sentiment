UV := $(HOME)/.local/bin/uv

.PHONY: install test lint collect analyze report pipeline clean

install:
	$(UV) sync --extra dev

install-ml:
	$(UV) sync --all-extras

test:
	$(UV) run pytest tests/ -v --tb=short

test-cov:
	$(UV) run pytest tests/ --cov=src/reddit_sentiment --cov-report=term-missing

lint:
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/

lint-fix:
	$(UV) run ruff check --fix src/ tests/
	$(UV) run ruff format src/ tests/

collect:
	$(UV) run reddit-sentiment collect

analyze:
	$(UV) run reddit-sentiment analyze

report:
	$(UV) run reddit-sentiment report

pipeline:
	$(UV) run reddit-sentiment pipeline

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
