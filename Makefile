UV := $(HOME)/.local/bin/uv

.PHONY: install install-ml install-dashboard test test-cov lint lint-fix \
        collect collect-public analyze report pipeline dashboard clean

install:
	$(UV) sync --extra dev

install-ml:
	$(UV) sync --all-extras

install-dashboard:
	$(UV) sync --extra dev --extra dashboard

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

# Collect with PRAW (requires Reddit API credentials in .env)
collect:
	$(UV) run reddit-sentiment collect

# Collect without credentials (public JSON API, includes comments)
collect-public:
	$(UV) run reddit-sentiment collect --public

analyze:
	$(UV) run reddit-sentiment analyze --no-transformer

analyze-ml:
	$(UV) run reddit-sentiment analyze

report:
	$(UV) run reddit-sentiment report

pipeline:
	$(UV) run reddit-sentiment pipeline --public --no-transformer

# Launch interactive Streamlit dashboard (install-dashboard first)
dashboard:
	$(UV) run reddit-sentiment dashboard

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
