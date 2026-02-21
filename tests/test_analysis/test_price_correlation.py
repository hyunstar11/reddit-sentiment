"""Tests for PriceCorrelationAnalyzer: aggregation, correlation, and empty-eBay handling."""

from __future__ import annotations

import pandas as pd
import pytest

from reddit_sentiment.analysis.price_correlation import (
    CorrelationResult,
    PriceCorrelationAnalyzer,
)


@pytest.fixture
def analyzer():
    return PriceCorrelationAnalyzer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reddit_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal annotated Reddit DataFrame."""
    return pd.DataFrame(rows)


def _ebay_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _reddit_row(models: list[str], hybrid_score: float = 0.3) -> dict:
    return {"models": models, "hybrid_score": hybrid_score}


# Enough mentions to pass MIN_MENTIONS=3
_AJ1_ROWS = [_reddit_row(["Air Jordan 1"], score) for score in [0.5, 0.6, 0.4]]
_DUNK_ROWS = [_reddit_row(["Dunk Low"], score) for score in [-0.1, -0.2, -0.15]]


# ---------------------------------------------------------------------------
# Empty / missing data
# ---------------------------------------------------------------------------


def test_empty_reddit_df_returns_no_signals(analyzer):
    result = analyzer.analyze(pd.DataFrame(), pd.DataFrame())
    assert isinstance(result, CorrelationResult)
    assert result.signals == []
    assert result.correlation_sentiment_premium is None


def test_missing_models_column_returns_no_signals(analyzer):
    df = pd.DataFrame([{"hybrid_score": 0.3}])
    result = analyzer.analyze(df, pd.DataFrame())
    assert result.signals == []


def test_empty_ebay_df_still_returns_reddit_signals(analyzer):
    df = _reddit_df(_AJ1_ROWS)
    result = analyzer.analyze(df, pd.DataFrame())
    assert len(result.signals) >= 1
    aj1 = next((s for s in result.signals if s.model == "Air Jordan 1"), None)
    assert aj1 is not None
    assert aj1.num_sales == 0  # no eBay data
    assert aj1.avg_sold_price == 0.0


# ---------------------------------------------------------------------------
# Reddit aggregation
# ---------------------------------------------------------------------------


def test_min_mentions_filter(analyzer):
    """Models with fewer than MIN_MENTIONS=3 should be excluded."""
    rows = [
        _reddit_row(["Air Jordan 1"]),  # only 1 mention → excluded
        _reddit_row(["Dunk Low"]),
        _reddit_row(["Dunk Low"]),
        _reddit_row(["Dunk Low"]),  # 3 mentions → included
    ]
    result = analyzer.analyze(_reddit_df(rows), pd.DataFrame())
    models = [s.model for s in result.signals]
    assert "Dunk Low" in models
    assert "Air Jordan 1" not in models


def test_sentiment_aggregated_correctly(analyzer):
    rows = [
        _reddit_row(["Air Jordan 1"], 0.6),
        _reddit_row(["Air Jordan 1"], 0.4),
        _reddit_row(["Air Jordan 1"], 0.2),
    ]
    result = analyzer.analyze(_reddit_df(rows), pd.DataFrame())
    aj1 = next(s for s in result.signals if s.model == "Air Jordan 1")
    assert aj1.avg_sentiment == pytest.approx(0.4, abs=1e-4)


def test_positive_pct_computed(analyzer):
    rows = [
        _reddit_row(["Dunk Low"], 0.8),   # positive (>0.05)
        _reddit_row(["Dunk Low"], 0.0),   # neutral
        _reddit_row(["Dunk Low"], -0.3),  # negative
    ]
    result = analyzer.analyze(_reddit_df(rows), pd.DataFrame())
    dunk = next(s for s in result.signals if s.model == "Dunk Low")
    assert dunk.positive_pct == pytest.approx(100 / 3, rel=0.01)


def test_signals_sorted_by_mention_count(analyzer):
    rows = _AJ1_ROWS + _DUNK_ROWS + [
        _reddit_row(["Air Force 1"], 0.1),
        _reddit_row(["Air Force 1"], 0.1),
        _reddit_row(["Air Force 1"], 0.1),
        _reddit_row(["Air Force 1"], 0.1),
        _reddit_row(["Air Force 1"], 0.1),  # 5 mentions
    ]
    result = analyzer.analyze(_reddit_df(rows), pd.DataFrame())
    counts = [s.mention_count for s in result.signals]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# eBay aggregation
# ---------------------------------------------------------------------------


def test_ebay_prices_joined(analyzer):
    reddit = _reddit_df(_AJ1_ROWS)
    ebay = _ebay_df([
        {"model": "Air Jordan 1", "sold_price_usd": 300.0},
        {"model": "Air Jordan 1", "sold_price_usd": 360.0},
    ])
    result = analyzer.analyze(reddit, ebay)
    aj1 = next(s for s in result.signals if s.model == "Air Jordan 1")
    assert aj1.num_sales == 2
    assert aj1.avg_sold_price == pytest.approx(330.0)
    assert aj1.min_sold_price == pytest.approx(300.0)
    assert aj1.max_sold_price == pytest.approx(360.0)


def test_price_premium_computed(analyzer):
    """Air Jordan 1 retail = $180; avg sold $270 → premium = 0.5 (50%)."""
    reddit = _reddit_df(_AJ1_ROWS)
    ebay = _ebay_df([
        {"model": "Air Jordan 1", "sold_price_usd": 270.0},
        {"model": "Air Jordan 1", "sold_price_usd": 270.0},
        {"model": "Air Jordan 1", "sold_price_usd": 270.0},
    ])
    result = analyzer.analyze(reddit, ebay)
    aj1 = next(s for s in result.signals if s.model == "Air Jordan 1")
    assert aj1.price_premium == pytest.approx(0.5, rel=0.01)


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------


def test_correlation_none_without_ebay(analyzer):
    result = analyzer.analyze(_reddit_df(_AJ1_ROWS + _DUNK_ROWS), pd.DataFrame())
    assert result.correlation_sentiment_premium is None


def test_correlation_none_with_fewer_than_3_paired(analyzer):
    reddit = _reddit_df(_AJ1_ROWS + _DUNK_ROWS)
    ebay = _ebay_df([{"model": "Air Jordan 1", "sold_price_usd": 300.0}])
    result = analyzer.analyze(reddit, ebay)
    # Only one model has eBay data → <3 pairs → None
    assert result.correlation_sentiment_premium is None


# ---------------------------------------------------------------------------
# summary_df
# ---------------------------------------------------------------------------


def test_summary_df_columns(analyzer):
    result = analyzer.analyze(_reddit_df(_AJ1_ROWS), pd.DataFrame())
    expected_cols = {
        "model", "brand", "retail_price", "mentions",
        "avg_sentiment", "positive_%", "negative_%",
        "num_sales", "avg_sold_price", "price_premium_%",
    }
    assert expected_cols.issubset(set(result.summary_df.columns))


def test_summary_df_empty_when_no_signals(analyzer):
    result = analyzer.analyze(pd.DataFrame(), pd.DataFrame())
    assert result.summary_df.empty
