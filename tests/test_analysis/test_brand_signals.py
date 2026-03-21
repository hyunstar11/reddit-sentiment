"""Tests for BrandIntelligenceAnalyzer."""

from __future__ import annotations

import pandas as pd
import pytest

from reddit_sentiment.analysis.brand_signals import (
    BrandIntelligenceAnalyzer,
    BrandIntelligenceResult,
)

_BRANDS = BrandIntelligenceAnalyzer.BRANDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n_per_brand: int = 20, seed: int = 42) -> pd.DataFrame:
    """Synthetic annotated DataFrame covering all five tracked brands."""
    import random

    random.seed(seed)
    rows = []
    base_ts = pd.Timestamp("2025-10-01", tz="UTC")
    for i, brand in enumerate(_BRANDS):
        for j in range(n_per_brand):
            score = random.gauss(0.1 + i * 0.05, 0.3)
            rows.append({
                "id": f"{brand}-{j}",
                "subreddit": "Sneakers",
                "record_type": "post",
                "hybrid_score": min(max(score, -1.0), 1.0),
                "brands": [brand],
                "channels": [],
                "models": [],
                "created_utc": base_ts + pd.Timedelta(hours=(i * n_per_brand + j) * 12),
                "primary_intent": "seeking_purchase" if j % 3 == 0 else None,
                "all_intents": [],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def full_df():
    return _make_df()


@pytest.fixture
def analyzer():
    return BrandIntelligenceAnalyzer()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_returns_result_type(analyzer, full_df):
    result = analyzer.analyze(full_df)
    assert isinstance(result, BrandIntelligenceResult)


def test_returns_all_five_brands(analyzer, full_df):
    result = analyzer.analyze(full_df)
    returned = {r.brand for r in result.brands}
    assert returned == set(_BRANDS)


def test_summary_df_has_correct_columns(analyzer, full_df):
    result = analyzer.analyze(full_df)
    required = {
        "Brand", "Trend", "Signal", "Health Score",
        "StockX Premium %", "Avg Deadstock $", "Volatility",
        "Avg Sentiment", "Positive %", "Negative %",
        "Mentions", "Purchase Intent %", "Data Quality",
    }
    assert required.issubset(set(result.summary_df.columns))


def test_summary_df_row_count(analyzer, full_df):
    result = analyzer.analyze(full_df)
    assert len(result.summary_df) == 5


# ---------------------------------------------------------------------------
# Health score properties
# ---------------------------------------------------------------------------

def test_health_scores_in_range(analyzer, full_df):
    result = analyzer.analyze(full_df)
    for row in result.brands:
        assert 0.0 <= row.health_score <= 1.0, f"{row.brand}: {row.health_score}"


def test_adidas_score_beats_asics(analyzer, full_df):
    """Adidas has higher premium and deadstock than Asics — should score higher."""
    result = analyzer.analyze(full_df)
    by_brand = {r.brand: r.health_score for r in result.brands}
    assert by_brand["Adidas"] > by_brand["Asics"]


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def test_signal_classification_scale_up():
    assert BrandIntelligenceAnalyzer._classify_signal(0.65) == "🟢 Scale Up"
    assert BrandIntelligenceAnalyzer._classify_signal(1.0) == "🟢 Scale Up"
    assert BrandIntelligenceAnalyzer._classify_signal(0.6) == "🟢 Scale Up"


def test_signal_classification_hold():
    assert BrandIntelligenceAnalyzer._classify_signal(0.5) == "🟡 Hold"
    assert BrandIntelligenceAnalyzer._classify_signal(0.4) == "🟡 Hold"


def test_signal_classification_watch():
    assert BrandIntelligenceAnalyzer._classify_signal(0.39) == "🔴 Watch"
    assert BrandIntelligenceAnalyzer._classify_signal(0.0) == "🔴 Watch"


# ---------------------------------------------------------------------------
# Low-data / graceful degradation
# ---------------------------------------------------------------------------

def test_low_mention_brand_still_in_result(analyzer):
    """Brand with no Reddit mentions should still appear (market data drives score)."""
    # Only Nike data — other brands have 0 mentions
    rows = [
        {
            "id": f"n{i}",
            "subreddit": "Sneakers",
            "record_type": "post",
            "hybrid_score": 0.2,
            "brands": ["Nike"],
            "channels": [],
            "models": [],
            "created_utc": pd.Timestamp("2025-10-01", tz="UTC") + pd.Timedelta(hours=i),
            "primary_intent": None,
            "all_intents": [],
        }
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    result = analyzer.analyze(df)
    assert len(result.brands) == 5


def test_low_mention_brand_flagged(analyzer):
    """Brand with fewer than min_mentions should have '⚠' in Data Quality."""
    rows = [
        {
            "id": f"n{i}",
            "subreddit": "Sneakers",
            "record_type": "post",
            "hybrid_score": 0.2,
            "brands": ["Nike"],
            "channels": [],
            "models": [],
            "created_utc": pd.Timestamp("2025-10-01", tz="UTC") + pd.Timedelta(hours=i),
            "primary_intent": None,
            "all_intents": [],
        }
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    result = analyzer.analyze(df, min_mentions=5)
    low_data = result.summary_df[result.summary_df["Brand"] != "Nike"]["Data Quality"]
    assert all("⚠" in v for v in low_data)


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

def test_trend_up_detected():
    """Rising weekly sentiment scores should produce '↑' trend direction."""
    rows = []
    base = pd.Timestamp("2025-10-01", tz="UTC")
    # 3 weeks: week 0 low, week 1 mid, week 2 high → rising
    for week, score in [(0, -0.3), (1, 0.0), (2, 0.4)]:
        for j in range(5):
            rows.append({
                "id": f"w{week}-{j}",
                "subreddit": "Sneakers",
                "record_type": "post",
                "hybrid_score": score,
                "brands": ["Nike"],
                "channels": [],
                "models": [],
                "created_utc": base + pd.Timedelta(weeks=week, hours=j),
                "primary_intent": None,
                "all_intents": [],
            })
    df = pd.DataFrame(rows)
    analyzer = BrandIntelligenceAnalyzer()
    directions = analyzer._trend_directions(df)
    assert directions.get("Nike") == "↑"


def test_trend_insufficient_data_returns_neutral():
    """Only one week of data → not enough for trend → '→'."""
    rows = [
        {
            "id": f"r{i}",
            "subreddit": "Sneakers",
            "record_type": "post",
            "hybrid_score": 0.2,
            "brands": ["Nike"],
            "channels": [],
            "models": [],
            "created_utc": pd.Timestamp("2025-10-01", tz="UTC") + pd.Timedelta(hours=i),
            "primary_intent": None,
            "all_intents": [],
        }
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    analyzer = BrandIntelligenceAnalyzer()
    directions = analyzer._trend_directions(df)
    assert directions.get("Nike", "→") == "→"


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_df_returns_five_degraded_brands(analyzer):
    """Empty input → 5 rows returned, all mention_count == 0."""
    empty_df = pd.DataFrame(columns=[
        "id", "subreddit", "record_type", "hybrid_score",
        "brands", "channels", "models", "created_utc",
        "primary_intent", "all_intents",
    ])
    result = analyzer.analyze(empty_df)
    assert len(result.brands) == 5
    assert all(r.mention_count == 0 for r in result.brands)
