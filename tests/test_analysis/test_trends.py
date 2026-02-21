"""Tests for SentimentTrendAnalyzer."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from reddit_sentiment.analysis.trends import SentimentTrendAnalyzer


@pytest.fixture
def analyzer():
    return SentimentTrendAnalyzer()


def _ts(year, month, day):
    return datetime(year, month, day, tzinfo=UTC)


def _sample_df():
    return pd.DataFrame(
        [
            {
                "hybrid_score": 0.5,
                "created_utc": _ts(2024, 1, 5),
                "brands": ["Nike"],
                "subreddit": "Sneakers",
            },
            {
                "hybrid_score": -0.3,
                "created_utc": _ts(2024, 1, 8),
                "brands": ["Adidas"],
                "subreddit": "Adidas",
            },
            {
                "hybrid_score": 0.2,
                "created_utc": _ts(2024, 1, 15),
                "brands": ["Nike"],
                "subreddit": "Sneakers",
            },
            {
                "hybrid_score": 0.8,
                "created_utc": _ts(2024, 2, 3),
                "brands": ["Nike"],
                "subreddit": "Sneakers",
            },
            {
                "hybrid_score": 0.1,
                "created_utc": _ts(2024, 2, 10),
                "brands": ["Adidas"],
                "subreddit": "Adidas",
            },
        ]
    )


def test_weekly_returns_dataframe(analyzer):
    result = analyzer.analyze(_sample_df())
    assert isinstance(result.weekly, pd.DataFrame)
    assert "period" in result.weekly.columns
    assert "avg_sentiment" in result.weekly.columns


def test_monthly_returns_dataframe(analyzer):
    result = analyzer.analyze(_sample_df())
    assert isinstance(result.monthly, pd.DataFrame)
    assert len(result.monthly) >= 2  # at least Jan + Feb


def test_monthly_count_correct(analyzer):
    result = analyzer.analyze(_sample_df())
    months = result.monthly.set_index("period")["count"]
    assert months.get("2024-01", 0) == 3
    assert months.get("2024-02", 0) == 2


def test_by_brand_trend(analyzer):
    result = analyzer.analyze(_sample_df(), by_brand=True)
    assert "brands" in result.monthly.columns or "brands" in result.weekly.columns


def test_missing_utc_column(analyzer):
    df = pd.DataFrame({"hybrid_score": [0.5, -0.3]})
    result = analyzer.analyze(df)
    assert result.weekly.empty
    assert result.monthly.empty


def test_sorted_ascending(analyzer):
    result = analyzer.analyze(_sample_df())
    periods = result.monthly["period"].tolist()
    assert periods == sorted(periods)
