"""Tests for BrandComparisonAnalyzer."""

import pandas as pd
import pytest

from reddit_sentiment.analysis.brand_comparison import BrandComparisonAnalyzer, BrandMetrics


def _make_df(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def analyzer():
    return BrandComparisonAnalyzer()


def _sample_df():
    return _make_df(
        [
            {
                "brands": ["Nike"],
                "hybrid_score": 0.8,
                "vader_score": 0.7,
                "score": 100,
                "subreddit": "Sneakers",
            },
            {
                "brands": ["Nike"],
                "hybrid_score": 0.3,
                "vader_score": 0.2,
                "score": 50,
                "subreddit": "Sneakers",
            },
            {
                "brands": ["Adidas"],
                "hybrid_score": -0.5,
                "vader_score": -0.4,
                "score": 30,
                "subreddit": "Adidas",
            },
            {
                "brands": ["Nike", "Adidas"],
                "hybrid_score": 0.1,
                "vader_score": 0.05,
                "score": 20,
                "subreddit": "Sneakers",
            },
        ]
    )


def test_compute_returns_brand_metrics(analyzer):
    df = _sample_df()
    result = analyzer.compute(df)
    assert "Nike" in result
    assert "Adidas" in result
    assert isinstance(result["Nike"], BrandMetrics)


def test_nike_mention_count(analyzer):
    df = _sample_df()
    result = analyzer.compute(df)
    assert result["Nike"].mention_count == 3  # rows 0, 1, 3


def test_adidas_mention_count(analyzer):
    df = _sample_df()
    result = analyzer.compute(df)
    assert result["Adidas"].mention_count == 2  # rows 2, 3


def test_positive_sentiment_label(analyzer):
    df = _sample_df()
    result = analyzer.compute(df)
    assert result["Nike"].sentiment_label == "Positive"


def test_negative_sentiment_label(analyzer):
    df = _sample_df()
    result = analyzer.compute(df)
    # Adidas: avg = (-0.5 + 0.1) / 2 = -0.2 → Negative
    assert result["Adidas"].sentiment_label == "Negative"


def test_comparison_table_sorted(analyzer):
    df = _sample_df()
    table = analyzer.comparison_table(df)
    assert list(table.columns[:2]) == ["brand", "mentions"]
    # Nike avg > Adidas avg → Nike first
    assert table["brand"].iloc[0] == "Nike"


def test_empty_df_returns_empty(analyzer):
    df = pd.DataFrame(columns=["brands", "hybrid_score", "vader_score", "score", "subreddit"])
    assert analyzer.compute(df) == {}
    assert analyzer.comparison_table(df).empty
