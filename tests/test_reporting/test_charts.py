"""Tests for Plotly chart functions."""

import json

import pandas as pd

from reddit_sentiment.analysis.brand_comparison import BrandMetrics
from reddit_sentiment.analysis.channel_attribution import (
    ChannelAttribution,
)
from reddit_sentiment.reporting.charts import (
    brand_sentiment_bar,
    channel_share_pie,
    intent_funnel,
    sentiment_distribution_pie,
    sentiment_trend_line,
)


def _sample_metrics():
    return {
        "Nike": BrandMetrics("Nike", 50, 0.4, 0.35, 70, 10, 20, 80),
        "Adidas": BrandMetrics("Adidas", 30, -0.2, -0.15, 30, 45, 25, 50),
    }


def _sample_attribution():
    return ChannelAttribution(
        channel_share={"StockX": 60.0, "GOAT": 40.0},
        channel_counts={"StockX": 30, "GOAT": 20},
        channel_by_brand={"Nike": {"StockX": 20}},
        intent_funnel={"completed_purchase": 10, "seeking_purchase": 5},
        top_channels=["StockX", "GOAT"],
    )


def test_brand_bar_returns_valid_json():
    metrics = _sample_metrics()
    result = brand_sentiment_bar(metrics)
    parsed = json.loads(result)
    assert "data" in parsed
    assert len(parsed["data"]) > 0


def test_brand_bar_empty():
    result = brand_sentiment_bar({})
    assert result == "{}"


def test_sentiment_pie_returns_valid_json():
    result = sentiment_distribution_pie(_sample_metrics())
    parsed = json.loads(result)
    assert "data" in parsed


def test_channel_pie_returns_valid_json():
    result = channel_share_pie(_sample_attribution())
    parsed = json.loads(result)
    assert "data" in parsed


def test_channel_pie_empty():
    empty = ChannelAttribution({}, {}, {}, {}, [])
    assert channel_share_pie(empty) == "{}"


def test_funnel_returns_valid_json():
    result = intent_funnel(_sample_attribution())
    parsed = json.loads(result)
    assert "data" in parsed


def test_trend_line_empty():
    result = sentiment_trend_line(pd.DataFrame())
    assert result == "{}"


def test_trend_line_returns_json():
    df = pd.DataFrame(
        {
            "period": ["2024-W01", "2024-W02", "2024-W03"],
            "avg_sentiment": [0.2, -0.1, 0.3],
            "count": [10, 8, 12],
        }
    )
    result = sentiment_trend_line(df)
    parsed = json.loads(result)
    assert "data" in parsed
