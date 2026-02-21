"""Tests for ChannelAttributionAnalyzer."""

import pandas as pd
import pytest

from reddit_sentiment.analysis.channel_attribution import ChannelAttributionAnalyzer


@pytest.fixture
def analyzer():
    return ChannelAttributionAnalyzer()


def _df(rows):
    return pd.DataFrame(rows)


def _sample_df():
    return _df(
        [
            {
                "brands": ["Nike"],
                "channels": ["StockX", "Nike Direct"],
                "primary_intent": "completed_purchase",
            },
            {"brands": ["Adidas"], "channels": ["GOAT"], "primary_intent": "seeking_purchase"},
            {"brands": ["Nike"], "channels": ["StockX"], "primary_intent": "price_discussion"},
            {"brands": ["New Balance"], "channels": [], "primary_intent": None},
        ]
    )


def test_channel_counts(analyzer):
    df = _sample_df()
    result = analyzer.analyze(df)
    assert result.channel_counts["StockX"] == 2
    assert result.channel_counts["Nike Direct"] == 1
    assert result.channel_counts["GOAT"] == 1


def test_channel_share_sums_to_100(analyzer):
    df = _sample_df()
    result = analyzer.analyze(df)
    total = sum(result.channel_share.values())
    assert abs(total - 100.0) < 0.1


def test_top_channels_ordered(analyzer):
    df = _sample_df()
    result = analyzer.analyze(df)
    assert result.top_channels[0] == "StockX"


def test_intent_funnel(analyzer):
    df = _sample_df()
    result = analyzer.analyze(df)
    assert result.intent_funnel.get("completed_purchase", 0) == 1
    assert result.intent_funnel.get("seeking_purchase", 0) == 1


def test_channel_by_brand(analyzer):
    df = _sample_df()
    result = analyzer.analyze(df)
    assert "Nike" in result.channel_by_brand
    assert result.channel_by_brand["Nike"].get("StockX", 0) >= 1


def test_empty_df(analyzer):
    df = pd.DataFrame(columns=["brands", "channels", "primary_intent"])
    result = analyzer.analyze(df)
    assert result.channel_counts == {}
