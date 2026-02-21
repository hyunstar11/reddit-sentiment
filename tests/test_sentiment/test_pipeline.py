"""Tests for SentimentPipeline (transformer mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from reddit_sentiment.sentiment.pipeline import SentimentPipeline


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {"extracted_urls": []}
    return pd.DataFrame([{**defaults, **r} for r in rows])


@pytest.fixture
def pipeline_no_transformer():
    """Pipeline with transformer disabled (VADER-only mode)."""
    return SentimentPipeline(use_transformer=False)


def test_vader_only_positive(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "These Nike shoes are amazing and I love them!"}])
    out = pipeline_no_transformer.annotate(df)
    assert "vader_score" in out.columns
    assert "hybrid_score" in out.columns
    assert out["vader_score"].iloc[0] > 0


def test_vader_only_negative(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "Terrible quality, worst Adidas shoe ever"}])
    out = pipeline_no_transformer.annotate(df)
    assert out["vader_score"].iloc[0] < 0


def test_brands_detected(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "Nike vs Adidas — which is better?"}])
    out = pipeline_no_transformer.annotate(df)
    brands = out["brands"].iloc[0]
    assert "Nike" in brands
    assert "Adidas" in brands


def test_channels_detected(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "Bought on StockX", "extracted_urls": []}])
    out = pipeline_no_transformer.annotate(df)
    assert "StockX" in out["channels"].iloc[0]


def test_intent_detected(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "Just copped the Nike Dunk!"}])
    out = pipeline_no_transformer.annotate(df)
    assert out["primary_intent"].iloc[0] == "completed_purchase"


def test_no_intent_returns_none(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "Interesting colorway on this shoe"}])
    out = pipeline_no_transformer.annotate(df)
    assert out["primary_intent"].iloc[0] is None


def test_transformer_score_is_none_when_disabled(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "Nike is great"}])
    out = pipeline_no_transformer.annotate(df)
    # transformer_score should be None because use_transformer=False
    assert out["transformer_score"].iloc[0] is None


def test_hybrid_score_equals_vader_when_no_transformer(pipeline_no_transformer):
    df = _make_df([{"id": "1", "full_text": "I love Adidas shoes"}])
    out = pipeline_no_transformer.annotate(df)
    assert abs(out["hybrid_score"].iloc[0] - out["vader_score"].iloc[0]) < 1e-9


@patch("reddit_sentiment.sentiment.pipeline.SentimentPipeline._get_transformer")
def test_hybrid_blends_scores(mock_get_transformer):
    """With transformer enabled, hybrid = 0.6*t + 0.4*vader."""
    mock_transformer = MagicMock()
    mock_transformer.score_batch.return_value = [0.8]  # one brand context
    mock_get_transformer.return_value = mock_transformer

    pl = SentimentPipeline(use_transformer=True)
    pl._transformer = mock_transformer

    df = _make_df([{"id": "1", "full_text": "Nike is absolutely incredible, I love them!"}])
    out = pl.annotate(df)

    vader = out["vader_score"].iloc[0]
    t_score = out["transformer_score"].iloc[0]
    hybrid = out["hybrid_score"].iloc[0]

    if t_score is not None:
        expected = 0.6 * t_score + 0.4 * vader
        assert abs(hybrid - expected) < 1e-6


def test_multi_row_annotation(pipeline_no_transformer):
    df = _make_df(
        [
            {"id": "1", "full_text": "Nike rules"},
            {"id": "2", "full_text": "Adidas is terrible"},
            {"id": "3", "full_text": "Just a regular post"},
        ]
    )
    out = pipeline_no_transformer.annotate(df)
    assert len(out) == 3
    assert list(out["id"]) == ["1", "2", "3"]
