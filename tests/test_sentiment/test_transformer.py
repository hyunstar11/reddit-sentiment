"""Tests for TransformerAnalyzer (transformer mocked — no real model download)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reddit_sentiment.sentiment.transformer import TransformerAnalyzer


def _make_pipeline_output(pos: float = 0.7, neg: float = 0.1, neu: float = 0.2):
    """Simulate HuggingFace pipeline output format."""
    return [
        {"label": "positive", "score": pos},
        {"label": "negative", "score": neg},
        {"label": "neutral", "score": neu},
    ]


@pytest.fixture
def analyzer():
    return TransformerAnalyzer(model_name="mock-model")


def test_labels_to_score_positive(analyzer):
    labels = _make_pipeline_output(pos=0.8, neg=0.1)
    score = analyzer._labels_to_score(labels)
    assert abs(score - 0.7) < 1e-6


def test_labels_to_score_negative(analyzer):
    labels = _make_pipeline_output(pos=0.1, neg=0.8)
    score = analyzer._labels_to_score(labels)
    assert abs(score - (-0.7)) < 1e-6


def test_labels_to_score_neutral(analyzer):
    labels = _make_pipeline_output(pos=0.33, neg=0.33)
    score = analyzer._labels_to_score(labels)
    assert abs(score) < 0.01


def test_score_returns_float(analyzer):
    """score() returns a float in [-1, 1] when _pipeline is pre-loaded."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [_make_pipeline_output(pos=0.9, neg=0.05)]
    analyzer._pipeline = mock_pipe  # bypass _load()

    score = analyzer.score("These shoes are fantastic!")
    assert isinstance(score, float)
    assert -1 <= score <= 1


def test_score_empty_returns_zero(analyzer):
    # Should NOT call the model for empty text
    assert analyzer.score("") == 0.0
    assert analyzer.score("   ") == 0.0


def test_score_batch_preserves_length(analyzer):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        _make_pipeline_output(pos=0.8, neg=0.1),
        _make_pipeline_output(pos=0.1, neg=0.8),
    ]
    analyzer._pipeline = mock_pipe  # bypass _load()

    texts = ["great!", "terrible"]
    scores = analyzer.score_batch(texts)
    assert len(scores) == 2
    assert scores[0] > 0
    assert scores[1] < 0


def test_import_error_without_ml():
    """TransformerAnalyzer should raise ImportError if torch missing."""
    analyzer = TransformerAnalyzer(model_name="mock-model")
    with patch("reddit_sentiment.sentiment.transformer._check_ml", return_value=False):
        with pytest.raises(ImportError, match="torch and transformers"):
            analyzer._load()
