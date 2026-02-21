"""Tests for VaderAnalyzer."""

import pytest

from reddit_sentiment.sentiment.vader import VaderAnalyzer


@pytest.fixture
def analyzer():
    return VaderAnalyzer()


def test_positive_text(analyzer):
    score = analyzer.score("I absolutely love these shoes! Amazing quality!")
    assert score > 0.3


def test_negative_text(analyzer):
    score = analyzer.score("These shoes are terrible, worst quality ever")
    assert score < -0.3


def test_neutral_text(analyzer):
    score = analyzer.score("The shoe is a size 10")
    # neutral should be close to 0
    assert -0.3 <= score <= 0.3


def test_empty_text(analyzer):
    assert analyzer.score("") == 0.0
    assert analyzer.score("   ") == 0.0


def test_score_batch(analyzer):
    texts = ["great!", "terrible", "ok"]
    scores = analyzer.score_batch(texts)
    assert len(scores) == 3
    assert scores[0] > scores[1]  # great > terrible


def test_full_scores_keys(analyzer):
    scores = analyzer.full_scores("I love this!")
    assert set(scores.keys()) == {"neg", "neu", "pos", "compound"}


def test_full_scores_empty(analyzer):
    scores = analyzer.full_scores("")
    assert scores["compound"] == 0.0
    assert scores["neu"] == 1.0
