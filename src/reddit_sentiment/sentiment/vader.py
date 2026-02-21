"""VADER-based sentiment analyzer for fast baseline scoring."""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderAnalyzer:
    """Thin wrapper around VaderSentiment for batch text analysis."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        """Return compound score in [-1, 1]."""
        if not text or not text.strip():
            return 0.0
        return self._analyzer.polarity_scores(text)["compound"]

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a list of texts; returns compound scores."""
        return [self.score(t) for t in texts]

    def full_scores(self, text: str) -> dict[str, float]:
        """Return all VADER scores (neg, neu, pos, compound)."""
        if not text or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self._analyzer.polarity_scores(text)
