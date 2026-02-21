"""Transformer-based sentiment analyzer (cardiffnlp/twitter-roberta-base-sentiment-latest).

Requires the [ml] optional extra:
    uv sync --extra ml

Falls back gracefully with ImportError if torch/transformers are not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid heavy imports at type-check time

_ML_AVAILABLE: bool | None = None


def _check_ml() -> bool:
    global _ML_AVAILABLE
    if _ML_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            _ML_AVAILABLE = True
        except ImportError:
            _ML_AVAILABLE = False
    return _ML_AVAILABLE


class TransformerAnalyzer:
    """Score short text snippets using a fine-tuned Twitter sentiment model.

    The model outputs three logits (negative / neutral / positive) mapped to a
    float score in [-1, 1]:  score = P(positive) - P(negative).

    On machines without torch/transformers installed the class can still be
    instantiated; calling ``score`` / ``score_batch`` will raise ``ImportError``
    with a clear install message.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from reddit_sentiment.config import SentimentConfig

        cfg = SentimentConfig()
        self._model_name = model_name or cfg.transformer_model
        self._batch_size = cfg.transformer_batch_size
        self._pipeline = None  # lazy-loaded on first use

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        if not _check_ml():
            raise ImportError(
                "torch and transformers are required for TransformerAnalyzer. "
                "Install with:  uv sync --extra ml"
            )
        from transformers import pipeline  # type: ignore[import]

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            tokenizer=self._model_name,
            top_k=None,  # return all label scores
            truncation=True,
            max_length=512,
        )

    def _labels_to_score(self, label_scores: list[dict]) -> float:
        """Convert [{label, score}, ...] → float in [-1, 1]."""
        mapping: dict[str, float] = {}
        for item in label_scores:
            label: str = item["label"].lower()
            if "positive" in label or label == "pos" or label == "label_2":
                mapping["positive"] = item["score"]
            elif "negative" in label or label == "neg" or label == "label_0":
                mapping["negative"] = item["score"]
            else:
                mapping["neutral"] = item["score"]
        pos = mapping.get("positive", 0.0)
        neg = mapping.get("negative", 0.0)
        return float(pos - neg)

    def score(self, text: str) -> float:
        """Score a single text snippet; returns float in [-1, 1]."""
        if not text or not text.strip():
            return 0.0
        self._load()
        result = self._pipeline([text])  # type: ignore[misc]
        return self._labels_to_score(result[0])

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a batch of texts (uses pipeline batching for efficiency)."""
        if not texts:
            return []
        self._load()
        # Filter empties, preserving index
        scores = []
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        results = self._pipeline(  # type: ignore[misc]
            [t for _, t in non_empty],
            batch_size=self._batch_size,
        )
        idx_map = {i: self._labels_to_score(r) for (i, _), r in zip(non_empty, results)}
        for i in range(len(texts)):
            scores.append(idx_map.get(i, 0.0))
        return scores
