"""Hybrid sentiment pipeline: VADER on all texts + transformer on brand contexts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reddit_sentiment.config import SentimentConfig
from reddit_sentiment.detection.brands import BrandDetector, BrandMention
from reddit_sentiment.detection.channels import ChannelDetector
from reddit_sentiment.detection.intent import IntentResult, PurchaseIntentClassifier
from reddit_sentiment.detection.models import ModelDetector
from reddit_sentiment.sentiment.vader import VaderAnalyzer


@dataclass
class TextAnnotation:
    """Full annotation for one Reddit text."""

    text_id: str
    vader_score: float
    transformer_score: float | None  # None if transformer not available
    hybrid_score: float  # final blended score
    brands: list[str]
    channels: list[str]
    intent: str | None
    all_intents: list[str]
    brand_mentions: list[BrandMention]


class SentimentPipeline:
    """Orchestrates detection + VADER + optional transformer scoring."""

    def __init__(self, use_transformer: bool = True) -> None:
        cfg = SentimentConfig()
        self._transformer_weight = cfg.transformer_weight
        self._vader_weight = cfg.vader_weight
        self._use_transformer = use_transformer

        self._vader = VaderAnalyzer()
        self._brand_detector = BrandDetector()
        self._model_detector = ModelDetector()
        self._channel_detector = ChannelDetector()
        self._intent_clf = PurchaseIntentClassifier()

        # Lazy-load transformer only when needed
        self._transformer = None

    def _get_transformer(self):
        if self._transformer is None:
            from reddit_sentiment.sentiment.transformer import TransformerAnalyzer

            self._transformer = TransformerAnalyzer()
        return self._transformer

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate a DataFrame that has 'full_text', 'id', 'extracted_urls' columns.

        Returns the input DataFrame with additional columns added in-place.
        """
        df = df.copy()
        texts = df["full_text"].fillna("").tolist()
        urls_col = (
            df["extracted_urls"].tolist() if "extracted_urls" in df.columns else [[] for _ in texts]
        )

        # ------------------------------------------------------------------
        # 1. VADER scores for all texts (fast)
        # ------------------------------------------------------------------
        vader_scores = self._vader.score_batch(texts)

        # ------------------------------------------------------------------
        # 2. Brand detection → collect brand-context windows for transformer
        # ------------------------------------------------------------------
        all_mentions: list[list[BrandMention]] = []
        brand_contexts: list[str] = []
        context_text_indices: list[int] = []  # which row each context belongs to

        for i, text in enumerate(texts):
            mentions = self._brand_detector.detect(text)
            all_mentions.append(mentions)
            for mention in mentions:
                brand_contexts.append(mention.context)
                context_text_indices.append(i)

        # ------------------------------------------------------------------
        # 3. Transformer on brand contexts (if enabled + available)
        # ------------------------------------------------------------------
        context_transformer_scores: list[float] = []
        transformer_available = False

        if self._use_transformer and brand_contexts:
            try:
                transformer = self._get_transformer()
                context_transformer_scores = transformer.score_batch(brand_contexts)
                transformer_available = True
            except ImportError:
                # Fall back to VADER for contexts too
                context_transformer_scores = self._vader.score_batch(brand_contexts)

        # ------------------------------------------------------------------
        # 4. Per-row aggregation
        # ------------------------------------------------------------------
        transformer_scores: list[float | None] = []
        hybrid_scores: list[float] = []
        brand_lists: list[list[str]] = []
        model_lists: list[list[str]] = []
        channel_lists: list[list[str]] = []
        intent_primaries: list[str | None] = []
        all_intents_col: list[list[str]] = []

        # Build per-text transformer scores by averaging over brand contexts
        text_transformer_scores: dict[int, list[float]] = {}
        for ctx_idx, row_idx in enumerate(context_text_indices):
            score = context_transformer_scores[ctx_idx] if context_transformer_scores else 0.0
            text_transformer_scores.setdefault(row_idx, []).append(score)

        for i, (text, vader, urls) in enumerate(zip(texts, vader_scores, urls_col)):
            # Transformer: average of brand-context scores for this row
            t_scores = text_transformer_scores.get(i, [])
            if t_scores and transformer_available:
                t_score: float | None = sum(t_scores) / len(t_scores)
            else:
                t_score = None

            # Hybrid blend
            if t_score is not None:
                hybrid = self._transformer_weight * t_score + self._vader_weight * vader
            else:
                hybrid = vader

            # Brand names
            brands = list({m.brand for m in all_mentions[i]})

            # Shoe models
            models = self._model_detector.detect_models(text)

            # Channels
            channels = self._channel_detector.detect(text, urls if isinstance(urls, list) else [])

            # Intent
            intent_result: IntentResult = self._intent_clf.classify(text)

            transformer_scores.append(t_score)
            hybrid_scores.append(hybrid)
            brand_lists.append(brands)
            model_lists.append(models)
            channel_lists.append(channels)
            intent_primaries.append(intent_result.primary_intent)
            all_intents_col.append(intent_result.all_intents)

        df["vader_score"] = vader_scores
        df["transformer_score"] = transformer_scores
        df["hybrid_score"] = hybrid_scores
        df["brands"] = brand_lists
        df["models"] = model_lists
        df["channels"] = channel_lists
        df["primary_intent"] = intent_primaries
        df["all_intents"] = all_intents_col

        return df
