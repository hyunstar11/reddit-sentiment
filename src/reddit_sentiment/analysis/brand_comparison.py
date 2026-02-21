"""Brand sentiment comparison analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BrandMetrics:
    """Aggregated sentiment metrics for one brand."""

    brand: str
    mention_count: int
    avg_hybrid_score: float
    avg_vader_score: float
    positive_pct: float  # % of texts with hybrid_score > 0.05
    negative_pct: float  # % of texts with hybrid_score < -0.05
    neutral_pct: float
    avg_post_score: float  # Reddit upvote score of posts/comments mentioning brand
    top_subreddits: list[str] = field(default_factory=list)

    @property
    def sentiment_label(self) -> str:
        if self.avg_hybrid_score > 0.05:
            return "Positive"
        if self.avg_hybrid_score < -0.05:
            return "Negative"
        return "Neutral"


class BrandComparisonAnalyzer:
    """Compute per-brand sentiment metrics from an annotated DataFrame."""

    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05

    def compute(self, df: pd.DataFrame) -> dict[str, BrandMetrics]:
        """Return brand → BrandMetrics mapping.

        Expected columns: brands (list), hybrid_score, vader_score, score, subreddit.
        """
        if df.empty:
            return {}

        # Explode brand list so each row has one brand
        exploded = df.explode("brands").rename(columns={"brands": "brand"})
        exploded = exploded[exploded["brand"].notna() & (exploded["brand"] != "")]

        metrics: dict[str, BrandMetrics] = {}
        for brand, group in exploded.groupby("brand"):
            n = len(group)
            hybrid = group["hybrid_score"].fillna(0.0)
            vader = group["vader_score"].fillna(0.0)
            post_score = (
                group["score"].fillna(0) if "score" in group.columns else pd.Series([0] * n)
            )

            pos_pct = (hybrid > self.POSITIVE_THRESHOLD).mean() * 100
            neg_pct = (hybrid < self.NEGATIVE_THRESHOLD).mean() * 100
            neu_pct = 100 - pos_pct - neg_pct

            top_subs = []
            if "subreddit" in group.columns:
                top_subs = group["subreddit"].value_counts().head(3).index.tolist()

            metrics[brand] = BrandMetrics(
                brand=brand,
                mention_count=n,
                avg_hybrid_score=float(hybrid.mean()),
                avg_vader_score=float(vader.mean()),
                positive_pct=float(pos_pct),
                negative_pct=float(neg_pct),
                neutral_pct=float(neu_pct),
                avg_post_score=float(post_score.mean()),
                top_subreddits=top_subs,
            )

        return metrics

    def comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a tidy DataFrame sorted by avg_hybrid_score descending."""
        metrics = self.compute(df)
        rows = [
            {
                "brand": m.brand,
                "mentions": m.mention_count,
                "avg_sentiment": round(m.avg_hybrid_score, 4),
                "sentiment": m.sentiment_label,
                "positive_%": round(m.positive_pct, 1),
                "negative_%": round(m.negative_pct, 1),
                "neutral_%": round(m.neutral_pct, 1),
                "avg_post_score": round(m.avg_post_score, 1),
            }
            for m in metrics.values()
        ]
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows).sort_values("avg_sentiment", ascending=False)
        return result.reset_index(drop=True)
