"""Brand Intelligence: unified signal combining StockX market data + Reddit sentiment."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reddit_sentiment.analysis.price_correlation import (
    STOCKX_BRAND_MARKET,
    PriceCorrelationAnalyzer,
)


@dataclass
class BrandIntelRow:
    """Combined market + Reddit intelligence for one brand."""

    brand: str
    stockx_premium: float       # e.g. 0.307
    avg_deadstock: float        # e.g. 9630 USD
    volatility: str             # "low" | "medium" | "high"
    avg_sentiment: float        # hybrid_score mean
    positive_pct: float
    negative_pct: float
    mention_count: int
    purchase_intent_pct: float  # purchase_intent_ratio * 100
    trend_direction: str        # "↑" | "↓" | "→"
    health_score: float         # 0.0–1.0
    signal: str                 # "🟢 Scale Up" | "🟡 Hold" | "🔴 Watch"


@dataclass
class BrandIntelligenceResult:
    """Output of BrandIntelligenceAnalyzer.analyze()."""

    brands: list[BrandIntelRow]
    summary_df: pd.DataFrame    # display-ready


class BrandIntelligenceAnalyzer:
    """Combine StockX market data with Reddit sentiment into a unified brand health score."""

    BRANDS = list(STOCKX_BRAND_MARKET.keys())
    MIN_TREND_WEEKS = 2
    VOLATILITY_NORM = {"low": 0.33, "medium": 0.67, "high": 1.0}

    def analyze(self, df: pd.DataFrame, min_mentions: int = 5) -> BrandIntelligenceResult:
        """Analyze all five tracked brands and compute health scores.

        Args:
            df: Annotated Reddit DataFrame.
            min_mentions: Brands with fewer mentions are flagged as low-data but still included.

        Returns:
            BrandIntelligenceResult with per-brand rows and display-ready DataFrame.
        """
        # Reddit signals via existing analyzer
        brand_corr = PriceCorrelationAnalyzer().analyze_brand_level(df)
        reddit_by_brand: dict[str, dict] = {
            s.brand: {
                "avg_sentiment": s.avg_sentiment,
                "positive_pct": s.positive_pct,
                "negative_pct": s.negative_pct,
                "mention_count": s.mention_count,
                "purchase_intent_ratio": s.purchase_intent_ratio,
            }
            for s in brand_corr.signals
        }

        # Trend directions
        trend_dirs = self._trend_directions(df)

        # Build raw rows for scoring
        rows_data: list[dict] = []
        for brand in self.BRANDS:
            market = STOCKX_BRAND_MARKET[brand]
            reddit = reddit_by_brand.get(brand, {
                "avg_sentiment": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "mention_count": 0,
                "purchase_intent_ratio": 0.0,
            })
            rows_data.append({
                "brand": brand,
                "stockx_premium": float(market["premium"]),
                "avg_deadstock": float(market["avg_deadstock"]),
                "volatility": str(market["volatility"]),
                "avg_sentiment": reddit["avg_sentiment"],
                "positive_pct": reddit["positive_pct"],
                "negative_pct": reddit["negative_pct"],
                "mention_count": reddit["mention_count"],
                "purchase_intent_pct": round(reddit["purchase_intent_ratio"] * 100, 1),
                "trend_direction": trend_dirs.get(brand, "→"),
            })

        health_scores = self._compute_health_scores(rows_data)

        brand_rows: list[BrandIntelRow] = []
        for row, score in zip(rows_data, health_scores):
            brand_rows.append(BrandIntelRow(
                brand=row["brand"],
                stockx_premium=row["stockx_premium"],
                avg_deadstock=row["avg_deadstock"],
                volatility=row["volatility"],
                avg_sentiment=row["avg_sentiment"],
                positive_pct=row["positive_pct"],
                negative_pct=row["negative_pct"],
                mention_count=row["mention_count"],
                purchase_intent_pct=row["purchase_intent_pct"],
                trend_direction=row["trend_direction"],
                health_score=round(score, 4),
                signal=self._classify_signal(score),
            ))

        summary_df = self._to_summary_df(brand_rows, min_mentions)
        return BrandIntelligenceResult(brands=brand_rows, summary_df=summary_df)

    def _trend_directions(self, df: pd.DataFrame) -> dict[str, str]:
        """Compute weekly trend direction per brand from sentiment time series."""
        from reddit_sentiment.analysis.trends import SentimentTrendAnalyzer

        try:
            trend_result = SentimentTrendAnalyzer().analyze(df, by_brand=True)
            weekly = trend_result.weekly
        except Exception:
            return {}

        if weekly.empty or "brands" not in weekly.columns:
            return {}

        directions: dict[str, str] = {}
        for brand in self.BRANDS:
            brand_weekly = weekly[weekly["brands"] == brand].sort_values("period")
            if len(brand_weekly) < self.MIN_TREND_WEEKS:
                directions[brand] = "→"
                continue
            last = brand_weekly["avg_sentiment"].iloc[-1]
            prev = brand_weekly["avg_sentiment"].iloc[-2]
            slope = last - prev
            if slope > 0.02:
                directions[brand] = "↑"
            elif slope < -0.02:
                directions[brand] = "↓"
            else:
                directions[brand] = "→"

        return directions

    @staticmethod
    def _compute_health_scores(rows_data: list[dict]) -> list[float]:
        """Min-max normalise key signals and compute weighted health score."""
        eps = 1e-9

        def norm(vals: list[float]) -> list[float]:
            lo, hi = min(vals), max(vals)
            return [(v - lo) / (hi - lo + eps) for v in vals]

        premiums = norm([r["stockx_premium"] for r in rows_data])
        deadstocks = norm([r["avg_deadstock"] for r in rows_data])
        sentiments = norm([r["avg_sentiment"] for r in rows_data])
        intents = norm([r["purchase_intent_pct"] for r in rows_data])

        scores = []
        for p, d, s, i in zip(premiums, deadstocks, sentiments, intents):
            scores.append(0.40 * p + 0.25 * d + 0.20 * s + 0.15 * i)
        return scores

    @staticmethod
    def _classify_signal(score: float) -> str:
        if score >= 0.6:
            return "🟢 Scale Up"
        if score >= 0.4:
            return "🟡 Hold"
        return "🔴 Watch"

    @staticmethod
    def _to_summary_df(rows: list[BrandIntelRow], min_mentions: int) -> pd.DataFrame:
        records = []
        for r in rows:
            records.append({
                "Brand": r.brand,
                "Trend": r.trend_direction,
                "Signal": r.signal,
                "Health Score": r.health_score,
                "StockX Premium %": round(r.stockx_premium * 100, 1),
                "Avg Deadstock $": r.avg_deadstock,
                "Volatility": r.volatility,
                "Avg Sentiment": r.avg_sentiment,
                "Positive %": r.positive_pct,
                "Negative %": r.negative_pct,
                "Mentions": r.mention_count,
                "Purchase Intent %": r.purchase_intent_pct,
                "Data Quality": "✓" if r.mention_count >= min_mentions else "⚠ Low data",
            })
        return pd.DataFrame(records)
