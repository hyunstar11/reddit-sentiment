"""Sentiment-price correlation: joins Reddit model signals with eBay sold prices."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from reddit_sentiment.detection.models import MODEL_INFO


@dataclass
class ModelSignal:
    """Combined Reddit + eBay signal for one shoe model."""

    model: str
    brand: str
    retail_price: float

    # Reddit signals
    mention_count: int = 0
    avg_sentiment: float = 0.0
    positive_pct: float = 0.0
    negative_pct: float = 0.0

    # eBay signals
    num_sales: int = 0
    avg_sold_price: float = 0.0
    min_sold_price: float = 0.0
    max_sold_price: float = 0.0
    price_premium: float = 0.0  # avg_sold / retail - 1


@dataclass
class CorrelationResult:
    """Full correlation analysis output."""

    signals: list[ModelSignal]
    correlation_sentiment_premium: float | None  # Pearson r
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class PriceCorrelationAnalyzer:
    """Join Reddit model-level sentiment with eBay sold prices and compute correlation."""

    # Minimum mentions required to include a model in correlation analysis
    MIN_MENTIONS = 3

    def analyze(
        self,
        reddit_df: pd.DataFrame,
        ebay_df: pd.DataFrame,
    ) -> CorrelationResult:
        """
        Args:
            reddit_df: Annotated Reddit DataFrame with 'models' and 'hybrid_score' columns.
            ebay_df: eBay sold listings DataFrame with 'model' and 'sold_price_usd' columns.

        Returns:
            CorrelationResult with per-model signals and correlation coefficient.
        """
        reddit_signals = self._aggregate_reddit(reddit_df)
        ebay_signals = self._aggregate_ebay(ebay_df)

        signals: list[ModelSignal] = []
        for model, reddit in reddit_signals.items():
            brand, retail = MODEL_INFO.get(model, ("Unknown", 0.0))
            sig = ModelSignal(
                model=model,
                brand=brand,
                retail_price=retail,
                **reddit,
            )
            if model in ebay_signals:
                eb = ebay_signals[model]
                sig.num_sales = eb["num_sales"]
                sig.avg_sold_price = eb["avg_sold_price"]
                sig.min_sold_price = eb["min_sold_price"]
                sig.max_sold_price = eb["max_sold_price"]
                if retail > 0:
                    sig.price_premium = round(sig.avg_sold_price / retail - 1, 4)
            signals.append(sig)

        # Sort by mention count descending
        signals.sort(key=lambda s: s.mention_count, reverse=True)

        # Pearson correlation between avg_sentiment and price_premium
        # (only for models with both Reddit mentions and eBay sales)
        corr = self._compute_correlation(signals)
        summary_df = self._to_dataframe(signals)

        return CorrelationResult(
            signals=signals,
            correlation_sentiment_premium=corr,
            summary_df=summary_df,
        )

    def _aggregate_reddit(self, df: pd.DataFrame) -> dict[str, dict]:
        """Aggregate sentiment per shoe model from annotated Reddit DataFrame."""
        if "models" not in df.columns or "hybrid_score" not in df.columns:
            return {}

        # Normalise: Parquet round-trips numpy arrays, so convert to plain lists first
        df = df.copy()
        df["models"] = df["models"].apply(
            lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []
        )
        exploded = df.explode("models")
        exploded = exploded[
            exploded["models"].notna()
            & (exploded["models"].astype(str).str.strip() != "")
            & (exploded["models"].astype(str) != "nan")
        ]

        if exploded.empty:
            return {}

        result = {}
        for model, grp in exploded.groupby("models"):
            if len(grp) < self.MIN_MENTIONS:
                continue
            scores = grp["hybrid_score"].dropna()
            pos = (scores > 0.05).sum()
            neg = (scores < -0.05).sum()
            n = len(scores)
            result[model] = {
                "mention_count": n,
                "avg_sentiment": round(float(scores.mean()), 4),
                "positive_pct": round(pos / n * 100, 1) if n else 0.0,
                "negative_pct": round(neg / n * 100, 1) if n else 0.0,
            }
        return result

    def _aggregate_ebay(self, df: pd.DataFrame) -> dict[str, dict]:
        """Aggregate sold price stats per shoe model from eBay DataFrame."""
        if df.empty or "model" not in df.columns or "sold_price_usd" not in df.columns:
            return {}

        result = {}
        for model, grp in df.groupby("model"):
            prices = grp["sold_price_usd"].dropna()
            if prices.empty:
                continue
            result[model] = {
                "num_sales": len(prices),
                "avg_sold_price": round(float(prices.mean()), 2),
                "min_sold_price": round(float(prices.min()), 2),
                "max_sold_price": round(float(prices.max()), 2),
            }
        return result

    def _compute_correlation(self, signals: list[ModelSignal]) -> float | None:
        """Pearson r between avg_sentiment and price_premium for models with both signals."""
        paired = [
            (s.avg_sentiment, s.price_premium)
            for s in signals
            if s.num_sales > 0 and s.retail_price > 0
        ]
        if len(paired) < 3:
            return None
        df = pd.DataFrame(paired, columns=["sentiment", "premium"])
        return round(float(df["sentiment"].corr(df["premium"])), 4)

    @staticmethod
    def _to_dataframe(signals: list[ModelSignal]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "model": s.model,
                "brand": s.brand,
                "retail_price": s.retail_price,
                "mentions": s.mention_count,
                "avg_sentiment": s.avg_sentiment,
                "positive_%": s.positive_pct,
                "negative_%": s.negative_pct,
                "num_sales": s.num_sales,
                "avg_sold_price": s.avg_sold_price,
                "price_premium_%": round(s.price_premium * 100, 1),
            }
            for s in signals
        ])
