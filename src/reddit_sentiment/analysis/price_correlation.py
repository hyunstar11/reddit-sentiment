"""Sentiment-price correlation: Reddit sentiment vs. resale price premium.

Primary path: joins model-level Reddit signals with eBay sold prices (requires EBAY_APP_ID).
Fallback path: brand-level correlation using StockX 2023 market data (no API needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from reddit_sentiment.detection.models import MODEL_INFO

# Brand-level resale premiums derived from StockX/sneakers2023 market snapshot.
# Used as fallback when eBay data is unavailable.
# Source: sneakers2023.csv — median pricePremium per brand (price / retail - 1).
STOCKX_BRAND_PREMIUMS: dict[str, float] = {
    "Adidas":      0.3070,
    "Asics":       0.1290,
    "New Balance": 0.2185,
    "Nike":        0.1770,
    "Puma":        0.4500,
}

# Extended market snapshot: StockX premium + deadstock volume from sneakers2023.csv (NB08).
# volatility tier: "low" | "medium" | "high"
STOCKX_BRAND_MARKET: dict[str, dict[str, object]] = {
    "Adidas":      {"premium": 0.3070, "avg_deadstock": 9630,  "volatility": "medium"},
    "Asics":       {"premium": 0.1290, "avg_deadstock": 1000,  "volatility": "low"},
    "New Balance": {"premium": 0.2185, "avg_deadstock": 2886,  "volatility": "low"},
    "Nike":        {"premium": 0.1770, "avg_deadstock": 6265,  "volatility": "medium"},
    "Puma":        {"premium": 0.4500, "avg_deadstock": 2967,  "volatility": "low"},
}


@dataclass
class BrandSignal:
    """Combined Reddit sentiment + StockX resale premium for one brand."""

    brand: str
    avg_sentiment: float
    mention_count: int
    positive_pct: float
    negative_pct: float
    stockx_premium: float        # median price / retail - 1 from sneakers2023
    purchase_intent_ratio: float = 0.0


@dataclass
class BrandCorrelationResult:
    """Brand-level correlation output (StockX fallback)."""

    signals: list[BrandSignal]
    correlation_sentiment_premium: float | None
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)


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

    def analyze_brand_level(self, reddit_df: pd.DataFrame) -> BrandCorrelationResult:
        """Brand-level correlation using StockX premiums as the price signal.

        Fallback when eBay data is unavailable. Aggregates Reddit sentiment per brand
        and joins with StockX 2023 median resale premiums.
        """
        if "brands" not in reddit_df.columns or "hybrid_score" not in reddit_df.columns:
            return BrandCorrelationResult(signals=[], correlation_sentiment_premium=None)

        df = reddit_df.copy()
        df["brands"] = df["brands"].apply(
            lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []
        )
        exploded = df.explode("brands").rename(columns={"brands": "brand"})
        exploded = exploded[
            exploded["brand"].notna()
            & (exploded["brand"].astype(str).str.strip() != "")
            & exploded["brand"].isin(STOCKX_BRAND_PREMIUMS)
        ]

        if exploded.empty:
            return BrandCorrelationResult(signals=[], correlation_sentiment_premium=None)

        signals: list[BrandSignal] = []
        for brand, grp in exploded.groupby("brand"):
            scores = grp["hybrid_score"].dropna()
            if len(scores) < 3:
                continue
            n = len(scores)
            pos = (scores > 0.05).sum()
            neg = (scores < -0.05).sum()
            completed = (grp.get("primary_intent", pd.Series()) == "completed_purchase").sum()
            seeking = (grp.get("primary_intent", pd.Series()) == "seeking_purchase").sum()
            intent_ratio = round(completed / (seeking + completed + 1e-9), 3)
            signals.append(BrandSignal(
                brand=str(brand),
                avg_sentiment=round(float(scores.mean()), 4),
                mention_count=n,
                positive_pct=round(pos / n * 100, 1),
                negative_pct=round(neg / n * 100, 1),
                stockx_premium=STOCKX_BRAND_PREMIUMS[str(brand)],
                purchase_intent_ratio=intent_ratio,
            ))

        signals.sort(key=lambda s: s.mention_count, reverse=True)

        # Pearson r: sentiment vs StockX premium
        corr = None
        if len(signals) >= 3:
            pairs = pd.DataFrame([
                {"sentiment": s.avg_sentiment, "premium": s.stockx_premium}
                for s in signals
            ])
            corr = round(float(pairs["sentiment"].corr(pairs["premium"])), 4)

        summary_df = pd.DataFrame([
            {
                "brand": s.brand,
                "mentions": s.mention_count,
                "avg_sentiment": s.avg_sentiment,
                "positive_%": s.positive_pct,
                "negative_%": s.negative_pct,
                "stockx_premium_%": round(s.stockx_premium * 100, 1),
                "purchase_intent_%": round(s.purchase_intent_ratio * 100, 1),
            }
            for s in signals
        ])

        return BrandCorrelationResult(
            signals=signals,
            correlation_sentiment_premium=corr,
            summary_df=summary_df,
        )

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
