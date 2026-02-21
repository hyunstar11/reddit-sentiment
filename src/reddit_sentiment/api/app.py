"""FastAPI application for the Reddit Sneaker Sentiment service."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from reddit_sentiment.analysis.brand_comparison import BrandComparisonAnalyzer
from reddit_sentiment.analysis.channel_attribution import ChannelAttributionAnalyzer
from reddit_sentiment.analysis.narrative import NarrativeThemeExtractor
from reddit_sentiment.analysis.trends import SentimentTrendAnalyzer
from reddit_sentiment.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    BrandEntry,
    BrandsResponse,
    ChannelsResponse,
    HealthResponse,
    ThemeEntry,
    ThemesResponse,
    TrendPoint,
    TrendsResponse,
)
from reddit_sentiment.config import collection_config
from reddit_sentiment.detection.brands import BrandDetector
from reddit_sentiment.detection.models import ModelDetector

_ANNOTATED = collection_config.processed_data_dir / "annotated.parquet"

# ---------------------------------------------------------------------------
# Data loading — cached for the lifetime of the process
# ---------------------------------------------------------------------------

_df_cache: pd.DataFrame | None = None


def _load_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if not _ANNOTATED.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Annotated data not found. "
                    "Run 'reddit-sentiment analyze' first."
                ),
            )
        _df_cache = pd.read_parquet(_ANNOTATED)
    return _df_cache


# ---------------------------------------------------------------------------
# Detectors — initialised once at startup
# ---------------------------------------------------------------------------

_brand_detector = BrandDetector()
_model_detector = ModelDetector()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: RUF029
    # Warm up: pre-load data if it exists so first request is fast
    if _ANNOTATED.exists():
        _load_df()
    yield


app = FastAPI(
    title="Reddit Sneaker Sentiment API",
    description=(
        "Analyse Reddit sentiment data for sneaker brands. "
        "Run 'reddit-sentiment serve' to start the server."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness check — returns record count from annotated dataset."""
    try:
        df = _load_df()
        records = len(df)
    except HTTPException:
        records = 0
    return HealthResponse(
        status="ok",
        records=records,
        data_path=str(_ANNOTATED),
    )


@app.get("/brands", response_model=BrandsResponse, tags=["Analysis"])
def brands(min_mentions: int = 5) -> BrandsResponse:
    """Brand sentiment ranking.

    Args:
        min_mentions: Exclude brands with fewer than this many mentions.
    """
    df = _load_df()
    analyzer = BrandComparisonAnalyzer()
    table = analyzer.comparison_table(df, min_mentions=min_mentions)
    if table.empty:
        return BrandsResponse(brands=[], total_brands=0)

    entries = [
        BrandEntry(
            brand=row["brand"],
            mentions=int(row["mentions"]),
            avg_sentiment=float(row["avg_sentiment"]),
            sentiment=row["sentiment"],
            positive_pct=float(row["positive_%"]),
            negative_pct=float(row["negative_%"]),
        )
        for _, row in table.iterrows()
    ]
    return BrandsResponse(brands=entries, total_brands=len(entries))


@app.get("/themes", response_model=ThemesResponse, tags=["Analysis"])
def themes() -> ThemesResponse:
    """Narrative theme frequency across the corpus."""
    df = _load_df()
    extractor = NarrativeThemeExtractor()
    narrative = extractor.extract(df)

    theme_entries = [
        ThemeEntry(
            theme=theme,
            count=cnt,
            pct=round(narrative.theme_percentages.get(theme, 0.0), 2),
        )
        for theme, cnt in sorted(
            narrative.theme_counts.items(), key=lambda x: x[1], reverse=True
        )
    ]
    return ThemesResponse(
        themes=theme_entries,
        top_tfidf=narrative.top_tfidf_terms[:20],
    )


@app.get("/channels", response_model=ChannelsResponse, tags=["Analysis"])
def channels() -> ChannelsResponse:
    """Retail channel attribution and purchase intent funnel."""
    df = _load_df()
    analyzer = ChannelAttributionAnalyzer()
    attribution = analyzer.analyze(df)
    return ChannelsResponse(
        channel_counts=dict(attribution.channel_counts),
        channel_share={k: round(v, 2) for k, v in attribution.channel_share.items()},
        top_channels=list(attribution.top_channels),
        intent_funnel=dict(attribution.intent_funnel),
    )


@app.get("/trends", response_model=TrendsResponse, tags=["Analysis"])
def trends() -> TrendsResponse:
    """Weekly and monthly sentiment trends."""
    df = _load_df()
    analyzer = SentimentTrendAnalyzer()
    result = analyzer.analyze(df)

    def _to_points(trend_df: pd.DataFrame) -> list[TrendPoint]:
        if trend_df.empty:
            return []
        return [
            TrendPoint(
                period=str(row["period"]),
                avg_sentiment=round(float(row["avg_sentiment"]), 4),
                count=int(row["count"]),
            )
            for _, row in trend_df.iterrows()
        ]

    return TrendsResponse(
        weekly=_to_points(result.weekly),
        monthly=_to_points(result.monthly),
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze arbitrary text: detect brands, shoe models, and score sentiment.

    Useful for scoring individual Reddit posts or any freeform sneaker text.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    brands_found = _brand_detector.detect_brands(text)
    models_found = _model_detector.detect_models(text)

    vader = SentimentIntensityAnalyzer()
    score = vader.polarity_scores(text)["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return AnalyzeResponse(
        text=text,
        brands=brands_found,
        models=models_found,
        vader_score=round(score, 4),
        sentiment_label=label,
    )
