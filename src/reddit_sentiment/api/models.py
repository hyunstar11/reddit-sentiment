"""Pydantic response/request models for the Reddit Sentiment API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    records: int
    data_path: str


class BrandEntry(BaseModel):
    brand: str
    mentions: int
    avg_sentiment: float
    sentiment: str
    positive_pct: float
    negative_pct: float


class BrandsResponse(BaseModel):
    brands: list[BrandEntry]
    total_brands: int


class ThemeEntry(BaseModel):
    theme: str
    count: int
    pct: float


class ThemesResponse(BaseModel):
    themes: list[ThemeEntry]
    top_tfidf: list[str]


class ChannelsResponse(BaseModel):
    channel_counts: dict[str, int]
    channel_share: dict[str, float]
    top_channels: list[str]
    intent_funnel: dict[str, int]


class TrendPoint(BaseModel):
    period: str
    avg_sentiment: float
    count: int


class TrendsResponse(BaseModel):
    weekly: list[TrendPoint]
    monthly: list[TrendPoint]


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    text: str
    brands: list[str]
    models: list[str]
    vader_score: float
    sentiment_label: str
