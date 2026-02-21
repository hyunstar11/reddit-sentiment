"""Configuration via pydantic-settings (reads from .env or environment variables)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/reddit_sentiment/config.py)
_ROOT = Path(__file__).parent.parent.parent


class RedditConfig(BaseSettings):
    """Reddit API credentials."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")
    user_agent: str = Field(
        default="reddit-sentiment/0.1 by sneaker_researcher",
        alias="REDDIT_USER_AGENT",
    )


class CollectionConfig(BaseSettings):
    """Data-collection parameters."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="COLLECTION_",
    )

    subreddits: list[str] = Field(
        default=[
            "Sneakers",
            "SneakerMarket",
            "Nike",
            "Adidas",
            "Yeezy",
            "Jordans",
            "malefashionadvice",
            "Running",
            "Basketball",
        ]
    )
    posts_per_subreddit: int = Field(default=500, alias="COLLECTION_POSTS_PER_SUBREDDIT")
    comments_per_post: int = Field(default=50, alias="COLLECTION_COMMENTS_PER_POST")
    sort_methods: list[str] = Field(default=["hot", "top", "new"])
    raw_data_dir: Path = Field(default=_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default=_ROOT / "data" / "processed")
    reports_dir: Path = Field(default=_ROOT / "data" / "reports")

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SentimentConfig(BaseSettings):
    """Sentiment analysis parameters."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    transformer_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        alias="SENTIMENT_TRANSFORMER_MODEL",
    )
    transformer_weight: float = 0.6
    vader_weight: float = 0.4
    context_window: int = 15  # words each side of brand mention
    transformer_batch_size: int = 32
    # Score threshold for "positive" classification
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05


class EbayConfig(BaseSettings):
    """eBay Finding API credentials."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_id: str = Field(default="", alias="EBAY_APP_ID")
    # eBay sneakers category ID
    category_id: str = Field(default="15709", alias="EBAY_CATEGORY_ID")
    max_results_per_model: int = Field(default=100, alias="EBAY_MAX_RESULTS")


# Singleton instances (import these in application code)
reddit_config = RedditConfig()
collection_config = CollectionConfig()
sentiment_config = SentimentConfig()
ebay_config = EbayConfig()
