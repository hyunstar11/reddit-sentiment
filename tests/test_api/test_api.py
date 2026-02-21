"""Tests for the FastAPI REST API endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from reddit_sentiment.api.app import app

# ---------------------------------------------------------------------------
# Shared sample DataFrame fixture
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {
        "id": f"t3_{i}",
        "subreddit": sub,
        "record_type": rtype,
        "score": 100,
        "created_utc": pd.Timestamp("2026-01-15", tz="UTC") + pd.Timedelta(days=i),
        "full_text": text,
        "vader_score": sent,
        "hybrid_score": sent,
        "transformer_score": None,
        "brands": brands,
        "channels": channels,
        "primary_intent": intent,
        "all_intents": [intent] if intent else [],
        "models": [],
    }
    for i, (sub, rtype, text, sent, brands, channels, intent) in enumerate([
        ("Sneakers", "post", "Love my Nike Air Max", 0.8, ["Nike"], ["StockX"],
         "completed_purchase"),
        ("Sneakers", "post", "Adidas Samba is fire", 0.6, ["Adidas"], ["GOAT"],
         "seeking_purchase"),
        ("Nike", "post", "Nike Dunk Low review", 0.5, ["Nike"], [], None),
        ("Nike", "comment", "These Dunks are great", 0.7, ["Nike"], ["Foot Locker"],
         "completed_purchase"),
        ("Adidas", "post", "Yeezy 350 hype is real", 0.3, ["Adidas"], ["eBay"], "marketplace"),
        ("Adidas", "comment", "Adidas quality dropped", -0.4, ["Adidas"], [], None),
        ("Jordans", "post", "Air Jordan 1 Chicago colorway", 0.9, ["Nike"], [], None),
        ("Jordans", "comment", "Jordan 1s never go out of style", 0.8, ["Nike"], ["GOAT"],
         "purchase_consideration"),
        ("Sneakers", "post", "New Balance 990 comfort test", 0.5, ["New Balance"], [], None),
        ("Sneakers", "comment", "NB990 fit wide feet", 0.6, ["New Balance"], [],
         "seeking_purchase"),
    ])
]

_SAMPLE_DF = pd.DataFrame(SAMPLE_ROWS)


@pytest.fixture(autouse=True)
def patch_load_df():
    """Patch _load_df so all endpoints use the sample DataFrame."""
    with patch("reddit_sentiment.api.app._load_df", return_value=_SAMPLE_DF):
        yield


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_status(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["records"] == len(_SAMPLE_DF)


def test_health_data_path(client):
    resp = client.get("/health")
    assert "data_path" in resp.json()


# ---------------------------------------------------------------------------
# /brands
# ---------------------------------------------------------------------------


def test_brands_returns_list(client):
    resp = client.get("/brands", params={"min_mentions": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert "brands" in data
    assert isinstance(data["brands"], list)


def test_brands_contains_nike(client):
    resp = client.get("/brands", params={"min_mentions": 1})
    brand_names = [b["brand"] for b in resp.json()["brands"]]
    assert "Nike" in brand_names


def test_brands_entry_fields(client):
    resp = client.get("/brands", params={"min_mentions": 1})
    entry = resp.json()["brands"][0]
    required = ("brand", "mentions", "avg_sentiment", "sentiment", "positive_pct", "negative_pct")
    for field in required:
        assert field in entry, f"Missing field: {field}"


def test_brands_min_mentions_filter(client):
    # Nike has 4 mentions, New Balance has 2 — with min_mentions=3 only Nike included
    resp = client.get("/brands", params={"min_mentions": 3})
    brand_names = [b["brand"] for b in resp.json()["brands"]]
    assert "Nike" in brand_names
    assert "New Balance" not in brand_names


def test_brands_total_brands_matches_list(client):
    resp = client.get("/brands", params={"min_mentions": 1})
    data = resp.json()
    assert data["total_brands"] == len(data["brands"])


# ---------------------------------------------------------------------------
# /themes
# ---------------------------------------------------------------------------


def test_themes_returns_themes(client):
    resp = client.get("/themes")
    assert resp.status_code == 200
    data = resp.json()
    assert "themes" in data
    assert isinstance(data["themes"], list)


def test_themes_entry_fields(client):
    resp = client.get("/themes")
    if resp.json()["themes"]:
        entry = resp.json()["themes"][0]
        for field in ("theme", "count", "pct"):
            assert field in entry


def test_themes_top_tfidf_is_list(client):
    resp = client.get("/themes")
    assert isinstance(resp.json()["top_tfidf"], list)


# ---------------------------------------------------------------------------
# /channels
# ---------------------------------------------------------------------------


def test_channels_returns_data(client):
    resp = client.get("/channels")
    assert resp.status_code == 200
    data = resp.json()
    assert "channel_counts" in data
    assert "intent_funnel" in data


def test_channels_intent_funnel_has_counts(client):
    resp = client.get("/channels")
    funnel = resp.json()["intent_funnel"]
    assert isinstance(funnel, dict)
    # Our sample data has marketplace and completed_purchase intents
    total = sum(funnel.values())
    assert total > 0


# ---------------------------------------------------------------------------
# /trends
# ---------------------------------------------------------------------------


def test_trends_returns_weekly_monthly(client):
    resp = client.get("/trends")
    assert resp.status_code == 200
    data = resp.json()
    assert "weekly" in data
    assert "monthly" in data


def test_trends_point_fields(client):
    resp = client.get("/trends")
    for series in ("weekly", "monthly"):
        points = resp.json()[series]
        if points:
            for field in ("period", "avg_sentiment", "count"):
                assert field in points[0]


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


def test_analyze_detects_nike(client):
    resp = client.post("/analyze", json={"text": "Just copped the Nike Air Force 1 low"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Nike" in data["brands"]


def test_analyze_positive_sentiment(client):
    resp = client.post("/analyze", json={"text": "Absolutely love these sneakers, perfect fit!"})
    data = resp.json()
    assert data["sentiment_label"] == "Positive"
    assert data["vader_score"] > 0


def test_analyze_negative_sentiment(client):
    resp = client.post("/analyze", json={"text": "Terrible quality, fell apart after one wear."})
    data = resp.json()
    assert data["sentiment_label"] == "Negative"
    assert data["vader_score"] < 0


def test_analyze_detects_shoe_model(client):
    resp = client.post("/analyze", json={"text": "My AJ1 Chicago is pristine"})
    data = resp.json()
    assert "Air Jordan 1" in data["models"]


def test_analyze_empty_text_returns_422(client):
    resp = client.post("/analyze", json={"text": ""})
    assert resp.status_code == 422


def test_analyze_response_fields(client):
    resp = client.post("/analyze", json={"text": "Nike Dunk Low looks great"})
    data = resp.json()
    for field in ("text", "brands", "models", "vader_score", "sentiment_label"):
        assert field in data
