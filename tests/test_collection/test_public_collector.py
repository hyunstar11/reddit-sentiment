"""Tests for PublicSubredditCollector — PullPush.io-based collection."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from reddit_sentiment.collection.public_collector import (
    PublicSubredditCollector,
    _extract_urls,
    _parse_pullpush_post,
)
from reddit_sentiment.config import CollectionConfig

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _cfg(tmp_path, subreddits=("Sneakers",), posts_per=6):
    return CollectionConfig(
        subreddits=list(subreddits),
        posts_per_subreddit=posts_per,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )


def _make_collector(tmp_path, **kwargs) -> PublicSubredditCollector:
    return PublicSubredditCollector(config=_cfg(tmp_path, **kwargs))


def _post_data(pid="abc123", score=100, num_comments=10, subreddit="Sneakers"):
    return {
        "id": pid,
        "title": "Test post about Nike Air Jordan 1",
        "selftext": "Great shoe, love it",
        "author": "testuser",
        "score": score,
        "upvote_ratio": 0.95,
        "num_comments": num_comments,
        "created_utc": datetime(2025, 1, 1, tzinfo=UTC).timestamp(),
        "url": f"https://reddit.com/{pid}",
        "permalink": f"/r/{subreddit}/{pid}",
        "is_self": True,
        "link_flair_text": None,
        "subreddit": subreddit,
    }


def _pullpush_response(posts: list[dict], has_more: bool = False) -> dict:
    """Wrap raw post dicts in PullPush API response structure."""
    return {"data": posts}


# ---------------------------------------------------------------------------
# _extract_urls
# ---------------------------------------------------------------------------


def test_extract_urls_finds_links():
    text = "Check https://stockx.com and https://goat.com for prices"
    urls = _extract_urls(text)
    assert "https://stockx.com" in urls
    assert "https://goat.com" in urls


def test_extract_urls_empty():
    assert _extract_urls("") == []
    assert _extract_urls("no links here") == []


# ---------------------------------------------------------------------------
# _parse_pullpush_post
# ---------------------------------------------------------------------------


def test_parse_pullpush_post_full_text():
    data = _post_data()
    post = _parse_pullpush_post(data)
    assert "Nike Air Jordan 1" in post.full_text
    assert "Great shoe" in post.full_text


def test_parse_pullpush_post_record_type():
    data = _post_data()
    post = _parse_pullpush_post(data)
    assert post.to_dict()["record_type"] == "post"


def test_parse_pullpush_post_fields():
    data = _post_data(pid="xyz", score=42, subreddit="Nike")
    post = _parse_pullpush_post(data)
    assert post.id == "xyz"
    assert post.score == 42
    assert post.subreddit == "Nike"


# ---------------------------------------------------------------------------
# _fetch_subreddit
# ---------------------------------------------------------------------------


def test_fetch_subreddit_returns_records(tmp_path):
    collector = _make_collector(tmp_path)
    posts = [_post_data("p1"), _post_data("p2")]
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = _pullpush_response(posts)
    collector._session = MagicMock()
    collector._session.get.return_value = fake_resp

    with patch("reddit_sentiment.collection.public_collector.time.sleep"):
        records = collector._fetch_subreddit("Sneakers", limit=10)

    assert len(records) == 2
    assert all(r["record_type"] == "post" for r in records)


def test_fetch_subreddit_handles_request_error(tmp_path):
    import requests as req
    collector = _make_collector(tmp_path)
    collector._session = MagicMock()
    collector._session.get.side_effect = req.RequestException("timeout")

    records = collector._fetch_subreddit("Sneakers", limit=10)
    assert records == []


def test_fetch_subreddit_empty_response(tmp_path):
    collector = _make_collector(tmp_path)
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {"data": []}
    collector._session = MagicMock()
    collector._session.get.return_value = fake_resp

    records = collector._fetch_subreddit("Sneakers", limit=10)
    assert records == []


# ---------------------------------------------------------------------------
# Full collect()
# ---------------------------------------------------------------------------


def test_collect_saves_parquet(tmp_path):
    collector = _make_collector(tmp_path)
    posts = [_post_data("p1"), _post_data("p2")]
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = _pullpush_response(posts)
    collector._session = MagicMock()
    collector._session.get.return_value = fake_resp

    with patch("reddit_sentiment.collection.public_collector.time.sleep"):
        out = collector.collect(output_path=tmp_path / "raw" / "test.parquet")

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert "record_type" in df.columns


def test_collect_multi_subreddit(tmp_path):
    collector = _make_collector(tmp_path, subreddits=("Sneakers", "Nike"))
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = _pullpush_response([_post_data("p1")])
    collector._session = MagicMock()
    collector._session.get.return_value = fake_resp

    with patch("reddit_sentiment.collection.public_collector.time.sleep"):
        out = collector.collect(output_path=tmp_path / "raw" / "test.parquet")

    df = pd.read_parquet(out)
    assert len(df) == 2  # 1 post per subreddit


def test_load_latest(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    df = pd.DataFrame({"id": ["x"], "record_type": ["post"]})
    path = raw_dir / "posts_20260101_000000.parquet"
    df.to_parquet(path, index=False)

    loaded = PublicSubredditCollector.load_latest(raw_dir)
    assert len(loaded) == 1
