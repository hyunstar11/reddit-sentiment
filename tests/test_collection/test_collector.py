"""Tests for SubredditCollector using mocked PRAW."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from reddit_sentiment.collection.collector import SubredditCollector, _extract_urls

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_submission(sid: str = "s1", title: str = "Test Post", body: str = "", score: int = 10):
    sub = MagicMock()
    sub.id = sid
    sub.title = title
    sub.selftext = body
    sub.author = MagicMock()
    sub.author.__str__ = lambda self: "testuser"
    sub.score = score
    sub.upvote_ratio = 0.9
    sub.num_comments = 5
    sub.created_utc = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
    sub.url = f"https://reddit.com/{sid}"
    sub.permalink = f"/r/Sneakers/{sid}"
    sub.is_self = True
    sub.link_flair_text = None
    # comments mock
    comments_mock = MagicMock()
    comments_mock.replace_more.return_value = None
    comments_mock.list.return_value = []
    sub.comments = comments_mock
    return sub


def _make_comment(cid: str = "c1", body: str = "Nice!"):
    import praw

    cmt = MagicMock(spec=praw.models.Comment)
    cmt.id = cid
    cmt.body = body
    cmt.author = MagicMock()
    cmt.author.__str__ = lambda self: "commenter"
    cmt.score = 2
    cmt.created_utc = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
    cmt.permalink = f"/r/Sneakers/{cid}"
    cmt.parent_id = "t3_s1"
    cmt.depth = 0
    return cmt


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------


def test_extract_urls_finds_https():
    text = "Check this out https://stockx.com/buy/shoe and https://goat.com"
    urls = _extract_urls(text)
    assert "https://stockx.com/buy/shoe" in urls
    assert "https://goat.com" in urls


def test_extract_urls_empty():
    assert _extract_urls("") == []
    assert _extract_urls("no urls here!") == []


# ---------------------------------------------------------------------------
# SubredditCollector
# ---------------------------------------------------------------------------


def _build_collector(tmp_path: Path) -> SubredditCollector:
    from reddit_sentiment.config import CollectionConfig

    cfg = CollectionConfig(
        subreddits=["Sneakers"],
        posts_per_subreddit=6,
        comments_per_post=5,
        sort_methods=["hot"],
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )
    collector = SubredditCollector.__new__(SubredditCollector)
    collector._cfg = cfg
    collector._cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
    collector._checkpoint_path = cfg.raw_data_dir / "checkpoint.json"
    return collector


@patch("reddit_sentiment.collection.collector.RedditClient")
def test_collect_creates_parquet(MockClient, tmp_path):
    collector = _build_collector(tmp_path)

    # Patch the PRAW subreddit call
    sub_mock = MagicMock()
    sub_mock.hot.return_value = [_make_submission("s1"), _make_submission("s2")]
    collector._client = MagicMock()
    collector._client.subreddit.return_value = sub_mock

    out = collector.collect(output_path=tmp_path / "raw" / "test.parquet")
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert "full_text" in df.columns
    assert "record_type" in df.columns


@patch("reddit_sentiment.collection.collector.RedditClient")
def test_collect_skips_checkpointed(MockClient, tmp_path):
    collector = _build_collector(tmp_path)

    # Pre-write checkpoint saying Sneakers is done
    collector._checkpoint_path.write_text(json.dumps({"Sneakers": True}))

    collector._client = MagicMock()
    out = collector.collect(output_path=tmp_path / "raw" / "test.parquet")
    df = pd.read_parquet(out)
    # Should be empty because subreddit was skipped
    assert len(df) == 0
    # Client subreddit should NOT have been called
    collector._client.subreddit.assert_not_called()


@patch("reddit_sentiment.collection.collector.RedditClient")
def test_collect_includes_comments(MockClient, tmp_path):
    collector = _build_collector(tmp_path)

    comment = _make_comment("c1", "Great shoe!")
    submission = _make_submission("s1")
    submission.comments.list.return_value = [comment]

    sub_mock = MagicMock()
    sub_mock.hot.return_value = [submission]
    collector._client = MagicMock()
    collector._client.subreddit.return_value = sub_mock

    out = collector.collect(output_path=tmp_path / "raw" / "test.parquet")
    df = pd.read_parquet(out)
    assert len(df) == 2  # 1 post + 1 comment
    types = set(df["record_type"].tolist())
    assert types == {"post", "comment"}
