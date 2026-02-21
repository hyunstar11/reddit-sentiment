"""Tests for PublicSubredditCollector — post listing and comment collection."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from reddit_sentiment.collection.public_collector import (
    PublicSubredditCollector,
    _parse_comment_json,
    _parse_post_json,
)
from reddit_sentiment.config import CollectionConfig

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _cfg(tmp_path, subreddits=("Sneakers",), posts_per=6, comments_per=5):
    return CollectionConfig(
        subreddits=list(subreddits),
        posts_per_subreddit=posts_per,
        comments_per_post=comments_per,
        sort_methods=["hot"],
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )


def _make_collector(tmp_path, **kwargs) -> PublicSubredditCollector:
    cfg = _cfg(tmp_path, **kwargs)
    c = PublicSubredditCollector(config=cfg)
    return c


def _post_data(pid="abc123", score=100, num_comments=10):
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
        "permalink": f"/r/Sneakers/{pid}",
        "is_self": True,
        "link_flair_text": None,
    }


def _comment_data(cid="c1", post_id="abc123", score=10, depth=0):
    return {
        "id": cid,
        "body": "Totally agree, best shoe ever",
        "author": "commenter",
        "score": score,
        "created_utc": datetime(2025, 1, 1, tzinfo=UTC).timestamp(),
        "permalink": f"/r/Sneakers/{cid}",
        "parent_id": f"t3_{post_id}",
        "depth": depth,
        "link_id": f"t3_{post_id}",
    }


def _listing_response(posts: list[dict]) -> dict:
    """Wrap raw post dicts in Reddit listing JSON structure."""
    return {
        "data": {
            "children": [{"kind": "t3", "data": p} for p in posts],
            "after": None,
        }
    }


def _comment_response(post: dict, comments: list[dict]) -> list[dict]:
    """Return the 2-element list Reddit sends for comment endpoints."""
    return [
        {"data": {"children": [{"kind": "t3", "data": post}]}},
        {
            "data": {
                "children": [{"kind": "t1", "data": c} for c in comments]
            }
        },
    ]


# ---------------------------------------------------------------------------
# _parse_post_json
# ---------------------------------------------------------------------------


def test_parse_post_json_full_text():
    data = _post_data()
    post = _parse_post_json(data, "Sneakers")
    assert "Nike Air Jordan 1" in post.full_text
    assert "Great shoe" in post.full_text


def test_parse_post_json_record_type():
    data = _post_data()
    post = _parse_post_json(data, "Sneakers")
    assert post.to_dict()["record_type"] == "post"


# ---------------------------------------------------------------------------
# _parse_comment_json
# ---------------------------------------------------------------------------


def test_parse_comment_json_full_text():
    data = _comment_data()
    comment = _parse_comment_json(data, "abc123", "Sneakers")
    assert comment.body == "Totally agree, best shoe ever"
    assert comment.to_dict()["full_text"] == comment.body


def test_parse_comment_json_record_type():
    data = _comment_data()
    comment = _parse_comment_json(data, "abc123", "Sneakers")
    assert comment.to_dict()["record_type"] == "comment"


# ---------------------------------------------------------------------------
# _fetch_comments
# ---------------------------------------------------------------------------


def test_fetch_comments_returns_comment_dicts(tmp_path):
    collector = _make_collector(tmp_path)
    post = _post_data()
    comments = [_comment_data("c1"), _comment_data("c2")]
    fake_resp = MagicMock()
    fake_resp.json.return_value = _comment_response(post, comments)
    fake_resp.raise_for_status.return_value = None
    collector._session = MagicMock()
    collector._session.get.return_value = fake_resp

    result = collector._fetch_comments("Sneakers", "abc123", limit=10)
    assert len(result) == 2
    assert all(r["record_type"] == "comment" for r in result)


def test_fetch_comments_skips_deleted(tmp_path):
    collector = _make_collector(tmp_path)
    post = _post_data()
    comments = [
        _comment_data("c1"),
        {**_comment_data("c2"), "body": "[deleted]"},
        {**_comment_data("c3"), "body": "[removed]"},
    ]
    fake_resp = MagicMock()
    fake_resp.json.return_value = _comment_response(post, comments)
    fake_resp.raise_for_status.return_value = None
    collector._session = MagicMock()
    collector._session.get.return_value = fake_resp

    result = collector._fetch_comments("Sneakers", "abc123", limit=10)
    assert len(result) == 1  # only c1 survives


def test_fetch_comments_skips_non_t1(tmp_path):
    """'more' stubs (kind != t1) should be ignored."""
    collector = _make_collector(tmp_path)
    post = _post_data()
    fake_resp = MagicMock()
    fake_resp.json.return_value = [
        {"data": {"children": [{"kind": "t3", "data": post}]}},
        {
            "data": {
                "children": [
                    {"kind": "t1", "data": _comment_data("c1")},
                    {"kind": "more", "data": {"id": "more1", "children": []}},
                ]
            }
        },
    ]
    fake_resp.raise_for_status.return_value = None
    collector._session = MagicMock()
    collector._session.get.return_value = fake_resp

    result = collector._fetch_comments("Sneakers", "abc123", limit=10)
    assert len(result) == 1


def test_fetch_comments_handles_request_error(tmp_path):
    import requests as req
    collector = _make_collector(tmp_path)
    collector._session = MagicMock()
    collector._session.get.side_effect = req.RequestException("timeout")

    result = collector._fetch_comments("Sneakers", "abc123", limit=10)
    assert result == []


# ---------------------------------------------------------------------------
# Full collect() with comments
# ---------------------------------------------------------------------------


def test_collect_with_comments_includes_comment_records(tmp_path):
    collector = _make_collector(tmp_path)

    post = _post_data(pid="p1", score=500, num_comments=20)
    listing_resp = MagicMock()
    listing_resp.raise_for_status.return_value = None
    listing_resp.json.return_value = _listing_response([post])

    comment_api_resp = MagicMock()
    comment_api_resp.raise_for_status.return_value = None
    comment_api_resp.json.return_value = _comment_response(
        post, [_comment_data("c1"), _comment_data("c2")]
    )

    # First call = listing, second call = comments
    collector._session = MagicMock()
    collector._session.get.side_effect = [listing_resp, comment_api_resp]

    with patch("reddit_sentiment.collection.public_collector.time.sleep"):
        out = collector.collect(
            output_path=tmp_path / "raw" / "test.parquet",
            collect_comments=True,
            max_comment_posts=5,
        )

    df = pd.read_parquet(out)
    assert "comment" in df["record_type"].values
    assert "post" in df["record_type"].values
    assert len(df) == 3  # 1 post + 2 comments


def test_collect_no_comments_flag(tmp_path):
    """--no-comments should skip all comment fetching."""
    collector = _make_collector(tmp_path)

    post = _post_data(pid="p1", score=500, num_comments=20)
    listing_resp = MagicMock()
    listing_resp.raise_for_status.return_value = None
    listing_resp.json.return_value = _listing_response([post])

    collector._session = MagicMock()
    collector._session.get.return_value = listing_resp

    with patch("reddit_sentiment.collection.public_collector.time.sleep"):
        out = collector.collect(
            output_path=tmp_path / "raw" / "test.parquet",
            collect_comments=False,
        )

    df = pd.read_parquet(out)
    # Only the listing endpoint should have been called (once per sort method)
    assert all(df["record_type"] == "post")
    # Comment endpoint should NOT have been called for any post
    calls = [str(c) for c in collector._session.get.call_args_list]
    assert not any("comments" in c for c in calls)


def test_collect_only_fetches_comments_for_posts_with_comments(tmp_path):
    """Posts with num_comments=0 should be skipped for comment fetching."""
    collector = _make_collector(tmp_path)

    post_with = _post_data(pid="p1", score=500, num_comments=10)
    post_without = _post_data(pid="p2", score=400, num_comments=0)

    listing_resp = MagicMock()
    listing_resp.raise_for_status.return_value = None
    listing_resp.json.return_value = _listing_response([post_with, post_without])

    comment_api_resp = MagicMock()
    comment_api_resp.raise_for_status.return_value = None
    comment_api_resp.json.return_value = _comment_response(
        post_with, [_comment_data("c1")]
    )

    collector._session = MagicMock()
    collector._session.get.side_effect = [listing_resp, comment_api_resp]

    with patch("reddit_sentiment.collection.public_collector.time.sleep"):
        out = collector.collect(
            output_path=tmp_path / "raw" / "test.parquet",
            collect_comments=True,
            max_comment_posts=10,
        )

    df = pd.read_parquet(out)
    # 2 posts + 1 comment from p1 only
    assert len(df) == 3
