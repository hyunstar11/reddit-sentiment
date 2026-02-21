"""Tests for RedditPost and RedditComment dataclasses."""

from datetime import UTC, datetime

from reddit_sentiment.collection.schemas import RedditComment, RedditPost


def _make_post(**kwargs) -> RedditPost:
    defaults = dict(
        id="abc123",
        subreddit="Sneakers",
        title="Nike Air Max review",
        selftext="Great shoe!",
        author="user1",
        score=100,
        upvote_ratio=0.95,
        num_comments=10,
        created_utc=datetime(2024, 1, 1, tzinfo=UTC),
        url="https://reddit.com/r/Sneakers/abc",
        permalink="/r/Sneakers/abc",
        is_self=True,
        flair="Review",
        full_text="Nike Air Max review Great shoe!",
        extracted_urls=[],
    )
    defaults.update(kwargs)
    return RedditPost(**defaults)


def _make_comment(**kwargs) -> RedditComment:
    defaults = dict(
        id="cmt1",
        post_id="abc123",
        subreddit="Sneakers",
        body="I agree!",
        author="user2",
        score=5,
        created_utc=datetime(2024, 1, 1, tzinfo=UTC),
        permalink="/r/Sneakers/abc/cmt1",
        parent_id="t3_abc123",
        depth=0,
        extracted_urls=[],
    )
    defaults.update(kwargs)
    return RedditComment(**defaults)


def test_post_to_dict_has_record_type():
    post = _make_post()
    d = post.to_dict()
    assert d["record_type"] == "post"
    assert d["id"] == "abc123"
    assert d["full_text"] == "Nike Air Max review Great shoe!"


def test_comment_to_dict_has_record_type():
    cmt = _make_comment()
    d = cmt.to_dict()
    assert d["record_type"] == "comment"
    assert d["full_text"] == "I agree!"


def test_post_extracted_urls_default_empty():
    post = _make_post()
    assert post.extracted_urls == []


def test_comment_extracted_urls():
    cmt = _make_comment(extracted_urls=["https://stockx.com/buy/shoe"])
    assert len(cmt.extracted_urls) == 1
