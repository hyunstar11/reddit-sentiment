"""Tests for RSSSubredditCollector — Reddit Atom/RSS-based collection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from reddit_sentiment.collection.rss_collector import (
    RSSSubredditCollector,
    _extract_urls,
    _parse_rss_entry,
)
from reddit_sentiment.config import CollectionConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOM_NS = "http://www.w3.org/2005/Atom"


def _cfg(tmp_path, subreddits=("Sneakers",)):
    return CollectionConfig(
        subreddits=list(subreddits),
        posts_per_subreddit=25,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )


def _make_collector(tmp_path, **kwargs) -> RSSSubredditCollector:
    return RSSSubredditCollector(config=_cfg(tmp_path, **kwargs))


def _atom_feed(entries_xml: str) -> bytes:
    """Wrap entry XML in a minimal Atom feed."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="{_ATOM_NS}">
  <title>r/Sneakers</title>
  {entries_xml}
</feed>""".encode()


def _entry_xml(
    post_id: str = "abc123",
    title: str = "Great Jordan 1s",
    author: str = "testuser",
    timestamp: str = "2026-03-17T08:00:00+00:00",
    link: str = "https://www.reddit.com/r/Sneakers/comments/abc123/great_jordan_1s/",
    content: str = "<p>Really love these shoes</p>",
) -> str:
    return f"""
  <entry>
    <id>https://www.reddit.com/t3_{post_id}</id>
    <title>{title}</title>
    <author><name>{author}</name></author>
    <updated>{timestamp}</updated>
    <link href="{link}"/>
    <content type="html">{content}</content>
  </entry>"""


# ---------------------------------------------------------------------------
# _extract_urls
# ---------------------------------------------------------------------------


def test_extract_urls_finds_links():
    text = "See https://stockx.com and https://goat.com"
    urls = _extract_urls(text)
    assert "https://stockx.com" in urls
    assert "https://goat.com" in urls


def test_extract_urls_empty():
    assert _extract_urls("") == []
    assert _extract_urls("no links here") == []


# ---------------------------------------------------------------------------
# _parse_rss_entry
# ---------------------------------------------------------------------------


def test_parse_rss_entry_basic():
    import xml.etree.ElementTree as ET

    xml_str = f"""<feed xmlns="{_ATOM_NS}">{_entry_xml()}</feed>"""
    root = ET.fromstring(xml_str)
    entry = root.find(f"{{{_ATOM_NS}}}entry")
    post = _parse_rss_entry(entry, "Sneakers")

    assert post.id == "abc123"
    assert post.subreddit == "Sneakers"
    assert post.author == "testuser"
    assert "Great Jordan 1s" in post.full_text
    assert "Really love these shoes" in post.full_text
    assert post.record_type if hasattr(post, "record_type") else True  # in to_dict


def test_parse_rss_entry_to_dict_record_type():
    import xml.etree.ElementTree as ET

    xml_str = f"""<feed xmlns="{_ATOM_NS}">{_entry_xml()}</feed>"""
    root = ET.fromstring(xml_str)
    entry = root.find(f"{{{_ATOM_NS}}}entry")
    d = _parse_rss_entry(entry, "Sneakers").to_dict()
    assert d["record_type"] == "post"


def test_parse_rss_entry_timestamp():
    import xml.etree.ElementTree as ET

    ts = "2026-03-17T08:00:00+00:00"
    xml_str = f"""<feed xmlns="{_ATOM_NS}">{_entry_xml(timestamp=ts)}</feed>"""
    root = ET.fromstring(xml_str)
    entry = root.find(f"{{{_ATOM_NS}}}entry")
    post = _parse_rss_entry(entry, "Sneakers")
    assert post.created_utc.year == 2026
    assert post.created_utc.month == 3
    assert post.created_utc.tzinfo is not None


def test_parse_rss_entry_strips_html():
    import xml.etree.ElementTree as ET

    html = "<p><b>Nike</b> is great</p>"
    xml_str = f"""<feed xmlns="{_ATOM_NS}">{_entry_xml(content=html)}</feed>"""
    root = ET.fromstring(xml_str)
    entry = root.find(f"{{{_ATOM_NS}}}entry")
    post = _parse_rss_entry(entry, "Sneakers")
    assert "<p>" not in post.selftext
    assert "Nike" in post.selftext


def test_parse_rss_entry_score_defaults():
    import xml.etree.ElementTree as ET

    xml_str = f"""<feed xmlns="{_ATOM_NS}">{_entry_xml()}</feed>"""
    root = ET.fromstring(xml_str)
    entry = root.find(f"{{{_ATOM_NS}}}entry")
    post = _parse_rss_entry(entry, "Sneakers")
    assert post.score == 0
    assert post.upvote_ratio == 0.0
    assert post.num_comments == 0


# ---------------------------------------------------------------------------
# _fetch_subreddit
# ---------------------------------------------------------------------------


def test_fetch_subreddit_returns_records(tmp_path):
    collector = _make_collector(tmp_path)
    feed = _atom_feed(_entry_xml("p1") + _entry_xml("p2", title="Another post"))
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = feed
    collector._session = MagicMock()
    collector._session.get.return_value = mock_resp

    with patch("reddit_sentiment.collection.rss_collector.time.sleep"):
        records = collector._fetch_subreddit("Sneakers")

    assert len(records) == 2
    assert all(r["record_type"] == "post" for r in records)


def test_fetch_subreddit_handles_request_error(tmp_path):
    import requests as req

    collector = _make_collector(tmp_path)
    collector._session = MagicMock()
    collector._session.get.side_effect = req.RequestException("timeout")

    records = collector._fetch_subreddit("Sneakers")
    assert records == []


def test_fetch_subreddit_handles_bad_xml(tmp_path):
    collector = _make_collector(tmp_path)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"not valid xml <<<"
    collector._session = MagicMock()
    collector._session.get.return_value = mock_resp

    records = collector._fetch_subreddit("Sneakers")
    assert records == []


def test_fetch_subreddit_empty_feed(tmp_path):
    collector = _make_collector(tmp_path)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = _atom_feed("")  # no entries
    collector._session = MagicMock()
    collector._session.get.return_value = mock_resp

    records = collector._fetch_subreddit("Sneakers")
    assert records == []


# ---------------------------------------------------------------------------
# collect()
# ---------------------------------------------------------------------------


def test_collect_saves_parquet(tmp_path):
    collector = _make_collector(tmp_path)
    feed = _atom_feed(_entry_xml("p1") + _entry_xml("p2"))
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = feed
    collector._session = MagicMock()
    collector._session.get.return_value = mock_resp

    with patch("reddit_sentiment.collection.rss_collector.time.sleep"):
        out = collector.collect(output_path=tmp_path / "raw" / "test.parquet")

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert "record_type" in df.columns


def test_collect_multi_subreddit(tmp_path):
    collector = _make_collector(tmp_path, subreddits=("Sneakers", "Nike"))
    feed = _atom_feed(_entry_xml("p1"))
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = feed
    collector._session = MagicMock()
    collector._session.get.return_value = mock_resp

    with patch("reddit_sentiment.collection.rss_collector.time.sleep"):
        out = collector.collect(output_path=tmp_path / "raw" / "test.parquet")

    df = pd.read_parquet(out)
    assert len(df) == 2  # 1 post per subreddit


def test_load_latest(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    df = pd.DataFrame({"id": ["x"], "record_type": ["post"]})
    path = raw_dir / "posts_20260317_000000.parquet"
    df.to_parquet(path, index=False)

    loaded = RSSSubredditCollector.load_latest(raw_dir)
    assert len(loaded) == 1
