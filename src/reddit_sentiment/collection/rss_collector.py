"""Collector using Reddit's public Atom/RSS feeds — no credentials required.

Each subreddit exposes an Atom 1.0 feed at:
    https://www.reddit.com/r/{subreddit}/new/.rss

Returns the ~25 most recent posts.  No authentication needed; rate-limit
to 2 seconds per subreddit to stay within Reddit's public crawl policy.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from reddit_sentiment.collection.schemas import RedditPost
from reddit_sentiment.config import CollectionConfig

_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")
_ATOM_NS = "http://www.w3.org/2005/Atom"
_RSS_URL = "https://www.reddit.com/r/{subreddit}/new/.rss"
_HEADERS = {
    "User-Agent": "reddit-sentiment/0.1 personal-research-project",
    "Accept": "application/atom+xml, application/rss+xml, text/xml",
}
_REQUEST_DELAY = 2.0  # seconds between subreddit requests


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


def _parse_rss_entry(entry: ET.Element, subreddit: str) -> RedditPost:
    """Parse a single Atom <entry> element into a RedditPost."""

    def tag(name: str) -> str:
        return f"{{{_ATOM_NS}}}{name}"

    # Post ID: <id>https://www.reddit.com/t3_abc123</id>
    id_el = entry.find(tag("id"))
    raw_id = (id_el.text or "") if id_el is not None else ""
    post_id = raw_id.split("_")[-1] if "_" in raw_id else raw_id.split("/")[-1]

    # Title
    title_el = entry.find(tag("title"))
    title = (title_el.text or "") if title_el is not None else ""

    # Author: <author><name>username</name></author>
    author_el = entry.find(f"{tag('author')}/{tag('name')}")
    author = (author_el.text or "[deleted]") if author_el is not None else "[deleted]"

    # Timestamp: <updated>2026-03-17T08:00:00+00:00</updated>
    updated_el = entry.find(tag("updated"))
    updated_str = (updated_el.text or "") if updated_el is not None else ""
    try:
        created_utc = datetime.fromisoformat(
            updated_str.replace("Z", "+00:00")
        ).astimezone(UTC)
    except (ValueError, AttributeError):
        created_utc = datetime.now(tz=UTC)

    # URL: <link href="https://www.reddit.com/r/sub/comments/ID/title/"/>
    link_el = entry.find(tag("link"))
    link = link_el.get("href", "") if link_el is not None else ""
    # Build a /r/... permalink from the full URL
    try:
        permalink = "/" + "/".join(link.split("/")[3:])
    except IndexError:
        permalink = link

    # Body: <content type="html"> ... </content>  (strip HTML tags)
    # Real RSS: HTML is entity-encoded → lives in content_el.text
    # Tests / some feeds: raw HTML is parsed as child XML elements → use itertext()
    content_el = entry.find(tag("content"))
    if content_el is not None:
        raw_content = "".join(content_el.itertext())
    else:
        raw_content = ""
    body = re.sub(r"<[^>]+>", " ", raw_content)
    body = re.sub(r"\s+", " ", body).strip()

    full_text = f"{title} {body}".strip()
    urls = _extract_urls(body)

    return RedditPost(
        id=post_id,
        subreddit=subreddit,
        title=title,
        selftext=body,
        author=author,
        score=0,            # not available in RSS
        upvote_ratio=0.0,   # not available in RSS
        num_comments=0,     # not available in RSS
        created_utc=created_utc,
        url=link,
        permalink=permalink,
        is_self=True,
        flair=None,
        full_text=full_text,
        extracted_urls=urls,
    )


class RSSSubredditCollector:
    """Collect posts via Reddit's public Atom/RSS feeds.

    No API credentials required.  Returns the ~25 most recent posts per
    subreddit per run.  Designed for incremental weekly refresh: merge the
    collected raw parquet with existing annotated.parquet to grow the dataset.
    """

    def __init__(self, config: CollectionConfig | None = None) -> None:
        self._cfg = config or CollectionConfig()
        self._cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _fetch_subreddit(self, subreddit: str) -> list[dict]:
        """Fetch the most recent posts from a subreddit via its RSS feed."""
        url = _RSS_URL.format(subreddit=subreddit)
        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [!] RSS error for r/{subreddit}: {exc}")
            return []

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            print(f"  [!] XML parse error for r/{subreddit}: {exc}")
            return []

        entries = root.findall(f"{{{_ATOM_NS}}}entry")
        return [_parse_rss_entry(e, subreddit).to_dict() for e in entries]

    def collect(self, output_path: Path | None = None) -> Path:
        """Collect all configured subreddits; return path to saved Parquet file."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = self._cfg.raw_data_dir / f"posts_{timestamp}.parquet"

        all_records: list[dict] = []
        for sub_name in self._cfg.subreddits:
            print(f"[collect] r/{sub_name} via RSS …")
            records = self._fetch_subreddit(sub_name)
            all_records.extend(records)
            print(f"  {len(records)} posts")
            time.sleep(_REQUEST_DELAY)

        df = pd.DataFrame(all_records)
        if not df.empty and "extracted_urls" in df.columns:
            df["extracted_urls"] = df["extracted_urls"].apply(
                lambda x: x if isinstance(x, list) else []
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\n[collect] {len(df)} rows saved → {output_path}")
        return output_path

    @classmethod
    def load_latest(cls, data_dir: Path) -> pd.DataFrame:
        """Load the most recently created Parquet file from data_dir."""
        files = sorted(data_dir.glob("posts_*.parquet"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        return pd.read_parquet(files[0])
