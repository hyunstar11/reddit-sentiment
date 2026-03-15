"""Collector using PullPush.io — a public Pushshift mirror, no credentials needed.

Reddit's own JSON API now requires OAuth for all endpoints. PullPush.io provides
a free, unauthenticated search API over Reddit's public post archive.
API docs: https://pullpush.io
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from reddit_sentiment.collection.schemas import RedditPost
from reddit_sentiment.config import CollectionConfig

_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")
_PULLPUSH_BASE = "https://api.pullpush.io/reddit/search/submission"
_HEADERS = {"User-Agent": "reddit-sentiment/0.1 personal-research-project"}
_REQUEST_DELAY = 1.0  # seconds between requests


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


def _parse_pullpush_post(data: dict) -> RedditPost:
    subreddit = data.get("subreddit", "")
    body = data.get("selftext") or ""
    title = data.get("title") or ""
    full_text = f"{title} {body}".strip()
    urls = _extract_urls(body)
    link_url = data.get("url", "")
    if link_url and not data.get("is_self"):
        urls.insert(0, link_url)

    return RedditPost(
        id=data.get("id", ""),
        subreddit=subreddit,
        title=title,
        selftext=body,
        author=data.get("author") or "[deleted]",
        score=data.get("score", 0),
        upvote_ratio=data.get("upvote_ratio", 0.0),
        num_comments=data.get("num_comments", 0),
        created_utc=datetime.fromtimestamp(data.get("created_utc", 0), tz=UTC),
        url=link_url,
        permalink=data.get("permalink", ""),
        is_self=data.get("is_self", True),
        flair=data.get("link_flair_text"),
        full_text=full_text,
        extracted_urls=urls,
    )


class PublicSubredditCollector:
    """Collect posts from subreddits via PullPush.io (public Pushshift mirror).

    No API credentials required. PullPush.io provides free unauthenticated
    access to Reddit's public post archive after Reddit locked down its own API.
    API docs: https://pullpush.io
    """

    def __init__(self, config: CollectionConfig | None = None) -> None:
        self._cfg = config or CollectionConfig()
        self._cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _fetch_subreddit(self, subreddit: str, limit: int) -> list[dict]:
        """Fetch up to `limit` recent posts from a subreddit via PullPush."""
        records: list[dict] = []
        before: int | None = None
        batch = min(limit, 100)

        while len(records) < limit:
            params: dict = {
                "subreddit": subreddit,
                "size": min(batch, limit - len(records)),
                "sort": "desc",
                "sort_type": "created_utc",
            }
            if before:
                params["before"] = before

            try:
                resp = self._session.get(_PULLPUSH_BASE, params=params, timeout=20)
                resp.raise_for_status()
                items = resp.json().get("data", [])
            except (requests.RequestException, ValueError) as exc:
                print(f"  [!] PullPush error for r/{subreddit}: {exc}")
                break

            if not items:
                break

            for item in items:
                post = _parse_pullpush_post(item)
                records.append(post.to_dict())

            before = items[-1].get("created_utc")
            if len(items) < batch:
                break

            time.sleep(_REQUEST_DELAY)

        return records

    def collect(
        self,
        output_path: Path | None = None,
        collect_comments: bool = False,
        max_comment_posts: int = 0,
    ) -> Path:
        """Collect all configured subreddits; return path to saved Parquet file."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = self._cfg.raw_data_dir / f"posts_{timestamp}.parquet"

        all_records: list[dict] = []

        for sub_name in self._cfg.subreddits:
            print(f"[collect] r/{sub_name} via PullPush …")
            records = self._fetch_subreddit(sub_name, self._cfg.posts_per_subreddit)
            all_records.extend(records)
            print(f"  {len(records)} posts")
            time.sleep(_REQUEST_DELAY)

        df = pd.DataFrame(all_records)
        if not df.empty and "extracted_urls" in df.columns:
            df["extracted_urls"] = df["extracted_urls"].apply(
                lambda x: x if isinstance(x, list) else []
            )

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
