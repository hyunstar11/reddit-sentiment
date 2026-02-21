"""Requests-based collector using Reddit's public JSON API (no credentials needed)."""

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
_BASE = "https://www.reddit.com"
_HEADERS = {"User-Agent": "reddit-sentiment/0.1 personal-research-project"}
_REQUEST_DELAY = 1.2  # seconds between requests (Reddit rate limit: 60/min without auth)


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


def _parse_post_json(data: dict, subreddit: str) -> RedditPost:
    body = data.get("selftext") or ""
    title = data.get("title") or ""
    full_text = f"{title} {body}".strip()
    urls = _extract_urls(body)
    link_url = data.get("url", "")
    if not data.get("is_self") and link_url:
        urls.insert(0, link_url)

    return RedditPost(
        id=data["id"],
        subreddit=subreddit,
        title=title,
        selftext=body,
        author=data.get("author") or "[deleted]",
        score=data.get("score", 0),
        upvote_ratio=data.get("upvote_ratio", 0.0),
        num_comments=data.get("num_comments", 0),
        created_utc=datetime.fromtimestamp(data["created_utc"], tz=UTC),
        url=link_url,
        permalink=data.get("permalink", ""),
        is_self=data.get("is_self", True),
        flair=data.get("link_flair_text"),
        full_text=full_text,
        extracted_urls=urls,
    )


class PublicSubredditCollector:
    """Collect posts from subreddits using Reddit's public JSON API.

    No API credentials required. Uses pagination (after token) to collect
    up to posts_per_subreddit posts per subreddit across configured sort methods.
    Comments are skipped to minimise request count; post titles + selftexts
    provide sufficient text for sentiment analysis.
    """

    def __init__(self, config: CollectionConfig | None = None) -> None:
        self._cfg = config or CollectionConfig()
        self._cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_listing(self, subreddit: str, sort: str, limit: int) -> list[dict]:
        """Paginate through a subreddit listing and return up to `limit` post dicts."""
        records: list[dict] = []
        after: str | None = None
        batch = 100  # max Reddit allows per request

        while len(records) < limit:
            url = f"{_BASE}/r/{subreddit}/{sort}.json"
            params: dict = {"limit": min(batch, limit - len(records)), "raw_json": 1}
            if sort == "top":
                params["t"] = "month"
            if after:
                params["after"] = after

            try:
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"  [!] request error: {exc}")
                break

            payload = resp.json().get("data", {})
            children = payload.get("children", [])
            if not children:
                break

            for child in children:
                post_data = child.get("data", {})
                # Skip stickied/pinned mod posts
                if post_data.get("stickied"):
                    continue
                post = _parse_post_json(post_data, subreddit)
                records.append(post.to_dict())

            after = payload.get("after")
            if not after:
                break  # no more pages

            time.sleep(_REQUEST_DELAY)

        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, output_path: Path | None = None) -> Path:
        """Collect all configured subreddits; return path to saved Parquet file."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = self._cfg.raw_data_dir / f"posts_{timestamp}.parquet"

        per_sort = max(1, self._cfg.posts_per_subreddit // len(self._cfg.sort_methods))
        all_records: list[dict] = []

        for sub_name in self._cfg.subreddits:
            print(f"[collect] r/{sub_name}")
            seen_ids: set[str] = set()
            sub_records: list[dict] = []

            for sort in self._cfg.sort_methods:
                records = self._fetch_listing(sub_name, sort, per_sort)
                for r in records:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        sub_records.append(r)
                print(f"  {sort}: {len(records)} posts")
                time.sleep(_REQUEST_DELAY)

            all_records.extend(sub_records)
            print(f"  total unique: {len(sub_records)}")

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
