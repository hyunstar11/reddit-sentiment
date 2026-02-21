"""SubredditCollector: pagination, deduplication, checkpointing, Parquet output."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import praw

from reddit_sentiment.collection.client import RedditClient
from reddit_sentiment.collection.schemas import RedditComment, RedditPost
from reddit_sentiment.config import CollectionConfig

# URL pattern for inline extraction
_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


def _parse_post(submission: praw.models.Submission, subreddit: str) -> RedditPost:
    body = submission.selftext or ""
    title = submission.title or ""
    full_text = f"{title} {body}".strip()
    urls = _extract_urls(body)
    if not submission.is_self and submission.url:
        urls.insert(0, submission.url)

    return RedditPost(
        id=submission.id,
        subreddit=subreddit,
        title=title,
        selftext=body,
        author=str(submission.author) if submission.author else "[deleted]",
        score=submission.score,
        upvote_ratio=submission.upvote_ratio,
        num_comments=submission.num_comments,
        created_utc=datetime.fromtimestamp(submission.created_utc, tz=UTC),
        url=submission.url,
        permalink=submission.permalink,
        is_self=submission.is_self,
        flair=submission.link_flair_text,
        full_text=full_text,
        extracted_urls=urls,
    )


def _parse_comment(comment: praw.models.Comment, post_id: str, subreddit: str) -> RedditComment:
    body = comment.body or ""
    return RedditComment(
        id=comment.id,
        post_id=post_id,
        subreddit=subreddit,
        body=body,
        author=str(comment.author) if comment.author else "[deleted]",
        score=comment.score,
        created_utc=datetime.fromtimestamp(comment.created_utc, tz=UTC),
        permalink=comment.permalink,
        parent_id=comment.parent_id,
        depth=getattr(comment, "depth", 0),
        extracted_urls=_extract_urls(body),
    )


class SubredditCollector:
    """Collect posts and comments from multiple subreddits with checkpointing."""

    def __init__(
        self,
        client: RedditClient | None = None,
        config: CollectionConfig | None = None,
    ) -> None:
        self._client = client or RedditClient()
        self._cfg = config or CollectionConfig()
        self._cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self._cfg.raw_data_dir / "checkpoint.json"

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict[str, bool]:
        if self._checkpoint_path.exists():
            return json.loads(self._checkpoint_path.read_text())
        return {}

    def _save_checkpoint(self, completed: dict[str, bool]) -> None:
        self._checkpoint_path.write_text(json.dumps(completed, indent=2))

    # ------------------------------------------------------------------
    # Per-subreddit collection
    # ------------------------------------------------------------------

    def _collect_subreddit(self, name: str) -> list[dict]:
        sub = self._client.subreddit(name)
        seen_ids: set[str] = set()
        records: list[dict] = []
        posts_limit = self._cfg.posts_per_subreddit // len(self._cfg.sort_methods)

        for sort in self._cfg.sort_methods:
            listing = {
                "hot": sub.hot,
                "top": sub.top,
                "new": sub.new,
            }.get(sort, sub.hot)

            kwargs: dict = {"limit": posts_limit}
            if sort == "top":
                kwargs["time_filter"] = "month"

            for submission in listing(**kwargs):
                if submission.id in seen_ids:
                    continue
                seen_ids.add(submission.id)

                post = _parse_post(submission, name)
                records.append(post.to_dict())

                # Fetch comments (replace MoreComments to avoid slow API calls)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[: self._cfg.comments_per_post]:
                    if not isinstance(comment, praw.models.Comment):
                        continue
                    records.append(_parse_comment(comment, submission.id, name).to_dict())

        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, output_path: Path | None = None) -> Path:
        """Collect all subreddits; returns path to saved Parquet file."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = self._cfg.raw_data_dir / f"posts_{timestamp}.parquet"

        completed = self._load_checkpoint()
        all_records: list[dict] = []

        for sub_name in self._cfg.subreddits:
            if completed.get(sub_name):
                print(f"[checkpoint] skipping {sub_name} (already collected)")
                continue

            print(f"[collect] r/{sub_name} …")
            try:
                records = self._collect_subreddit(sub_name)
                all_records.extend(records)
                completed[sub_name] = True
                self._save_checkpoint(completed)
                print(f"[collect] r/{sub_name}: {len(records)} records")
            except Exception as exc:  # noqa: BLE001
                print(f"[collect] r/{sub_name} failed: {exc}")

        df = self._to_dataframe(all_records)
        df.to_parquet(output_path, index=False)
        print(f"[collect] saved {len(df)} rows → {output_path}")

        # Clear checkpoint so next run starts fresh
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

        return output_path

    @staticmethod
    def _to_dataframe(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if df.empty:
            return df
        # Ensure list columns are stored as object (Parquet handles lists natively)
        if "extracted_urls" in df.columns:
            df["extracted_urls"] = df["extracted_urls"].apply(
                lambda x: x if isinstance(x, list) else []
            )
        return df

    @classmethod
    def load_latest(cls, data_dir: Path) -> pd.DataFrame:
        """Load the most recently created Parquet file from data_dir."""
        files = sorted(data_dir.glob("posts_*.parquet"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        return pd.read_parquet(files[0])
