"""PRAW wrapper for read-only Reddit access."""

from __future__ import annotations

import praw

from reddit_sentiment.config import RedditConfig


class RedditClient:
    """Thin wrapper around a read-only PRAW Reddit instance."""

    def __init__(self, config: RedditConfig | None = None) -> None:
        cfg = config or RedditConfig()
        self._reddit = praw.Reddit(
            client_id=cfg.client_id,
            client_secret=cfg.client_secret,
            user_agent=cfg.user_agent,
            # Explicitly read-only — never prompts for username/password
            read_only=True,
        )

    @property
    def reddit(self) -> praw.Reddit:
        return self._reddit

    def subreddit(self, name: str) -> praw.models.Subreddit:
        return self._reddit.subreddit(name)

    def is_authenticated(self) -> bool:
        """Return True if credentials look valid (client_id set)."""
        return bool(self._reddit.config.client_id)
