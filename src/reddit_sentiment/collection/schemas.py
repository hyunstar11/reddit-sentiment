"""Dataclasses for Reddit post and comment records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RedditPost:
    """A single Reddit post (submission)."""

    id: str
    subreddit: str
    title: str
    selftext: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    url: str
    permalink: str
    is_self: bool
    flair: str | None
    # Full text = title + selftext (populated by collector)
    full_text: str = ""
    # URLs extracted from text body
    extracted_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subreddit": self.subreddit,
            "title": self.title,
            "selftext": self.selftext,
            "author": self.author,
            "score": self.score,
            "upvote_ratio": self.upvote_ratio,
            "num_comments": self.num_comments,
            "created_utc": self.created_utc,
            "url": self.url,
            "permalink": self.permalink,
            "is_self": self.is_self,
            "flair": self.flair,
            "full_text": self.full_text,
            "extracted_urls": self.extracted_urls,
            "record_type": "post",
        }


@dataclass
class RedditComment:
    """A single Reddit comment."""

    id: str
    post_id: str
    subreddit: str
    body: str
    author: str
    score: int
    created_utc: datetime
    permalink: str
    parent_id: str
    depth: int = 0
    extracted_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "post_id": self.post_id,
            "subreddit": self.subreddit,
            "body": self.body,
            "author": self.author,
            "score": self.score,
            "created_utc": self.created_utc,
            "permalink": self.permalink,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "extracted_urls": self.extracted_urls,
            "record_type": "comment",
            # Normalise: comments use 'body' as full_text
            "full_text": self.body,
        }
