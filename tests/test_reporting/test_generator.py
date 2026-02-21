"""Tests for ReportGenerator (end-to-end with synthetic data)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from reddit_sentiment.reporting.generator import ReportGenerator


def _make_annotated_df() -> pd.DataFrame:
    """Synthetic annotated DataFrame that resembles pipeline output."""
    now = datetime(2024, 3, 15, tzinfo=UTC)
    rows = []
    brands_cycle = [["Nike"], ["Adidas"], ["New Balance"], ["Nike", "Adidas"], []]
    channels_cycle = [["StockX"], ["GOAT"], [], ["Foot Locker"], []]
    intents = ["completed_purchase", "seeking_purchase", "price_discussion", None, None]
    scores = [0.7, -0.3, 0.1, 0.5, -0.1]

    for i in range(20):
        rows.append(
            {
                "id": f"post_{i}",
                "subreddit": "Sneakers" if i % 2 == 0 else "Nike",
                "full_text": f"Sample text about sneakers {i}",
                "record_type": "post" if i < 10 else "comment",
                "created_utc": now,
                "score": 100 - i * 3,
                "vader_score": scores[i % 5],
                "hybrid_score": scores[i % 5],
                "transformer_score": None,
                "brands": brands_cycle[i % 5],
                "channels": channels_cycle[i % 5],
                "primary_intent": intents[i % 5],
                "all_intents": [intents[i % 5]] if intents[i % 5] else [],
            }
        )
    return pd.DataFrame(rows)


def test_generate_creates_html(tmp_path):
    df = _make_annotated_df()
    gen = ReportGenerator(reports_dir=tmp_path)
    html_path, md_path = gen.generate(df, timestamp="20240315_120000")

    assert html_path.exists()
    assert md_path.exists()

    html_content = html_path.read_text(encoding="utf-8")
    assert "Reddit Sneaker Sentiment Report" in html_content
    assert "chart-bar" in html_content
    assert "Plotly" in html_content or "plotly" in html_content


def test_generate_markdown_contains_brand_table(tmp_path):
    df = _make_annotated_df()
    gen = ReportGenerator(reports_dir=tmp_path)
    _, md_path = gen.generate(df, timestamp="20240315_120001")

    md_content = md_path.read_text(encoding="utf-8")
    assert "## Brand Rankings" in md_content
    assert "## Top Retail Channels" in md_content
    assert "## Purchase Intent" in md_content


def test_generate_empty_df(tmp_path):
    """Should not crash on an empty DataFrame."""
    df = pd.DataFrame(
        columns=[
            "id",
            "subreddit",
            "full_text",
            "record_type",
            "created_utc",
            "score",
            "vader_score",
            "hybrid_score",
            "transformer_score",
            "brands",
            "channels",
            "primary_intent",
            "all_intents",
        ]
    )
    gen = ReportGenerator(reports_dir=tmp_path)
    # Should complete without raising
    html_path, md_path = gen.generate(df, timestamp="20240315_120002")
    assert html_path.exists()
