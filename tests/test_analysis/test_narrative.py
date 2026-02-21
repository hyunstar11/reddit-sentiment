"""Tests for NarrativeThemeExtractor."""

import pandas as pd
import pytest

from reddit_sentiment.analysis.narrative import THEME_KEYWORDS, NarrativeThemeExtractor


@pytest.fixture
def extractor():
    return NarrativeThemeExtractor()


def _df(texts: list[str], brands: list[list[str]] | None = None) -> pd.DataFrame:
    data = {"full_text": texts}
    if brands is not None:
        data["brands"] = brands
    return pd.DataFrame(data)


def test_hype_theme_detected(extractor):
    df = _df(["Limited drop — entered the raffle for this collab!"])
    result = extractor.extract(df)
    assert "Hype & Exclusivity" in result.theme_counts


def test_quality_theme_detected(extractor):
    df = _df(["The cushioning and support on these are excellent"])
    result = extractor.extract(df)
    assert "Quality & Comfort" in result.theme_counts


def test_theme_percentage_sums_reasonably(extractor):
    df = _df(["hype drop", "comfortable quality", "price resale", "another text"])
    result = extractor.extract(df)
    # Each pct should be 0-100
    for pct in result.theme_percentages.values():
        assert 0 <= pct <= 100


def test_tfidf_terms_returned(extractor):
    texts = ["Nike drops limited edition colorway", "Adidas Yeezy hype resale prices"] * 3
    df = _df(texts)
    result = extractor.extract(df)
    assert len(result.top_tfidf_terms) > 0


def test_brand_themes(extractor):
    df = _df(
        ["comfortable Nike shoe", "Adidas hype drop raffle"],
        brands=[["Nike"], ["Adidas"]],
    )
    result = extractor.extract(df)
    assert "Nike" in result.brand_themes
    assert "Adidas" in result.brand_themes
    assert "Quality & Comfort" in result.brand_themes["Nike"]
    assert "Hype & Exclusivity" in result.brand_themes["Adidas"]


def test_empty_df(extractor):
    df = _df([])
    result = extractor.extract(df)
    assert result.theme_counts == {}


def test_all_themes_defined():
    assert len(THEME_KEYWORDS) >= 6
