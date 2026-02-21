"""Tests for ModelDetector alias mapping, overlap deduplication, and canonical names."""

from __future__ import annotations

import pytest

from reddit_sentiment.detection.models import MODEL_CATALOG, MODEL_INFO, ModelDetector


@pytest.fixture
def detector():
    return ModelDetector()


# ---------------------------------------------------------------------------
# Canonical name resolution via aliases
# ---------------------------------------------------------------------------


def test_aj1_aliases_resolve_to_canonical(detector):
    """Multiple AJ1 aliases should all resolve to 'Air Jordan 1'."""
    for alias in ["AJ1", "Jordan 1", "J1", "OG 1s"]:
        models = detector.detect_models(f"Just copped the {alias} in Chicago colourway")
        assert "Air Jordan 1" in models, f"Alias '{alias}' not mapped to Air Jordan 1"


def test_dunk_low_alias(detector):
    models = detector.detect_models("SB Dunk Low pandas are still selling out")
    assert "Dunk Low" in models


def test_af1_aliases(detector):
    for alias in ["AF1", "Air Force 1", "Air Force One", "Forces"]:
        models = detector.detect_models(f"wearing my {alias} today")
        assert "Air Force 1" in models, f"Alias '{alias}' not mapped to Air Force 1"


def test_yeezy_350_aliases(detector):
    for alias in ["Yeezy 350", "350 v2", "350v2", "Yeezy Boost 350"]:
        models = detector.detect_models(f"picked up the {alias}")
        assert "Yeezy 350" in models


def test_samba_alias(detector):
    models = detector.detect_models("Adidas Samba OG are everywhere this season")
    assert "Samba" in models


def test_nb_990_numeric_alias(detector):
    models = detector.detect_models("Just got the 990v6 for a run")
    assert "NB 990" in models


# ---------------------------------------------------------------------------
# Overlap deduplication
# ---------------------------------------------------------------------------


def test_no_duplicate_canonical_in_output(detector):
    """A text matching multiple aliases of the same model should return it once."""
    text = "The Jordan 1 AJ1 is the best — Air Jordan 1 forever"
    models = detector.detect_models(text)
    assert models.count("Air Jordan 1") == 1


def test_overlapping_span_not_double_counted(detector):
    """Overlapping character spans should not produce two ModelMention entries."""
    text = "Air Jordan 1 review"
    mentions = detector.detect(text)
    # All spans should be non-overlapping
    spans = [(m.start, m.end) for m in mentions]
    for i, (s1, e1) in enumerate(spans):
        for j, (s2, e2) in enumerate(spans):
            if i != j:
                assert not (s1 < e2 and s2 < e1), f"Overlapping spans: {spans[i]} vs {spans[j]}"


# ---------------------------------------------------------------------------
# Multiple models in one text
# ---------------------------------------------------------------------------


def test_multiple_distinct_models(detector):
    text = "Can't decide between the Dunk Low and the Air Max 90"
    models = detector.detect_models(text)
    assert "Dunk Low" in models
    assert "Air Max 90" in models


def test_empty_text_returns_empty(detector):
    assert detector.detect_models("") == []
    assert detector.detect("") == []


# ---------------------------------------------------------------------------
# MODEL_CATALOG / MODEL_INFO structure
# ---------------------------------------------------------------------------


def test_model_info_keys_match_catalog():
    assert set(MODEL_INFO.keys()) == set(MODEL_CATALOG.keys())


def test_model_info_brand_matches_catalog():
    for name, (brand, price) in MODEL_INFO.items():
        catalog_brand, catalog_price, _ = MODEL_CATALOG[name]
        assert brand == catalog_brand
        assert price == catalog_price


def test_all_retail_prices_positive():
    for name, (_brand, price) in MODEL_INFO.items():
        assert price > 0, f"{name} has non-positive retail price {price}"


def test_all_aliases_non_empty():
    for name, (_brand, _price, aliases) in MODEL_CATALOG.items():
        assert len(aliases) >= 1, f"{name} has no aliases"
