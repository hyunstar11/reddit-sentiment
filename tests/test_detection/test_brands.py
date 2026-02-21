"""Tests for BrandDetector."""

import pytest

from reddit_sentiment.detection.brands import BRAND_ALIASES, BrandDetector


@pytest.fixture
def detector():
    return BrandDetector(context_window=5)


def test_direct_brand_name(detector):
    brands = detector.detect_brands("I love Nike shoes")
    assert "Nike" in brands


def test_alias_three_stripes_resolves_to_adidas(detector):
    brands = detector.detect_brands("The Three Stripes collab is sick")
    assert "Adidas" in brands
    assert "Nike" not in brands


def test_alias_way_of_wade_resolves_to_lining(detector):
    brands = detector.detect_brands("Way of Wade 10 just dropped")
    assert "Li-Ning" in brands


def test_alias_ua_resolves_to_under_armour(detector):
    brands = detector.detect_brands("UA Curry 11 is underrated")
    assert "Under Armour" in brands


def test_alias_yeezy_resolves_to_adidas(detector):
    brands = detector.detect_brands("Yeezy 350 just restocked on Adidas")
    assert "Adidas" in brands


def test_multi_brand_detection(detector):
    text = "Comparing Nike Air Max with New Balance 990"
    brands = detector.detect_brands(text)
    assert "Nike" in brands
    assert "New Balance" in brands


def test_no_false_positive_on_partial_word(detector):
    # "UA" should not match "ULTRA" as Under Armour
    # Test that boundary matching works
    brands = detector.detect_brands("An ultralight design")
    # "ultralight" should not match "UA" as a word
    # UA appears as part of "ULTRA"—but word boundary \b protects against this
    # (this is a tricky case; "UA" regex: \bUA\b won't match inside "ultralight")
    assert "Under Armour" not in brands


def test_context_extraction(detector):
    text = "I just bought Nike Air Max and they are amazing"
    mentions = detector.detect(text)
    nike_mentions = [m for m in mentions if m.brand == "Nike"]
    assert len(nike_mentions) > 0
    ctx = nike_mentions[0].context
    # Context should contain nearby words
    assert "Nike" in ctx or "nike" in ctx.lower()


def test_empty_text(detector):
    assert detector.detect("") == []
    assert detector.detect_brands("") == []


def test_all_canonical_brands_present():
    expected = {
        "Nike",
        "Adidas",
        "Li-Ning",
        "Anta",
        "361 Degrees",
        "Under Armour",
        "New Balance",
        "Puma",
        "Asics",
        "Hoka",
    }
    assert expected == set(BRAND_ALIASES.keys())
