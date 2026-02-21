"""Tests for ChannelDetector."""

import pytest

from reddit_sentiment.detection.channels import ChannelDetector


@pytest.fixture
def detector():
    return ChannelDetector()


def test_url_domain_stockx(detector):
    channels = detector.detect_from_urls(["https://stockx.com/buy/nike-air-max"])
    assert "StockX" in channels


def test_url_domain_nike(detector):
    channels = detector.detect_from_urls(["https://www.nike.com/t/air-max-90"])
    assert "Nike Direct" in channels


def test_url_domain_goat(detector):
    channels = detector.detect_from_urls(["https://www.goat.com/sneakers/air-jordan-1"])
    assert "GOAT" in channels


def test_url_domain_foot_locker(detector):
    channels = detector.detect_from_urls(["https://www.footlocker.com/product/model/123"])
    assert "Foot Locker" in channels


def test_url_unknown_domain(detector):
    channels = detector.detect_from_urls(["https://unknownshop.xyz/product"])
    assert channels == []


def test_text_keyword_stockx(detector):
    channels = detector.detect_from_text("Bought it on StockX for $200")
    assert "StockX" in channels


def test_text_keyword_snkrs(detector):
    channels = detector.detect_from_text("Entered the SNKRS draw for these")
    assert "Nike Direct" in channels


def test_text_keyword_foot_locker_space(detector):
    channels = detector.detect_from_text("Got them at foot locker yesterday")
    assert "Foot Locker" in channels


def test_combined_detect_deduplicates(detector):
    # Same channel from URL and text should appear once
    channels = detector.detect("Check stockx prices", urls=["https://stockx.com/buy/shoe"])
    assert channels.count("StockX") == 1


def test_multiple_channels_in_one_text(detector):
    text = "Listed on StockX and GOAT, also on Grailed"
    channels = detector.detect_from_text(text)
    assert "StockX" in channels
    assert "GOAT" in channels
    assert "Grailed" in channels


def test_empty_inputs(detector):
    assert detector.detect("") == []
    assert detector.detect_from_urls([]) == []
