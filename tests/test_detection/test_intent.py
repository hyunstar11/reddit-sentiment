"""Tests for PurchaseIntentClassifier."""

import pytest

from reddit_sentiment.detection.intent import PurchaseIntentClassifier


@pytest.fixture
def clf():
    return PurchaseIntentClassifier()


def test_just_copped_is_completed_purchase(clf):
    result = clf.classify("Just copped the Nike Dunk Low in my size!")
    assert result.primary_intent == "completed_purchase"


def test_w2c_is_seeking_purchase(clf):
    result = clf.classify("W2C these Adidas Gazelles in size 10?")
    assert result.primary_intent == "seeking_purchase"


def test_should_i_cop_is_consideration(clf):
    result = clf.classify("Should I cop the Yeezy 350 for retail?")
    assert result.primary_intent == "purchase_consideration"


def test_drops_at_is_availability(clf):
    result = clf.classify("This drops at Nike SNKRS on Friday")
    assert result.primary_intent == "availability_info"


def test_wts_is_marketplace(clf):
    result = clf.classify("WTS Nike Air Jordan 1 size 10.5, asking $300")
    assert result.primary_intent == "marketplace"


def test_price_discussion(clf):
    result = clf.classify("Market price on the 990v4 seems fair at $150 retail $140")
    assert result.primary_intent == "price_discussion"


def test_no_intent(clf):
    result = clf.classify("These shoes look really cool I like the colorway")
    assert result.primary_intent is None
    assert result.all_intents == []


def test_empty_text(clf):
    result = clf.classify("")
    assert result.primary_intent is None


def test_multiple_intents_captured(clf):
    # Text has both completed_purchase and price_discussion
    text = "Just copped for retail $120 — market price is $180"
    result = clf.classify(text)
    assert "completed_purchase" in result.all_intents
    assert "price_discussion" in result.all_intents
    # Priority: completed_purchase wins
    assert result.primary_intent == "completed_purchase"


def test_priority_completed_over_marketplace(clf):
    # "just copped" + "for sale" — completed_purchase takes priority
    text = "Just copped a pair. Also have one for sale."
    result = clf.classify(text)
    assert result.primary_intent == "completed_purchase"
