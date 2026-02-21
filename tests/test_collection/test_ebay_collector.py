"""Tests for EbayCollector: load_latest, unconfigured error, and price parsing."""

from __future__ import annotations

import pandas as pd
import pytest

from reddit_sentiment.collection.ebay_collector import EbayCollector
from reddit_sentiment.config import EbayConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _unconfigured_collector() -> EbayCollector:
    """Return a collector with no API key set."""
    return EbayCollector(config=EbayConfig(EBAY_APP_ID=""))


def _configured_collector() -> EbayCollector:
    """Return a collector with a dummy API key (won't make real requests)."""
    return EbayCollector(config=EbayConfig(EBAY_APP_ID="dummy-key-for-tests"))


# ---------------------------------------------------------------------------
# Configuration guard
# ---------------------------------------------------------------------------


def test_collect_raises_when_unconfigured(tmp_path):
    collector = _unconfigured_collector()
    with pytest.raises(RuntimeError, match="EBAY_APP_ID"):
        collector.collect(["Air Jordan 1"], output_path=tmp_path / "out.parquet")


def test_is_configured_false_when_no_app_id():
    collector = _unconfigured_collector()
    assert not collector._is_configured()


def test_is_configured_true_when_app_id_set():
    collector = _configured_collector()
    assert collector._is_configured()


# ---------------------------------------------------------------------------
# load_latest
# ---------------------------------------------------------------------------


def test_load_latest_raises_when_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        EbayCollector.load_latest(tmp_path)


def test_load_latest_returns_most_recent(tmp_path):
    """Creates two parquet files; load_latest should return the newer one."""
    older = tmp_path / "ebay_20240101_000000.parquet"
    newer = tmp_path / "ebay_20240202_120000.parquet"

    df_old = pd.DataFrame([{"model": "old", "sold_price_usd": 100.0}])
    df_new = pd.DataFrame([{"model": "new", "sold_price_usd": 200.0}])

    df_old.to_parquet(older, index=False)
    df_new.to_parquet(newer, index=False)

    result = EbayCollector.load_latest(tmp_path)
    assert result["model"].iloc[0] == "new"


def test_load_latest_ignores_non_ebay_parquets(tmp_path):
    """Files not matching ebay_*.parquet should be ignored."""
    other = tmp_path / "annotated.parquet"
    other_df = pd.DataFrame([{"model": "annotated", "sold_price_usd": 99.0}])
    other_df.to_parquet(other, index=False)

    with pytest.raises(FileNotFoundError):
        EbayCollector.load_latest(tmp_path)


def test_load_latest_single_file(tmp_path):
    ebay_file = tmp_path / "ebay_20250101_000000.parquet"
    df = pd.DataFrame([
        {"model": "Dunk Low", "sold_price_usd": 250.0, "condition": "New"},
    ])
    df.to_parquet(ebay_file, index=False)

    result = EbayCollector.load_latest(tmp_path)
    assert len(result) == 1
    assert result["model"].iloc[0] == "Dunk Low"


# ---------------------------------------------------------------------------
# collect() saves parquet and returns path (network mocked)
# ---------------------------------------------------------------------------


def test_collect_saves_parquet(tmp_path, monkeypatch):
    """Monkeypatch _fetch_model to return canned records, verify parquet is saved."""
    collector = _configured_collector()

    fake_records = [
        {
            "model": "Dunk Low",
            "ebay_title": "Nike Dunk Low Panda Size 10",
            "sold_price_usd": 180.0,
            "condition": "New",
            "sold_date": None,
            "ebay_item_id": "123456",
        }
    ]

    monkeypatch.setattr(collector, "_fetch_model", lambda model, max_results: fake_records)

    out_path = tmp_path / "ebay_test.parquet"
    result_path = collector.collect(["Dunk Low"], output_path=out_path)

    assert result_path == out_path
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df["sold_price_usd"].iloc[0] == 180.0


def test_collect_empty_when_no_results(tmp_path, monkeypatch):
    collector = _configured_collector()
    monkeypatch.setattr(collector, "_fetch_model", lambda model, max_results: [])

    out_path = tmp_path / "ebay_empty.parquet"
    collector.collect(["NonExistentModel"], output_path=out_path)

    df = pd.read_parquet(out_path)
    assert df.empty
