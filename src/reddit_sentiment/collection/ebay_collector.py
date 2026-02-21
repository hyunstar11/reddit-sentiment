"""eBay Finding API collector — fetches completed/sold sneaker listings."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from reddit_sentiment.config import EbayConfig

_FINDING_API_URL = "https://svcs.ebay.com/services/search/FindingService/v1"
_REQUEST_DELAY = 0.5  # seconds between requests


def _parse_price(item: dict) -> float | None:
    try:
        return float(item["sellingStatus"][0]["currentPrice"][0]["__value__"])
    except (KeyError, IndexError, ValueError):
        return None


def _parse_date(item: dict) -> datetime | None:
    try:
        raw = item["listingInfo"][0]["endTime"][0]
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (KeyError, IndexError, ValueError):
        return None


def _parse_condition(item: dict) -> str:
    try:
        return item["condition"][0]["conditionDisplayName"][0]
    except (KeyError, IndexError):
        return "Unknown"


class EbayCollector:
    """Fetch completed (sold) eBay listings for a list of shoe models.

    Requires EBAY_APP_ID in .env. Register free at developer.ebay.com.
    """

    def __init__(self, config: EbayConfig | None = None) -> None:
        self._cfg = config or EbayConfig()
        self._session = requests.Session()

    def _is_configured(self) -> bool:
        return bool(self._cfg.app_id)

    def _fetch_model(self, model_name: str, max_results: int) -> list[dict]:
        """Fetch sold listings for one shoe model."""
        params = {
            "OPERATION-NAME": "findCompletedItems",
            "SERVICE-VERSION": "1.0.0",
            "SECURITY-APPNAME": self._cfg.app_id,
            "RESPONSE-DATA-FORMAT": "JSON",
            "REST-PAYLOAD": "",
            "keywords": model_name,
            "categoryId": self._cfg.category_id,
            "itemFilter(0).name": "SoldItemsOnly",
            "itemFilter(0).value": "true",
            "itemFilter(1).name": "ListingType",
            "itemFilter(1).value": "AuctionWithBIN,FixedPrice",
            "sortOrder": "EndTimeSoonest",
            "paginationInput.entriesPerPage": min(max_results, 100),
        }

        try:
            resp = self._session.get(_FINDING_API_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [!] eBay request failed for '{model_name}': {exc}")
            return []

        try:
            data = resp.json()
            items = (
                data
                .get("findCompletedItemsResponse", [{}])[0]
                .get("searchResult", [{}])[0]
                .get("item", [])
            )
        except (KeyError, IndexError, ValueError) as exc:
            print(f"  [!] eBay parse error for '{model_name}': {exc}")
            return []

        records = []
        for item in items:
            price = _parse_price(item)
            if price is None or price <= 0:
                continue
            records.append({
                "model": model_name,
                "ebay_title": item.get("title", [None])[0],
                "sold_price_usd": price,
                "condition": _parse_condition(item),
                "sold_date": _parse_date(item),
                "ebay_item_id": item.get("itemId", [None])[0],
            })

        return records

    def collect(
        self,
        models: list[str],
        output_path: Path | None = None,
    ) -> Path:
        """Fetch sold listings for each model; save to Parquet.

        Args:
            models: List of canonical shoe model names to query.
            output_path: Where to save the Parquet file.

        Returns:
            Path to saved Parquet file.
        """
        if not self._is_configured():
            raise RuntimeError(
                "EBAY_APP_ID not set. Register at developer.ebay.com and add "
                "EBAY_APP_ID=<your_app_id> to .env"
            )

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            from reddit_sentiment.config import CollectionConfig
            data_dir = CollectionConfig().raw_data_dir
            output_path = data_dir / f"ebay_{timestamp}.parquet"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_records: list[dict] = []

        for model_name in models:
            print(f"[ebay] {model_name} …")
            records = self._fetch_model(model_name, self._cfg.max_results_per_model)
            all_records.extend(records)
            print(f"  {len(records)} sold listings")
            time.sleep(_REQUEST_DELAY)

        df = pd.DataFrame(all_records)
        df.to_parquet(output_path, index=False)
        print(f"\n[ebay] {len(df)} total listings → {output_path}")
        return output_path

    @classmethod
    def load_latest(cls, data_dir: Path) -> pd.DataFrame:
        """Load the most recently created eBay Parquet file."""
        files = sorted(data_dir.glob("ebay_*.parquet"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No eBay parquet files found in {data_dir}")
        return pd.read_parquet(files[0])
