"""Retail channel detection via URL domain mapping and keyword fallback."""

from __future__ import annotations

import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Domain → channel mapping (40+ retailers)
# ---------------------------------------------------------------------------

DOMAIN_TO_CHANNEL: dict[str, str] = {
    # Nike Direct
    "nike.com": "Nike Direct",
    "snkrs.com": "Nike Direct",
    # Adidas Direct
    "adidas.com": "Adidas Direct",
    "adidas.us": "Adidas Direct",
    # Stockx
    "stockx.com": "StockX",
    # GOAT
    "goat.com": "GOAT",
    # Foot Locker group
    "footlocker.com": "Foot Locker",
    "footaction.com": "Foot Locker",
    "kidsfootlocker.com": "Foot Locker",
    "champssports.com": "Champs Sports",
    "eastbay.com": "Eastbay",
    # Dick's / Finish Line
    "dickssportinggoods.com": "Dick's Sporting Goods",
    "finishline.com": "Finish Line",
    # Amazon
    "amazon.com": "Amazon",
    "amazon.co.jp": "Amazon JP",
    # Farfetch / Ssense
    "farfetch.com": "Farfetch",
    "ssense.com": "SSENSE",
    # Size? / JD Sports group
    "jdsports.co.uk": "JD Sports",
    "jdsports.com": "JD Sports",
    "size.co.uk": "Size?",
    # Other boutiques
    "kith.com": "Kith",
    "solefly.com": "Solefly",
    "undefeated.com": "Undefeated",
    "consortium.adidas.com": "Adidas Consortium",
    "concepts.ltd": "Concepts",
    "bodega.com": "Bodega",
    "socialstatuspgh.com": "Social Status",
    "kicksusa.com": "KicksUSA",
    "sneakersnstuff.com": "Sneakersnstuff",
    "end.com": "END Clothing",
    "offspring.co.uk": "Offspring",
    # Resale
    "ebay.com": "eBay",
    "grailed.com": "Grailed",
    "depop.com": "Depop",
    "flightclub.com": "Flight Club",
    "klekt.com": "Klekt",
    # Running specialty
    "runningwarehouse.com": "Running Warehouse",
    "rei.com": "REI",
    "holabird.com": "Holabird Sports",
}

# Keyword → channel (for mentions without a URL)
# Note: "rei" intentionally omitted — too short, matches inside other words.
# REI.com is still detected via URL domain mapping above.
_KEYWORD_TO_CHANNEL: dict[str, str] = {
    "stockx": "StockX",
    "goat": "GOAT",
    "foot locker": "Foot Locker",
    "footlocker": "Foot Locker",
    "dick's": "Dick's Sporting Goods",
    "dicks": "Dick's Sporting Goods",
    "finish line": "Finish Line",
    "amazon": "Amazon",
    "nike direct": "Nike Direct",
    "snkrs": "Nike Direct",
    "adidas direct": "Adidas Direct",
    "kith": "Kith",
    "flight club": "Flight Club",
    "grailed": "Grailed",
    "depop": "Depop",
    "farfetch": "Farfetch",
    "ssense": "SSENSE",
    "jd sports": "JD Sports",
    "end clothing": "END Clothing",
    "undefeated": "Undefeated",
    "concepts": "Concepts",
    "bodega": "Bodega",
    "ebay": "eBay",
}

# Pre-compiled keyword pattern — longest first, word boundaries to prevent
# partial matches (e.g. "rei" inside "received", "kith" inside "skither")
_sorted_keywords = sorted(_KEYWORD_TO_CHANNEL, key=len, reverse=True)
_KEYWORD_PATTERN = re.compile(
    "|".join(r"\b" + re.escape(k) + r"\b" for k in _sorted_keywords),
    re.IGNORECASE,
)


def _domain_from_url(url: str) -> str:
    """Extract normalised domain (strip www.) from a URL string."""
    try:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:  # noqa: BLE001
        return ""


class ChannelDetector:
    """Detect retail channels from URLs and text keywords."""

    def detect_from_urls(self, urls: list[str]) -> list[str]:
        """Map URL list → deduplicated channel names."""
        channels: list[str] = []
        for url in urls:
            domain = _domain_from_url(url)
            channel = DOMAIN_TO_CHANNEL.get(domain)
            if channel and channel not in channels:
                channels.append(channel)
        return channels

    def detect_from_text(self, text: str) -> list[str]:
        """Scan text for retail keywords → channel names."""
        if not text:
            return []
        channels: list[str] = []
        for m in _KEYWORD_PATTERN.finditer(text):
            channel = _KEYWORD_TO_CHANNEL.get(m.group(0).lower())
            if channel and channel not in channels:
                channels.append(channel)
        return channels

    def detect(self, text: str, urls: list[str] | None = None) -> list[str]:
        """Combined detection: URL domains first, then text keywords."""
        channels: list[str] = []
        if urls:
            channels.extend(self.detect_from_urls(urls))
        for ch in self.detect_from_text(text):
            if ch not in channels:
                channels.append(ch)
        return channels
