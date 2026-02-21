"""Shoe model detection with alias mapping and retail price lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model alias registry
# canonical name → (brand, retail_price_usd, [aliases])
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, tuple[str, float, list[str]]] = {
    # ── Nike ────────────────────────────────────────────────────────────────
    "Air Jordan 1": ("Nike", 180.0, [
        "Air Jordan 1", "Jordan 1", "AJ1", "J1", "OG 1s", "Jordan 1s",
        "Jordan Ones", "High OG", "AJ 1",
    ]),
    "Air Jordan 3": ("Nike", 200.0, ["Air Jordan 3", "Jordan 3", "AJ3", "J3"]),
    "Air Jordan 4": ("Nike", 210.0, ["Air Jordan 4", "Jordan 4", "AJ4", "J4"]),
    "Air Jordan 5": ("Nike", 210.0, ["Air Jordan 5", "Jordan 5", "AJ5", "J5"]),
    "Air Jordan 11": ("Nike", 220.0, [
        "Air Jordan 11", "Jordan 11", "AJ11", "J11", "Concords", "Space Jams",
    ]),
    "Air Jordan 12": ("Nike", 200.0, ["Air Jordan 12", "Jordan 12", "AJ12", "J12"]),
    "Dunk Low": ("Nike", 110.0, [
        "Dunk Low", "Dunks Low", "SB Dunk Low", "Dunk Lows",
    ]),
    "Dunk High": ("Nike", 110.0, ["Dunk High", "Dunks High", "SB Dunk High"]),
    "Air Force 1": ("Nike", 110.0, [
        "Air Force 1", "Air Force One", "AF1", "AF-1", "Forces",
    ]),
    "Air Max 90": ("Nike", 120.0, ["Air Max 90", "AM90", "AM 90"]),
    "Air Max 95": ("Nike", 160.0, ["Air Max 95", "AM95", "AM 95"]),
    "Air Max 97": ("Nike", 175.0, ["Air Max 97", "AM97", "AM 97"]),
    "Air Max 1": ("Nike", 130.0, ["Air Max 1", "AM1", "AM 1"]),
    "Air Max Plus": ("Nike", 165.0, [
        "Air Max Plus", "TN", "Air Max TN", "Tuned Air",
    ]),
    "Pegasus": ("Nike", 130.0, [
        "Pegasus", "Peg 40", "Peg 41", "Pegasus 40", "Pegasus 41",
    ]),
    "Vaporfly": ("Nike", 260.0, [
        "Vaporfly", "Vaporfly 3", "Next%", "Next percent",
    ]),
    "Alphafly": ("Nike", 285.0, ["Alphafly", "Alphafly 3"]),
    # ── Adidas ──────────────────────────────────────────────────────────────
    "Yeezy 350": ("Adidas", 230.0, [
        "Yeezy 350", "350 v2", "350v2", "Yeezy Boost 350",
    ]),
    "Yeezy 380": ("Adidas", 230.0, ["Yeezy 380", "Yeezy Boost 380"]),
    "Yeezy 500": ("Adidas", 200.0, ["Yeezy 500"]),
    "Yeezy 700": ("Adidas", 300.0, ["Yeezy 700", "Yeezy Wave Runner"]),
    "Samba": ("Adidas", 100.0, [
        "Samba", "Samba OG", "Adidas Samba",
    ]),
    "Gazelle": ("Adidas", 100.0, ["Gazelle", "Adidas Gazelle"]),
    "Campus 00s": ("Adidas", 100.0, [
        "Campus 00s", "Campus 00", "Adidas Campus",
    ]),
    "NMD R1": ("Adidas", 130.0, ["NMD R1", "NMD_R1", "NMD"]),
    "Ultraboost": ("Adidas", 190.0, [
        "Ultraboost", "Ultra Boost", "UB 22", "UB 23", "UB22", "UB23",
    ]),
    "Stan Smith": ("Adidas", 100.0, ["Stan Smith", "Stans"]),
    "Forum Low": ("Adidas", 100.0, ["Forum Low", "Forum 84 Low"]),
    # ── New Balance ─────────────────────────────────────────────────────────
    "NB 550": ("New Balance", 110.0, [
        "550", "NB 550", "New Balance 550",
    ]),
    "NB 990": ("New Balance", 185.0, [
        "990", "990v6", "990 v6", "990v5", "990 v5", "990v4", "New Balance 990",
    ]),
    "NB 2002R": ("New Balance", 150.0, [
        "2002R", "2002", "New Balance 2002", "NB 2002",
    ]),
    "NB 1906R": ("New Balance", 150.0, [
        "1906R", "1906", "New Balance 1906", "NB 1906",
    ]),
    "NB 574": ("New Balance", 90.0, ["574", "NB 574", "New Balance 574"]),
    # ── Hoka ────────────────────────────────────────────────────────────────
    "Clifton 9": ("Hoka", 145.0, ["Clifton 9", "Clifton", "Hoka Clifton"]),
    "Bondi 8": ("Hoka", 165.0, ["Bondi 8", "Bondi", "Hoka Bondi"]),
    # ── ASICS ───────────────────────────────────────────────────────────────
    "Gel-Kayano": ("Asics", 160.0, [
        "Gel-Kayano", "Gel Kayano", "Kayano", "Kayano 30",
    ]),
    # ── Puma ────────────────────────────────────────────────────────────────
    "Speedcat": ("Puma", 100.0, ["Speedcat", "Puma Speedcat"]),
}

# Flat lookup: canonical → (brand, retail_price)
MODEL_INFO: dict[str, tuple[str, float]] = {
    name: (brand, price) for name, (brand, price, _) in MODEL_CATALOG.items()
}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

@dataclass
class ModelMention:
    model: str       # canonical model name
    brand: str
    alias: str       # matched text
    start: int
    end: int
    retail_price: float


class ModelDetector:
    """Detect specific shoe model mentions using pre-compiled word-boundary patterns."""

    def __init__(self) -> None:
        # Build: canonical_name → list of (pattern, alias, brand, retail_price)
        self._patterns: list[tuple[re.Pattern, str, str, str, float]] = []
        for model, (brand, retail, aliases) in MODEL_CATALOG.items():
            # Sort aliases longest-first so longer matches win over shorter ones
            for alias in sorted(aliases, key=len, reverse=True):
                pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
                self._patterns.append((pat, alias, model, brand, retail))

    def detect(self, text: str) -> list[ModelMention]:
        """Return all model mentions found in text, deduplicated by character span."""
        if not text:
            return []

        mentions: list[ModelMention] = []
        covered: list[tuple[int, int]] = []  # track matched spans to avoid overlap

        for pat, alias, model, brand, retail in self._patterns:
            for m in pat.finditer(text):
                # Skip if this span overlaps an already-matched span
                if any(s <= m.start() < e or s < m.end() <= e for s, e in covered):
                    continue
                covered.append((m.start(), m.end()))
                mentions.append(ModelMention(
                    model=model,
                    brand=brand,
                    alias=alias,
                    start=m.start(),
                    end=m.end(),
                    retail_price=retail,
                ))

        mentions.sort(key=lambda mn: mn.start)
        return mentions

    def detect_models(self, text: str) -> list[str]:
        """Return deduplicated canonical model names found in text."""
        seen: list[str] = []
        for m in self.detect(text):
            if m.model not in seen:
                seen.append(m.model)
        return seen
