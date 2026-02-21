"""Narrative theme extraction using keyword frequency and TF-IDF."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Curated theme → keyword seeds (keyword appears in text → theme is activated)
THEME_KEYWORDS: dict[str, list[str]] = {
    "Quality & Comfort": [
        "comfortable",
        "comfort",
        "quality",
        "material",
        "cushion",
        "support",
        "fit",
        "feels",
        "durable",
        "soft",
        "stiff",
        "heavy",
        "lightweight",
    ],
    "Hype & Exclusivity": [
        "hype",
        "limited",
        "exclusive",
        "grail",
        "cop",
        "drop",
        "raffle",
        "release",
        "collab",
        "collaboration",
        "sold out",
        "resell",
    ],
    "Value & Pricing": [
        "price",
        "retail",
        "resale",
        "markup",
        "expensive",
        "cheap",
        "value",
        "worth",
        "investment",
        "stockx",
        "goat",
        "market",
    ],
    "Aesthetics & Design": [
        "colorway",
        "design",
        "look",
        "style",
        "clean",
        "dope",
        "fire",
        "ugly",
        "beautiful",
        "classic",
        "retro",
        "heritage",
        "silhouette",
    ],
    "Performance & Sport": [
        "performance",
        "running",
        "basketball",
        "training",
        "grip",
        "traction",
        "responsive",
        "bounce",
        "energy return",
        "support",
        "court",
    ],
    "Brand Loyalty": [
        "fanboy",
        "loyal",
        "always",
        "forever",
        "brand",
        "team",
        "stan",
        "die-hard",
        "prefer",
        "switch",
    ],
    "Sustainability": [
        "sustainable",
        "recycled",
        "eco",
        "environment",
        "vegan",
        "carbon",
        "footprint",
        "green",
    ],
    "Authenticity & Fakes": [
        "fake",
        "rep",
        "replica",
        "auth",
        "authentic",
        "real",
        "legit",
        "UA",
        "unauthorized",
        "stockx verified",
    ],
}


@dataclass
class ThemeResult:
    """Theme analysis results for a corpus."""

    theme_counts: dict[str, int]  # theme → number of matching texts
    theme_percentages: dict[str, float]  # theme → % of corpus
    top_tfidf_terms: list[str]  # top TF-IDF terms (global)
    brand_themes: dict[str, dict[str, int]] = field(default_factory=dict)  # brand → theme_counts


class NarrativeThemeExtractor:
    """Extract narrative themes from annotated text data."""

    def __init__(self, max_tfidf_features: int = 50) -> None:
        self._max_features = max_tfidf_features

    def _match_themes(self, text: str) -> list[str]:
        """Return list of theme names whose keywords appear in text."""
        lower = text.lower()
        matched = []
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                matched.append(theme)
        return matched

    def extract(self, df: pd.DataFrame) -> ThemeResult:
        """Extract themes from 'full_text' column; optionally per brand."""
        texts = df["full_text"].fillna("").tolist()
        n = len(texts)

        # Per-text theme matching
        all_matched: list[list[str]] = [self._match_themes(t) for t in texts]

        # Count across corpus
        counter: Counter = Counter()
        for matched in all_matched:
            counter.update(matched)

        theme_counts = dict(counter)
        theme_pct = {k: round(v / n * 100, 2) for k, v in counter.items()} if n else {}

        # Global TF-IDF top terms
        top_terms: list[str] = []
        non_empty = [t for t in texts if t.strip()]
        if len(non_empty) >= 2:
            try:
                tfidf = TfidfVectorizer(
                    max_features=self._max_features,
                    stop_words="english",
                    ngram_range=(1, 2),
                )
                tfidf.fit(non_empty)
                top_terms = list(tfidf.get_feature_names_out())
            except Exception:  # noqa: BLE001
                pass

        # Per-brand themes
        brand_themes: dict[str, dict[str, int]] = {}
        if "brands" in df.columns:
            exploded = df.copy()
            exploded["_themes"] = all_matched
            exploded = exploded.explode("brands")
            exploded = exploded[exploded["brands"].notna() & (exploded["brands"] != "")]
            for brand, group in exploded.groupby("brands"):
                brand_counter: Counter = Counter()
                for themes in group["_themes"]:
                    if isinstance(themes, list):
                        brand_counter.update(themes)
                brand_themes[brand] = dict(brand_counter)

        return ThemeResult(
            theme_counts=theme_counts,
            theme_percentages=theme_pct,
            top_tfidf_terms=top_terms,
            brand_themes=brand_themes,
        )
