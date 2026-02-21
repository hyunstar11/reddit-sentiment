"""Brand detection with alias mapping and context-window extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from reddit_sentiment.config import SentimentConfig

# ---------------------------------------------------------------------------
# Brand alias registry
# ---------------------------------------------------------------------------

# canonical name → list of aliases (case-insensitive matching)
BRAND_ALIASES: dict[str, list[str]] = {
    "Nike": [
        "Nike",
        "Swoosh",
        "Just Do It",
        "Air Force",
        "Air Max",
        "Air Jordan",
        "Jordan Brand",
        "Jordan",
        "Dunk",
        "React",
    ],
    "Adidas": [
        "Adidas",
        "Adidas Originals",
        "Three Stripes",
        "3 Stripes",
        "Yeezy",
        "Ultraboost",
        "NMD",
        "Stan Smith",
        "Samba",
        "Gazelle",
    ],
    "Li-Ning": [
        "Li-Ning",
        "Li Ning",
        "LiNing",
        "LN",
        "Way of Wade",
        "WoW",
        "ANTA Sports Li-Ning",
    ],
    "Anta": [
        "Anta",
        "ANTA",
        "Klay Thompson",
        "KT",
    ],
    "361 Degrees": [
        "361",
        "361 Degrees",
        "361°",
    ],
    "Under Armour": [
        "Under Armour",
        "UA",
        "Curry Brand",
        "Curry N",
        "Curry shoes",
    ],
    "New Balance": [
        "New Balance",
        "NB",
        "990",
        "993",
        "1906",
        "2002",
    ],
    "Puma": [
        "Puma",
        "PUMA",
    ],
    "Asics": [
        "Asics",
        "ASICS",
        "Gel-Kayano",
        "Gel Kayano",
        "GT-2000",
    ],
    "Hoka": [
        "Hoka",
        "HOKA",
        "Hoka One One",
        "Clifton",
        "Bondi",
    ],
}


@dataclass
class BrandMention:
    """A single detected mention of a brand within a text."""

    brand: str  # canonical brand name
    alias: str  # the alias string that was matched
    start: int  # character offset in original text
    end: int
    context: str  # ±context_window words around the mention
    context_words_before: list[str] = field(default_factory=list)
    context_words_after: list[str] = field(default_factory=list)


class BrandDetector:
    """Detect sneaker brand mentions in text using pre-compiled regex patterns."""

    def __init__(self, context_window: int | None = None) -> None:
        cfg = SentimentConfig()
        self._window = context_window if context_window is not None else cfg.context_window
        # Build: brand → list of (compiled_pattern, alias_string)
        self._patterns: dict[str, list[tuple[re.Pattern, str]]] = {}
        for brand, aliases in BRAND_ALIASES.items():
            compiled = []
            for alias in aliases:
                # Word-boundary match, case-insensitive
                pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
                compiled.append((pat, alias))
            self._patterns[brand] = compiled

    def detect(self, text: str) -> list[BrandMention]:
        """Return all brand mentions found in *text*, in order of occurrence."""
        if not text:
            return []

        words = text.split()
        mentions: list[BrandMention] = []

        for brand, pat_list in self._patterns.items():
            for pat, alias in pat_list:
                for m in pat.finditer(text):
                    context, before, after = self._extract_context(words, text, m)
                    mentions.append(
                        BrandMention(
                            brand=brand,
                            alias=alias,
                            start=m.start(),
                            end=m.end(),
                            context=context,
                            context_words_before=before,
                            context_words_after=after,
                        )
                    )

        # Sort by character position
        mentions.sort(key=lambda mn: mn.start)
        return mentions

    def detect_brands(self, text: str) -> list[str]:
        """Return deduplicated canonical brand names found in text."""
        seen: list[str] = []
        for m in self.detect(text):
            if m.brand not in seen:
                seen.append(m.brand)
        return seen

    def _extract_context(
        self,
        words: list[str],
        text: str,
        match: re.Match,
    ) -> tuple[str, list[str], list[str]]:
        """Extract ±window words around a regex match."""
        # Find which word index the match start falls in
        pos = 0
        match_word_idx: int | None = None
        for idx, word in enumerate(words):
            if pos + len(word) >= match.start():
                match_word_idx = idx
                break
            pos += len(word) + 1  # +1 for space

        if match_word_idx is None:
            return text[max(0, match.start() - 50) : match.end() + 50], [], []

        start_idx = max(0, match_word_idx - self._window)
        end_idx = min(len(words), match_word_idx + self._window + 1)
        before = words[start_idx:match_word_idx]
        after = words[match_word_idx + 1 : end_idx]
        context_words = words[start_idx:end_idx]
        return " ".join(context_words), before, after
