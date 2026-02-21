"""Purchase intent classification via regex pattern matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Intent definitions (priority order — first match wins for primary intent)
# ---------------------------------------------------------------------------

# intent_type → list of pattern strings
_INTENT_PATTERNS: dict[str, list[str]] = {
    "completed_purchase": [
        r"\bjust\s+copped\b",
        r"\bjust\s+bought\b",
        r"\bjust\s+got\b",
        r"\bpicked\s+up\b",
        r"\bsecured\s+the\s+bag\b",
        r"\bpulled\b.{0,20}\bpair\b",
        r"\bgot\s+(my|a|the)\s+(pair|set|cop)\b",
        r"\bcopped\b",
        r"\bjust\s+copped\b",
        r"\bordered\b.{0,10}\bshipping\b",
    ],
    "marketplace": [
        r"\bWTS\b",
        r"\bWTB\b",
        r"\bWTT\b",
        r"\bfor\s+sale\b",
        r"\bselling\b.{0,20}\bpair\b",
        r"\bprice\s+check\b",
        r"\bPC\b.{0,10}\b(pair|shoe|sneaker)\b",
    ],
    "selling": [
        r"\bfor\s+sale\b",
        r"\bselling\b",
        r"\blisting\b.{0,15}\b(pair|sneaker|shoe)\b",
        r"\bDM\s+for\s+price\b",
    ],
    "seeking_purchase": [
        r"\bW2C\b",
        r"\bwhere\s+to\s+cop\b",
        r"\bwhere\s+can\s+I\s+(get|buy|find|cop)\b",
        r"\bwhere\s+to\s+buy\b",
        r"\bany\s+(chance|way)\s+to\s+(get|cop|buy)\b",
        r"\blooking\s+(for|to\s+buy|to\s+cop)\b",
    ],
    "purchase_consideration": [
        r"\bshould\s+I\s+(cop|buy|get)\b",
        r"\bworth\s+(it|the\s+price|copping)\b",
        r"\bconsidering\b.{0,20}\b(cop|buy|get|purchase)\b",
        r"\bthinking\s+(of|about)\s+(buying|copping|getting)\b",
        r"\btempted\b.{0,20}\b(cop|buy)\b",
    ],
    "availability_info": [
        r"\bdrops?\s+at\b",
        r"\breleases?\s+on\b",
        r"\bavailable\s+(at|on|in)\b",
        r"\brestocked?\b",
        r"\bsold\s+out\b",
        r"\bin\s+stock\b",
        r"\brelease\s+date\b",
    ],
    "price_discussion": [
        r"\bpaid\s+\$[\d,]+",
        r"\basks?\s+\$[\d,]+",
        r"\bretail\s+\$[\d,]+",
        r"\bmarket\s+(price|value)\b",
        r"\bpremium\b.{0,15}\b(price|retail)\b",
        r"\b(price|cost|value)\b.{0,10}\b(thoughts?|opinion|fair|high|low)\b",
    ],
}

# Pre-compile all patterns for performance
_COMPILED: dict[str, list[re.Pattern]] = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in _INTENT_PATTERNS.items()
}

# Priority order for primary_intent resolution
_PRIORITY = [
    "completed_purchase",
    "marketplace",
    "selling",
    "seeking_purchase",
    "purchase_consideration",
    "availability_info",
    "price_discussion",
]


@dataclass
class IntentResult:
    """Classification result for one text."""

    primary_intent: str | None  # highest-priority matched intent
    all_intents: list[str]  # all matched intent types
    matched_patterns: dict[str, list[str]]  # intent → matching snippets


class PurchaseIntentClassifier:
    """Classify purchase intent signals in Reddit text."""

    def classify(self, text: str) -> IntentResult:
        if not text:
            return IntentResult(None, [], {})

        all_intents: list[str] = []
        matched_patterns: dict[str, list[str]] = {}

        for intent in _PRIORITY:
            pats = _COMPILED[intent]
            snippets = []
            for pat in pats:
                for m in pat.finditer(text):
                    snippets.append(m.group(0))
            if snippets:
                all_intents.append(intent)
                matched_patterns[intent] = snippets

        primary = all_intents[0] if all_intents else None
        return IntentResult(
            primary_intent=primary,
            all_intents=all_intents,
            matched_patterns=matched_patterns,
        )
