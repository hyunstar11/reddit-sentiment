"""Retail channel attribution analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ChannelAttribution:
    """Aggregated channel attribution results."""

    channel_share: dict[str, float]  # channel → % of mentions
    channel_counts: dict[str, int]  # channel → raw count
    channel_by_brand: dict[str, dict[str, int]]  # brand → {channel: count}
    intent_funnel: dict[str, int]  # intent → count
    top_channels: list[str] = field(default_factory=list)


class ChannelAttributionAnalyzer:
    """Analyse channel mentions in the annotated DataFrame."""

    def analyze(self, df: pd.DataFrame) -> ChannelAttribution:
        """Compute channel share, per-brand breakdown, and intent funnel."""
        # Explode channel list
        exploded = df.explode("channels") if "channels" in df.columns else df.copy()
        exploded = (
            exploded[exploded["channels"].notna() & (exploded["channels"].astype(str) != "")]
            if "channels" in exploded.columns
            else exploded
        )

        # Channel counts & share
        channel_counts: dict[str, int] = {}
        if "channels" in exploded.columns:
            vc = exploded["channels"].value_counts()
            channel_counts = vc.to_dict()

        total = sum(channel_counts.values()) or 1
        channel_share = {ch: round(cnt / total * 100, 2) for ch, cnt in channel_counts.items()}
        top_channels = sorted(channel_counts, key=channel_counts.get, reverse=True)[:5]  # type: ignore[arg-type]

        # Per-brand channel breakdown
        channel_by_brand: dict[str, dict[str, int]] = {}
        if "brands" in df.columns and "channels" in df.columns:
            temp = df.copy()
            temp = temp.explode("brands")
            temp = temp.explode("channels")
            temp = temp[
                temp["brands"].notna()
                & (temp["brands"] != "")
                & temp["channels"].notna()
                & (temp["channels"] != "")
            ]
            for brand, group in temp.groupby("brands"):
                channel_by_brand[brand] = group["channels"].value_counts().to_dict()

        # Intent funnel
        intent_funnel: dict[str, int] = {}
        if "primary_intent" in df.columns:
            intent_funnel = df["primary_intent"].dropna().value_counts().to_dict()

        return ChannelAttribution(
            channel_share=channel_share,
            channel_counts=channel_counts,
            channel_by_brand=channel_by_brand,
            intent_funnel=intent_funnel,
            top_channels=top_channels,
        )
