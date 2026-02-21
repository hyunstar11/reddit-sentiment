"""Sentiment trend analysis over time (weekly/monthly)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrendResult:
    """Time-series sentiment trend data."""

    weekly: pd.DataFrame  # columns: period, avg_sentiment, count, brand (opt)
    monthly: pd.DataFrame  # columns: period, avg_sentiment, count, brand (opt)


class SentimentTrendAnalyzer:
    """Aggregate sentiment scores over time periods."""

    def analyze(self, df: pd.DataFrame, by_brand: bool = False) -> TrendResult:
        """Compute weekly and monthly sentiment trends.

        Requires 'created_utc' and 'hybrid_score' columns.
        """
        df = df.copy()

        if "created_utc" not in df.columns:
            empty = pd.DataFrame(columns=["period", "avg_sentiment", "count"])
            return TrendResult(weekly=empty, monthly=empty)

        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["created_utc"])

        if df.empty:
            empty = pd.DataFrame(columns=["period", "avg_sentiment", "count"])
            return TrendResult(weekly=empty, monthly=empty)

        # Drop tz info before to_period to avoid pandas UserWarning
        utc_naive = df["created_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["week"] = utc_naive.dt.to_period("W").astype(str)
        df["month"] = utc_naive.dt.to_period("M").astype(str)

        if by_brand and "brands" in df.columns:
            base = df.explode("brands")
            base = base[base["brands"].notna() & (base["brands"] != "")]
            group_cols_w = ["week", "brands"]
            group_cols_m = ["month", "brands"]
        else:
            base = df
            group_cols_w = ["week"]
            group_cols_m = ["month"]

        weekly = self._aggregate(base, group_cols_w, "week")
        monthly = self._aggregate(base, group_cols_m, "month")

        return TrendResult(weekly=weekly, monthly=monthly)

    @staticmethod
    def _aggregate(df: pd.DataFrame, group_cols: list[str], period_col: str) -> pd.DataFrame:
        agg = (
            df.groupby(group_cols)["hybrid_score"]
            .agg(avg_sentiment="mean", count="count")
            .reset_index()
            .rename(columns={period_col: "period"})
        )
        agg["avg_sentiment"] = agg["avg_sentiment"].round(4)
        return agg.sort_values("period")
