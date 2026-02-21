"""Plotly chart functions returning JSON-serialisable dicts for embedding."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from reddit_sentiment.analysis.brand_comparison import BrandMetrics
from reddit_sentiment.analysis.channel_attribution import ChannelAttribution


def brand_sentiment_bar(metrics: dict[str, BrandMetrics]) -> str:
    """Horizontal bar chart: avg sentiment per brand, coloured by label."""
    if not metrics:
        return "{}"

    brands = sorted(metrics.values(), key=lambda m: m.avg_hybrid_score, reverse=True)
    names = [m.brand for m in brands]
    scores = [m.avg_hybrid_score for m in brands]
    colours = ["#22c55e" if s > 0.05 else "#ef4444" if s < -0.05 else "#94a3b8" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colours,
            text=[f"{s:+.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Brand Sentiment Comparison",
        xaxis_title="Avg. Sentiment Score",
        yaxis_title="",
        xaxis=dict(range=[-1, 1]),
        height=400,
        margin=dict(l=120, r=60, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig.to_json()


def sentiment_distribution_pie(metrics: dict[str, BrandMetrics]) -> str:
    """Stacked sentiment distribution for all brands combined."""
    if not metrics:
        return "{}"

    pos = sum(m.positive_pct * m.mention_count for m in metrics.values()) / max(
        sum(m.mention_count for m in metrics.values()), 1
    )
    neg = sum(m.negative_pct * m.mention_count for m in metrics.values()) / max(
        sum(m.mention_count for m in metrics.values()), 1
    )
    neu = 100 - pos - neg

    fig = go.Figure(
        go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[pos, neu, neg],
            marker_colors=["#22c55e", "#94a3b8", "#ef4444"],
            hole=0.35,
        )
    )
    fig.update_layout(
        title="Overall Sentiment Distribution",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig.to_json()


def channel_share_pie(attribution: ChannelAttribution) -> str:
    """Pie chart of retail channel share."""
    if not attribution.channel_counts:
        return "{}"

    # Collapse small channels into "Other"
    sorted_ch = sorted(attribution.channel_counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = [], []
    other = 0
    for name, cnt in sorted_ch:
        if len(labels) < 7:
            labels.append(name)
            values.append(cnt)
        else:
            other += cnt
    if other:
        labels.append("Other")
        values.append(other)

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
        )
    )
    fig.update_layout(
        title="Retail Channel Share",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig.to_json()


def sentiment_trend_line(weekly_df: pd.DataFrame) -> str:
    """Line chart of weekly average sentiment."""
    if weekly_df.empty:
        return "{}"

    if "brands" in weekly_df.columns:
        fig = px.line(
            weekly_df,
            x="period",
            y="avg_sentiment",
            color="brands",
            title="Weekly Sentiment Trend by Brand",
        )
    else:
        fig = px.line(weekly_df, x="period", y="avg_sentiment", title="Weekly Sentiment Trend")

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Avg. Sentiment",
        yaxis=dict(range=[-1, 1]),
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig.to_json()


def intent_funnel(attribution: ChannelAttribution) -> str:
    """Funnel chart of purchase intent stages."""
    if not attribution.intent_funnel:
        return "{}"

    # Ordered funnel stages
    stage_order = [
        "availability_info",
        "purchase_consideration",
        "seeking_purchase",
        "completed_purchase",
        "price_discussion",
        "marketplace",
        "selling",
    ]
    labels, values = [], []
    for stage in stage_order:
        if stage in attribution.intent_funnel:
            labels.append(stage.replace("_", " ").title())
            values.append(attribution.intent_funnel[stage])

    # Add any remaining stages not in our list
    for stage, cnt in attribution.intent_funnel.items():
        if stage not in stage_order:
            labels.append(stage.replace("_", " ").title())
            values.append(cnt)

    fig = go.Figure(go.Funnel(y=labels, x=values, textinfo="value+percent total"))
    fig.update_layout(
        title="Purchase Intent Funnel",
        height=400,
        margin=dict(l=150, r=60, t=60, b=40),
    )
    return fig.to_json()
