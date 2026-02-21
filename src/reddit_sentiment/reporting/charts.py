"""Plotly chart functions returning JSON-serialisable dicts for embedding."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from reddit_sentiment.analysis.brand_comparison import BrandMetrics
from reddit_sentiment.analysis.channel_attribution import ChannelAttribution
from reddit_sentiment.analysis.price_correlation import ModelSignal


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

    fig = go.Figure()

    if "brands" in weekly_df.columns:
        for brand, grp in weekly_df.groupby("brands"):
            grp = grp.sort_values("period")
            fig.add_trace(go.Scatter(
                x=grp["period"].tolist(),
                y=grp["avg_sentiment"].tolist(),
                mode="lines+markers",
                name=str(brand),
            ))
        title = "Weekly Sentiment Trend by Brand"
    else:
        df_sorted = weekly_df.sort_values("period")
        fig.add_trace(go.Scatter(
            x=df_sorted["period"].tolist(),
            y=df_sorted["avg_sentiment"].tolist(),
            mode="lines+markers",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=6),
            name="Avg. Sentiment",
        ))
        title = "Weekly Sentiment Trend"

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title="Avg. Sentiment",
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


def model_mentions_bar(signals: list[ModelSignal]) -> str:
    """Horizontal bar: Reddit mention count per shoe model, coloured by sentiment."""
    filtered = [s for s in signals if s.mention_count >= 3]
    if not filtered:
        return "{}"

    filtered.sort(key=lambda s: s.mention_count)
    names = [s.model for s in filtered]
    counts = [s.mention_count for s in filtered]
    colours = [
        "#22c55e" if s.avg_sentiment > 0.05 else "#ef4444" if s.avg_sentiment < -0.05 else "#94a3b8"
        for s in filtered
    ]

    fig = go.Figure(go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker_color=colours,
        text=[f"{s.avg_sentiment:+.2f}" for s in filtered],
        textposition="outside",
    ))
    fig.update_layout(
        title="Shoe Model Mentions (colour = sentiment)",
        xaxis_title="Reddit Mentions",
        height=max(350, len(filtered) * 28),
        margin=dict(l=160, r=80, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig.to_json()


def sentiment_price_scatter(signals: list[ModelSignal]) -> str:
    """Scatter: avg_sentiment (x) vs price_premium (y), bubble size = mentions."""
    paired = [s for s in signals if s.num_sales > 0 and s.retail_price > 0]
    if len(paired) < 3:
        return "{}"

    fig = go.Figure(go.Scatter(
        x=[s.avg_sentiment for s in paired],
        y=[s.price_premium * 100 for s in paired],
        mode="markers+text",
        text=[s.model for s in paired],
        textposition="top center",
        marker=dict(
            size=[max(8, min(s.mention_count * 1.5, 40)) for s in paired],
            color=[s.avg_sentiment for s in paired],
            colorscale="RdYlGn",
            cmin=-0.5,
            cmax=0.5,
            showscale=True,
            colorbar=dict(title="Sentiment"),
        ),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Reddit Sentiment vs. eBay Resale Premium",
        xaxis_title="Avg. Sentiment Score (Reddit)",
        yaxis_title="Price Premium over Retail (%)",
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig.to_json()
