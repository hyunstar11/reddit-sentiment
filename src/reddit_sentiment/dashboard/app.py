"""Streamlit dashboard for Reddit Sneaker Sentiment Analysis.

Run with:
    reddit-sentiment dashboard
    # or directly:
    streamlit run src/reddit_sentiment/dashboard/app.py
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path

# Make the reddit_sentiment package importable when run directly by Streamlit Cloud
# (without a pip-installed editable install)
_SRC = Path(__file__).parent.parent.parent  # .../src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sneaker Sentiment Intel",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent.parent.parent  # repo root
_ANNOTATED = _ROOT / "data" / "processed" / "annotated.parquet"
_EBAY_GLOB = "ebay_*.parquet"
_RAW_DIR = _ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading annotated data…")
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_ebay(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob(_EBAY_GLOB), reverse=True)
    if files:
        return pd.read_parquet(files[0])
    return pd.DataFrame()


def _synthetic_demo() -> pd.DataFrame:
    """Generate 500-row synthetic data for demo when no parquet exists."""
    random.seed(42)
    brands_pool = [
        ["Nike"], ["Adidas"], ["New Balance"], ["Hoka"], ["Under Armour"],
        ["Nike", "Adidas"], ["Puma"], ["Asics"], [],
    ]
    channels_pool = [
        ["StockX"], ["GOAT"], ["Nike Direct"], ["Foot Locker"],
        ["StockX", "GOAT"], [], ["Amazon"], ["Grailed"],
    ]
    intents = [
        "completed_purchase", "seeking_purchase", "price_discussion",
        "availability_info", "purchase_consideration", None, None, None,
    ]
    subreddits = ["Sneakers", "Nike", "Adidas", "Running", "Jordans", "SneakerMarket"]
    rows = []
    for i in range(500):
        sentiment = random.gauss(0.15, 0.35)
        rows.append({
            "id": f"r{i}",
            "subreddit": random.choice(subreddits),
            "record_type": "post" if i < 250 else "comment",
            "score": random.randint(1, 500),
            "created_utc": pd.Timestamp("2025-10-01", tz="UTC") + pd.Timedelta(hours=i * 3),
            "full_text": f"Sample text about sneakers #{i}",
            "vader_score": sentiment,
            "hybrid_score": min(max(sentiment, -1.0), 1.0),
            "transformer_score": None,
            "brands": random.choice(brands_pool),
            "channels": random.choice(channels_pool),
            "models": [],
            "primary_intent": random.choice(intents),
            "all_intents": [],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis helpers (cached per filtered df hash)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def run_brand_analysis(df_json: str, min_mentions: int):
    from reddit_sentiment.analysis.brand_comparison import BrandComparisonAnalyzer
    df = pd.read_json(io.StringIO(df_json), orient="split")
    analyzer = BrandComparisonAnalyzer()
    metrics = analyzer.compute(df, min_mentions=min_mentions)
    table = analyzer.comparison_table(df, min_mentions=min_mentions)
    return metrics, table


@st.cache_data(show_spinner=False)
def run_channel_analysis(df_json: str):
    from reddit_sentiment.analysis.channel_attribution import ChannelAttributionAnalyzer
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return ChannelAttributionAnalyzer().analyze(df)


@st.cache_data(show_spinner=False)
def run_narrative_analysis(df_json: str):
    from reddit_sentiment.analysis.narrative import NarrativeThemeExtractor
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return NarrativeThemeExtractor().extract(df)


@st.cache_data(show_spinner=False)
def run_trend_analysis(df_json: str):
    from reddit_sentiment.analysis.trends import SentimentTrendAnalyzer
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return SentimentTrendAnalyzer().analyze(df)


@st.cache_data(show_spinner=False)
def run_model_analysis(df_json: str, ebay_json: str):
    from reddit_sentiment.analysis.price_correlation import PriceCorrelationAnalyzer
    df = pd.read_json(io.StringIO(df_json), orient="split")
    ebay = pd.read_json(io.StringIO(ebay_json), orient="split") if ebay_json else pd.DataFrame()
    return PriceCorrelationAnalyzer().analyze(df, ebay)


@st.cache_data(show_spinner=False)
def run_brand_correlation(df_json: str):
    from reddit_sentiment.analysis.price_correlation import PriceCorrelationAnalyzer
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return PriceCorrelationAnalyzer().analyze_brand_level(df)


@st.cache_data(show_spinner=False)
def run_brand_signals(df_json: str, min_mentions: int):
    from reddit_sentiment.analysis.brand_signals import BrandIntelligenceAnalyzer
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return BrandIntelligenceAnalyzer().analyze(df, min_mentions=min_mentions)


@st.cache_data(show_spinner=False)
def run_model_intelligence(df_json: str):
    from reddit_sentiment.analysis.model_intelligence import ModelIntelligenceAnalyzer
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return ModelIntelligenceAnalyzer().analyze(df)


@st.cache_data(show_spinner=False, ttl=3600)
def load_health_history() -> pd.DataFrame:
    path = _ROOT / "data" / "processed" / "health_score_history.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame()


def _detect_signal_changes(history: pd.DataFrame) -> list[dict]:
    """Compare the two most recent snapshots; return list of brand signal changes."""
    if history.empty or "signal" not in history.columns:
        return []
    timestamps = sorted(history["timestamp"].unique())
    if len(timestamps) < 2:
        return []
    prev_ts, curr_ts = timestamps[-2], timestamps[-1]
    prev = history[history["timestamp"] == prev_ts].set_index("brand")
    curr = history[history["timestamp"] == curr_ts].set_index("brand")
    changes = []
    for brand in curr.index:
        if brand not in prev.index:
            continue
        old_sig = prev.loc[brand, "signal"]
        new_sig = curr.loc[brand, "signal"]
        old_score = prev.loc[brand, "health_score"]
        new_score = curr.loc[brand, "health_score"]
        if old_sig != new_sig:
            changes.append({
                "brand": brand,
                "from": old_sig,
                "to": new_sig,
                "delta": new_score - old_score,
            })
    return changes


def _to_json(df: pd.DataFrame) -> str:
    """Serialize DataFrame to JSON for cache key (handles datetime columns)."""
    df = df.copy()
    for col in df.select_dtypes(include=["datetimetz", "datetime64[ns, UTC]", "datetime"]).columns:
        df[col] = df[col].astype(str)
    return df.to_json(orient="split", date_format="iso")


def _render(chart_json: str) -> None:
    if chart_json and chart_json != "{}":
        fig = pio.from_json(chart_json)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough data to render this chart.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Render sidebar filters; return (filtered_df, min_mentions, is_demo)."""
    is_demo = not _ANNOTATED.exists()

    with st.sidebar:
        st.title("👟 Sneaker Intel")

        if is_demo:
            st.warning(
                "No annotated.parquet found — showing **demo data**.\n\n"
                "Run:\n```\nreddit-sentiment pipeline --public\n```"
            )
        else:
            mtime = _ANNOTATED.stat().st_mtime
            mtime_str = pd.Timestamp(mtime, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
            st.success(f"Data loaded\n\n**Updated:** {mtime_str}")

        st.divider()
        st.subheader("Filters")

        # Subreddit filter
        all_subs = (
            sorted(df["subreddit"].dropna().unique().tolist())
            if "subreddit" in df.columns
            else []
        )
        selected_subs = st.multiselect(
            "Subreddits",
            options=all_subs,
            default=all_subs,
            help="Filter to specific subreddits",
        )

        # Date range
        filtered = df.copy()
        if "created_utc" in df.columns:
            try:
                dates = pd.to_datetime(df["created_utc"], utc=True)
                min_date = dates.min().date()
                max_date = dates.max().date()
                date_range = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_filter",
                )
                if isinstance(date_range, list | tuple) and len(date_range) == 2:
                    start, end = date_range
                    mask = (dates.dt.date >= start) & (dates.dt.date <= end)
                    filtered = filtered[mask]
            except Exception:
                pass

        # Apply subreddit filter
        if selected_subs and "subreddit" in filtered.columns:
            filtered = filtered[filtered["subreddit"].isin(selected_subs)]

        # Record type filter
        if "record_type" in df.columns:
            record_types = st.multiselect(
                "Record type",
                options=["post", "comment"],
                default=["post", "comment"],
            )
            if record_types:
                filtered = filtered[filtered["record_type"].isin(record_types)]

        st.divider()
        st.subheader("Analysis options")

        min_mentions = st.slider(
            "Min brand mentions",
            min_value=1,
            max_value=20,
            value=5,
            help="Exclude brands with fewer than this many mentions",
        )

        st.divider()
        st.caption(f"**Showing:** {len(filtered):,} / {len(df):,} records")

    return filtered, min_mentions, is_demo


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def _kpi_cards(df: pd.DataFrame, brand_metrics: dict, attribution) -> None:
    total = len(df)
    posts = int((df["record_type"] == "post").sum()) if "record_type" in df.columns else total
    avg_sent = float(df["hybrid_score"].mean()) if "hybrid_score" in df.columns else 0.0
    sent_label = "Positive" if avg_sent > 0.05 else "Negative" if avg_sent < -0.05 else "Neutral"
    sent_colour = "🟢" if avg_sent > 0.05 else "🔴" if avg_sent < -0.05 else "⚪"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{total:,}", help="Posts + comments in filtered set")
    c2.metric("Posts", f"{posts:,}", f"{total - posts:,} comments")
    c3.metric(
        "Avg Sentiment",
        f"{avg_sent:+.3f}",
        f"{sent_colour} {sent_label}",
        delta_color="normal",
    )
    c4.metric("Brands Tracked", len(brand_metrics))
    c5.metric("Retail Channels", len(attribution.channel_counts))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _tab_overview(df: pd.DataFrame, brand_metrics: dict, attribution, trends) -> None:
    from reddit_sentiment.reporting.charts import sentiment_distribution_pie, sentiment_trend_line

    col_l, col_r = st.columns([1, 2])
    with col_l:
        _render(sentiment_distribution_pie(brand_metrics))
    with col_r:
        _render(sentiment_trend_line(trends.weekly))

    st.divider()
    # Top-level subreddit breakdown
    if "subreddit" in df.columns and "hybrid_score" in df.columns:
        st.subheader("Sentiment by Subreddit")
        sub_df = (
            df.groupby("subreddit")["hybrid_score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_sentiment", "count": "mentions"})
            .sort_values("avg_sentiment", ascending=False)
            .reset_index()
        )
        sub_df["avg_sentiment"] = sub_df["avg_sentiment"].round(4)

        import plotly.express as px
        fig = px.bar(
            sub_df,
            x="avg_sentiment",
            y="subreddit",
            orientation="h",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            range_color=[-0.5, 0.5],
            text="avg_sentiment",
            labels={"avg_sentiment": "Avg Sentiment", "subreddit": ""},
            title="Average Sentiment by Subreddit",
        )
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch")


def _tab_brands(df: pd.DataFrame, brand_metrics: dict, brand_table: pd.DataFrame) -> None:
    from reddit_sentiment.reporting.charts import brand_sentiment_bar

    if not brand_metrics:
        st.info("No brands meet the minimum mentions threshold. Lower the slider in the sidebar.")
        return

    _render(brand_sentiment_bar(brand_metrics))

    st.divider()
    st.subheader("Brand comparison table")

    if not brand_table.empty:
        # Colour the sentiment column
        st.dataframe(
            brand_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "avg_sentiment": st.column_config.NumberColumn(format="%+.4f"),
                "positive_%": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100
                ),
                "negative_%": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )


def _tab_channels(attribution) -> None:
    from reddit_sentiment.reporting.charts import channel_share_pie, intent_funnel

    col_l, col_r = st.columns(2)
    with col_l:
        _render(channel_share_pie(attribution))
    with col_r:
        _render(intent_funnel(attribution))

    if attribution.top_channels:
        st.divider()
        st.subheader("Channel breakdown")
        ch_rows = [
            {
                "Channel": ch,
                "Mentions": attribution.channel_counts.get(ch, 0),
                "Share (%)": round(attribution.channel_share.get(ch, 0), 1),
            }
            for ch in attribution.top_channels
        ]
        st.dataframe(pd.DataFrame(ch_rows), use_container_width=True, hide_index=True)


def _tab_themes(narrative) -> None:
    import plotly.express as px

    theme_rows = [
        {"Theme": k, "Mentions": v, "%": round(narrative.theme_percentages.get(k, 0), 1)}
        for k, v in sorted(narrative.theme_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    if not theme_rows:
        st.info("No themes detected.")
        return

    theme_df = pd.DataFrame(theme_rows)
    fig = px.bar(
        theme_df.sort_values("Mentions"),
        x="Mentions",
        y="Theme",
        orientation="h",
        color="%",
        color_continuous_scale="Blues",
        text="%",
        title="Narrative Theme Frequency",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=10, r=60, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    if narrative.top_tfidf_terms:
        st.divider()
        st.subheader("Top TF-IDF keywords")
        terms = narrative.top_tfidf_terms[:30]
        # Display as a pill-style grid
        cols = st.columns(6)
        for i, term in enumerate(terms):
            cols[i % 6].markdown(
                f"<span style='background:#e0e7ff;padding:3px 8px;border-radius:12px;"
                f"font-size:0.85em'>{term}</span>",
                unsafe_allow_html=True,
            )


def _render_stockx_brand_correlation(df_json: str) -> None:
    """Show brand-level Reddit sentiment vs. StockX resale premium as eBay fallback."""
    import plotly.express as px

    brand_corr = run_brand_correlation(df_json)
    if not brand_corr.signals:
        st.info("Not enough brand data to compute correlation.")
        return

    st.divider()
    st.subheader("Sentiment vs. StockX Resale Premium — Brand Level")
    st.caption(
        "eBay data unavailable. Using StockX 2023 market snapshot as the resale price signal "
        "(median price premium per brand). This mirrors the combined-signal analysis in the "
        "companion [sneaker-intel](https://github.com/hyunstar11/sneaker-intel) project."
    )

    df = brand_corr.summary_df
    if df.empty:
        return

    fig = px.scatter(
        df,
        x="avg_sentiment",
        y="stockx_premium_%",
        text="brand",
        size="mentions",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        range_color=[-0.3, 0.5],
        labels={
            "avg_sentiment": "Avg Reddit Sentiment",
            "stockx_premium_%": "StockX Resale Premium (%)",
            "mentions": "Reddit Mentions",
        },
        title="Reddit Sentiment vs. StockX Resale Premium by Brand",
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    if brand_corr.correlation_sentiment_premium is not None:
        r = brand_corr.correlation_sentiment_premium
        interp = "positive" if r > 0.3 else "negative" if r < -0.3 else "weak"
        st.caption(f"Pearson r = **{r:.3f}** — {interp} correlation (brand level, n={len(df)})")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "avg_sentiment": st.column_config.NumberColumn(format="%+.4f"),
            "stockx_premium_%": st.column_config.NumberColumn(format="%.1f%%"),
            "positive_%": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
            "negative_%": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )


def _tab_models(corr_result, has_ebay: bool, df_json: str = "") -> None:
    import plotly.express as px

    from reddit_sentiment.reporting.charts import model_mentions_bar, sentiment_price_scatter

    if not corr_result.signals:
        st.info(
            "No shoe model signals detected. Make sure `full_text` column is present "
            "and run `reddit-sentiment analyze` to populate the `models` column."
        )
        return

    # Top-model KPI strip
    top3 = corr_result.signals[:3]
    cols = st.columns(len(top3))
    for col, sig in zip(cols, top3):
        label = "Positive" if sig.avg_sentiment > 0.05 else (
            "Negative" if sig.avg_sentiment < -0.05 else "Neutral"
        )
        col.metric(sig.model, f"{sig.mention_count} mentions", label)

    st.divider()
    _render(model_mentions_bar(corr_result.signals))

    if not corr_result.summary_df.empty:
        st.divider()
        st.subheader("Model signals")

        # Show eBay columns only when data is available
        reddit_cols = ["model", "brand", "retail_price", "mentions",
                       "avg_sentiment", "positive_%", "negative_%"]
        ebay_cols = ["num_sales", "avg_sold_price", "price_premium_%"]
        display_cols = reddit_cols + ebay_cols if has_ebay else reddit_cols
        display_df = corr_result.summary_df[display_cols].copy()

        col_cfg = {
            "avg_sentiment": st.column_config.NumberColumn(format="%+.4f"),
            "positive_%": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
            "negative_%": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
        }
        if has_ebay:
            col_cfg["price_premium_%"] = st.column_config.NumberColumn(format="%.1f%%")

        st.dataframe(display_df, use_container_width=True, hide_index=True,
                     column_config=col_cfg)

        # Sentiment breakdown bar chart per model (top 10)
        st.divider()
        st.subheader("Positive vs. Negative breakdown (top 10 models)")
        top10 = corr_result.summary_df.nlargest(10, "mentions")
        breakdown = pd.concat([
            pd.DataFrame({"model": r["model"], "type": t, "pct": v}, index=[0])
            for _, r in top10.iterrows()
            for t, v in [("Positive", r["positive_%"]), ("Negative", r["negative_%"])]
        ])
        fig = px.bar(
            breakdown,
            x="pct", y="model", color="type", orientation="h",
            barmode="group",
            color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444"},
            labels={"pct": "%", "model": "", "type": ""},
            title="Sentiment Split per Model",
        )
        fig.update_layout(height=420, plot_bgcolor="white",
                          margin=dict(l=10, r=40, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    if has_ebay:
        st.divider()
        st.subheader("Sentiment vs. Resale Premium (eBay)")
        _render(sentiment_price_scatter(corr_result.signals))
        if corr_result.correlation_sentiment_premium is not None:
            r = corr_result.correlation_sentiment_premium
            interp = "positive" if r > 0.3 else "negative" if r < -0.3 else "weak"
            st.caption(
                f"Pearson r = **{r:.3f}** — {interp} correlation between "
                "Reddit sentiment and eBay resale premium."
            )
    else:
        _render_stockx_brand_correlation(df_json)


# ---------------------------------------------------------------------------
# Brand Signals tab
# ---------------------------------------------------------------------------

_SIGNAL_COLORS = {
    "🟢 Scale Up": "#22c55e",
    "🟡 Hold": "#eab308",
    "🔴 Watch": "#ef4444",
}


def _tab_brand_signals(intel_result) -> None:
    import plotly.express as px

    st.subheader("Brand Health Intelligence")
    st.caption(
        "Composite score combining StockX resale premium, deadstock volume, "
        "Reddit sentiment, and purchase intent."
    )

    df = intel_result.summary_df
    if df.empty:
        st.info("No brand intelligence data available.")
        return

    # 1 — Hero table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Health Score": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0
            ),
            "Purchase Intent %": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
            "Avg Sentiment": st.column_config.NumberColumn(format="%+.4f"),
        },
    )

    st.divider()

    col_l, col_r = st.columns(2)

    # 2 — Health Score bar chart
    with col_l:
        bar_df = df[["Brand", "Health Score", "Signal"]].sort_values(
            "Health Score", ascending=True
        )
        fig_bar = px.bar(
            bar_df,
            x="Health Score",
            y="Brand",
            orientation="h",
            color="Signal",
            color_discrete_map=_SIGNAL_COLORS,
            text="Health Score",
            title="Brand Health Score",
        )
        fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_bar.update_layout(
            plot_bgcolor="white",
            height=350,
            margin=dict(l=10, r=60, t=50, b=10),
            showlegend=True,
        )
        st.plotly_chart(fig_bar, width="stretch")

    # 3 — Sentiment vs Premium scatter
    with col_r:
        scatter_df = df[["Brand", "Avg Sentiment", "StockX Premium %", "Mentions", "Signal"]].copy()
        # Give brands with 0 mentions a minimum bubble size for visibility
        scatter_df["_size"] = scatter_df["Mentions"].clip(lower=5)
        fig_scatter = px.scatter(
            scatter_df,
            x="Avg Sentiment",
            y="StockX Premium %",
            text="Brand",
            size="_size",
            color="Signal",
            color_discrete_map=_SIGNAL_COLORS,
            labels={
                "Avg Sentiment": "Avg Reddit Sentiment",
                "StockX Premium %": "StockX Resale Premium (%)",
            },
            title="Sentiment vs. StockX Premium",
        )
        fig_scatter.update_traces(
            textposition="top center",
            marker=dict(line=dict(width=1, color="white")),
        )
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.update_layout(
            plot_bgcolor="white",
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # 4 — Signal change alerts
    history = load_health_history()
    changes = _detect_signal_changes(history)
    if changes:
        st.divider()
        st.subheader("⚡ Signal Changes Since Last Refresh")
        for ch in changes:
            direction = "upgraded" if ch["delta"] > 0 else "downgraded"
            colour = "green" if ch["delta"] > 0 else "red"
            st.markdown(
                f":{colour}[**{ch['brand']}** {direction}: "
                f"{ch['from']} → {ch['to']} "
                f"(score {ch['delta']:+.3f})]"
            )

    # 5 — Health score trend + 30-day forecast
    if not history.empty and history["timestamp"].nunique() >= 2:
        import numpy as np

        st.divider()
        st.subheader("Brand Health Score — Trend & 30-Day Forecast")
        fig_trend = px.line(
            history.sort_values("timestamp"),
            x="timestamp",
            y="health_score",
            color="brand",
            markers=True,
            labels={"timestamp": "", "health_score": "Health Score", "brand": "Brand"},
            title="Brand Health Score Over Time (dashed = 30-day projection)",
        )

        # Add forecast traces per brand
        brand_colors = {
            trace.name: trace.line.color
            for trace in fig_trend.data
        }
        last_ts = history["timestamp"].max()
        future_ts = last_ts + pd.Timedelta(days=30)
        for brand in history["brand"].unique():
            bdf = history[history["brand"] == brand].sort_values("timestamp")
            if len(bdf) < 2:
                continue
            # Convert timestamps to numeric (days since first point)
            t0 = bdf["timestamp"].iloc[0]
            x_num = (bdf["timestamp"] - t0).dt.total_seconds() / 86400
            y_vals = bdf["health_score"].values
            slope, intercept = np.polyfit(x_num, y_vals, 1)
            x_future = (future_ts - t0).total_seconds() / 86400
            y_future = float(np.clip(intercept + slope * x_future, 0.0, 1.0))
            color = brand_colors.get(brand, "#888888")
            fig_trend.add_scatter(
                x=[last_ts, future_ts],
                y=[float(bdf["health_score"].iloc[-1]), y_future],
                mode="lines",
                line=dict(dash="dash", color=color, width=1.5),
                showlegend=False,
                hovertemplate=(
                    f"<b>{brand}</b> (projected)<br>"
                    f"Score: {y_future:.3f}<extra></extra>"
                ),
            )

        fig_trend.update_layout(
            plot_bgcolor="white",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(range=[0, 1]),
        )
        fig_trend.add_hline(
            y=0.6, line_dash="dot", line_color="#22c55e",
            opacity=0.4, annotation_text="Scale Up threshold",
        )
        fig_trend.add_hline(
            y=0.4, line_dash="dot", line_color="#eab308",
            opacity=0.4, annotation_text="Watch threshold",
        )
        st.plotly_chart(fig_trend, width="stretch")
        n_pts = history["timestamp"].nunique()
        st.caption(
            f"Forecast based on linear trend across {n_pts} snapshots. "
            "Accuracy improves as more data accumulates."
        )


# ---------------------------------------------------------------------------
# Model Intelligence tab
# ---------------------------------------------------------------------------

_TIER_COLORS = {"High": "#22c55e", "Medium": "#eab308", "Low": "#ef4444"}


def _tab_model_intelligence(result) -> None:
    import plotly.express as px

    st.subheader("Model Intelligence")
    st.caption(
        "ML demand signal (sneaker-intel ensemble trained on StockX 2017–2019) "
        "combined with live Reddit sentiment. Demand Score = global percentile rank "
        "by predicted sale/retail multiplier."
    )

    if result.predictions_meta:
        st.caption(
            f"Predictions: {result.predictions_meta.get('model_source', '')} · "
            f"Generated {result.predictions_meta.get('generated_at', '')}"
        )

    df = result.summary_df
    if df.empty:
        st.info("Model predictions not found. Ensure model_predictions.json exists.")
        return

    # Brand filter
    brands = sorted(df["Brand"].unique())
    selected = st.multiselect(
        "Filter by brand", options=brands, default=brands, key="mintel_brand_filter"
    )
    if selected:
        df = df[df["Brand"].isin(selected)]

    # 1 — Table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Demand Score": st.column_config.ProgressColumn(
                format="%.2f", min_value=0.0, max_value=1.0
            ),
            "Combined Score": st.column_config.ProgressColumn(
                format="%.2f", min_value=0.0, max_value=1.0
            ),
            "Avg Sentiment": st.column_config.NumberColumn(format="%+.4f"),
            "Positive %": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
            "Negative %": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )

    st.divider()
    col_l, col_r = st.columns(2)

    # 2 — Combined Score bar (top 15)
    with col_l:
        top = df.nlargest(15, "Combined Score")
        fig = px.bar(
            top.sort_values("Combined Score"),
            x="Combined Score",
            y="Model",
            color="Demand Tier",
            color_discrete_map=_TIER_COLORS,
            orientation="h",
            text="Combined Score",
            title="Top 15 Models — Combined Score",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(
            plot_bgcolor="white",
            height=450,
            margin=dict(l=10, r=60, t=50, b=10),
            showlegend=True,
        )
        st.plotly_chart(fig, width="stretch")

    # 3 — Demand Score vs Reddit Sentiment scatter
    with col_r:
        plot_df = df.copy()
        plot_df["_size"] = plot_df["Reddit Mentions"].clip(lower=3)
        fig2 = px.scatter(
            plot_df,
            x="Demand Score",
            y="Avg Sentiment",
            text="Model",
            size="_size",
            color="Demand Tier",
            color_discrete_map=_TIER_COLORS,
            labels={
                "Demand Score": "ML Demand Score (percentile)",
                "Avg Sentiment": "Avg Reddit Sentiment",
            },
            title="ML Demand vs. Reddit Sentiment",
        )
        fig2.update_traces(
            textposition="top center",
            marker=dict(line=dict(width=1, color="white")),
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig2.update_layout(
            plot_bgcolor="white",
            height=450,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load raw data
    if _ANNOTATED.exists():
        df_full = load_data(_ANNOTATED)
    else:
        df_full = _synthetic_demo()

    ebay_df = load_ebay(_RAW_DIR) if _RAW_DIR.exists() else pd.DataFrame()

    # Sidebar filters → filtered df
    df, min_mentions, is_demo = _sidebar(df_full)

    # Page header
    st.title("👟 Sneaker Sentiment Intelligence")
    if is_demo:
        st.caption("Demo mode — synthetic data")
    else:
        st.caption(f"{len(df):,} records from {df['subreddit'].nunique()} subreddits")

    st.divider()

    if df.empty:
        st.warning("No records match current filters. Adjust the sidebar.")
        return

    # Serialise filtered df once for cache-keyed analysis calls
    df_json = _to_json(df)
    ebay_json = _to_json(ebay_df) if not ebay_df.empty else ""

    # Run all analyses
    with st.spinner("Running analyses…"):
        brand_metrics, brand_table = run_brand_analysis(df_json, min_mentions)
        attribution = run_channel_analysis(df_json)
        narrative = run_narrative_analysis(df_json)
        trends = run_trend_analysis(df_json)
        corr_result = run_model_analysis(df_json, ebay_json)
        intel_result = run_brand_signals(df_json, min_mentions)
        model_intel_result = run_model_intelligence(df_json)

    # KPI strip
    _kpi_cards(df, brand_metrics, attribution)
    st.divider()

    # Tabs
    tab_ov, tab_br, tab_ch, tab_th, tab_mo, tab_intel, tab_mintel = st.tabs([
        "📊 Overview",
        "🏷️ Brands",
        "🛒 Channels",
        "💬 Themes",
        "👟 Models",
        "🎯 Brand Signals",
        "🔬 Model Intelligence",
    ])

    with tab_ov:
        _tab_overview(df, brand_metrics, attribution, trends)

    with tab_br:
        _tab_brands(df, brand_metrics, brand_table)

    with tab_ch:
        _tab_channels(attribution)

    with tab_th:
        _tab_themes(narrative)

    with tab_mo:
        _tab_models(corr_result, has_ebay=not ebay_df.empty, df_json=df_json)

    with tab_intel:
        _tab_brand_signals(intel_result)

    with tab_mintel:
        _tab_model_intelligence(model_intel_result)


if __name__ == "__main__":
    main()
