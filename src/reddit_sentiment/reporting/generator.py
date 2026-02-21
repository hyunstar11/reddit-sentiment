"""ReportGenerator: runs all analyses and renders HTML + Markdown output."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from reddit_sentiment.analysis.brand_comparison import BrandComparisonAnalyzer
from reddit_sentiment.analysis.channel_attribution import ChannelAttributionAnalyzer
from reddit_sentiment.analysis.narrative import NarrativeThemeExtractor
from reddit_sentiment.analysis.trends import SentimentTrendAnalyzer
from reddit_sentiment.analysis.price_correlation import PriceCorrelationAnalyzer
from reddit_sentiment.reporting.charts import (
    brand_sentiment_bar,
    channel_share_pie,
    intent_funnel,
    model_mentions_bar,
    sentiment_distribution_pie,
    sentiment_price_scatter,
    sentiment_trend_line,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class ReportGenerator:
    """Orchestrates all analysis passes and renders HTML + Markdown reports."""

    def __init__(self, reports_dir: Path) -> None:
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=False,
        )

    def _load_ebay_data(self) -> pd.DataFrame:
        """Load latest eBay parquet if available, else return empty DataFrame."""
        from reddit_sentiment.config import collection_config
        data_dir = collection_config.raw_data_dir
        files = sorted(data_dir.glob("ebay_*.parquet"), reverse=True)
        if files:
            return pd.read_parquet(files[0])
        return pd.DataFrame()

    def generate(self, df: pd.DataFrame, timestamp: str | None = None) -> tuple[Path, Path]:
        """Run analyses and write HTML + Markdown reports.

        Returns:
            (html_path, markdown_path)
        """
        if timestamp is None:
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

        report_date = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

        # ------------------------------------------------------------------
        # Run all analyses
        # ------------------------------------------------------------------
        brand_analyzer = BrandComparisonAnalyzer()
        brand_metrics = brand_analyzer.compute(df)
        brand_table_df = brand_analyzer.comparison_table(df)
        brand_table = brand_table_df.to_dict("records") if not brand_table_df.empty else []

        channel_analyzer = ChannelAttributionAnalyzer()
        attribution = channel_analyzer.analyze(df)

        narrative_extractor = NarrativeThemeExtractor()
        narrative = narrative_extractor.extract(df)

        trend_analyzer = SentimentTrendAnalyzer()
        trends = trend_analyzer.analyze(df)

        # ------------------------------------------------------------------
        # Summary stats
        # ------------------------------------------------------------------
        total_records = len(df)
        total_posts = (
            int((df["record_type"] == "post").sum())
            if "record_type" in df.columns
            else total_records
        )
        total_comments = total_records - total_posts
        brands_detected = len(brand_metrics)
        channels_detected = len(attribution.channel_counts)
        intent_signals = sum(attribution.intent_funnel.values())
        avg_sentiment = float(df["hybrid_score"].mean()) if "hybrid_score" in df.columns else 0.0
        subreddits = df["subreddit"].unique().tolist() if "subreddit" in df.columns else []

        # ------------------------------------------------------------------
        # Price correlation (uses eBay data if available)
        # ------------------------------------------------------------------
        corr_analyzer = PriceCorrelationAnalyzer()
        ebay_df = self._load_ebay_data()
        corr_result = corr_analyzer.analyze(df, ebay_df)
        corr_table = corr_result.summary_df.to_dict("records") if not corr_result.summary_df.empty else []

        # ------------------------------------------------------------------
        # Render charts
        # ------------------------------------------------------------------
        chart_bar = brand_sentiment_bar(brand_metrics)
        chart_pie_sentiment = sentiment_distribution_pie(brand_metrics)
        chart_channel_pie = channel_share_pie(attribution)
        chart_funnel = intent_funnel(attribution)
        chart_trend = sentiment_trend_line(trends.weekly)
        chart_model_bar = model_mentions_bar(corr_result.signals)
        chart_scatter = sentiment_price_scatter(corr_result.signals)

        # ------------------------------------------------------------------
        # HTML report
        # ------------------------------------------------------------------
        template = self._jinja.get_template("report.html.j2")
        html_content = template.render(
            report_date=report_date,
            total_records=total_records,
            total_posts=total_posts,
            total_comments=total_comments,
            brands_detected=brands_detected,
            channels_detected=channels_detected,
            intent_signals=intent_signals,
            avg_sentiment=avg_sentiment,
            subreddits=subreddits,
            brand_table=brand_table,
            theme_counts=narrative.theme_counts,
            theme_pct=narrative.theme_percentages,
            top_tfidf=narrative.top_tfidf_terms,
            top_channels=attribution.top_channels,
            channel_counts=attribution.channel_counts,
            channel_share=attribution.channel_share,
            chart_bar=chart_bar,
            chart_pie_sentiment=chart_pie_sentiment,
            chart_channel_pie=chart_channel_pie,
            chart_funnel=chart_funnel,
            chart_trend=chart_trend,
            chart_model_bar=chart_model_bar,
            chart_scatter=chart_scatter,
            corr_table=corr_table,
            corr_coefficient=corr_result.correlation_sentiment_premium,
            has_ebay_data=not ebay_df.empty,
        )
        html_path = self._reports_dir / f"report_{timestamp}.html"
        html_path.write_text(html_content, encoding="utf-8")

        # ------------------------------------------------------------------
        # Markdown summary
        # ------------------------------------------------------------------
        md_path = self._reports_dir / f"report_{timestamp}.md"
        md_path.write_text(
            self._render_markdown(
                report_date=report_date,
                total_records=total_records,
                total_posts=total_posts,
                total_comments=total_comments,
                avg_sentiment=avg_sentiment,
                brand_table=brand_table,
                attribution=attribution,
                narrative=narrative,
            ),
            encoding="utf-8",
        )

        print(f"[report] HTML  → {html_path}")
        print(f"[report] MD    → {md_path}")
        return html_path, md_path

    @staticmethod
    def _render_markdown(
        report_date: str,
        total_records: int,
        total_posts: int,
        total_comments: int,
        avg_sentiment: float,
        brand_table: list[dict],
        attribution,
        narrative,
    ) -> str:
        lines = [
            "# Reddit Sneaker Sentiment Report",
            "",
            f"Generated: {report_date}",
            "",
            "## Summary",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Records | {total_records:,} |",
            f"| Posts | {total_posts:,} |",
            f"| Comments | {total_comments:,} |",
            f"| Overall Sentiment | {avg_sentiment:+.4f} |",
            "",
            "## Brand Rankings",
            "| Rank | Brand | Mentions | Avg Score | Sentiment | Positive% | Negative% |",
            "|------|-------|----------|-----------|-----------|-----------|-----------|",
        ]
        for i, row in enumerate(brand_table, 1):
            lines.append(
                f"| {i} | {row['brand']} | {row['mentions']} | "
                f"{row['avg_sentiment']:+.4f} | {row['sentiment']} | "
                f"{row['positive_%']:.1f}% | {row['negative_%']:.1f}% |"
            )

        lines += [
            "",
            "## Top Retail Channels",
            "| Channel | Mentions | Share |",
            "|---------|----------|-------|",
        ]
        for ch in attribution.top_channels:
            cnt = attribution.channel_counts.get(ch, 0)
            share = attribution.channel_share.get(ch, 0)
            lines.append(f"| {ch} | {cnt} | {share:.1f}% |")

        lines += [
            "",
            "## Purchase Intent",
            "| Intent | Count |",
            "|--------|-------|",
        ]
        for intent, cnt in sorted(
            attribution.intent_funnel.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"| {intent.replace('_', ' ').title()} | {cnt} |")

        lines += [
            "",
            "## Narrative Themes",
            "| Theme | Mentions | % |",
            "|-------|----------|---|",
        ]
        for theme, cnt in sorted(narrative.theme_counts.items(), key=lambda x: x[1], reverse=True):
            pct = narrative.theme_percentages.get(theme, 0)
            lines.append(f"| {theme} | {cnt} | {pct:.1f}% |")

        if narrative.top_tfidf_terms:
            lines += [
                "",
                f"**Top TF-IDF terms:** {', '.join(narrative.top_tfidf_terms[:20])}",
            ]

        return "\n".join(lines) + "\n"
