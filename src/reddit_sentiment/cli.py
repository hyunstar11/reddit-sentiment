"""Click CLI: collect | analyze | report | pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from reddit_sentiment.config import CollectionConfig, collection_config


@click.group()
def cli() -> None:
    """Reddit Sneaker Sentiment Analysis Pipeline."""


@cli.command()
@click.option("--output", "-o", type=click.Path(), default=None, help="Output Parquet path")
@click.option(
    "--subreddits",
    "-s",
    multiple=True,
    help="Subreddits to collect (repeatable; defaults to config list)",
)
@click.option(
    "--public",
    is_flag=True,
    default=False,
    help="Use public JSON API (no credentials required)",
)
def collect(output: str | None, subreddits: tuple[str, ...], public: bool) -> None:
    """Fetch posts from Reddit → data/raw/*.parquet

    Use --public to collect without API credentials (uses Reddit's public JSON API).
    """
    cfg = CollectionConfig()
    if subreddits:
        cfg = CollectionConfig(subreddits=list(subreddits))

    out_path = Path(output) if output else None

    if public:
        from reddit_sentiment.collection.public_collector import PublicSubredditCollector
        collector = PublicSubredditCollector(config=cfg)
    else:
        from reddit_sentiment.collection.collector import SubredditCollector
        collector = SubredditCollector(config=cfg)  # type: ignore[assignment]

    try:
        result = collector.collect(output_path=out_path)
        click.echo(f"Saved: {result}")
    except Exception as exc:
        click.echo(f"Collection failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    default=None,
    help="Input Parquet file (default: latest in data/raw/)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output Parquet path (default: data/processed/annotated.parquet)",
)
@click.option(
    "--no-transformer", is_flag=True, default=False, help="Use VADER only (faster, no GPU needed)"
)
def analyze(input_path: str | None, output: str | None, no_transformer: bool) -> None:
    """Run brand detection + sentiment → data/processed/annotated.parquet"""
    import pandas as pd

    from reddit_sentiment.collection.collector import SubredditCollector
    from reddit_sentiment.sentiment.pipeline import SentimentPipeline

    cfg = collection_config
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if input_path:
        df = pd.read_parquet(input_path)
    else:
        df = SubredditCollector.load_latest(cfg.raw_data_dir)

    click.echo(f"Loaded {len(df)} records")

    pipeline = SentimentPipeline(use_transformer=not no_transformer)
    annotated = pipeline.annotate(df)

    out_path = Path(output) if output else cfg.processed_data_dir / "annotated.parquet"
    annotated.to_parquet(out_path, index=False)
    click.echo(f"Saved annotated data: {out_path}")


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    default=None,
    help="Input annotated Parquet (default: data/processed/annotated.parquet)",
)
def report(input_path: str | None) -> None:
    """Generate HTML + Markdown report → data/reports/"""
    import pandas as pd

    from reddit_sentiment.reporting.generator import ReportGenerator

    cfg = collection_config

    if input_path:
        df = pd.read_parquet(input_path)
    else:
        annotated_path = cfg.processed_data_dir / "annotated.parquet"
        if not annotated_path.exists():
            click.echo("No annotated.parquet found. Run 'analyze' first.", err=True)
            sys.exit(1)
        df = pd.read_parquet(annotated_path)

    click.echo(f"Generating report for {len(df)} records…")
    generator = ReportGenerator(reports_dir=cfg.reports_dir)
    html_path, md_path = generator.generate(df)
    click.echo(f"HTML report: {html_path}")
    click.echo(f"Markdown:    {md_path}")


@cli.command("ebay-collect")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output Parquet path")
@click.option(
    "--models",
    "-m",
    multiple=True,
    help="Specific model names to collect (default: all detected models from latest annotated data)",
)
def ebay_collect(output: str | None, models: tuple[str, ...]) -> None:
    """Fetch eBay sold listings for shoe models → data/raw/ebay_*.parquet

    Requires EBAY_APP_ID in .env. Register free at developer.ebay.com.
    """
    import pandas as pd

    from reddit_sentiment.collection.ebay_collector import EbayCollector
    from reddit_sentiment.config import ebay_config
    from reddit_sentiment.detection.models import MODEL_CATALOG

    cfg = collection_config

    # Determine models to collect
    if models:
        model_list = list(models)
    else:
        # Auto-detect: use models found in the latest annotated data
        annotated_path = cfg.processed_data_dir / "annotated.parquet"
        if annotated_path.exists():
            df = pd.read_parquet(annotated_path)
            if "models" in df.columns:
                found = set()
                for row in df["models"].dropna():
                    if hasattr(row, "__iter__") and not isinstance(row, str):
                        found.update(row)
                model_list = sorted(
                    {str(m) for m in found if m and str(m) != "nan"}
                    & set(MODEL_CATALOG.keys())
                )
                click.echo(f"Auto-detected {len(model_list)} models from annotated data")
            else:
                model_list = list(MODEL_CATALOG.keys())
        else:
            model_list = list(MODEL_CATALOG.keys())

    if not model_list:
        click.echo("No models to collect.", err=True)
        sys.exit(1)

    click.echo(f"Collecting eBay sold listings for {len(model_list)} models…")
    collector = EbayCollector(config=ebay_config)

    try:
        out_path = Path(output) if output else None
        result = collector.collect(models=model_list, output_path=out_path)
        click.echo(f"Saved: {result}")
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--no-transformer", is_flag=True, default=False)
@click.option("--public", is_flag=True, default=False, help="Use public JSON API (no credentials)")
def pipeline(no_transformer: bool, public: bool) -> None:
    """Run collect → analyze → report in sequence."""
    from reddit_sentiment.reporting.generator import ReportGenerator
    from reddit_sentiment.sentiment.pipeline import SentimentPipeline

    cfg = collection_config

    click.echo("=== Step 1/3: Collect ===")
    if public:
        from reddit_sentiment.collection.public_collector import PublicSubredditCollector
        collector = PublicSubredditCollector()
    else:
        from reddit_sentiment.collection.collector import SubredditCollector
        collector = SubredditCollector()
    raw_path = collector.collect()

    click.echo("=== Step 2/3: Analyze ===")
    import pandas as pd

    df = pd.read_parquet(raw_path)
    pl = SentimentPipeline(use_transformer=not no_transformer)
    annotated = pl.annotate(df)
    out = cfg.processed_data_dir / "annotated.parquet"
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_parquet(out, index=False)
    click.echo(f"Annotated: {out}")

    click.echo("=== Step 3/3: Report ===")
    generator = ReportGenerator(reports_dir=cfg.reports_dir)
    html_path, md_path = generator.generate(annotated)
    click.echo(f"Done! Report: {html_path}")


if __name__ == "__main__":
    cli()
