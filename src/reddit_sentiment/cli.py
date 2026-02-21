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
def collect(output: str | None, subreddits: tuple[str, ...]) -> None:
    """Fetch posts and comments from Reddit → data/raw/*.parquet"""
    from reddit_sentiment.collection.collector import SubredditCollector

    cfg = CollectionConfig()
    if subreddits:
        cfg = CollectionConfig(subreddits=list(subreddits))

    collector = SubredditCollector(config=cfg)
    out_path = Path(output) if output else None
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


@cli.command()
@click.option("--no-transformer", is_flag=True, default=False)
def pipeline(no_transformer: bool) -> None:
    """Run collect → analyze → report in sequence."""
    from reddit_sentiment.collection.collector import SubredditCollector
    from reddit_sentiment.reporting.generator import ReportGenerator
    from reddit_sentiment.sentiment.pipeline import SentimentPipeline

    cfg = collection_config

    click.echo("=== Step 1/3: Collect ===")
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
