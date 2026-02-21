# Reddit Sneaker Sentiment Analysis

A standalone NLP pipeline that extracts consumer narrative signals from Reddit — brand sentiment, narrative themes, and retail channel attribution across 10 major sneaker brands.

```
reddit-sentiment collect → analyze → report
```

---

## Architecture

```
reddit-sentiment/
├── src/reddit_sentiment/
│   ├── config.py                   # pydantic-settings: API keys + thresholds
│   ├── collection/
│   │   ├── client.py               # PRAW read-only wrapper
│   │   ├── collector.py            # SubredditCollector (checkpoint + Parquet)
│   │   └── schemas.py              # RedditPost / RedditComment dataclasses
│   ├── detection/
│   │   ├── brands.py               # BrandDetector (alias map + context window)
│   │   ├── channels.py             # ChannelDetector (URL domain + keyword)
│   │   └── intent.py               # PurchaseIntentClassifier (7 intent types)
│   ├── sentiment/
│   │   ├── vader.py                # VaderAnalyzer (fast baseline, all texts)
│   │   ├── transformer.py          # TransformerAnalyzer (cardiffnlp, brand contexts)
│   │   └── pipeline.py             # SentimentPipeline (hybrid 0.6t + 0.4v)
│   ├── analysis/
│   │   ├── brand_comparison.py     # BrandMetrics + comparison_table()
│   │   ├── narrative.py            # TF-IDF + keyword theme extraction
│   │   ├── channel_attribution.py  # Channel share + intent funnel
│   │   └── trends.py               # Weekly / monthly sentiment trends
│   ├── reporting/
│   │   ├── charts.py               # 5 Plotly chart functions → JSON
│   │   ├── generator.py            # ReportGenerator (HTML + Markdown)
│   │   └── templates/report.html.j2
│   └── cli.py                      # Click CLI: collect|analyze|report|pipeline
└── tests/                          # 101 tests, all passing
```

### Sentiment Scoring

| Layer | Model | Scope | Weight |
|-------|-------|-------|--------|
| Fast baseline | VADER (`vaderSentiment`) | All texts | 0.4 |
| Nuanced | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Brand context windows (±15 words) only | 0.6 |
| **Final** | **Hybrid** | `0.6 × transformer + 0.4 × VADER` | — |

The transformer runs only on short brand-context snippets (not full posts), keeping inference time manageable without losing accuracy where it matters most.

---

## Quick Start

### 1. Prerequisites

Create a Reddit "script" app at <https://www.reddit.com/prefs/apps> and copy credentials:

```bash
cp .env.example .env
# Edit .env: add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
```

### 2. Install

```bash
# Core + dev dependencies (no GPU needed)
make install

# With transformer model (requires ~2 GB free disk + GPU optional)
make install-ml
```

### 3. Run Pipeline

```bash
# All-in-one
make pipeline

# Or step by step:
make collect   # → data/raw/posts_YYYYMMDD_HHMMSS.parquet
make analyze   # → data/processed/annotated.parquet
make report    # → data/reports/report_YYYYMMDD.html
```

### 4. CLI Reference

```bash
reddit-sentiment collect   [--output PATH] [--subreddits r1 r2 ...]
reddit-sentiment analyze   [--input PATH] [--output PATH] [--no-transformer]
reddit-sentiment report    [--input PATH]
reddit-sentiment pipeline  [--no-transformer]
```

---

## Brands Tracked

| Brand | Key Aliases |
|-------|-------------|
| Nike | Swoosh, Air Max, Air Jordan, Dunk, Jordan |
| Adidas | Three Stripes, Yeezy, Ultraboost, NMD |
| Li-Ning | Way of Wade, WoW, LN |
| Anta | KT, Klay Thompson |
| 361 Degrees | 361° |
| Under Armour | UA, Curry Brand, Curry N |
| New Balance | NB, 990, 993, 1906 |
| Puma | PUMA |
| Asics | ASICS, Gel-Kayano |
| Hoka | Hoka One One, Clifton, Bondi |

## Subreddits

`r/Sneakers` · `r/SneakerMarket` · `r/Nike` · `r/Adidas` · `r/Yeezy` · `r/FashionReps` · `r/malefashionadvice` · `r/Running` · `r/Basketball`

## Retail Channels

40+ domains mapped including Nike Direct, StockX, GOAT, Foot Locker, Adidas Direct, Farfetch, SSENSE, JD Sports, Kith, Flight Club, Grailed, and more.

## Purchase Intent Types

| Intent | Example trigger |
|--------|----------------|
| `completed_purchase` | "just copped", "picked up" |
| `seeking_purchase` | "W2C", "where to cop" |
| `purchase_consideration` | "should I cop", "worth it?" |
| `availability_info` | "drops at SNKRS", "restocked" |
| `marketplace` | "WTS", "WTB", "WTT" |
| `selling` | "for sale", "listing" |
| `price_discussion` | "retail $120", "market price" |

---

## Report Output

The HTML report is fully self-contained (embedded Plotly CDN) and includes:

- **Brand Sentiment Rankings** table with avg score, sentiment label, positive/negative %
- **Sentiment Bar Chart** — horizontal bar coloured green/red/grey by label
- **Sentiment Distribution Pie** — overall corpus positive/neutral/negative split
- **Channel Share Pie** — top retail channel breakdown
- **Purchase Intent Funnel** — staged buyer journey signals
- **Sentiment Trend Line** — weekly average over collection period
- **Narrative Themes** table with TF-IDF top terms

A companion Markdown summary is also generated for quick reading.

---

## Development

```bash
make test        # run 101 tests
make lint        # ruff check + format check
make lint-fix    # auto-fix lint issues
make clean       # remove caches and build artifacts
```

### Project Relation to sneaker-intel

This is a sibling package to the [Sneaker Demand Intelligence Platform](../README.md):

| | sneaker-intel | reddit-sentiment |
|---|---|---|
| Data source | StockX historical CSVs (static) | Reddit live API |
| Domain | Tabular price prediction (ML) | NLP sentiment pipeline |
| Output | Aftermarket price forecasts | Brand narrative reports |
| Models | XGBoost / Ridge regression | VADER + Twitter-RoBERTa |

---

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_data_collection_eda.ipynb` | Collection architecture, EDA, URL analysis |
| `02_sentiment_analysis.ipynb` | Pipeline walkthrough, brand detection, intent signals |
| `03_brand_insights.ipynb` | Brand comparison, themes, channel attribution, report generation |

---

## License

MIT — see [Portfolio repo](../README.md).
