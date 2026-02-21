# Reddit Sneaker Sentiment Analysis

A standalone NLP pipeline that collects public Reddit discussions from sneaker communities and extracts **brand sentiment**, **retail channel attribution**, and **purchase intent signals** across major footwear brands.

Part of a two-project portfolio system — see [Integration with sneaker-intel](#integration-with-sneaker-intel).

```
reddit-sentiment collect → analyze → report
```

---

## What This Project Does

Sneaker brands like Nike and Adidas spend heavily on product launches, but consumer reaction is largely measured after the fact through sales data. This project extracts **pre- and post-launch consumer signals** from Reddit — where real buyers discuss what they want, what they bought, and where they bought it.

Three core questions it answers:

1. **Which brands have positive vs. negative narrative momentum?**
   VADER + Twitter-RoBERTa sentiment scoring on brand-mention context windows, aggregated into weekly trends.

2. **Where are consumers being directed to buy?**
   URL domain extraction and keyword matching identify 40+ retail channels (Nike Direct, StockX, GOAT, Foot Locker, etc.) and map how purchase intent flows across them.

3. **What are consumers actually talking about?**
   TF-IDF and keyword theme extraction surface dominant narratives: hype, quality, value, authenticity, performance.

**Data source:** 9 subreddits — `r/Sneakers`, `r/Nike`, `r/Adidas`, `r/SneakerMarket`, `r/Yeezy`, `r/FashionReps`, `r/malefashionadvice`, `r/Running`, `r/Basketball`

---

## Integration with sneaker-intel

This project is one half of a two-part portfolio platform. Together they form an end-to-end **demand signal system**:

```
┌─────────────────────────────────────────────────────────────────┐
│              Sneaker Demand Intelligence Platform               │
├────────────────────────────┬────────────────────────────────────┤
│       sneaker-intel        │       reddit-sentiment             │
├────────────────────────────┼────────────────────────────────────┤
│ StockX historical data     │ Reddit live discussions            │
│ (99K+ transactions)        │ (2,400+ posts, 9 subreddits)       │
│                            │                                    │
│ Tabular ML pipeline        │ NLP sentiment pipeline             │
│ XGBoost / LightGBM / RF    │ VADER + Twitter-RoBERTa            │
│                            │                                    │
│ Output: demand tiers,      │ Output: brand narratives,          │
│ price signals, sell-       │ channel attribution,               │
│ through risk               │ purchase intent funnel             │
└────────────────────────────┴────────────────────────────────────┘
                         │
                         ▼
          Combined signal: "Product X has high aftermarket
          demand (sneaker-intel) AND strong positive Reddit
          narrative with 188 active seekers (reddit-sentiment)
          → recommend: increase production allocation,
            prioritize Nike Direct channel"
```

**Why two projects instead of one?**

- `sneaker-intel` answers: *"How strong was demand historically?"* — backward-looking, quantitative
- `reddit-sentiment` answers: *"What is the current consumer narrative?"* — forward-looking, qualitative

The combination mirrors how footwear demand planners actually work: historical sell-through data calibrates production models, while social listening captures real-time intent shifts before they show up in sales figures.

**Example combined insight from this run:**
- `sneaker-intel`: Nike products cluster in the High demand tier, trading at 1.3–1.8× retail
- `reddit-sentiment`: Nike has 456 mentions, +0.245 avg sentiment, 188 active purchase-seekers, with eBay and GOAT as the dominant secondary market channels
- **Planning implication**: Strong aftermarket signal + active secondary market activity → demand exceeds supply; increase production or tighten channel allocation to Nike Direct

---

## Architecture

```
reddit-sentiment/
├── src/reddit_sentiment/
│   ├── config.py                   # pydantic-settings: API keys + thresholds
│   ├── collection/
│   │   ├── client.py               # PRAW read-only wrapper (requires API credentials)
│   │   ├── public_collector.py     # Requests-based collector (no credentials needed)
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

### Option A — No credentials (public JSON API)

```bash
make install
reddit-sentiment collect --public   # pulls real Reddit data, no API key needed
reddit-sentiment analyze --no-transformer
reddit-sentiment report
```

### Option B — With Reddit API credentials (full pipeline)

```bash
cp .env.example .env
# Edit .env: add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
make install
make pipeline
```

### Install

```bash
# Core + dev dependencies (no GPU needed)
make install

# With transformer model (requires ~2 GB free disk, GPU optional)
make install-ml
```

### CLI Reference

```bash
reddit-sentiment collect   [--output PATH] [--subreddits r1 r2 ...] [--public]
reddit-sentiment analyze   [--input PATH] [--output PATH] [--no-transformer]
reddit-sentiment report    [--input PATH]
reddit-sentiment pipeline  [--no-transformer] [--public]
```

---

## Sample Findings (Feb 2026, 2,452 posts)

### Brand Sentiment Rankings

| Brand | Mentions | Avg Sentiment | Positive% | Negative% |
|-------|----------|--------------|-----------|-----------|
| Nike | 456 | +0.245 | 52.4% | 22.4% |
| Adidas | 260 | +0.293 | 54.6% | 19.6% |
| New Balance | 42 | +0.168 | 47.6% | 23.8% |
| Hoka | 9 | +0.319 | 55.6% | 22.2% |

### Top Retail Channels

eBay · Kith · Undefeated · GOAT · Amazon · Foot Locker · StockX · Nike Direct

### Purchase Intent Funnel

| Stage | Count |
|-------|-------|
| Marketplace activity (WTS/WTB) | 278 |
| Actively seeking purchase | 188 |
| Completed purchase | 91 |
| Considering purchase | 30 |
| Availability / drop info | 16 |

### Dominant Narrative Themes

Aesthetics & Design (27.9%) · Authenticity & Fakes (20.7%) · Quality & Comfort (19.5%) · Brand Loyalty (19.2%) · Value & Pricing (14.6%)

---

## Brands Tracked

| Brand | Key Aliases |
|-------|-------------|
| Nike | Swoosh, Air Max, Air Jordan, Dunk, Jordan |
| Adidas | Three Stripes, Yeezy, Ultraboost, NMD, Samba |
| New Balance | NB, 990, 993, 1906 |
| Hoka | Hoka One One, Clifton, Bondi |
| Under Armour | UA, Curry Brand |
| Puma | PUMA |
| Asics | ASICS, Gel-Kayano |
| Li-Ning | Way of Wade, LN |
| Anta | KT, Klay Thompson |

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

The HTML report is fully self-contained (Plotly CDN) and includes:

- **Brand Sentiment Rankings** table — avg score, sentiment label, positive/negative %
- **Sentiment Bar Chart** — horizontal bar coloured by sentiment label
- **Sentiment Distribution Pie** — corpus-wide positive/neutral/negative split
- **Channel Share Pie** — top retail channel breakdown
- **Purchase Intent Funnel** — staged buyer journey signals
- **Weekly Sentiment Trend** — avg sentiment over collection window
- **Narrative Themes** table with TF-IDF top terms

A companion Markdown summary is also generated.

---

## Development

```bash
make test        # run 101 tests
make lint        # ruff check + format check
make lint-fix    # auto-fix lint issues
make clean       # remove caches and build artifacts
```

---

## License

MIT — see [Portfolio repo](../README.md).
