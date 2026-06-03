# Non-Financial Attention Strategy

This strategy is a separate research sleeve. It tries to find listed companies whose products or brands are becoming unusually visible before the financial press has already turned the story into an earnings narrative.

## Objective

The target signal is not "good news about the stock." The target signal is rising public attention around products, services, brands, or consumer behavior that may later appear in revenue, user growth, same-store sales, subscriber growth, or guidance.

Examples:

- Beauty: makeup hauls, new product launches, viral cosmetics.
- Apparel: shoes, athleisure, branded bags, seasonal collaborations.
- Restaurants and beverages: menu items, energy drinks, store traffic themes.
- Media and digital: shows, playlists, games, creator and user discussion.

## Current Data Sources

The first version uses:

- GDELT DOC 2.0 timeline data as a no-key historical proxy for broad web/news attention.
- Wikimedia Pageviews as a no-key fallback proxy for public search/interest when GDELT is rate-limited.
- Finance terms are excluded from keyword queries, including stock, earnings, revenue, guidance, analyst, price target, NYSE, and NASDAQ.
- Optional CSV imports under `data/attention_sources/` for future YouTube, Reddit, TikTok, Google Trends, or paid vendor data.

CSV import schema:

```text
date,symbol,source,mentions,engagement
2026-05-01,ELF,youtube,128,42000
```

`date` and `symbol` are required. `source`, `mentions`, and `engagement` are optional but recommended.

## Signal Logic

For each candidate symbol:

1. Build a non-financial keyword query from product and brand terms.
2. Aggregate daily mention volume.
3. Compute a 7-day recent mention sum.
4. Compare the recent sum against the prior 60-day baseline.
5. Score the symbol with spike z-score plus capped percentage growth.
6. Every 5 trading days, select the TopN eligible names from the selected research configuration.
7. If no symbol passes attention thresholds, stay in cash.

The backtest uses prior-day attention features and enters at the next market open with trading cost assumptions. That avoids using same-day information that would not have been known at the open.

## Universe

The initial universe is intentionally focused on U.S. listed companies where non-financial attention can plausibly lead financial results:

- Beauty: `ELF`, `ULTA`, `EL`, `COTY`
- Apparel: `LULU`, `DECK`, `CROX`, `NKE`
- Consumer: `SBUX`, `CMG`, `CAVA`, `CELH`
- Media: `NFLX`, `DIS`
- Digital: `RBLX`, `SPOT`

This is not a full-market scan yet. A broader version should add sector-specific keyword templates and liquidity filters before expanding the universe.

## Backtest Command

```powershell
.\.venv\Scripts\python.exe research_attention_strategy.py --range 2y --refresh
```

Outputs are written to `outputs/attention_strategy/`:

- `attention_report.json`
- `attention_metrics.csv`
- `attention_config_search.csv`
- `latest_attention_candidates.csv`
- `attention_rebalances.csv`
- `attention_equity.csv`
- `attention_timeline.csv`
- `attention_features.csv`
- `attention_failures.csv`

## Latest Research Result

Latest 2-year run generated on 2026-06-03:

- Selected config: Top 5, 5-trading-day rebalance, 7-day recent attention, 60-day baseline, minimum spike z-score 1.5.
- Usable symbols: 16 of 16.
- Attention strategy return: `+80.57%`.
- SPY return over the same aligned window: `+38.32%`.
- Excess return: `+42.25%`.
- Max drawdown: strategy `-18.55%`, SPY `-19.00%`.
- Sharpe: strategy `1.14`, SPY `1.16`.

This passes the first benchmark test on the current proxy data, but it remains research-only because source quality is not yet strong enough for live execution.

## Production Boundary

This strategy is research-only for now.

It needs stronger historical social/search data before live sizing:

- YouTube search and channel/topic trend history.
- Reddit subreddit and keyword trend history.
- TikTok or short-video trend proxies.
- Google Trends or vendor search-volume history.
- Forward paper tracking to confirm the signal still works after implementation.

The current GDELT/Wikimedia version is useful for architecture and an initial historical test, but it should not be treated as complete evidence for live trading.
