# Cross-Platform App Guide

This project ships as a Streamlit web app. It is cross-platform because it runs in a browser on Windows, macOS, Linux, iOS, and Android.

Project entrypoint:

- `PROJECT.md`: consolidated project brief, current strategy state, live boundaries, and next steps.
- `docs/scan_operations.md`: crypto/Taiwan/U.S. scan rules, schedules, commands, outputs, and notification rules.
- `docs/equity_strategy_review.md`: equity strategy review, backtest status, and live execution boundary.
- `docs/attention_strategy.md`: non-financial attention strategy design, data sources, backtest command, and production boundary.

## What This App Provides

- Dashboard: market snapshot, news, crypto live-readiness, and tracked account/position status.
- Crypto: signal center, forward paper tracking, allocation chart, candlestick strategy chart, and notification controls.
- Stocks: Taiwan and U.S. strategy ranking charts, company names, selectable candlestick charts, rotation/rebalance signals, and benchmark comparison.
- Attention: separate research page for non-financial news, keyword, search, and social-attention signals.
- Live Desk: strategy-controlled crypto, U.S. 10,000 USD, and Taiwan 300,000 TWD sleeves that generate live rebalance intents from the current backtested strategy.
- Accounts: Pionex crypto account tracking, Cathay Securities Taiwan stock tracking, Firstrade U.S. stock tracking, position records, and order tracking.
- Research: current optimization stance and generated research files.
- Records: generated CSV/JSON outputs for audit and review.
- Deployment: local URLs and Streamlit Cloud deployment settings.

## Live Trading Boundary

The app currently records account state and order plans. It does not place live orders. Pionex execution should only be enabled after API keys, canary capital limits, max order size, daily loss limit, and kill-switch rules are configured outside Git.

Cathay Securities and Firstrade execution remains manual by design. The app can now sync exported holding CSV files, record account equity, positions, and planned orders, but the actual stock orders remain user-executed.

The Live Desk turns the latest strategy-selected rows into broker-ready rebalance intents for crypto, U.S., and Taiwan sleeves. It records target weight, target value, current value, reference price, delta, and order quantity. It does not show entry, stop, TP, or RR for live execution because those overlays are not part of the current equity backtest.

## Strategy Signal Boundary

The strategy supports cash/no-trade states. `HOLD_CASH` and `HOLD_CASH_OR_EXIT` are valid outputs, not errors.

The current live equity strategy is a dynamic rotation/rebalance model. It does not have a separately backtested entry, take-profit, or stop-loss layer. For live order planning, the reference price is the latest available scan/market price used to estimate order quantity; the real fill price is whatever the broker/exchange executes.

The detailed equity strategy review is in `docs/equity_strategy_review.md`. It separates the backtested selection/rebalance model from the removed execution overlay that used entry, stop loss, TP, and RR.

## Data Reliability

Crypto data first attempts Binance USD-M perpetual candles. If that endpoint is blocked by region policy such as HTTP 451, the loader tries Bybit USDT perpetual candles before falling back to Binance spot candles. Bybit V5 candles are fetched backwards in pages, so one-to-five-year 4h backtests are not limited to the latest 1000 candles. The returned frame records the active source in `data_source`, and spot fallback carries a `data_warning` so the app can distinguish true perpetual data from a degraded source.

Crypto broad scans use CoinGecko `/coins/markets` as the market-cap universe source, requesting up to the top 100 coins by USD market cap. The app filters stablecoins and wrapped duplicates, maps each remaining coin ticker to a centralized `USDT` pair, then keeps only symbols with usable Binance/Bybit market data. Failed symbols are written as scan failures instead of breaking the whole scan.

Equity data first uses Yahoo Finance chart data. If Yahoo rate-limits with HTTP 429, the app returns usable cache when available and falls back to Stooq daily data for daily charts when no cache exists.

Dashboard news uses RSS feeds with a local cache. If all feeds are temporarily unavailable, the app keeps the page alive and shows the latest cached items when present.

Selected stock news is shown for daily Taiwan/U.S. scan picks on the Dashboard and Stocks page. It uses per-symbol RSS caches under `outputs/news/equity_symbols/`; clicking refresh fetches Yahoo Finance and Google News results for each selected symbol. This news layer is informational only and does not alter strategy scores, target weights, or order plans.

## Local Storage Root

Generated market data, scan outputs, account imports, Discord secrets, and research artifacts can be stored outside the Git workspace. The app reads `storage.local.json` first, then `AI_STRATEGY_STORAGE_ROOT` or `SMI_LAB_STORAGE_DIR`, and falls back to the project folder only when no local storage root is configured.

The current local target is:

```text
E:\AI_Strategy_Command_Center
```

Run the migration helper from the project folder:

```powershell
.\.venv\Scripts\python.exe migrate_storage.py --target E:\AI_Strategy_Command_Center
```

This copies local `data/` and `outputs/` into the target and writes a Git-ignored `storage.local.json`. It does not delete the original C-drive folders unless `--move` is passed intentionally. Keep the C-drive copy as a backup until the app and scheduled scans are verified against E-drive storage.

## Investing.com Research Monitor

If Investing.com is subscribed, the practical integration should start from exportable or user-authorized data rather than scraping a logged-in subscription page. Place curated CSV exports under:

```text
E:\AI_Strategy_Command_Center\data\investing\
```

Supported columns are:

```text
market,symbol,company,source,as_of,rating,fair_value,analyst_target,upside_pct,technical_summary,fundamental_summary,risk_summary,notes,url
```

Only `symbol` is required. `market` can be `tw`, `us`, or blank; blank rows are matched to the current scan market when the symbol matches. After each Taiwan or U.S. scan, the app writes selected-symbol research monitors to:

```text
E:\AI_Strategy_Command_Center\outputs\external_research\tw_investing_monitor.csv
E:\AI_Strategy_Command_Center\outputs\external_research\us_investing_monitor.csv
```

These files are exposed in the Records page. The next production step is to automate the source feed if the subscription provides a permitted export, alert email, API, or browser-assisted capture flow.

## Non-Financial Attention Strategy

The `Attention` page is a separate research sleeve for early attention signals. It looks for product, brand, search, and social discussion spikes that may precede earnings narratives. The first implementation uses GDELT DOC 2.0 timeline data as a no-key historical proxy, falls back to Wikimedia Pageviews when GDELT is rate-limited, and excludes finance-related terms from keyword queries.

Run manually:

```powershell
.\.venv\Scripts\python.exe research_attention_strategy.py --range 2y --refresh
```

The current selected research config uses Top 5, 5-trading-day rebalance, 7-day recent attention, 60-day baseline, and minimum spike z-score 1.5. It can hold cash when no symbol passes the thresholds. Outputs are written to `outputs/attention_strategy/` and are shown in the `Attention`, `Research`, and `Records` pages.

Latest 2-year proxy-data backtest generated on 2026-06-03:

- Attention strategy: `+80.57%`.
- SPY aligned benchmark: `+38.32%`.
- Excess return: `+42.25%`.
- Max drawdown: strategy `-18.55%`, SPY `-19.00%`.

This sleeve is research-only until stronger historical YouTube, Reddit, TikTok, Google Trends, or vendor search-volume data is connected and forward paper tracking confirms the signal.

## Broker Holding Tracking Plan

Firstrade holdings use an automated CSV-first workflow because a stable public API is not assumed. Export positions/account value from Firstrade, place the CSV under `data/broker_imports/firstrade/`, then click `Sync broker exports` on the Accounts page. The importer maps common English column names to `symbol`, `quantity`, `average_price`, `current_price`, `market_value`, and `unrealized_pnl`.

Cathay Securities holdings follow the same statement/CSV import approach. Place exported CSV files under `data/broker_imports/cathay/`; local Taiwan symbols such as `2330` are normalized to Yahoo-style symbols such as `2330.TW`. Stock execution remains manual, as requested.

Pionex starts with account and order tracking. Live API execution should only be enabled after canary capital, max order size, daily loss limit, and kill-switch controls are implemented.

The same sync can run from the terminal:

```powershell
.\.venv\Scripts\python.exe sync_holdings.py
```

The importer upserts by `account_id` and `symbol` into `outputs/accounts/positions.csv`, so repeated exports update existing holdings instead of duplicating them.

## Automated Position Planning

The Accounts page includes a `Position Planner` tab for crypto. It reads the latest account snapshot or a temporary portfolio equity override, tracked positions, current strategy target allocation, and latest prices, then calculates target value, current value, delta value, side, and estimated order quantity.

Generated plans can be appended to the order tracker as `PLANNED` rows. The app still does not submit live Pionex orders; this keeps execution reviewable until API keys, canary sizing, daily loss limits, and a kill switch are configured.

## Equity Strategy Scan

The stock page now treats the earlier small Taiwan/U.S. lists as seed watchlists, not final recommendations. `scan_equity_signals.py` scans broader liquid Taiwan and U.S. universes, ranks symbols with the strategy, and writes current recommendations to `outputs/equity_scan/`.

Current default equity scan coverage is intentionally larger than the original seed lists:

- Taiwan: 137 large and mid-large liquid Taiwan-listed symbols, with `0050.TW` kept as the market gate.
- U.S.: 155 liquid U.S.-listed symbols across mega-cap, semiconductors, software, financials, healthcare, industrials, and consumer leaders, with `SPY` kept as the market gate.

This is still not a full exchange-wide scan. It is a practical liquid universe designed to avoid rate-limit failures and thin-liquidity noise while giving the strategy a much larger candidate pool than the prior 54 Taiwan and 63 U.S. symbols.

Run manually:

```powershell
.\.venv\Scripts\python.exe scan_equity_signals.py --market both --interval 1d --range 2y --refresh
```

Run the broader crypto market-cap scan manually:

```powershell
.\.venv\Scripts\python.exe scan_signals.py --strategy allocation --crypto-universe top100 --crypto-limit 100 --interval 4h --bars 1200 --refresh
```

Use `--crypto-universe core` to force the old BTC/ETH/DOGE/SOL universe, or `--crypto-universe symbols --symbols BTCUSDT ETHUSDT SOLUSDT` to scan an explicit custom list.

Active Codex automations:

- Taiwan equity scans: weekdays at 08:45, 11:30, and 14:10 Taiwan time. Premarket and intraday are observation scans; post-close is the official daily strategy update.
- U.S. equity scans: 21:00 Taiwan time on U.S. trading nights, then 01:00 and 05:30 Taiwan time on the following Taiwan calendar day. Premarket and intraday are observation scans; post-close is the official daily strategy update.
- `Crypto 4h allocation scan`: every 4 hours, matching the crypto allocation strategy candle interval.

Stock scans are intentionally limited to premarket, intraday, and post-close checkpoints. Taiwan and U.S. strategies use daily bars, so official strategy recommendations should be based on post-close data. Intraday observations can be sent to Discord for risk awareness, but they should not update Live Desk order intents unless a separate intraday strategy is designed and backtested.

Small live equity usage rule:

- Treat post-close recommendations as the source of truth for new orders.
- Use premarket and intraday scans only to confirm whether the selected sleeve is still stable or whether a risk-off/cash state has appeared.
- Start with a canary sleeve, for example 5-10% of the intended strategy capital.
- Hold the selected Top 3 names at equal target weights unless the strategy configuration changes.
- Rebalance only when the selected symbols or target weights change materially; avoid reacting to every observation scan.

The latest equity optimization selected Top 3 sleeves for both markets. Taiwan uses 40/60-day momentum with 40-day rebalance, EMA200 trend, and 0050.TW as the gate. U.S. uses 63/126-day momentum with 40-day rebalance and SPY as the gate. The latest two-year broad-universe backtests were:

- Taiwan strategy: `+472.23%` versus 0050.TW `+149.73%`, max drawdown `-19.35%`.
- U.S. strategy: `+209.40%` versus SPY `+43.27%`, max drawdown `-20.54%`.

Stock trade plans no longer expose entry, stop, TP, or RR because that execution overlay is not part of the current backtested live strategy. The legacy price-level alert script is disabled by default:

```powershell
.\.venv\Scripts\python.exe check_price_alerts.py
```

It will print a disabled message unless `--legacy-level-alerts` is passed intentionally. Future production alerts should be based on strategy rebalance events, such as new selected symbols, target-weight changes, or holdings dropping out of the selected sleeve.

Discord alerts require:

```powershell
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
$env:DISCORD_MENTION = "<@your_user_id>"
```

The notifier also reads local Git-ignored fallbacks:

- `data/secrets/discord_webhook_url.txt`
- `data/secrets/discord_mention.txt`

The old `Equity price alert check` automation is paused. Future production alerts should be rebalance-event alerts instead of entry/stop/TP price-level alerts.

## Strategy Optimization Status

The current crypto strategy was re-tested against a broader staggered trend allocation grid in `research_crypto_optimization.py`. Selection uses calibration and validation under triple trading costs, while holdout remains report-only to reduce overfitting.

Latest optimization output is written to `outputs/crypto_optimization/`. The strongest candidate increased full-period return but carried materially worse drawdown and weaker holdout return than the current paper strategy, so the script keeps the current production candidate and records `recommendation: keep_current`.

## Start Locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To restart the local website in the background on Windows, use the launcher:

```powershell
.\.venv\Scripts\python.exe start_streamlit.py
```

Open:

```text
http://localhost:8501
```

The current local deployment command uses port `8501` and binds to `0.0.0.0`, as configured in `.streamlit/config.toml`.

For mobile on the same Wi-Fi network, open the LAN URL printed by Streamlit, for example:

```text
http://192.168.x.x:8501
```

For this machine, the current LAN URL is:

```text
http://192.168.1.108:8501
```

This LAN URL only works when the phone and computer are on the same local network, and the firewall allows inbound traffic on port `8501`.

## Streamlit Community Cloud Deployment

Streamlit Community Cloud deploys from a GitHub repository. The required settings for this project are:

- Repository: `https://github.com/HaaaNaaaBiii/ai-strategy-command-center`.
- Branch: `master`.
- Main file path: `app.py`.
- Python version: `3.12`.
- Dependencies: `requirements.txt`.

Prefilled deployment URL:

```text
https://share.streamlit.io/deploy?owner=HaaaNaaaBiii&repo=ai-strategy-command-center&branch=master&mainModule=app.py
```

Deployment flow:

1. Push this repository to GitHub.
2. Open `https://share.streamlit.io`.
3. Create a new app from an existing GitHub repository.
4. Select the repo, branch, and `app.py`.
5. Deploy.

The app does not require API keys for public market data. If Discord or Telegram notifications are used in a hosted deployment, configure those webhook/token values as Streamlit secrets instead of committing them to Git.

## Deployment Note

For internet access from a phone outside the local network, deploy the same app to a small VPS, Streamlit Community Cloud, or another internal web host. Do not expose exchange/API secrets in environment variables on a public machine without access controls.
