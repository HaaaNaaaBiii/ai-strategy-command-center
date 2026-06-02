# Cross-Platform App Guide

This project ships as a Streamlit web app. It is cross-platform because it runs in a browser on Windows, macOS, Linux, iOS, and Android.

## What This App Provides

- Dashboard: market snapshot, news, crypto live-readiness, and tracked account/position status.
- Crypto: signal center, forward paper tracking, allocation donut chart, candlestick chart levels, and notification controls.
- Stocks: Taiwan and U.S. strategy ranking charts, company names, selectable candlestick charts, entry/exit/SL/TP levels, and benchmark comparison.
- Accounts: Pionex crypto account tracking, Cathay Securities Taiwan stock tracking, Firstrade U.S. stock tracking, position records, and order tracking.
- Research: current optimization stance and generated research files.
- Records: generated CSV/JSON outputs for audit and review.
- Deployment: local URLs and Streamlit Cloud deployment settings.

## Live Trading Boundary

The app currently records account state and order plans. It does not place live orders. Pionex execution should only be enabled after API keys, canary capital limits, max order size, daily loss limit, and kill-switch rules are configured outside Git.

Cathay Securities and Firstrade execution remains manual by design. The app can now sync exported holding CSV files, record account equity, positions, and planned orders, but the actual stock orders remain user-executed.

## Strategy Signal Boundary

The strategy supports cash/no-trade states. `HOLD_CASH` and `HOLD_CASH_OR_EXIT` are valid outputs, not errors.

Entry prices are strategy trigger prices. The app does not use the latest close as a recommended entry. For selected assets, the entry waits for a breakout trigger derived from recent highs, trend level, and ATR. Stop loss and TP levels are then calculated from that trigger price.

## Data Reliability

Crypto data first attempts Binance USD-M perpetual candles. If that endpoint is blocked by region policy such as HTTP 451, the loader tries Bybit USDT perpetual candles before falling back to Binance spot candles. Bybit V5 candles are fetched backwards in pages, so one-to-five-year 4h backtests are not limited to the latest 1000 candles. The returned frame records the active source in `data_source`, and spot fallback carries a `data_warning` so the app can distinguish true perpetual data from a degraded source.

Equity data first uses Yahoo Finance chart data. If Yahoo rate-limits with HTTP 429, the app returns usable cache when available and falls back to Stooq daily data for daily charts when no cache exists.

Dashboard news uses RSS feeds with a local cache. If all feeds are temporarily unavailable, the app keeps the page alive and shows the latest cached items when present.

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

Run manually:

```powershell
.\.venv\Scripts\python.exe scan_equity_signals.py --market both --interval 1d --range 2y --refresh
```

Active Codex automations:

- `Taiwan daily equity scan`: every weekday at 09:00 Taiwan time.
- `U.S. daily equity scan`: every weekday at 21:00 Taiwan time.

The latest equity optimization selected Top 3 sleeves for both markets. Taiwan uses 20/126-day momentum with 20-day rebalance and 0050.TW as the gate. U.S. uses 63/126-day momentum with 40-day rebalance and SPY as the gate. The latest two-year broad-universe backtests were:

- Taiwan strategy: `+492.37%` versus 0050.TW `+148.55%`, max drawdown `-19.55%`.
- U.S. strategy: `+207.82%` versus SPY `+43.27%`, max drawdown `-18.41%`.

Stock trade plans include RR to TP1 and TP2. Price alerts can be checked manually:

```powershell
.\.venv\Scripts\python.exe check_price_alerts.py
```

Discord alerts require:

```powershell
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
$env:DISCORD_MENTION = "<@your_user_id>"
```

The `Equity price alert check` automation is created but paused until Discord is configured.

## Strategy Optimization Status

The current crypto strategy was re-tested against a broader staggered trend allocation grid in `research_crypto_optimization.py`. Selection uses calibration and validation under triple trading costs, while holdout remains report-only to reduce overfitting.

Latest optimization output is written to `outputs/crypto_optimization/`. The strongest candidate increased full-period return but carried materially worse drawdown and weaker holdout return than the current paper strategy, so the script keeps the current production candidate and records `recommendation: keep_current`.

## Start Locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
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
