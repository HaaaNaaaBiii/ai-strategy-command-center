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

Cathay Securities and Firstrade are tracked manually by design. The app can record account equity, positions, and planned orders, but the actual stock orders remain user-executed.

## Strategy Signal Boundary

The strategy supports cash/no-trade states. `HOLD_CASH` and `HOLD_CASH_OR_EXIT` are valid outputs, not errors.

Entry prices are strategy trigger prices. The app does not use the latest close as a recommended entry. For selected assets, the entry waits for a breakout trigger derived from recent highs, trend level, and ATR. Stop loss and TP levels are then calculated from that trigger price.

## Data Reliability

Crypto data first attempts Binance USD-M perpetual candles. If that endpoint is blocked by region policy such as HTTP 451, the app falls back to Binance spot candles for signal generation and treats funding as unavailable instead of failing the page.

Equity data first uses Yahoo Finance chart data. If Yahoo rate-limits with HTTP 429, the app returns usable cache when available and falls back to Stooq daily data for daily charts when no cache exists.

Dashboard news uses RSS feeds with a local cache. If all feeds are temporarily unavailable, the app keeps the page alive and shows the latest cached items when present.

## Broker Holding Tracking Plan

Firstrade holdings should be tracked with a CSV-first workflow because a stable public API is not assumed. Export positions/account value from Firstrade, normalize columns to `symbol`, `quantity`, `average_price`, and `current_price`, then import or manually enter them into the Accounts page.

Cathay Securities holdings should follow the same statement/CSV import approach. Local Taiwan symbols must be normalized to Yahoo-style symbols such as `2330.TW`. Stock execution remains manual, as requested.

Pionex starts with account and order tracking. Live API execution should only be enabled after canary capital, max order size, daily loss limit, and kill-switch controls are implemented.

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
