# Cross-Platform App Guide

This project ships as a Streamlit web app. It is cross-platform because it runs in a browser on Windows, macOS, Linux, iOS, and Android.

## Start Locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

For mobile on the same Wi-Fi network, open the LAN URL printed by Streamlit, for example:

```text
http://192.168.x.x:8501
```

## Pages

- Crypto Strategy: current crypto technical view, allocation snapshot, and notification send-out.
- Forward Paper Tracking: live-readiness gates, paper equity, benchmark comparison, and blockers.
- Taiwan Stocks: TW stock technical summary, market-adjusted ranking, and backtest versus 0050.TW.
- U.S. Stocks: U.S. stock technical summary, market-adjusted ranking, and backtest versus SPY.
- Strategy Architecture: design rules and deployment stages.
- Records: generated CSV/JSON outputs.

## Deployment Note

For internet access from a phone outside the local network, deploy the same app to a small VPS, Streamlit Community Cloud, or another internal web host. Do not expose exchange/API secrets in environment variables on a public machine without access controls.
