from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
DEFAULT_TW_SYMBOLS = ("2330.TW", "2317.TW", "2454.TW", "2308.TW", "2603.TW")
DEFAULT_US_SYMBOLS = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL")


def normalize_equity_symbol(symbol: str, market: str) -> str:
    value = symbol.strip().upper()
    if market == "tw" and "." not in value:
        return f"{value}.TW"
    return value


def _cache_file(cache_dir: str | Path, symbol: str, interval: str, range_: str) -> Path:
    safe = symbol.replace("^", "INDEX_").replace(".", "_")
    return Path(cache_dir) / f"{safe}_{interval}_{range_}.csv"


def fetch_yahoo_chart(
    symbol: str,
    interval: str = "1d",
    range_: str = "1y",
    cache_dir: str | Path = "data/equities",
    refresh: bool = False,
) -> pd.DataFrame:
    target = _cache_file(cache_dir, symbol, interval, range_)
    if target.exists() and not refresh:
        cached = pd.read_csv(target)
        cached["open_time"] = pd.to_datetime(cached["open_time"], utc=True, format="mixed")
        cached["close_time"] = pd.to_datetime(cached["close_time"], utc=True, format="mixed")
        return cached.set_index("open_time").sort_index()
    params = urlencode(
        {
            "range": range_,
            "interval": interval,
            "includePrePost": "false",
            "events": "history",
        }
    )
    request = Request(
        f"{YAHOO_CHART_URL}/{symbol}?{params}",
        headers={"User-Agent": "crypto-signal-lab/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)
    result = payload["chart"]["result"]
    if not result:
        raise RuntimeError(f"No Yahoo chart data returned for {symbol}.")
    result = result[0]
    timestamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    if not timestamps:
        raise RuntimeError(f"No Yahoo timestamps returned for {symbol}.")
    frame = pd.DataFrame(
        {
            "open_time": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    ).dropna(subset=["open", "high", "low", "close"])
    frame["close_time"] = frame["open_time"]
    frame = frame.set_index("open_time").sort_index()
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(target, index=False)
    return frame


def load_equity_universe(
    symbols: list[str],
    market: str,
    interval: str = "1d",
    range_: str = "1y",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    return {
        normalize_equity_symbol(symbol, market): fetch_yahoo_chart(
            normalize_equity_symbol(symbol, market),
            interval=interval,
            range_=range_,
            refresh=refresh,
        )
        for symbol in symbols
    }
