from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_CHART_FALLBACK_URL = "https://query2.finance.yahoo.com/v8/finance/chart"
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
DEFAULT_TW_SYMBOLS = ("2330.TW", "2317.TW", "2454.TW", "2308.TW", "2603.TW")
DEFAULT_US_SYMBOLS = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL")
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def normalize_equity_symbol(symbol: str, market: str) -> str:
    value = symbol.strip().upper()
    if market == "tw" and "." not in value:
        return f"{value}.TW"
    return value


def _cache_file(cache_dir: str | Path, symbol: str, interval: str, range_: str) -> Path:
    safe = symbol.replace("^", "INDEX_").replace(".", "_")
    return Path(cache_dir) / f"{safe}_{interval}_{range_}.csv"


def _read_cached_chart(target: Path) -> pd.DataFrame:
    cached = pd.read_csv(target)
    cached["open_time"] = pd.to_datetime(cached["open_time"], utc=True, format="mixed")
    cached["close_time"] = pd.to_datetime(cached["close_time"], utc=True, format="mixed")
    return cached.set_index("open_time").sort_index()


def _write_chart_cache(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(target, index=False)


def _range_start(range_: str) -> pd.Timestamp | None:
    now = datetime.now(timezone.utc)
    if range_.endswith("mo"):
        return pd.Timestamp(now - timedelta(days=31 * int(range_[:-2])))
    if range_.endswith("y"):
        return pd.Timestamp(now - timedelta(days=366 * int(range_[:-1])))
    return None


def _stooq_symbol(symbol: str) -> str:
    value = symbol.strip().lower()
    if value.endswith(".tw"):
        return value
    if "." not in value:
        return f"{value}.us"
    return value


def _fetch_stooq_daily(symbol: str, range_: str) -> pd.DataFrame:
    params = urlencode({"s": _stooq_symbol(symbol), "i": "d"})
    request = Request(
        f"{STOOQ_DAILY_URL}?{params}",
        headers={"User-Agent": YAHOO_HEADERS["User-Agent"]},
    )
    try:
        with urlopen(request, timeout=20) as response:
            frame = pd.read_csv(response)
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"Stooq daily response is not parseable for {symbol}: {exc}") from exc
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Stooq daily request failed for {symbol}: {exc}") from exc
    if frame.empty or "Date" not in frame:
        raise RuntimeError(f"No Stooq daily data returned for {symbol}.")
    frame = frame.rename(
        columns={
            "Date": "open_time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["open_time", "open", "high", "low", "close"])
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["close_time"] = frame["open_time"]
    frame = frame.set_index("open_time").sort_index()
    start = _range_start(range_)
    if start is not None:
        frame = frame.loc[frame.index >= start]
    if frame.empty:
        raise RuntimeError(f"Stooq data is empty after range filter for {symbol}.")
    return frame[["open", "high", "low", "close", "volume", "close_time"]]


def _fetch_yahoo_payload(symbol: str, interval: str, range_: str) -> object:
    params = urlencode(
        {
            "range": range_,
            "interval": interval,
            "includePrePost": "false",
            "events": "history",
        }
    )
    last_error: Exception | None = None
    for base_url in (YAHOO_CHART_URL, YAHOO_CHART_FALLBACK_URL):
        request = Request(f"{base_url}/{symbol}?{params}", headers=YAHOO_HEADERS)
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code == 429:
                time.sleep(1.5)
            continue
    raise RuntimeError(f"Yahoo chart request failed for {symbol}: {last_error}") from last_error


def fetch_yahoo_chart(
    symbol: str,
    interval: str = "1d",
    range_: str = "1y",
    cache_dir: str | Path = "data/equities",
    refresh: bool = False,
) -> pd.DataFrame:
    target = _cache_file(cache_dir, symbol, interval, range_)
    cached = pd.DataFrame()
    if target.exists() and not refresh:
        return _read_cached_chart(target)
    if target.exists():
        cached = _read_cached_chart(target)
    try:
        payload = _fetch_yahoo_payload(symbol, interval, range_)
    except RuntimeError:
        if not cached.empty:
            return cached
        if interval == "1d":
            frame = _fetch_stooq_daily(symbol, range_)
            _write_chart_cache(frame, target)
            return frame
        raise
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
    _write_chart_cache(frame, target)
    return frame


def load_equity_universe(
    symbols: list[str],
    market: str,
    interval: str = "1d",
    range_: str = "1y",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    universe: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(symbols):
        normalized = normalize_equity_symbol(symbol, market)
        universe[normalized] = fetch_yahoo_chart(
            normalized,
            interval=interval,
            range_=range_,
            refresh=refresh,
        )
        if index < len(symbols) - 1:
            time.sleep(0.35)
    return universe
