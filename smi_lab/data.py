from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT")
API_BASE = "https://data-api.binance.vision"
FUTURES_API_BASE = "https://fapi.binance.com"
INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
DEFAULT_RESEARCH_YEARS = 5
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def bars_for_years(interval: str, years: int = DEFAULT_RESEARCH_YEARS) -> int:
    """Return candle count for a trailing research horizon of one to five years."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    if not 1 <= years <= 5:
        raise ValueError("Research history must be between one and five years.")
    milliseconds = years * 365.25 * 24 * 60 * 60 * 1000
    return int(-(-milliseconds // INTERVAL_MS[interval]))


def _request_json(
    path: str, params: dict[str, object], api_base: str = API_BASE
) -> object:
    url = f"{api_base}{path}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "smi-signal-lab/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Market data request failed for {url}: {exc}") from exc
    if isinstance(payload, dict) and "code" in payload:
        raise RuntimeError(f"Binance returned an API error: {payload}")
    return payload


def _as_frame(rows: list[list[object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    numeric = [c for c in KLINE_COLUMNS if c not in {"open_time", "close_time"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.set_index("open_time").sort_index()
    return frame[["open", "high", "low", "close", "volume", "close_time"]]


def fetch_klines(
    symbol: str,
    interval: str = "4h",
    bars: int = 4500,
    end_time_ms: int | None = None,
    market: str = "spot",
) -> pd.DataFrame:
    """Download closed spot or USD-M perpetual candles from Binance."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    if bars < 50:
        raise ValueError("At least 50 candles are required.")
    if market not in {"spot", "perpetual"}:
        raise ValueError(f"Unsupported market: {market}")

    interval_ms = INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)
    requested_end = end_time_ms or now_ms
    start_ms = requested_end - (bars + 10) * interval_ms
    rows: list[list[object]] = []
    cursor = start_ms

    request_market = market
    while cursor < requested_end and len(rows) < bars + 10:
        try:
            batch = _request_json(
                "/api/v3/klines" if request_market == "spot" else "/fapi/v1/klines",
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": requested_end,
                    "limit": 1000,
                },
                API_BASE if request_market == "spot" else FUTURES_API_BASE,
            )
        except RuntimeError:
            if request_market != "perpetual":
                raise
            # Some regions return HTTP 451 for Binance futures endpoints.
            # Spot candles are still usable for signal generation, while funding is optional.
            request_market = "spot"
            rows.clear()
            cursor = start_ms
            continue
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected kline response for {symbol} {request_market}.")
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break

    frame = _as_frame(rows)
    if frame.empty:
        raise RuntimeError(f"No candles returned for {symbol} {interval}.")
    closed_before = pd.to_datetime(requested_end, unit="ms", utc=True)
    frame = frame[frame["close_time"] < closed_before]
    return frame[~frame.index.duplicated(keep="last")].tail(bars)


def cache_path(
    cache_dir: str | Path, symbol: str, interval: str, market: str = "spot"
) -> Path:
    suffix = "" if market == "spot" else "_perpetual"
    return Path(cache_dir) / f"{symbol.upper()}_{interval}{suffix}.csv"


def funding_cache_path(cache_dir: str | Path, symbol: str) -> Path:
    return Path(cache_dir) / f"{symbol.upper()}_funding.csv"


def fetch_funding_rates(
    symbol: str, start_time_ms: int, end_time_ms: int
) -> pd.DataFrame:
    """Download USD-M perpetual funding charges in ascending time order."""
    rows: list[dict[str, object]] = []
    cursor = start_time_ms
    while cursor <= end_time_ms:
        batch = _request_json(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol.upper(),
                "startTime": cursor,
                "endTime": end_time_ms,
                "limit": 1000,
            },
            FUTURES_API_BASE,
        )
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["funding_rate", "mark_price"])
    frame = pd.DataFrame(rows)
    frame["funding_time"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame["mark_price"] = pd.to_numeric(frame["markPrice"], errors="coerce")
    return (
        frame.set_index("funding_time")[["funding_rate", "mark_price"]]
        .sort_index()
        .loc[lambda item: ~item.index.duplicated(keep="last")]
    )


def get_funding_rates(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> pd.DataFrame:
    target = funding_cache_path(cache_dir, symbol)
    cached = pd.DataFrame()
    if target.exists():
        cached = pd.read_csv(target)
        cached["funding_time"] = pd.to_datetime(
            cached["funding_time"], utc=True, format="mixed"
        )
        cached = cached.set_index("funding_time").sort_index()
        funding_interval = pd.Timedelta(hours=8)
        covers_window = (
            cached.index.min() <= start + funding_interval
            and cached.index.max() >= end - funding_interval
        )
        if not refresh and covers_window:
            return cached.loc[start:end]
    downloaded = fetch_funding_rates(
        symbol, int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    )
    frame = pd.concat([cached, downloaded]).sort_index() if not cached.empty else downloaded
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(target, index=False)
    return frame.loc[start:end]


def attach_funding_rates(frame: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["funding_rate"] = 0.0
    result["funding_mark_price"] = pd.NA
    if not funding.empty:
        if len(result.index) < 2:
            raise ValueError("At least two candles are required to align funding rates.")
        candle_interval = result.index[1] - result.index[0]
        if candle_interval > pd.Timedelta(hours=8):
            raise ValueError("Funding backtests require an interval of eight hours or less.")
        aligned = funding.copy()
        aligned.index = aligned.index.round(candle_interval)
        aligned = aligned.groupby(level=0).agg(
            funding_rate=("funding_rate", "sum"),
            mark_price=("mark_price", "last"),
        )
        matched = aligned["funding_rate"].reindex(result.index).fillna(0.0)
        result["funding_rate"] = matched.astype(float)
        result["funding_mark_price"] = aligned["mark_price"].reindex(result.index)
    return result


def get_klines(
    symbol: str,
    interval: str = "4h",
    bars: int = 4500,
    cache_dir: str | Path = "data",
    refresh: bool = False,
    market: str = "spot",
    include_funding: bool = False,
) -> pd.DataFrame:
    if include_funding and market != "perpetual":
        raise ValueError("Funding rates are only available for perpetual markets.")
    target = cache_path(cache_dir, symbol, interval, market)
    cached = pd.DataFrame()
    if target.exists():
        cached = pd.read_csv(target)
        cached["open_time"] = pd.to_datetime(cached["open_time"], utc=True, format="mixed")
        cached["close_time"] = pd.to_datetime(cached["close_time"], utc=True, format="mixed")
        cached = cached.set_index("open_time").sort_index()
        if not refresh and len(cached) >= bars:
            frame = cached.tail(bars)
            if not include_funding:
                return frame

    if cached.empty or refresh or len(cached) < bars:
        frame = fetch_klines(symbol, interval=interval, bars=bars, market=market)
        if not cached.empty:
            frame = pd.concat([cached, frame]).sort_index()
            frame = frame[~frame.index.duplicated(keep="last")]
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index().to_csv(target, index=False)
        frame = frame.tail(bars)
    if include_funding:
        try:
            funding = get_funding_rates(
                symbol,
                frame.index[0],
                frame["close_time"].iloc[-1],
                cache_dir=cache_dir,
                refresh=refresh,
            )
        except RuntimeError as exc:
            funding = pd.DataFrame(columns=["funding_rate", "mark_price"])
            frame.attrs["funding_warning"] = str(exc)
        frame = attach_funding_rates(frame, funding)
    return frame


def load_universe(
    symbols: tuple[str, ...] | list[str] = DEFAULT_SYMBOLS,
    interval: str = "4h",
    bars: int = 4500,
    cache_dir: str | Path = "data",
    refresh: bool = False,
    market: str = "spot",
    include_funding: bool = False,
) -> dict[str, pd.DataFrame]:
    return {
        symbol.upper(): get_klines(
            symbol,
            interval=interval,
            bars=bars,
            cache_dir=cache_dir,
            refresh=refresh,
            market=market,
            include_funding=include_funding,
        )
        for symbol in symbols
    }


def data_window(frame: pd.DataFrame) -> str:
    start = frame.index[0].to_pydatetime().astimezone(timezone.utc)
    end = frame["close_time"].iloc[-1].to_pydatetime().astimezone(timezone.utc)
    return f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
