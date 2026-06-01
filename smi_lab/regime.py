from __future__ import annotations

from datetime import datetime, time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


CBOE_SERIES_URLS = {
    "SPX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SPX_History.csv",
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
}


def _series_cache_path(cache_dir: str | Path, series: str) -> Path:
    return Path(cache_dir) / f"CBOE_{series}_daily.csv"


def fetch_cboe_series(series: str) -> pd.DataFrame:
    url = CBOE_SERIES_URLS[series]
    request = Request(url, headers={"User-Agent": "smi-signal-lab/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            frame = pd.read_csv(BytesIO(response.read()))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cboe data request failed for {url}: {exc}") from exc
    frame.columns = [column.lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    value_column = "close" if "close" in frame else series.lower()
    frame[series.lower()] = pd.to_numeric(frame[value_column], errors="coerce")
    return frame[["date", series.lower()]].dropna().sort_values("date")


def get_cboe_series(
    series: str, cache_dir: str | Path = "data", refresh: bool = False
) -> pd.DataFrame:
    if series not in CBOE_SERIES_URLS:
        raise ValueError(f"Unsupported Cboe series: {series}")
    target = _series_cache_path(cache_dir, series)
    if target.exists() and not refresh:
        frame = pd.read_csv(target)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
    frame = fetch_cboe_series(series)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return frame


def cboe_regime_history(
    cache_dir: str | Path = "data", refresh: bool = False
) -> pd.DataFrame:
    """Calculate a U.S.-equity stress state available only after each close."""
    spx = get_cboe_series("SPX", cache_dir, refresh)
    vix = get_cboe_series("VIX", cache_dir, refresh)
    result = spx.merge(vix, on="date", how="inner").sort_values("date")
    result["spx_ema100"] = result["spx"].ewm(span=100, adjust=False).mean()
    result["vix_ema20"] = result["vix"].ewm(span=20, adjust=False).mean()
    result["risk_off"] = (result["spx"] < result["spx_ema100"]) & (
        result["vix"] > result["vix_ema20"]
    )
    eastern = ZoneInfo("America/New_York")
    result["available_time"] = result["date"].map(
        lambda value: pd.Timestamp(
            datetime.combine(value.date(), time(16, 15), tzinfo=eastern)
        ).tz_convert("UTC")
    )
    return result[
        ["available_time", "spx", "vix", "spx_ema100", "vix_ema20", "risk_off"]
    ]


def attach_cboe_regime(frame: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    left = frame.sort_index().reset_index()
    index_name = frame.index.name or "open_time"
    if left.columns[0] != index_name:
        left = left.rename(columns={left.columns[0]: index_name})
    left[index_name] = pd.to_datetime(left[index_name], utc=True).astype(
        "datetime64[ms, UTC]"
    )
    aligned_regime = regime.copy()
    aligned_regime["available_time"] = pd.to_datetime(
        aligned_regime["available_time"], utc=True
    ).astype("datetime64[ms, UTC]")
    merged = pd.merge_asof(
        left,
        aligned_regime.sort_values("available_time"),
        left_on=index_name,
        right_on="available_time",
        direction="backward",
    ).set_index(index_name)
    merged["risk_off"] = merged["risk_off"].fillna(False).astype(bool)
    return merged


def attach_btc_momentum_regime(
    universe: dict[str, pd.DataFrame],
    btc_ema_period: int = 100,
    momentum_period: int = 180,
    top_n: int = 1,
) -> dict[str, pd.DataFrame]:
    """Permit risk-on longs only in the strongest assets while BTC trends higher."""
    if "BTCUSDT" not in universe:
        raise ValueError("BTCUSDT is required for the BTC momentum regime.")
    if not 1 <= top_n <= len(universe):
        raise ValueError("top_n must be within the symbol universe.")
    closes = pd.concat(
        {symbol: frame["close"] for symbol, frame in universe.items()}, axis=1
    )
    btc = closes["BTCUSDT"]
    btc_trending_up = btc >= btc.ewm(
        span=btc_ema_period, adjust=False, min_periods=btc_ema_period
    ).mean()
    momentum_rank = closes.pct_change(momentum_period).rank(
        axis=1, ascending=False, method="min"
    )
    result: dict[str, pd.DataFrame] = {}
    for symbol, frame in universe.items():
        long_allowed = (btc_trending_up & (momentum_rank[symbol] <= top_n)).fillna(
            False
        )
        aligned = long_allowed.reindex(frame.index).ffill().fillna(False)
        result[symbol] = frame.assign(risk_off=(~aligned).astype(bool))
    return result
