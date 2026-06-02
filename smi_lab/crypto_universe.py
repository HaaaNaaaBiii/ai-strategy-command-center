from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import DEFAULT_SYMBOLS, get_klines


COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
DEFAULT_CRYPTO_MARKET_CAP_LIMIT = 100
DEFAULT_CRYPTO_UNIVERSE_CACHE = Path("data/crypto_universe/coingecko_top_markets.json")

STABLE_OR_WRAPPED_IDS = {
    "binance-bridged-usdt-bnb-smart-chain",
    "binance-usd",
    "beldex",
    "bridged-usdc-polygon-pos-bridge",
    "blackrock-usd-institutional-digital-liquidity-fund",
    "blockchain-capital",
    "bitget-token",
    "coinbase-wrapped-btc",
    "dai",
    "ethena-usde",
    "eutbl",
    "falcon-finance",
    "first-digital-usd",
    "frax",
    "global-dollar",
    "gatechain-token",
    "hashnote-usyc",
    "htx-dao",
    "kucoin-shares",
    "leo-token",
    "liquity-usd",
    "janus-henderson-anemoy-treasury-fund",
    "ondo-us-dollar-yield",
    "paypal-usd",
    "pi-network",
    "rain",
    "rocket-pool-eth",
    "staked-ether",
    "superstate-short-duration-us-government-securities-fund-ustb",
    "susde",
    "true-usd",
    "usual-usd",
    "tether",
    "usd-coin",
    "usdd",
    "usds",
    "usdtb",
    "apxusd",
    "whitebit",
    "wrapped-bitcoin",
    "wrapped-eeth",
    "wrapped-steth",
    "wrapped-usdt",
    "weth",
}
STABLE_OR_WRAPPED_SYMBOLS = {
    "BUSD",
    "BCAP",
    "BDX",
    "BGB",
    "BUIDL",
    "CBBTC",
    "DAI",
    "EUTBL",
    "FDUSD",
    "FRAX",
    "GHO",
    "GT",
    "HTX",
    "JTRSY",
    "KCS",
    "LEO",
    "LUSD",
    "PI",
    "PYUSD",
    "RAIN",
    "RETH",
    "SUSDE",
    "TUSD",
    "APXUSD",
    "USDF",
    "USDG",
    "USD0",
    "USDY",
    "USDC",
    "USDD",
    "USDE",
    "USDS",
    "USDT",
    "USDTB",
    "USTB",
    "USYC",
    "WBT",
    "WBTC",
    "WEETH",
    "WETH",
    "WSTETH",
}


def _request_markets_page(page: int, per_page: int, vs_currency: str) -> list[dict[str, object]]:
    params = urlencode(
        {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d,30d",
        }
    )
    request = Request(
        f"{COINGECKO_MARKETS_URL}?{params}",
        headers={
            "User-Agent": "smi-signal-lab/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"CoinGecko market-cap request failed: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected CoinGecko market-cap response: {payload}")
    return [item for item in payload if isinstance(item, dict)]


def _fetch_coingecko_markets(
    limit: int = DEFAULT_CRYPTO_MARKET_CAP_LIMIT,
    vs_currency: str = "usd",
) -> list[dict[str, object]]:
    if limit < 1:
        raise ValueError("Crypto market-cap limit must be positive.")
    rows: list[dict[str, object]] = []
    per_page = min(250, max(1, limit))
    page = 1
    while len(rows) < limit:
        batch = _request_markets_page(page, per_page, vs_currency)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        time.sleep(0.35)
    return rows[:limit]


def _read_cached_markets(cache_path: str | Path = DEFAULT_CRYPTO_UNIVERSE_CACHE) -> list[dict[str, object]]:
    target = Path(cache_path)
    if not target.exists():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = payload.get("markets", [])
    return [item for item in rows if isinstance(item, dict)]


def _write_cached_markets(
    markets: list[dict[str, object]],
    cache_path: str | Path = DEFAULT_CRYPTO_UNIVERSE_CACHE,
) -> None:
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source": "coingecko_coins_markets",
                "markets": markets,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def coingecko_top_markets(
    limit: int = DEFAULT_CRYPTO_MARKET_CAP_LIMIT,
    refresh: bool = False,
    cache_path: str | Path = DEFAULT_CRYPTO_UNIVERSE_CACHE,
    cache_only: bool = False,
) -> list[dict[str, object]]:
    cached = _read_cached_markets(cache_path)
    if cache_only:
        return cached[:limit]
    if cached and not refresh and len(cached) >= min(limit, DEFAULT_CRYPTO_MARKET_CAP_LIMIT):
        return cached[:limit]
    try:
        markets = _fetch_coingecko_markets(limit=limit)
    except RuntimeError:
        return cached[:limit]
    if markets:
        _write_cached_markets(markets, cache_path)
    return markets[:limit] if markets else cached[:limit]


def _is_strategy_candidate(item: dict[str, object], include_stable: bool) -> bool:
    if include_stable:
        return True
    coin_id = str(item.get("id", "")).lower()
    symbol = str(item.get("symbol", "")).upper()
    if coin_id in STABLE_OR_WRAPPED_IDS:
        return False
    return symbol not in STABLE_OR_WRAPPED_SYMBOLS


def crypto_scan_symbols(
    limit: int = DEFAULT_CRYPTO_MARKET_CAP_LIMIT,
    quote: str = "USDT",
    refresh: bool = False,
    cache_path: str | Path = DEFAULT_CRYPTO_UNIVERSE_CACHE,
    cache_only: bool = False,
    include_stable: bool = False,
) -> tuple[str, ...]:
    """Return a market-cap-ranked crypto scan universe mapped to centralized USDT pairs."""
    quote = quote.upper()
    markets = coingecko_top_markets(
        limit=limit,
        refresh=refresh,
        cache_path=cache_path,
        cache_only=cache_only,
    )
    symbols: list[str] = []
    for item in markets:
        if not _is_strategy_candidate(item, include_stable=include_stable):
            continue
        base = str(item.get("symbol", "")).strip().upper()
        if not base or not base.replace("-", "").isalnum():
            continue
        symbols.append(f"{base}{quote}")
    resolved = list(dict.fromkeys([*DEFAULT_SYMBOLS, *symbols]))
    return tuple(resolved[:limit] if len(resolved) > limit else resolved) or DEFAULT_SYMBOLS


def load_crypto_scan_universe(
    symbols: list[str] | tuple[str, ...] | None = None,
    *,
    limit: int = DEFAULT_CRYPTO_MARKET_CAP_LIMIT,
    interval: str = "4h",
    bars: int = 500,
    refresh: bool = False,
    cache_dir: str | Path = "data",
    market: str = "perpetual",
    include_funding: bool = False,
    min_bars: int = 0,
    request_pause: float = 0.08,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    source_symbols = list(symbols or crypto_scan_symbols(limit=limit, refresh=refresh))
    source_symbols = list(dict.fromkeys([*DEFAULT_SYMBOLS, *[symbol.upper() for symbol in source_symbols]]))
    universe: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, object]] = []
    for index, symbol in enumerate(source_symbols):
        try:
            frame = get_klines(
                symbol,
                interval=interval,
                bars=bars,
                cache_dir=cache_dir,
                refresh=refresh,
                market=market,
                include_funding=include_funding,
            )
            if frame.empty:
                raise RuntimeError("empty chart")
            if min_bars and len(frame) < min_bars:
                raise RuntimeError(f"only {len(frame)} bars, need at least {min_bars}")
            universe[symbol] = frame
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
        if index < len(source_symbols) - 1:
            time.sleep(request_pause)
    return universe, pd.DataFrame(failures)
