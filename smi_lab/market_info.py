from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import cache_path
from .equity_data import _cache_file
from .equity_signals import company_name


NEWS_FEEDS = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?"
    + urlencode(
        {
            "s": "BTC-USD,ETH-USD,AAPL,NVDA,TSM,SPY",
            "region": "US",
            "lang": "en-US",
        }
    ),
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
)
NEWS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class MarketSnapshot:
    market: str
    symbol: str
    name: str
    close: float
    change_pct: float
    as_of: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    link: str
    published_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_cached_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "open_time" not in frame:
        return pd.DataFrame()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    if "close_time" in frame:
        frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, format="mixed")
    return frame.set_index("open_time").sort_index()


def _snapshot_from_frame(
    market: str,
    symbol: str,
    name: str,
    frame: pd.DataFrame,
    lookback_rows: int,
) -> MarketSnapshot | None:
    if frame.empty or "close" not in frame:
        return None
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        return None
    prior_index = max(0, len(close) - 1 - lookback_rows)
    prior = float(close.iloc[prior_index])
    latest = float(close.iloc[-1])
    change_pct = 0.0 if prior == 0 else (latest / prior - 1.0) * 100.0
    as_of = frame.index[-1].isoformat()
    return MarketSnapshot(market, symbol, name, latest, change_pct, as_of)


def cached_crypto_snapshots(
    symbols: list[str] | tuple[str, ...],
    interval: str = "4h",
    cache_dir: str | Path = "data",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        target = cache_path(cache_dir, symbol, interval, "perpetual")
        frame = _read_cached_frame(target)
        if frame.empty:
            frame = _read_cached_frame(cache_path(cache_dir, symbol, interval, "spot"))
        snapshot = _snapshot_from_frame("Crypto", symbol.upper(), symbol.upper(), frame, 6)
        if snapshot is not None:
            rows.append(snapshot.to_dict())
    return pd.DataFrame(rows)


def cached_equity_snapshots(
    symbols: list[str] | tuple[str, ...],
    market: str,
    interval: str = "1d",
    range_: str = "2y",
    cache_dir: str | Path = "data/equities",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    label = "Taiwan" if market == "tw" else "U.S."
    for symbol in symbols:
        target = _cache_file(cache_dir, symbol, interval, range_)
        frame = _read_cached_frame(target)
        snapshot = _snapshot_from_frame(label, symbol.upper(), company_name(symbol), frame, 1)
        if snapshot is not None:
            rows.append(snapshot.to_dict())
    return pd.DataFrame(rows)


def _parse_rss(feed_url: str, payload: bytes, max_items: int) -> list[NewsItem]:
    root = ET.fromstring(payload)
    channel_title = root.findtext("./channel/title") or feed_url
    items: list[NewsItem] = []
    for item in root.findall("./channel/item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        if published:
            try:
                published = parsedate_to_datetime(published).astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError, IndexError):
                pass
        if title and link:
            items.append(NewsItem(title, channel_title, link, published))
    return items


def _load_news_cache(path: Path) -> list[NewsItem]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [NewsItem(**item) for item in payload.get("items", [])]


def fetch_market_news(
    cache_path_: str | Path = "outputs/news/market_news.json",
    refresh: bool = False,
    max_items: int = 8,
) -> list[NewsItem]:
    target = Path(cache_path_)
    if target.exists() and not refresh:
        cached = _load_news_cache(target)
        if cached:
            return cached[:max_items]
    items: list[NewsItem] = []
    for feed_url in NEWS_FEEDS:
        try:
            request = Request(feed_url, headers=NEWS_HEADERS)
            with urlopen(request, timeout=12) as response:
                items.extend(_parse_rss(feed_url, response.read(), max_items))
        except Exception:
            continue
        if len(items) >= max_items:
            break
    if not items:
        return _load_news_cache(target)[:max_items]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "items": [item.to_dict() for item in items[:max_items]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return items[:max_items]
