from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .backtest import _metrics
from .equity_data import fetch_yahoo_chart


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
WIKIMEDIA_PAGEVIEWS_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
DEFAULT_OUTPUT_DIR = Path("outputs/attention_strategy")
DEFAULT_CACHE_DIR = Path("data/attention_strategy")

FINANCIAL_NOISE_TERMS = (
    "stock",
    "stocks",
    "share",
    "shares",
    "earnings",
    "revenue",
    "profit",
    "analyst",
    "price target",
    "quarter",
    "guidance",
    "investor",
    "investors",
    "NYSE",
    "NASDAQ",
)


@dataclass(frozen=True)
class AttentionCandidate:
    symbol: str
    company: str
    keywords: tuple[str, ...]
    category: str
    wiki_titles: tuple[str, ...] = ()

    def query(self) -> str:
        terms = " OR ".join(f'"{keyword}"' for keyword in self.keywords)
        exclusions = " ".join(f'-"{term}"' if " " in term else f"-{term}" for term in FINANCIAL_NOISE_TERMS)
        return f"({terms}) {exclusions}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["keywords"] = list(self.keywords)
        return payload


@dataclass(frozen=True)
class AttentionConfig:
    top_n: int = 3
    rebalance_days: int = 5
    spike_lookback_days: int = 60
    recent_days: int = 7
    min_recent_mentions: float = 3.0
    min_spike_z: float = 0.5
    max_selected_spike_z: float = 8.0
    fee_bps: float = 1.0
    slippage_bps: float = 3.0

    def validate(self) -> "AttentionConfig":
        if min(self.top_n, self.rebalance_days, self.spike_lookback_days, self.recent_days) < 1:
            raise ValueError("Attention strategy periods must be positive.")
        if self.recent_days >= self.spike_lookback_days:
            raise ValueError("recent_days must be shorter than spike_lookback_days.")
        if min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("Trading costs must not be negative.")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AttentionBacktestResult:
    equity: pd.Series
    events: pd.DataFrame
    ranking: pd.DataFrame
    metrics: dict[str, float]


def default_attention_candidates() -> tuple[AttentionCandidate, ...]:
    return (
        AttentionCandidate("ELF", "e.l.f. Beauty", ("e.l.f. cosmetics", "elf halo glow", "elf beauty"), "beauty", ("E.l.f. Beauty",)),
        AttentionCandidate("ULTA", "Ulta Beauty", ("Ulta Beauty", "ulta haul", "ulta sale"), "beauty", ("Ulta Beauty",)),
        AttentionCandidate("EL", "Estee Lauder", ("Estee Lauder", "La Mer skincare", "Clinique skincare"), "beauty", ("The Estee Lauder Companies",)),
        AttentionCandidate("COTY", "Coty", ("CoverGirl makeup", "Rimmel London", "Coty fragrance"), "beauty", ("Coty", "CoverGirl")),
        AttentionCandidate("LULU", "Lululemon", ("lululemon", "lululemon leggings", "lululemon belt bag"), "apparel", ("Lululemon Athletica",)),
        AttentionCandidate("DECK", "Deckers Outdoor", ("Hoka shoes", "UGG boots", "Hoka running"), "apparel", ("Hoka One One", "UGG (brand)")),
        AttentionCandidate("CROX", "Crocs", ("Crocs", "crocs clogs", "crocs collaboration"), "apparel", ("Crocs",)),
        AttentionCandidate("NKE", "Nike", ("Nike sneakers", "Nike running", "Air Jordan"), "apparel", ("Nike, Inc.", "Air Jordan")),
        AttentionCandidate("SBUX", "Starbucks", ("Starbucks drink", "Starbucks menu", "Starbucks seasonal"), "consumer", ("Starbucks",)),
        AttentionCandidate("CMG", "Chipotle", ("Chipotle menu", "Chipotle bowl", "Chipotle hack"), "consumer", ("Chipotle Mexican Grill",)),
        AttentionCandidate("CAVA", "Cava Group", ("Cava restaurant", "Cava bowl", "Cava menu"), "consumer", ("Cava Group",)),
        AttentionCandidate("CELH", "Celsius Holdings", ("Celsius drink", "Celsius energy", "Celsius beverage"), "consumer", ("Celsius Holdings",)),
        AttentionCandidate("NFLX", "Netflix", ("Netflix series", "Netflix show", "Netflix documentary"), "media", ("Netflix",)),
        AttentionCandidate("DIS", "Disney", ("Disney movie", "Disney park", "Disney plus"), "media", ("The Walt Disney Company", "Disney+")),
        AttentionCandidate("RBLX", "Roblox", ("Roblox", "Roblox game", "Roblox avatar"), "digital", ("Roblox",)),
        AttentionCandidate("SPOT", "Spotify", ("Spotify playlist", "Spotify wrapped", "Spotify podcast"), "digital", ("Spotify",)),
    )


def _date_range_for(range_: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    now = datetime.now(timezone.utc)
    if range_.endswith("y"):
        start = now - timedelta(days=366 * int(range_[:-1]))
    elif range_.endswith("mo"):
        start = now - timedelta(days=31 * int(range_[:-2]))
    else:
        raise ValueError("Attention range must end with y or mo.")
    return pd.Timestamp(start).floor("D"), pd.Timestamp(now).floor("D")


def _cache_file(cache_dir: str | Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return Path(cache_dir) / f"{symbol.upper()}_{start:%Y%m%d}_{end:%Y%m%d}_gdelt.json"


def _wikimedia_cache_file(cache_dir: str | Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return Path(cache_dir) / f"{symbol.upper()}_{start:%Y%m%d}_{end:%Y%m%d}_wikimedia.json"


def _parse_gdelt_date(value: object) -> pd.Timestamp:
    text = str(value)
    if text.isdigit() and len(text) == 14:
        return pd.to_datetime(text, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d", utc=True, errors="coerce")
    return pd.to_datetime(text, utc=True, errors="coerce")


def _extract_timeline_rows(item: dict[str, object]) -> list[dict[str, object]]:
    raw_rows = item.get("data")
    if isinstance(raw_rows, list):
        series = str(item.get("series", "")).lower()
        if series and not any(token in series for token in ("volume", "raw", "count", "article")):
            return []
        return [row for row in raw_rows if isinstance(row, dict)]
    return [item]


def _parse_gdelt_timeline(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected GDELT response: {payload}")
    raw = payload.get("timeline") or payload.get("Timeline") or []
    rows: list[dict[str, object]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for row in _extract_timeline_rows(item):
                date_value = row.get("date") or row.get("datetime")
                if not date_value:
                    continue
                rows.append(
                    {
                        "date": _parse_gdelt_date(date_value),
                        "mentions": pd.to_numeric(
                            row.get("value")
                            or row.get("count")
                            or row.get("Volume Intensity")
                            or row.get("mentions")
                            or 0,
                            errors="coerce",
                        ),
                        "norm": pd.to_numeric(
                            row.get("norm") or row.get("All Articles") or np.nan,
                            errors="coerce",
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["mentions", "norm"])
    frame = frame.dropna(subset=["date"])
    frame["date"] = frame["date"].dt.floor("D")
    frame["mentions"] = pd.to_numeric(frame["mentions"], errors="coerce").fillna(0.0)
    frame["norm"] = pd.to_numeric(frame["norm"], errors="coerce")
    return frame.groupby("date", as_index=True).agg(mentions=("mentions", "sum"), norm=("norm", "sum")).sort_index()


def fetch_gdelt_attention_timeline(
    candidate: AttentionCandidate,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    target = _cache_file(cache_dir, candidate.symbol, start, end)
    if target.exists() and not refresh:
        return _parse_gdelt_timeline(json.loads(target.read_text(encoding="utf-8")))
    params = {
        "query": candidate.query(),
        "mode": "TimelineVolRaw",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "timelinesmooth": 0,
    }
    request = Request(
        f"{GDELT_DOC_URL}?{urlencode(params)}",
        headers={"User-Agent": "ai-strategy-command-center/1.0"},
    )
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        if target.exists():
            return _parse_gdelt_timeline(json.loads(target.read_text(encoding="utf-8")))
        raise RuntimeError(f"GDELT request failed for {candidate.symbol}: {exc}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return _parse_gdelt_timeline(payload)


def _parse_wikimedia_pageviews(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Wikimedia response: {payload}")
    rows: list[dict[str, object]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp")
        if not timestamp:
            continue
        rows.append(
            {
                "date": pd.to_datetime(str(timestamp), format="%Y%m%d%H", utc=True, errors="coerce"),
                "mentions": pd.to_numeric(item.get("views", 0), errors="coerce"),
                "norm": pd.to_numeric(item.get("views", np.nan), errors="coerce"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["mentions", "norm"])
    frame = frame.dropna(subset=["date"])
    frame["date"] = frame["date"].dt.floor("D")
    frame["mentions"] = pd.to_numeric(frame["mentions"], errors="coerce").fillna(0.0)
    frame["norm"] = pd.to_numeric(frame["norm"], errors="coerce")
    return frame.groupby("date", as_index=True).agg(mentions=("mentions", "sum"), norm=("norm", "sum")).sort_index()


def fetch_wikimedia_attention_timeline(
    candidate: AttentionCandidate,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    target = _wikimedia_cache_file(cache_dir, candidate.symbol, start, end)
    if target.exists() and not refresh:
        return _parse_wikimedia_pageviews(json.loads(target.read_text(encoding="utf-8")))
    start_text = start.strftime("%Y%m%d00")
    end_text = end.strftime("%Y%m%d00")
    titles = candidate.wiki_titles or (candidate.company,)
    combined: list[dict[str, object]] = []
    failures: list[str] = []
    for title in titles:
        encoded = quote(title.replace(" ", "_"), safe="")
        request = Request(
            f"{WIKIMEDIA_PAGEVIEWS_URL}/{encoded}/daily/{start_text}/{end_text}",
            headers={"User-Agent": "ai-strategy-command-center/1.0 (attention research)"},
        )
        try:
            with urlopen(request, timeout=25) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            failures.append(f"{title}: {exc}")
            continue
        combined.extend(payload.get("items", []))
        time.sleep(0.1)
    if not combined:
        if target.exists():
            return _parse_wikimedia_pageviews(json.loads(target.read_text(encoding="utf-8")))
        raise RuntimeError("; ".join(failures) or "empty Wikimedia pageviews")
    payload = {"items": combined}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return _parse_wikimedia_pageviews(payload)


def load_imported_attention(
    path: str | Path = "data/attention_sources",
) -> pd.DataFrame:
    directory = Path(path)
    if not directory.exists():
        return pd.DataFrame(columns=["date", "symbol", "source", "mentions", "engagement"])
    frames = []
    for file in directory.glob("*.csv"):
        try:
            frame = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            continue
        required = {"date", "symbol"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["source"] = frame.get("source", file.stem)
        frame["mentions"] = pd.to_numeric(frame.get("mentions", 0.0), errors="coerce").fillna(0.0)
        frame["engagement"] = pd.to_numeric(frame.get("engagement", 0.0), errors="coerce").fillna(0.0)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.floor("D")
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frames.append(frame.dropna(subset=["date", "symbol"]))
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "source", "mentions", "engagement"])
    return pd.concat(frames, ignore_index=True)


def build_attention_dataset(
    candidates: tuple[AttentionCandidate, ...] | list[AttentionCandidate],
    range_: str = "2y",
    refresh: bool = False,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    request_pause: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = _date_range_for(range_)
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        gdelt_cache = _cache_file(cache_dir, candidate.symbol, start, end)
        wikimedia_cache = _wikimedia_cache_file(cache_dir, candidate.symbol, start, end)
        source = "gdelt_nonfinancial_web"
        if not refresh and not gdelt_cache.exists() and wikimedia_cache.exists():
            source = "wikimedia_pageviews"
            frame = fetch_wikimedia_attention_timeline(candidate, start, end, cache_dir=cache_dir, refresh=False)
        else:
            try:
                frame = fetch_gdelt_attention_timeline(candidate, start, end, cache_dir=cache_dir, refresh=refresh)
                if frame.empty:
                    raise RuntimeError("empty GDELT timeline")
            except Exception as exc:
                gdelt_error = str(exc)
                source = "wikimedia_pageviews"
                try:
                    frame = fetch_wikimedia_attention_timeline(candidate, start, end, cache_dir=cache_dir, refresh=refresh)
                    if frame.empty:
                        raise RuntimeError("empty Wikimedia pageviews")
                except Exception as fallback_exc:
                    failures.append(
                        {
                            "symbol": candidate.symbol,
                            "company": candidate.company,
                            "error": f"GDELT: {gdelt_error}; Wikimedia: {fallback_exc}",
                        }
                    )
                    frame = pd.DataFrame()
        if not frame.empty:
            item = frame.reset_index().rename(columns={"index": "date"})
            item["symbol"] = candidate.symbol
            item["company"] = candidate.company
            item["category"] = candidate.category
            item["source"] = source
            rows.append(item)
        if index < len(candidates) - 1:
            time.sleep(request_pause)
    imported = load_imported_attention()
    if not imported.empty:
        imported = imported.rename(columns={"engagement": "norm"})
        imported["company"] = imported["symbol"].map({candidate.symbol: candidate.company for candidate in candidates})
        imported["category"] = imported["symbol"].map({candidate.symbol: candidate.category for candidate in candidates})
        rows.append(imported[["date", "symbol", "company", "category", "source", "mentions", "norm"]])
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "company", "category", "source", "mentions", "norm"]), pd.DataFrame(failures)
    data = pd.concat(rows, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce").dt.floor("D")
    data["mentions"] = pd.to_numeric(data["mentions"], errors="coerce").fillna(0.0)
    data["norm"] = pd.to_numeric(data.get("norm", np.nan), errors="coerce")
    data = data.dropna(subset=["date", "symbol"])
    return data.sort_values(["symbol", "date"]), pd.DataFrame(failures)


def _load_price_universe(
    candidates: tuple[AttentionCandidate, ...] | list[AttentionCandidate],
    range_: str,
    refresh: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    universe: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            frame = fetch_yahoo_chart(candidate.symbol, interval="1d", range_=range_, refresh=refresh)
            if frame.empty:
                raise RuntimeError("empty price chart")
            universe[candidate.symbol] = frame
        except Exception as exc:
            failures.append({"symbol": candidate.symbol, "company": candidate.company, "error": str(exc)})
        time.sleep(0.25)
    benchmark = "SPY"
    if benchmark not in universe:
        universe[benchmark] = fetch_yahoo_chart(benchmark, interval="1d", range_=range_, refresh=refresh)
    return universe, pd.DataFrame(failures)


def attention_features(
    attention: pd.DataFrame,
    config: AttentionConfig,
) -> pd.DataFrame:
    config.validate()
    frames: list[pd.DataFrame] = []
    for symbol, group in attention.groupby("symbol"):
        daily = (
            group.groupby("date", as_index=True)
            .agg(
                mentions=("mentions", "sum"),
                company=("company", "last"),
                category=("category", "last"),
            )
            .sort_index()
        )
        full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="1D", tz="UTC")
        daily = daily.reindex(full_index)
        daily["mentions"] = pd.to_numeric(daily["mentions"], errors="coerce").fillna(0.0)
        daily["company"] = daily["company"].ffill().bfill()
        daily["category"] = daily["category"].ffill().bfill()
        recent = daily["mentions"].rolling(config.recent_days, min_periods=1).sum()
        baseline = recent.shift(config.recent_days).rolling(config.spike_lookback_days, min_periods=10)
        mean = baseline.mean()
        std = baseline.std().replace(0.0, np.nan)
        z_score = (recent - mean) / std
        growth = recent / mean.replace(0.0, np.nan) - 1.0
        output = daily.copy()
        output["symbol"] = str(symbol)
        output["recent_mentions"] = recent
        output["baseline_mentions"] = mean
        output["spike_z"] = z_score.replace([np.inf, -np.inf], np.nan)
        output["attention_growth_pct"] = growth.replace([np.inf, -np.inf], np.nan) * 100.0
        output["attention_score"] = (
            output["spike_z"].clip(lower=-3.0, upper=config.max_selected_spike_z).fillna(0.0) * 1.0
            + output["attention_growth_pct"].clip(lower=-100.0, upper=500.0).fillna(0.0) / 100.0 * 0.35
        )
        frames.append(output.reset_index().rename(columns={"index": "date"}))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]) if frames else pd.DataFrame()


def _aligned_price_frames(universe: dict[str, pd.DataFrame]) -> tuple[list[str], pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    symbols = [symbol for symbol in universe if symbol != "SPY"]
    index = universe["SPY"].index
    for symbol in symbols:
        index = index.intersection(universe[symbol].index)
    index = index.sort_values()
    closes = pd.DataFrame({symbol: universe[symbol].reindex(index)["close"] for symbol in symbols}, index=index).astype(float)
    opens = pd.DataFrame({symbol: universe[symbol].reindex(index)["open"] for symbol in symbols}, index=index).astype(float)
    return symbols, index, closes, opens


def _target_weights(ranking: pd.DataFrame, config: AttentionConfig) -> dict[str, float]:
    eligible = ranking[
        (ranking["recent_mentions"] >= config.min_recent_mentions)
        & (ranking["spike_z"] >= config.min_spike_z)
        & (ranking["spike_z"] <= config.max_selected_spike_z)
    ].sort_values("attention_score", ascending=False)
    selected = eligible.head(config.top_n)
    if selected.empty:
        return {}
    return {symbol: 1.0 / len(selected) for symbol in selected["symbol"].astype(str).tolist()}


def backtest_attention_strategy(
    universe: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    config: AttentionConfig = AttentionConfig(),
    initial_equity: float = 10_000.0,
) -> AttentionBacktestResult:
    config.validate()
    if "SPY" not in universe:
        raise ValueError("SPY benchmark prices are required.")
    symbols, index, closes, opens = _aligned_price_frames(universe)
    feature_frame = features.copy()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"], utc=True).dt.floor("D")
    feature_lookup = {
        date: frame
        for date, frame in feature_frame[feature_frame["symbol"].isin(symbols)].groupby("date")
    }
    start_offset = config.spike_lookback_days + config.recent_days + 2
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    cash = initial_equity
    quantities = np.zeros(len(symbols), dtype=float)
    prior_weights = np.zeros(len(symbols), dtype=float)
    curve: dict[pd.Timestamp, float] = {}
    events: list[dict[str, object]] = []
    latest_ranking = pd.DataFrame()
    for i in range(start_offset, len(index)):
        timestamp = index[i]
        prior = index[i - 1].floor("D")
        should_rebalance = (i - start_offset) % config.rebalance_days == 0
        if should_rebalance:
            ranking = feature_lookup.get(prior, pd.DataFrame()).copy()
            latest_ranking = ranking.sort_values("attention_score", ascending=False) if not ranking.empty else ranking
            target_weights = _target_weights(ranking, config) if not ranking.empty else {}
            weights = np.zeros(len(symbols), dtype=float)
            for symbol, weight in target_weights.items():
                if symbol in symbols:
                    weights[symbols.index(symbol)] = weight
            if np.any(weights != prior_weights):
                open_equity = cash + float(np.sum(quantities * opens.iloc[i].to_numpy(dtype=float)))
                desired_values = open_equity * weights
                current_values = quantities * opens.iloc[i].to_numpy(dtype=float)
                turnover = float(np.abs(desired_values - current_values).sum())
                cost = turnover * cost_rate
                cash = open_equity - float(desired_values.sum()) - cost
                quantities = np.divide(
                    desired_values,
                    opens.iloc[i].to_numpy(dtype=float),
                    out=np.zeros(len(symbols), dtype=float),
                    where=opens.iloc[i].to_numpy(dtype=float) != 0.0,
                )
                prior_weights = weights
                selected_symbols = [symbol for symbol, weight in target_weights.items() if weight > 0]
                events.append(
                    {
                        "timestamp": timestamp,
                        "decision_date": prior,
                        "selected_symbols": ",".join(selected_symbols) or "CASH",
                        "target_weights": ",".join(f"{symbol}:{target_weights[symbol]:.6f}" for symbol in selected_symbols) or "CASH:0.000000",
                        "turnover": turnover,
                        "turnover_pct": turnover / open_equity * 100.0 if open_equity else 0.0,
                        "cost": cost,
                        "equity_before_cost": open_equity,
                    }
                )
        curve[timestamp] = cash + float(np.sum(quantities * closes.iloc[i].to_numpy(dtype=float)))
    equity = pd.Series(curve, name="attention_strategy", dtype=float).sort_index()
    events_frame = pd.DataFrame(events)
    metrics = _metrics(equity, pd.DataFrame(), initial_equity)
    metrics.update(
        {
            "rebalances": float(len(events_frame)),
            "avg_turnover_pct": float(events_frame["turnover_pct"].mean()) if not events_frame.empty else 0.0,
            "total_cost": float(events_frame["cost"].sum()) if not events_frame.empty else 0.0,
        }
    )
    if latest_ranking.empty and not feature_frame.empty:
        latest_ranking = feature_frame.sort_values(["date", "attention_score"], ascending=[False, False])
    return AttentionBacktestResult(equity, events_frame, latest_ranking, metrics)


def benchmark_buy_and_hold_spy(
    universe: dict[str, pd.DataFrame],
    initial_equity: float = 10_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 3.0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> AttentionBacktestResult:
    frame = universe["SPY"].copy()
    if start is not None:
        frame = frame.loc[frame.index >= start]
    if end is not None:
        frame = frame.loc[frame.index <= end]
    if frame.empty:
        empty = pd.Series(dtype=float)
        return AttentionBacktestResult(empty, pd.DataFrame(), pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity))
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    invested = initial_equity / (1.0 + cost_rate)
    quantity = invested / float(frame["open"].iloc[0])
    equity = frame["close"].astype(float) * quantity
    equity.iloc[-1] *= 1.0 - cost_rate
    equity.name = "SPY"
    return AttentionBacktestResult(equity, pd.DataFrame(), pd.DataFrame(), _metrics(equity, pd.DataFrame(), initial_equity))


def attention_config_candidates() -> tuple[AttentionConfig, ...]:
    configs: list[AttentionConfig] = []
    for top_n in (1, 2, 3, 5):
        for rebalance_days in (5, 10, 20):
            for recent_days, lookback in ((3, 30), (5, 30), (7, 60), (14, 90)):
                for min_z in (0.5, 1.0, 1.5, 2.0):
                    configs.append(
                        AttentionConfig(
                            top_n=top_n,
                            rebalance_days=rebalance_days,
                            recent_days=recent_days,
                            spike_lookback_days=lookback,
                            min_spike_z=min_z,
                        )
                    )
    return tuple(configs)


def optimize_attention_configs(
    universe: dict[str, pd.DataFrame],
    attention: pd.DataFrame,
    usable_symbols: list[str],
    configs: tuple[AttentionConfig, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_configs = configs or attention_config_candidates()
    scoped_universe = {symbol: universe[symbol] for symbol in [*usable_symbols, "SPY"] if symbol in universe}
    scoped_attention = attention[attention["symbol"].isin(usable_symbols)]
    for config in selected_configs:
        features = attention_features(scoped_attention, config)
        result = backtest_attention_strategy(scoped_universe, features, config)
        if result.equity.empty:
            continue
        benchmark = benchmark_buy_and_hold_spy(universe, start=result.equity.index[0], end=result.equity.index[-1])
        excess = result.metrics["return_pct"] - benchmark.metrics["return_pct"]
        score = excess + 0.25 * result.metrics["sharpe"] + 0.20 * result.metrics["max_drawdown_pct"]
        rows.append(
            {
                **config.to_dict(),
                "return_pct": result.metrics["return_pct"],
                "cagr_pct": result.metrics["cagr_pct"],
                "sharpe": result.metrics["sharpe"],
                "max_drawdown_pct": result.metrics["max_drawdown_pct"],
                "rebalances": result.metrics["rebalances"],
                "avg_turnover_pct": result.metrics["avg_turnover_pct"],
                "total_cost": result.metrics["total_cost"],
                "spy_return_pct": benchmark.metrics["return_pct"],
                "excess_return_pct": excess,
                "selection_score": score,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)


def _attention_config_from_row(row: pd.Series) -> AttentionConfig:
    return AttentionConfig(
        top_n=int(row["top_n"]),
        rebalance_days=int(row["rebalance_days"]),
        spike_lookback_days=int(row["spike_lookback_days"]),
        recent_days=int(row["recent_days"]),
        min_recent_mentions=float(row["min_recent_mentions"]),
        min_spike_z=float(row["min_spike_z"]),
        max_selected_spike_z=float(row["max_selected_spike_z"]),
        fee_bps=float(row["fee_bps"]),
        slippage_bps=float(row["slippage_bps"]),
    )


def select_attention_config(search: pd.DataFrame) -> tuple[AttentionConfig, str]:
    if search.empty:
        return AttentionConfig(), "default_config_no_search_results"
    diversified = search[
        (search["top_n"] >= 3)
        & (search["excess_return_pct"] > 0)
        & (search["sharpe"] >= 1.0)
        & (search["max_drawdown_pct"] >= -30.0)
    ]
    if not diversified.empty:
        return _attention_config_from_row(diversified.iloc[0]), "best_diversified_positive_excess_config"
    positive = search[search["excess_return_pct"] > 0]
    if not positive.empty:
        return _attention_config_from_row(positive.iloc[0]), "best_positive_excess_config"
    return _attention_config_from_row(search.iloc[0]), "best_available_config_still_research_only"


def run_attention_research(
    range_: str = "2y",
    refresh: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = default_attention_candidates()
    attention, attention_failures = build_attention_dataset(candidates, range_=range_, refresh=refresh)
    universe, price_failures = _load_price_universe(candidates, range_, refresh)
    usable_symbols = sorted(set(attention["symbol"].astype(str)) & (set(universe) - {"SPY"})) if not attention.empty else []
    usable_candidates = [candidate for candidate in candidates if candidate.symbol in usable_symbols]
    config_search = optimize_attention_configs(universe, attention, usable_symbols) if usable_symbols else pd.DataFrame()
    selected_config, selected_config_reason = select_attention_config(config_search)
    features = attention_features(attention[attention["symbol"].isin(usable_symbols)], selected_config) if usable_symbols else pd.DataFrame()
    result = backtest_attention_strategy(
        {symbol: universe[symbol] for symbol in [*usable_symbols, "SPY"] if symbol in universe},
        features,
        selected_config,
    ) if usable_symbols else AttentionBacktestResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), _metrics(pd.Series(dtype=float), pd.DataFrame(), 10_000.0))
    benchmark_start = result.equity.index[0] if not result.equity.empty else None
    benchmark_end = result.equity.index[-1] if not result.equity.empty else None
    benchmark = benchmark_buy_and_hold_spy(universe, start=benchmark_start, end=benchmark_end)
    attention.to_csv(output / "attention_timeline.csv", index=False)
    features.to_csv(output / "attention_features.csv", index=False)
    config_search.to_csv(output / "attention_config_search.csv", index=False)
    result.events.to_csv(output / "attention_rebalances.csv", index=False)
    result.ranking.to_csv(output / "latest_attention_ranking.csv", index=False)
    pd.concat({"attention_strategy": result.equity, "SPY": benchmark.equity}, axis=1, sort=False).to_csv(output / "attention_equity.csv")
    pd.DataFrame(
        [
            {"strategy": "attention_strategy", **result.metrics},
            {"strategy": "SPY", **benchmark.metrics},
        ]
    ).to_csv(output / "attention_metrics.csv", index=False)
    failures = pd.concat(
        [
            attention_failures.assign(stage="attention") if not attention_failures.empty else pd.DataFrame(),
            price_failures.assign(stage="price") if not price_failures.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    failures.to_csv(output / "attention_failures.csv", index=False)
    latest = result.ranking.copy()
    if not latest.empty:
        latest["eligible"] = (
            (latest["recent_mentions"] >= selected_config.min_recent_mentions)
            & (latest["spike_z"] >= selected_config.min_spike_z)
            & (latest["spike_z"] <= selected_config.max_selected_spike_z)
        )
        selected_symbols = set(_target_weights(latest, selected_config))
        latest["selected"] = latest["symbol"].astype(str).isin(selected_symbols)
        latest = latest.sort_values(["selected", "eligible", "attention_score"], ascending=[False, False, False]).head(10)
        latest.insert(0, "rank", range(1, len(latest) + 1))
    latest.to_csv(output / "latest_attention_candidates.csv", index=False)
    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "range": range_,
        "data_source": "GDELT DOC 2.0 TimelineVolRaw with Wikimedia Pageviews fallback plus optional CSV imports under data/attention_sources/",
        "candidate_count": len(candidates),
        "usable_symbols": usable_symbols,
        "attention_failures": len(attention_failures),
        "price_failures": len(price_failures),
        "config": selected_config.to_dict(),
        "config_selection": selected_config_reason,
        "metrics": {
            "attention_strategy": result.metrics,
            "SPY": benchmark.metrics,
            "excess_return_pct": result.metrics["return_pct"] - benchmark.metrics["return_pct"],
        },
        "live_status": "research_only",
        "live_blocker": "Social/search attention data needs stronger historical sources and forward tracking before live deployment.",
        "candidates": [candidate.to_dict() for candidate in usable_candidates],
    }
    (output / "attention_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
