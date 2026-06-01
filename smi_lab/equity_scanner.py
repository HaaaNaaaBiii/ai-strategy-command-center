from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import pandas as pd

from .equity_data import fetch_yahoo_chart, normalize_equity_symbol
from .equity_signals import add_company_names, build_equity_trade_plan
from .equity_strategy import (
    EquitySelectionConfig,
    backtest_equity_selection,
    benchmark_buy_and_hold,
    default_equity_config,
    rank_equities,
)
from .equity_universe import equity_scan_symbols


SCAN_COLUMNS = [
    "symbol",
    "company",
    "action",
    "selected",
    "eligible",
    "rank",
    "score",
    "close",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "strategy_exit",
    "short_momentum_pct",
    "long_momentum_pct",
    "annualized_volatility_pct",
    "above_trend",
    "risk_on",
    "reason",
]


def scan_config(market: str, top_n: int | None = None) -> EquitySelectionConfig:
    config = default_equity_config(market)
    if top_n is not None:
        config = replace(config, top_n=top_n)
    return config


def load_equity_scan_universe(
    market: str,
    symbols: list[str] | None = None,
    interval: str = "1d",
    range_: str = "2y",
    refresh: bool = False,
    cache_dir: str | Path = "data/equities",
    request_pause: float = 0.35,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    source_symbols = symbols or list(equity_scan_symbols(market))
    config = default_equity_config(market)
    normalized_symbols = list(
        dict.fromkeys(
            [normalize_equity_symbol(symbol, market) for symbol in [*source_symbols, config.market_symbol]]
        )
    )
    universe: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, object]] = []
    for index, symbol in enumerate(normalized_symbols):
        try:
            frame = fetch_yahoo_chart(
                symbol,
                interval=interval,
                range_=range_,
                cache_dir=cache_dir,
                refresh=refresh,
            )
            if frame.empty:
                raise RuntimeError("empty chart")
            universe[symbol] = frame
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
        if index < len(normalized_symbols) - 1:
            time.sleep(request_pause)
    return universe, pd.DataFrame(failures)


def build_scan_recommendations(
    universe: dict[str, pd.DataFrame],
    config: EquitySelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking = add_company_names(rank_equities(universe, config))
    ranking["scan_rank"] = range(1, len(ranking) + 1)
    eligible = ranking[ranking["eligible"]].head(config.top_n)
    rows: list[dict[str, object]] = []
    ranking_lookup = ranking.set_index("symbol").to_dict("index")
    for symbol in eligible["symbol"].tolist():
        plan = build_equity_trade_plan(symbol, universe, config, ranking)
        ranked = ranking_lookup.get(symbol, {})
        row = plan.to_dict()
        row.update(
            {
                "score": ranked.get("score"),
                "short_momentum_pct": ranked.get("short_momentum_pct"),
                "long_momentum_pct": ranked.get("long_momentum_pct"),
                "annualized_volatility_pct": ranked.get("annualized_volatility_pct"),
                "above_trend": ranked.get("above_trend"),
                "risk_on": ranked.get("risk_on"),
            }
        )
        rows.append(row)
    recommendations = pd.DataFrame(rows, columns=SCAN_COLUMNS)
    result = backtest_equity_selection(universe, config)
    benchmark = benchmark_buy_and_hold(
        universe[config.market_symbol],
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
    )
    metrics = pd.DataFrame(
        [
            {"strategy": "equity_selection_scan", **result.metrics},
            {"strategy": config.market_symbol, **benchmark.metrics},
        ]
    )
    return ranking, recommendations, metrics


def run_equity_scan(
    market: str,
    interval: str = "1d",
    range_: str = "2y",
    refresh: bool = False,
    top_n: int = 5,
    output_dir: str | Path = "outputs/equity_scan",
    symbols: list[str] | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = scan_config(market, top_n=top_n)
    universe, failures = load_equity_scan_universe(
        market,
        symbols=symbols,
        interval=interval,
        range_=range_,
        refresh=refresh,
    )
    if config.market_symbol not in universe:
        raise RuntimeError(f"{market} benchmark is unavailable: {config.market_symbol}")
    if len(universe) <= 1:
        raise RuntimeError(f"{market} scan has no usable symbols.")
    ranking, recommendations, metrics = build_scan_recommendations(universe, config)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ranking.insert(0, "scan_time_utc", timestamp)
    recommendations.insert(0, "scan_time_utc", timestamp)
    metrics.insert(0, "scan_time_utc", timestamp)
    ranking.to_csv(output / f"{market}_scan_ranking.csv", index=False)
    recommendations.to_csv(output / f"{market}_recommendations.csv", index=False)
    metrics.to_csv(output / f"{market}_scan_metrics.csv", index=False)
    failures.to_csv(output / f"{market}_scan_failures.csv", index=False)
    summary = {
        "market": market,
        "scan_time_utc": timestamp,
        "interval": interval,
        "range": range_,
        "configured_top_n": top_n,
        "loaded_symbols": len(universe),
        "failed_symbols": len(failures),
        "eligible_symbols": int(ranking["eligible"].sum()) if "eligible" in ranking else 0,
        "recommended_symbols": recommendations["symbol"].tolist() if not recommendations.empty else [],
        "status": "HAS_RECOMMENDATIONS" if not recommendations.empty else "HOLD_CASH",
        "config": config.to_dict(),
    }
    (output / f"{market}_scan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
