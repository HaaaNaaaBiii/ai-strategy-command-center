from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace

import pandas as pd

from smi_lab.equity_data import load_equity_universe
from smi_lab.equity_strategy import (
    backtest_equity_selection,
    benchmark_buy_and_hold,
    default_equity_config,
    rank_equities,
)
from smi_lab.equity_universe import equity_scan_symbols


OUTPUT_DIR = Path("outputs/equity_selection")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Rank and backtest TW/US equity selection strategies.")
    command.add_argument("--market", choices=["tw", "us", "both"], default="both")
    command.add_argument("--interval", default="1d", choices=["1d", "1wk", "1h"])
    command.add_argument("--range", dest="range_", default="2y")
    command.add_argument("--weighting", choices=["equal", "score", "capped_score", "all"], default="equal")
    command.add_argument("--refresh", action="store_true")
    return command


def weighting_configs(config, weighting: str):
    variants = {
        "equal": config,
        "score": replace(config, weighting_method="score"),
        "capped_score": replace(
            config,
            weighting_method="capped_score",
            min_position_weight=0.20,
            max_position_weight=0.40,
        ),
    }
    if weighting == "all":
        return variants
    return {weighting: variants[weighting]}


def run_market(market: str, interval: str, range_: str, refresh: bool, weighting: str) -> None:
    config = default_equity_config(market)
    symbols = list(dict.fromkeys([*equity_scan_symbols(market), config.market_symbol]))
    universe = load_equity_universe(
        symbols,
        market=market,
        interval=interval,
        range_=range_,
        refresh=refresh,
    )
    ranking = rank_equities(universe, config)
    results = {
        label: backtest_equity_selection(universe, variant_config)
        for label, variant_config in weighting_configs(config, weighting).items()
    }
    benchmark = benchmark_buy_and_hold(
        universe[config.market_symbol],
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_DIR / f"{market}_ranking.csv", index=False)
    for label, result in results.items():
        suffix = "" if label == "equal" else f"_{label}"
        result.rebalances.to_csv(OUTPUT_DIR / f"{market}{suffix}_rebalances.csv", index=False)
    pd.DataFrame(
        [
            *[
                {"strategy": f"equity_selection_{label}", **result.metrics}
                for label, result in results.items()
            ],
            {"strategy": config.market_symbol, **benchmark.metrics},
        ]
    ).to_csv(OUTPUT_DIR / f"{market}_metrics.csv", index=False)
    (OUTPUT_DIR / f"{market}_strategy_metadata.json").write_text(
        json.dumps(
            {
                "market": market,
                "role": "long-only relative-strength equity selection research candidate",
                "market_adjustments": {
                    "tw": {
                        "benchmark": "0050.TW",
                        "default_rebalance": "40 trading days",
                        "cost_model": "14.25 bps commission proxy plus 5 bps slippage",
                        "rationale": "Taiwan market is more concentrated and has different fee/tax frictions; use slower turnover and ETF market gate.",
                    },
                    "us": {
                        "benchmark": "SPY",
                        "default_rebalance": "20 trading days",
                        "cost_model": "1 bps commission proxy plus 3 bps slippage",
                        "rationale": "U.S. mega-cap basket is deeper and more liquid; use faster monthly turnover and SPY market gate.",
                    },
                }[market],
                "config": config.to_dict(),
                "benchmark_return_pct": benchmark.metrics["return_pct"],
                "strategy_variants": {
                    label: {
                        "config": weighting_configs(config, weighting)[label].to_dict(),
                        "metrics": result.metrics,
                        "excess_pct": result.metrics["return_pct"] - benchmark.metrics["return_pct"],
                    }
                    for label, result in results.items()
                },
                "deployment_status": "research_only",
                "deployment_blocker": "Needs larger universe, survivorship-bias controls, corporate actions/dividend handling, and forward paper tracking before notifications.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.concat(
        {
            **{f"equity_selection_{label}": result.equity for label, result in results.items()},
            config.market_symbol: benchmark.equity,
        },
        axis=1,
        sort=False,
    ).to_csv(OUTPUT_DIR / f"{market}_equity.csv")
    print(f"[{market}]")
    print(ranking.head(10).to_string(index=False))
    print(pd.read_csv(OUTPUT_DIR / f"{market}_metrics.csv").to_string(index=False))


def main() -> None:
    args = parser().parse_args()
    markets = ("tw", "us") if args.market == "both" else (args.market,)
    for market in markets:
        run_market(market, args.interval, args.range_, args.refresh, args.weighting)


if __name__ == "__main__":
    main()
