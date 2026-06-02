from __future__ import annotations

from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import pandas as pd

from smi_lab.equity_scanner import load_equity_scan_universe
from smi_lab.equity_strategy import (
    EquitySelectionConfig,
    backtest_equity_selection,
    benchmark_buy_and_hold,
    default_equity_config,
)


OUTPUT_DIR = Path("outputs/equity_optimization")


def candidate_configs(market: str) -> list[EquitySelectionConfig]:
    base = default_equity_config(market)
    rebalance_values = (20, 40, 60) if market == "tw" else (10, 20, 40)
    return [
        replace(
            base,
            top_n=top_n,
            rebalance_bars=rebalance,
            short_momentum_period=short_momentum,
            long_momentum_period=long_momentum,
            trend_period=trend,
            max_volatility_pct=max_volatility,
        )
        for top_n, rebalance, short_momentum, long_momentum, trend, max_volatility in product(
            (3, 5, 8),
            rebalance_values,
            (20, 40, 63),
            (60, 126),
            (100, 200),
            (60.0, 80.0),
        )
    ]


def robust_score(row: pd.Series) -> float:
    return (
        float(row["excess_return_pct"])
        + 0.45 * float(row["sharpe"])
        + 0.35 * float(row["max_drawdown_pct"])
        - 0.02 * float(row["rebalances"])
    )


def run_market(market: str, refresh: bool = False) -> dict[str, object]:
    universe, failures = load_equity_scan_universe(market, range_="2y", refresh=refresh)
    benchmark_config = default_equity_config(market)
    benchmark = benchmark_buy_and_hold(
        universe[benchmark_config.market_symbol],
        fee_bps=benchmark_config.fee_bps,
        slippage_bps=benchmark_config.slippage_bps,
    )
    rows: list[dict[str, object]] = []
    for config in candidate_configs(market):
        try:
            result = backtest_equity_selection(universe, config)
        except Exception as exc:
            rows.append({"market": market, "error": str(exc), **config.to_dict()})
            continue
        rows.append(
            {
                "market": market,
                **config.to_dict(),
                **result.metrics,
                "benchmark_return_pct": benchmark.metrics["return_pct"],
                "excess_return_pct": result.metrics["return_pct"] - benchmark.metrics["return_pct"],
            }
        )
    frame = pd.DataFrame(rows)
    valid = frame.dropna(subset=["return_pct", "max_drawdown_pct", "sharpe"]).copy()
    valid["robust_score"] = valid.apply(robust_score, axis=1)
    valid = valid.sort_values("robust_score", ascending=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / f"{market}_candidate_metrics.csv", index=False)
    valid.head(20).to_csv(OUTPUT_DIR / f"{market}_top_candidates.csv", index=False)
    selected = valid.iloc[0].to_dict()
    report = {
        "market": market,
        "loaded_symbols": len(universe),
        "failed_symbols": len(failures),
        "benchmark_return_pct": benchmark.metrics["return_pct"],
        "selected": {
            key: selected[key]
            for key in (
                "top_n",
                "rebalance_bars",
                "short_momentum_period",
                "long_momentum_period",
                "trend_period",
                "max_volatility_pct",
                "return_pct",
                "max_drawdown_pct",
                "sharpe",
                "excess_return_pct",
                "robust_score",
            )
        },
        "selection_note": "Ranked by excess return, Sharpe, drawdown, and turnover penalty on the current 2y broad scan universe.",
    }
    (OUTPUT_DIR / f"{market}_optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    reports = [run_market("tw"), run_market("us")]
    (OUTPUT_DIR / "latest_optimization_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
