from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

import pandas as pd

from smi_lab.allocation import (
    AllocationResult,
    TrendAllocationConfig,
    backtest_buy_and_hold,
    backtest_trend_allocation,
)
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe


OUTPUT_DIR = Path("outputs/market_alpha")
SCENARIOS = {
    "normal": (10.0, 5.0),
    "double_cost": (20.0, 10.0),
    "triple_cost": (30.0, 15.0),
}


def candidate_configs() -> list[TrendAllocationConfig]:
    return [
        TrendAllocationConfig(
            momentum_period=momentum,
            asset_ema_period=asset_ema,
            btc_ema_period=btc_ema,
            top_n=top_n,
            rebalance_bars=rebalance,
            gross_exposure=exposure,
        )
        for momentum, asset_ema, btc_ema, top_n, rebalance, exposure in product(
            (42, 90, 180),
            (42, 90, 180),
            (24, 50, 100),
            (1, 2),
            (6, 42),
            (0.25, 0.40, 0.50),
        )
    ]


def config_name(config: TrendAllocationConfig) -> str:
    return (
        f"mom{config.momentum_period}_assetema{config.asset_ema_period}_"
        f"btcema{config.btc_ema_period}_top{config.top_n}_"
        f"reb{config.rebalance_bars}_exp{config.gross_exposure:.2f}"
    )


def windows(index: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    length = len(index)

    def at(ratio: float, final: bool = False) -> pd.Timestamp:
        offset = min(int(length * ratio) - (1 if final else 0), length - 1)
        return index[max(offset, 1)]

    return [
        ("calibration", at(0.0), at(0.6, final=True)),
        ("validation", at(0.6), at(0.8, final=True)),
        ("holdout", at(0.8), at(1.0, final=True)),
        ("evaluation", at(0.6), at(1.0, final=True)),
        ("full_period", at(0.0), at(1.0, final=True)),
    ]


def annual_windows(index: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    length = len(index)
    return [
        (
            f"year_{year + 1}",
            index[max(int(length * year / 5), 1)],
            index[min(int(length * (year + 1) / 5) - 1, length - 1)],
        )
        for year in range(5)
    ]


def result_row(
    strategy: str,
    scenario: str,
    phase: str,
    result: AllocationResult,
) -> dict[str, float | str]:
    return {
        "strategy": strategy,
        "scenario": scenario,
        "phase": phase,
        **result.metrics,
    }


def compare_to_benchmarks(
    result: AllocationResult,
    equal_weight: AllocationResult,
    btc: AllocationResult,
) -> dict[str, float | bool]:
    strategy_return = result.metrics["return_pct"]
    equal_weight_return = equal_weight.metrics["return_pct"]
    btc_return = btc.metrics["return_pct"]
    return {
        "return_pct": strategy_return,
        "max_drawdown_pct": result.metrics["max_drawdown_pct"],
        "excess_vs_equal_weight_pct": strategy_return - equal_weight_return,
        "excess_vs_btc_pct": strategy_return - btc_return,
        "beats_equal_weight": strategy_return > equal_weight_return,
        "beats_btc": strategy_return > btc_return,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_universe(
        DEFAULT_SYMBOLS,
        interval="4h",
        bars=bars_for_years("4h", 5),
        market="perpetual",
        include_funding=True,
    )
    index = universe["BTCUSDT"].index
    phases = windows(index)
    benchmark_rows: list[dict[str, float | str]] = []
    benchmarks: dict[tuple[str, str], AllocationResult] = {}
    equal_weights = {symbol: 1.0 / len(DEFAULT_SYMBOLS) for symbol in DEFAULT_SYMBOLS}
    btc_weights = {symbol: (1.0 if symbol == "BTCUSDT" else 0.0) for symbol in DEFAULT_SYMBOLS}
    for phase, start, end in phases + annual_windows(index):
        for name, weights in (("equal_weight_market", equal_weights), ("BTCUSDT", btc_weights)):
            result = backtest_buy_and_hold(
                universe,
                weights,
                trade_start=start,
                trade_end=end,
                fee_bps=10.0,
                slippage_bps=5.0,
            )
            benchmarks[(phase, name)] = result
            benchmark_rows.append(result_row(name, "standard_cost", phase, result))
    benchmarks_frame = pd.DataFrame(benchmark_rows)
    benchmarks_frame.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)

    triple_costs = SCENARIOS["triple_cost"]
    screen_rows: list[dict[str, float | str | bool]] = []
    for config in candidate_configs():
        stressed = replace(
            config, fee_bps=triple_costs[0], slippage_bps=triple_costs[1]
        )
        comparisons: dict[str, dict[str, float | bool]] = {}
        for phase, start, end in phases[:2]:
            result = backtest_trend_allocation(
                universe, stressed, trade_start=start, trade_end=end
            )
            comparisons[phase] = compare_to_benchmarks(
                result,
                benchmarks[(phase, "equal_weight_market")],
                benchmarks[(phase, "BTCUSDT")],
            )
        passes_selection = all(
            comparison["beats_equal_weight"] and comparison["beats_btc"]
            for comparison in comparisons.values()
        )
        screen_rows.append(
            {
                "candidate": config_name(config),
                **asdict(config),
                "passes_calibration_and_validation": passes_selection,
                "calibration_return_pct": comparisons["calibration"]["return_pct"],
                "calibration_excess_vs_market_pct": comparisons["calibration"][
                    "excess_vs_equal_weight_pct"
                ],
                "calibration_excess_vs_btc_pct": comparisons["calibration"][
                    "excess_vs_btc_pct"
                ],
                "calibration_max_drawdown_pct": comparisons["calibration"][
                    "max_drawdown_pct"
                ],
                "validation_return_pct": comparisons["validation"]["return_pct"],
                "validation_excess_vs_market_pct": comparisons["validation"][
                    "excess_vs_equal_weight_pct"
                ],
                "validation_excess_vs_btc_pct": comparisons["validation"][
                    "excess_vs_btc_pct"
                ],
                "validation_max_drawdown_pct": comparisons["validation"][
                    "max_drawdown_pct"
                ],
                "worst_selection_drawdown_pct": min(
                    comparisons["calibration"]["max_drawdown_pct"],
                    comparisons["validation"]["max_drawdown_pct"],
                ),
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        [
            "passes_calibration_and_validation",
            "worst_selection_drawdown_pct",
            "validation_excess_vs_btc_pct",
            "calibration_excess_vs_btc_pct",
        ],
        ascending=[False, False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_name = str(screen.iloc[0]["candidate"])
    selected = next(
        config for config in candidate_configs() if config_name(config) == selected_name
    )

    metrics_rows: list[dict[str, float | str]] = []
    selected_results: dict[tuple[str, str], AllocationResult] = {}
    for scenario, costs in SCENARIOS.items():
        configured = replace(selected, fee_bps=costs[0], slippage_bps=costs[1])
        for phase, start, end in phases + annual_windows(index):
            result = backtest_trend_allocation(
                universe, configured, trade_start=start, trade_end=end
            )
            selected_results[(scenario, phase)] = result
            metrics_rows.append(result_row(selected_name, scenario, phase, result))
        selected_results[(scenario, "full_period")].rebalances.to_csv(
            OUTPUT_DIR / f"{scenario}_full_period_rebalances.csv", index=False
        )
    selected_metrics = pd.DataFrame(metrics_rows)
    selected_metrics.to_csv(OUTPUT_DIR / "selected_metrics.csv", index=False)

    comparison_rows: list[dict[str, float | str | bool]] = []
    for scenario in SCENARIOS:
        for phase, _, _ in phases:
            result = selected_results[(scenario, phase)]
            comparison_rows.append(
                {
                    "scenario": scenario,
                    "phase": phase,
                    **compare_to_benchmarks(
                        result,
                        benchmarks[(phase, "equal_weight_market")],
                        benchmarks[(phase, "BTCUSDT")],
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(OUTPUT_DIR / "selected_benchmark_comparison.csv", index=False)

    stressed_comparisons = comparisons[comparisons["scenario"] == "triple_cost"].set_index(
        "phase"
    )
    selected_pool = screen[screen["passes_calibration_and_validation"]]
    holdout_pass = bool(
        stressed_comparisons.loc["holdout", "beats_equal_weight"]
        and stressed_comparisons.loc["holdout", "beats_btc"]
    )
    full_pass = bool(
        stressed_comparisons.loc["full_period", "beats_equal_weight"]
        and stressed_comparisons.loc["full_period", "beats_btc"]
    )
    payload = {
        "data_window": data_window(universe["BTCUSDT"]),
        "strategy_role": "unlevered long/cash trend and relative-strength allocation candidate",
        "primary_benchmark": (
            "Static equal-weight buy-and-hold of BTCUSDT, ETHUSDT, DOGEUSDT, "
            "and SOLUSDT with standard one-entry/one-exit costs and no funding drag."
        ),
        "secondary_benchmark": (
            "BTCUSDT buy-and-hold with standard one-entry/one-exit costs and no "
            "funding drag."
        ),
        "selection_protocol": (
            "Screen configurations under triple trading costs using calibration "
            "(first 60%) and validation (next 20%) only; choose the qualifying "
            "configuration with the smallest worst drawdown. Holdout (last 20%) "
            "is evaluated after selection."
        ),
        "candidate_count": len(screen),
        "selection_pass_count": int(
            screen["passes_calibration_and_validation"].sum()
        ),
        "selected_candidate": selected_name,
        "selected_config": asdict(selected),
        "triple_cost_holdout_beats_both_benchmarks": holdout_pass,
        "triple_cost_full_period_beats_both_benchmarks": full_pass,
        "notification_ready": False,
        "deployment_blocker": (
            "Only one five-year history and one post-selection holdout are "
            "available; this allocation must be tracked forward before it can "
            "replace the active SMI notification strategy."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "paper_strategy.json").write_text(
        json.dumps(
            {
                "name": "market_alpha_trend_allocation",
                "status": "shadow_only",
                "config": asdict(selected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(benchmarks_frame[benchmarks_frame["phase"].isin([p[0] for p in phases])].to_string(index=False))
    print(screen.head(12).to_string(index=False))
    print(json.dumps(payload, indent=2))
    print(
        comparisons[
            comparisons["phase"].isin(["calibration", "validation", "holdout", "full_period"])
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
