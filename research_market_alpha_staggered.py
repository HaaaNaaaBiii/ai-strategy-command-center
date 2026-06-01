from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

import pandas as pd

from research_market_alpha import (
    SCENARIOS,
    annual_windows,
    compare_to_benchmarks,
    result_row,
    windows,
)
from smi_lab.allocation import (
    AllocationResult,
    TrendAllocationConfig,
    backtest_buy_and_hold,
    backtest_staggered_trend_allocation,
)
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe


OUTPUT_DIR = Path("outputs/market_alpha_staggered")
OFFSETS = tuple(range(42))


def candidate_configs() -> list[TrendAllocationConfig]:
    return [
        TrendAllocationConfig(
            momentum_period=180,
            asset_ema_period=asset_ema,
            btc_ema_period=btc_ema,
            top_n=1,
            rebalance_bars=42,
            gross_exposure=exposure,
        )
        for asset_ema, btc_ema, exposure in product(
            (42, 90, 180),
            (24, 50, 100),
            (0.35, 0.40, 0.45, 0.50, 0.60),
        )
    ]


def config_name(config: TrendAllocationConfig) -> str:
    return (
        f"mom{config.momentum_period}_assetema{config.asset_ema_period}_"
        f"btcema{config.btc_ema_period}_top{config.top_n}_"
        f"reb{config.rebalance_bars}_exp{config.gross_exposure:.2f}"
    )


def benchmark_results(
    universe: dict[str, pd.DataFrame],
    phases: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], AllocationResult]]:
    benchmark_rows: list[dict[str, float | str]] = []
    benchmarks: dict[tuple[str, str], AllocationResult] = {}
    equal_weights = {symbol: 1.0 / len(DEFAULT_SYMBOLS) for symbol in DEFAULT_SYMBOLS}
    btc_weights = {
        symbol: (1.0 if symbol == "BTCUSDT" else 0.0)
        for symbol in DEFAULT_SYMBOLS
    }
    for phase, start, end in phases:
        for name, weights in (
            ("equal_weight_market", equal_weights),
            ("BTCUSDT", btc_weights),
        ):
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
    return pd.DataFrame(benchmark_rows), benchmarks


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
    split_phases = windows(index)
    report_phases = split_phases + annual_windows(index)
    benchmarks_frame, benchmarks = benchmark_results(universe, report_phases)
    benchmarks_frame.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)

    triple_costs = SCENARIOS["triple_cost"]
    screen_rows: list[dict[str, float | str | bool]] = []
    for config in candidate_configs():
        stressed = replace(
            config, fee_bps=triple_costs[0], slippage_bps=triple_costs[1]
        )
        comparisons: dict[str, dict[str, float | bool]] = {}
        for phase, start, end in split_phases[:2]:
            result = backtest_staggered_trend_allocation(
                universe, stressed, OFFSETS, trade_start=start, trade_end=end
            )
            comparisons[phase] = compare_to_benchmarks(
                result,
                benchmarks[(phase, "equal_weight_market")],
                benchmarks[(phase, "BTCUSDT")],
            )
        passes_primary = all(
            comparison["beats_equal_weight"] for comparison in comparisons.values()
        )
        screen_rows.append(
            {
                "candidate": config_name(config),
                **asdict(stressed),
                "passes_primary_calibration_and_validation": passes_primary,
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
            "passes_primary_calibration_and_validation",
            "worst_selection_drawdown_pct",
            "validation_excess_vs_market_pct",
            "calibration_excess_vs_market_pct",
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
        for phase, start, end in report_phases:
            result = backtest_staggered_trend_allocation(
                universe, configured, OFFSETS, trade_start=start, trade_end=end
            )
            selected_results[(scenario, phase)] = result
            metrics_rows.append(result_row(selected_name, scenario, phase, result))
        selected_results[(scenario, "full_period")].rebalances.to_csv(
            OUTPUT_DIR / f"{scenario}_full_period_rebalances.csv", index=False
        )
    pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "selected_metrics.csv", index=False)

    comparison_rows: list[dict[str, float | str | bool]] = []
    for scenario in SCENARIOS:
        for phase, _, _ in split_phases:
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
    stressed = comparisons[comparisons["scenario"] == "triple_cost"].set_index("phase")

    payload = {
        "data_window": data_window(universe["BTCUSDT"]),
        "strategy_role": (
            "Unlevered long/cash trend and relative-strength allocation shadow candidate."
        ),
        "primary_benchmark": (
            "Static equal-weight buy-and-hold of BTCUSDT, ETHUSDT, DOGEUSDT, "
            "and SOLUSDT with standard one-entry/one-exit costs and no funding drag."
        ),
        "secondary_benchmark": (
            "BTCUSDT buy-and-hold with standard one-entry/one-exit costs and no "
            "funding drag."
        ),
        "schedule_design": (
            "Split capital equally into all 42 fixed four-hour offsets within a "
            "weekly rebalance cycle; this avoids selecting a favorable weekly phase."
        ),
        "selection_protocol": (
            "After broad exploratory research, screen the constrained staggered grid "
            "under triple trading costs using calibration (first 60%) and validation "
            "(next 20%) only. Select the primary-benchmark passing configuration with "
            "the smallest worst selection-period drawdown. Evaluate holdout last "
            "within this screen; it is not pristine unseen data because earlier "
            "research already examined the same five-year history."
        ),
        "candidate_count": len(screen),
        "selection_pass_count": int(
            screen["passes_primary_calibration_and_validation"].sum()
        ),
        "selected_candidate": selected_name,
        "selected_config": asdict(selected),
        "schedule_sleeves": len(OFFSETS),
        "triple_cost_holdout_beats_primary_benchmark": bool(
            stressed.loc["holdout", "beats_equal_weight"]
        ),
        "triple_cost_full_period_beats_primary_benchmark": bool(
            stressed.loc["full_period", "beats_equal_weight"]
        ),
        "triple_cost_full_period_beats_btc_benchmark": bool(
            stressed.loc["full_period", "beats_btc"]
        ),
        "triple_cost_validation_beats_btc_benchmark": bool(
            stressed.loc["validation", "beats_btc"]
        ),
        "notification_ready": False,
        "deployment_blocker": (
            "The allocation signal is not wired into the notifier and needs forward "
            "paper tracking before funded use."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "paper_strategy.json").write_text(
        json.dumps(
            {
                "name": "market_alpha_staggered_trend_allocation",
                "status": "shadow_only",
                "config": asdict(selected),
                "rebalance_offsets": list(OFFSETS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.head(10).to_string(index=False))
    print(json.dumps(payload, indent=2))
    print(
        comparisons[
            comparisons["phase"].isin(
                ["calibration", "validation", "holdout", "full_period"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
