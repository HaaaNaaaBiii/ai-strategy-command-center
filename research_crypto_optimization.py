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
from smi_lab.paper import DEFAULT_STRATEGY_PATH, load_allocation_strategy


OUTPUT_DIR = Path("outputs/crypto_optimization")
FALLBACK_CURRENT_CONFIG = TrendAllocationConfig(
    momentum_period=180,
    asset_ema_period=42,
    btc_ema_period=100,
    top_n=1,
    rebalance_bars=42,
    gross_exposure=0.35,
)


def offsets_for(config: TrendAllocationConfig) -> tuple[int, ...]:
    return tuple(range(config.rebalance_bars))


def screening_offsets_for(config: TrendAllocationConfig) -> tuple[int, ...]:
    step = max(1, config.rebalance_bars // 7)
    offsets = tuple(range(0, config.rebalance_bars, step))
    return offsets if offsets else (0,)


def config_name(config: TrendAllocationConfig) -> str:
    return (
        f"mom{config.momentum_period}_assetema{config.asset_ema_period}_"
        f"btcema{config.btc_ema_period}_top{config.top_n}_"
        f"reb{config.rebalance_bars}_exp{config.gross_exposure:.2f}"
    )


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
            (90, 180, 240),
            (42, 90, 150),
            (24, 50),
            (1, 2),
            (21, 42),
            (0.35, 0.55),
        )
    ]


def benchmark_results(
    universe: dict[str, pd.DataFrame],
    phases: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], AllocationResult]]:
    benchmark_rows: list[dict[str, float | str]] = []
    benchmarks: dict[tuple[str, str], AllocationResult] = {}
    equal_weights = {symbol: 1.0 / len(DEFAULT_SYMBOLS) for symbol in DEFAULT_SYMBOLS}
    btc_weights = {symbol: (1.0 if symbol == "BTCUSDT" else 0.0) for symbol in DEFAULT_SYMBOLS}
    for phase, start, end in phases:
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
    return pd.DataFrame(benchmark_rows), benchmarks


def score_candidate(row: pd.Series) -> float:
    return (
        float(row["validation_excess_vs_market_pct"]) * 1.0
        + float(row["validation_excess_vs_btc_pct"]) * 0.35
        + float(row["calibration_excess_vs_market_pct"]) * 0.25
        + float(row["worst_selection_drawdown_pct"]) * 0.75
        - abs(float(row["calibration_return_pct"]) - float(row["validation_return_pct"])) * 0.15
    )


def screen_candidates(
    universe: dict[str, pd.DataFrame],
    phases: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    benchmarks: dict[tuple[str, str], AllocationResult],
) -> pd.DataFrame:
    triple_costs = SCENARIOS["triple_cost"]
    screen_rows: list[dict[str, float | str | bool]] = []
    for config in candidate_configs():
        stressed = replace(config, fee_bps=triple_costs[0], slippage_bps=triple_costs[1])
        comparisons: dict[str, dict[str, float | bool]] = {}
        for phase, start, end in phases[:2]:
            result = backtest_staggered_trend_allocation(
                universe,
                stressed,
                screening_offsets_for(stressed),
                trade_start=start,
                trade_end=end,
            )
            comparisons[phase] = compare_to_benchmarks(
                result,
                benchmarks[(phase, "equal_weight_market")],
                benchmarks[(phase, "BTCUSDT")],
            )
        passes_selection = all(
            comparison["beats_equal_weight"] for comparison in comparisons.values()
        )
        screen_rows.append(
            {
                "candidate": config_name(config),
                **asdict(stressed),
                "passes_calibration_and_validation": passes_selection,
                "calibration_return_pct": comparisons["calibration"]["return_pct"],
                "calibration_excess_vs_market_pct": comparisons["calibration"]["excess_vs_equal_weight_pct"],
                "calibration_excess_vs_btc_pct": comparisons["calibration"]["excess_vs_btc_pct"],
                "calibration_max_drawdown_pct": comparisons["calibration"]["max_drawdown_pct"],
                "validation_return_pct": comparisons["validation"]["return_pct"],
                "validation_excess_vs_market_pct": comparisons["validation"]["excess_vs_equal_weight_pct"],
                "validation_excess_vs_btc_pct": comparisons["validation"]["excess_vs_btc_pct"],
                "validation_max_drawdown_pct": comparisons["validation"]["max_drawdown_pct"],
                "worst_selection_drawdown_pct": min(
                    comparisons["calibration"]["max_drawdown_pct"],
                    comparisons["validation"]["max_drawdown_pct"],
                ),
            }
        )
    screen = pd.DataFrame(screen_rows)
    screen["robust_score"] = screen.apply(score_candidate, axis=1)
    return screen.sort_values(
        ["passes_calibration_and_validation", "robust_score"],
        ascending=[False, False],
    )


def evaluate_config(
    name: str,
    config: TrendAllocationConfig,
    universe: dict[str, pd.DataFrame],
    phases: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    benchmarks: dict[tuple[str, str], AllocationResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, float | str]] = []
    comparison_rows: list[dict[str, float | str | bool]] = []
    for scenario, costs in SCENARIOS.items():
        configured = replace(config, fee_bps=costs[0], slippage_bps=costs[1])
        for phase, start, end in phases:
            result = backtest_staggered_trend_allocation(
                universe,
                configured,
                offsets_for(configured),
                trade_start=start,
                trade_end=end,
            )
            metrics_rows.append(result_row(name, scenario, phase, result))
            if phase in {"calibration", "validation", "holdout", "full_period"}:
                comparison_rows.append(
                    {
                        "strategy": name,
                        "scenario": scenario,
                        "phase": phase,
                        **compare_to_benchmarks(
                            result,
                            benchmarks[(phase, "equal_weight_market")],
                            benchmarks[(phase, "BTCUSDT")],
                        ),
                    }
                )
    return pd.DataFrame(metrics_rows), pd.DataFrame(comparison_rows)


def load_current_strategy() -> tuple[TrendAllocationConfig, str]:
    if Path(DEFAULT_STRATEGY_PATH).exists():
        config, _, payload = load_allocation_strategy(DEFAULT_STRATEGY_PATH)
        return config, str(payload.get("name", "current_paper_strategy"))
    return FALLBACK_CURRENT_CONFIG, "fallback_current_strategy"


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
    report_phases = phases + annual_windows(index)
    benchmarks_frame, benchmarks = benchmark_results(universe, report_phases)
    benchmarks_frame.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    screen = screen_candidates(universe, phases, benchmarks)
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_row = screen.iloc[0]
    selected = TrendAllocationConfig(
        momentum_period=int(selected_row["momentum_period"]),
        asset_ema_period=int(selected_row["asset_ema_period"]),
        btc_ema_period=int(selected_row["btc_ema_period"]),
        top_n=int(selected_row["top_n"]),
        rebalance_bars=int(selected_row["rebalance_bars"]),
        gross_exposure=float(selected_row["gross_exposure"]),
    )
    selected_name = str(selected_row["candidate"])
    current_config, current_name = load_current_strategy()
    selected_metrics, selected_comparisons = evaluate_config(
        selected_name, selected, universe, report_phases, benchmarks
    )
    current_metrics, current_comparisons = evaluate_config(
        current_name, current_config, universe, report_phases, benchmarks
    )
    metrics = pd.concat([selected_metrics, current_metrics], ignore_index=True)
    comparisons = pd.concat([selected_comparisons, current_comparisons], ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "benchmark_comparison.csv", index=False)
    stressed = comparisons[comparisons["scenario"] == "triple_cost"]
    selected_holdout = stressed[(stressed["strategy"] == selected_name) & (stressed["phase"] == "holdout")]
    current_holdout = stressed[(stressed["strategy"] == current_name) & (stressed["phase"] == "holdout")]
    recommendation = "keep_current"
    if not selected_holdout.empty and not current_holdout.empty:
        if (
            bool(selected_holdout.iloc[0]["beats_equal_weight"])
            and float(selected_holdout.iloc[0]["return_pct"]) > float(current_holdout.iloc[0]["return_pct"])
            and float(selected_holdout.iloc[0]["max_drawdown_pct"]) >= float(current_holdout.iloc[0]["max_drawdown_pct"]) - 5.0
        ):
            recommendation = "promote_candidate_after_forward_tracking"
    report = {
        "data_window": data_window(universe["BTCUSDT"]),
        "candidate_count": int(len(screen)),
        "selected_candidate": selected_name,
        "selected_config": asdict(selected),
        "current_strategy": current_name,
        "current_config": asdict(current_config),
        "selection_rule": "Screen calibration and validation under triple trading costs; holdout is report-only.",
        "recommendation": recommendation,
    }
    (OUTPUT_DIR / "optimization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(screen.head(10).to_string(index=False))
    print(json.dumps(report, indent=2))
    print(stressed.to_string(index=False))


if __name__ == "__main__":
    main()
