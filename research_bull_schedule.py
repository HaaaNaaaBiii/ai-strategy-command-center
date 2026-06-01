from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from research_bull_basket import combine_basket, maximum_initial_risk as basket_risk, signal_config
from research_bull_combined import mature_sleeves
from research_bull_offense import benchmark_rows, phase_ranges
from research_bull_portfolio_brake import (
    MAXIMUM_EVALUATION_DRAWDOWN_PCT,
    MAXIMUM_INITIAL_STOP_RISK_PCT,
    MINIMUM_BENCHMARK_EXCESS_PCT,
    combine_with_brake,
    maximum_initial_risk,
)
from research_bull_rotation import SCENARIOS, with_costs
from smi_lab.backtest import backtest_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.regime import attach_btc_momentum_regime
from smi_lab.rotation import RotationConfig, RotationResult, backtest_rotation


OUTPUT_DIR = Path("outputs/bull_schedule")
ROTATION_WEIGHT = 0.72
PORTFOLIO_REBALANCE_BARS = 42
DRAWDOWN_BRAKE_PCT = 12.0
REDUCED_ROTATION_WEIGHT = 0.35
STAGGERED_GROUPS = {
    "daily_7": tuple(range(0, 42, 6)),
    "half_day_14": tuple(range(0, 42, 3)),
    "all_42": tuple(range(42)),
}


def qualified(
    returns_by_scenario: dict[str, dict[str, float]],
    bull_benchmark: pd.Series,
    worst_drawdown: float,
    maximum_risk: float,
) -> dict[str, float | bool]:
    minimum_excess = min(
        float(
            (
                pd.Series(returns_by_scenario[scenario]).loc[bull_benchmark.index]
                - bull_benchmark
            ).min()
        )
        for scenario in SCENARIOS
    )
    target = all(
        bool(
            (
                pd.Series(returns_by_scenario[scenario]).loc[bull_benchmark.index]
                >= 50.0
            ).all()
        )
        for scenario in SCENARIOS
    ) and minimum_excess >= MINIMUM_BENCHMARK_EXCESS_PCT
    risk_pass = bool(
        maximum_risk <= MAXIMUM_INITIAL_STOP_RISK_PCT
        and worst_drawdown >= -MAXIMUM_EVALUATION_DRAWDOWN_PCT
    )
    return {
        "bull_target_all_cost_scenarios": target,
        "risk_pass_all_evaluations": risk_pass,
        "qualified": target and risk_pass,
        "minimum_excess_vs_btc_pct": minimum_excess,
        "normal_full_return_pct": returns_by_scenario["normal"]["full_period"],
        "double_full_return_pct": returns_by_scenario["double_cost"]["full_period"],
        "triple_full_return_pct": returns_by_scenario["triple_cost"]["full_period"],
        "worst_evaluation_drawdown_pct": worst_drawdown,
        "estimated_maximum_initial_risk_pct": maximum_risk,
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
    benchmark = benchmark_rows(universe)
    benchmark.to_csv(OUTPUT_DIR / "benchmark_annual.csv", index=False)
    bull_benchmark = benchmark[benchmark["btc_bull_market"]].set_index("phase")[
        "BTCUSDT"
    ]
    mature_universe = attach_btc_momentum_regime(
        universe, btc_ema_period=100, momentum_period=240, top_n=1
    )
    phases = phase_ranges() + [("full_period", 0.0, 1.0)]
    mature_cache = {
        (scenario, phase): backtest_portfolio(
            mature_universe,
            mature_sleeves(scenario),
            start_ratio=start,
            end_ratio=end,
        )
        for scenario in SCENARIOS
        for phase, start, end in phases
    }
    base = signal_config(40, 100, 180)
    configurations = {
        offset: replace(base, rebalance_offset_bars=offset) for offset in range(42)
    }
    rotation_cache: dict[tuple[int, str, str], RotationResult] = {}
    for offset, config in configurations.items():
        for scenario, costs in SCENARIOS.items():
            configured = with_costs(config, costs)
            for phase, start, end in phases:
                rotation_cache[(offset, scenario, phase)] = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
    offset_rows: list[dict[str, float | int | bool]] = []
    for offset in range(42):
        returns_by_scenario: dict[str, dict[str, float]] = {}
        worst_drawdown = 0.0
        maximum_risk_value = 0.0
        for scenario, (fee, slippage) in SCENARIOS.items():
            returns: dict[str, float] = {}
            for phase, _, _ in phases:
                rotation = rotation_cache[(offset, scenario, phase)]
                metrics = combine_with_brake(
                    mature_cache[(scenario, phase)].equity,
                    rotation.equity,
                    ROTATION_WEIGHT,
                    PORTFOLIO_REBALANCE_BARS,
                    DRAWDOWN_BRAKE_PCT,
                    REDUCED_ROTATION_WEIGHT,
                    fee,
                    slippage,
                )
                returns[phase] = metrics["return_pct"]
                worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                maximum_risk_value = max(
                    maximum_risk_value,
                    maximum_initial_risk(rotation, ROTATION_WEIGHT),
                )
            returns_by_scenario[scenario] = returns
        offset_rows.append(
            {
                "rebalance_offset_bars": offset,
                **qualified(
                    returns_by_scenario,
                    bull_benchmark,
                    worst_drawdown,
                    maximum_risk_value,
                ),
            }
        )
    staggered_rows: list[dict[str, float | str | bool]] = []
    annual_rows: list[dict[str, float | str]] = []
    for group_name, offsets in STAGGERED_GROUPS.items():
        returns_by_scenario: dict[str, dict[str, float]] = {}
        worst_drawdown = 0.0
        maximum_risk_value = 0.0
        for scenario, (fee, slippage) in SCENARIOS.items():
            returns: dict[str, float] = {}
            for phase, _, _ in phases:
                rotations = [
                    rotation_cache[(offset, scenario, phase)] for offset in offsets
                ]
                metrics = combine_basket(
                    mature_cache[(scenario, phase)].equity,
                    [rotation.equity for rotation in rotations],
                    fee,
                    slippage,
                )
                returns[phase] = metrics["return_pct"]
                worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                maximum_risk_value = max(maximum_risk_value, basket_risk(rotations))
                if phase != "full_period":
                    annual_rows.append(
                        {"group": group_name, "scenario": scenario, "phase": phase, **metrics}
                    )
            returns_by_scenario[scenario] = returns
        staggered_rows.append(
            {
                "group": group_name,
                "offset_count": len(offsets),
                **qualified(
                    returns_by_scenario,
                    bull_benchmark,
                    worst_drawdown,
                    maximum_risk_value,
                ),
            }
        )
    offsets = pd.DataFrame(offset_rows).sort_values(
        ["qualified", "minimum_excess_vs_btc_pct", "worst_evaluation_drawdown_pct"],
        ascending=[False, False, False],
    )
    staggered = pd.DataFrame(staggered_rows).sort_values(
        ["qualified", "minimum_excess_vs_btc_pct", "worst_evaluation_drawdown_pct"],
        ascending=[False, False, False],
    )
    annual = pd.DataFrame(annual_rows)
    offsets.to_csv(OUTPUT_DIR / "single_offset_screen.csv", index=False)
    staggered.to_csv(OUTPUT_DIR / "staggered_summary.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "staggered_annual_metrics.csv", index=False)
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "base_rotation_config": {
            "btc_ema_period": base.btc_ema_period,
            "asset_ema_period": base.asset_ema_period,
            "momentum_period": base.momentum_period,
            "rebalance_bars": base.rebalance_bars,
            "max_initial_risk_pct": base.max_initial_risk_pct,
            "funding_lookback_bars": base.funding_lookback_bars,
            "max_cumulative_funding_rate": base.max_cumulative_funding_rate,
        },
        "single_offset_qualified_count": int(offsets["qualified"].sum()),
        "staggered_groups": {key: list(value) for key, value in STAGGERED_GROUPS.items()},
        "qualification_thresholds": {
            "minimum_excess_vs_btc_in_each_bull_year_pct": MINIMUM_BENCHMARK_EXCESS_PCT,
            "maximum_initial_stop_risk_pct": MAXIMUM_INITIAL_STOP_RISK_PCT,
            "maximum_drawdown_pct_across_annual_and_full_period_tests": MAXIMUM_EVALUATION_DRAWDOWN_PCT,
        },
        "activation_ready": False,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(offsets.head(12).to_string(index=False))
    print(staggered.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
