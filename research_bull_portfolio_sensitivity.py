from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import pandas as pd

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
from smi_lab.rotation import RotationConfig, backtest_rotation


OUTPUT_DIR = Path("outputs/bull_portfolio_brake")
ROTATION_WEIGHT = 0.72
PORTFOLIO_REBALANCE_BARS = 42
DRAWDOWN_BRAKE_PCT = 12.0
REDUCED_ROTATION_WEIGHT = 0.35


def candidate_configs() -> list[RotationConfig]:
    return [
        RotationConfig(
            btc_ema_period=btc_ema,
            asset_ema_period=asset_ema,
            momentum_period=momentum,
            rebalance_bars=42,
            exposure=1.0,
            max_initial_risk_pct=12.0,
            funding_lookback_bars=42,
            max_cumulative_funding_rate=0.006,
            stop_atr=3.0,
            tp1_r=3.0,
            tp2_r=8.0,
            tp3_r=20.0,
            tp1_fraction=0.10,
            tp2_fraction=0.10,
        )
        for btc_ema, asset_ema, momentum in product(
            (35, 40, 45),
            (90, 100, 110),
            (165, 180, 195),
        )
    ]


def name(config: RotationConfig) -> str:
    return (
        f"btc{config.btc_ema_period}_asset{config.asset_ema_period}_"
        f"mom{config.momentum_period}"
    )


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
    rows: list[dict[str, float | str | bool]] = []
    for config in candidate_configs():
        returns_by_scenario: dict[str, dict[str, float]] = {}
        worst_drawdown = 0.0
        maximum_risk = 0.0
        for scenario, (fee, slippage) in SCENARIOS.items():
            configured = with_costs(config, (fee, slippage))
            scenario_returns: dict[str, float] = {}
            for phase, start, end in phases:
                rotation = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
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
                scenario_returns[phase] = metrics["return_pct"]
                worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                maximum_risk = max(
                    maximum_risk, maximum_initial_risk(rotation, ROTATION_WEIGHT)
                )
            returns_by_scenario[scenario] = scenario_returns
        excess = {
            scenario: (
                pd.Series(scenario_returns).loc[bull_benchmark.index] - bull_benchmark
            )
            for scenario, scenario_returns in returns_by_scenario.items()
        }
        minimum_excess = min(float(values.min()) for values in excess.values())
        target = all(
            bool(
                (
                    pd.Series(returns_by_scenario[scenario]).loc[bull_benchmark.index]
                    >= 50.0
                ).all()
                and (values >= MINIMUM_BENCHMARK_EXCESS_PCT).all()
            )
            for scenario, values in excess.items()
        )
        risk_pass = bool(
            maximum_risk <= MAXIMUM_INITIAL_STOP_RISK_PCT
            and worst_drawdown >= -MAXIMUM_EVALUATION_DRAWDOWN_PCT
        )
        rows.append(
            {
                "candidate": name(config),
                **asdict(config),
                "bull_target_all_cost_scenarios": target,
                "risk_pass_all_evaluations": risk_pass,
                "qualified": target and risk_pass,
                "minimum_excess_vs_btc_pct": minimum_excess,
                "normal_full_return_pct": returns_by_scenario["normal"]["full_period"],
                "double_full_return_pct": returns_by_scenario["double_cost"][
                    "full_period"
                ],
                "triple_full_return_pct": returns_by_scenario["triple_cost"][
                    "full_period"
                ],
                "worst_evaluation_drawdown_pct": worst_drawdown,
                "estimated_maximum_initial_risk_pct": maximum_risk,
            }
        )
    results = pd.DataFrame(rows).sort_values(
        [
            "qualified",
            "bull_target_all_cost_scenarios",
            "worst_evaluation_drawdown_pct",
            "minimum_excess_vs_btc_pct",
        ],
        ascending=[False, False, False, False],
    )
    results.to_csv(OUTPUT_DIR / "signal_parameter_sensitivity.csv", index=False)
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "purpose": "fixed-allocation signal-period sensitivity around selected rule",
        "candidate_count": len(results),
        "qualified_count": int(results["qualified"].sum()),
        "fixed_portfolio_rule": {
            "rotation_weight": ROTATION_WEIGHT,
            "portfolio_rebalance_bars": PORTFOLIO_REBALANCE_BARS,
            "drawdown_brake_pct": DRAWDOWN_BRAKE_PCT,
            "reduced_rotation_weight": REDUCED_ROTATION_WEIGHT,
        },
        "qualification_thresholds": {
            "minimum_excess_vs_btc_in_each_bull_year_pct": MINIMUM_BENCHMARK_EXCESS_PCT,
            "maximum_initial_stop_risk_pct": MAXIMUM_INITIAL_STOP_RISK_PCT,
            "maximum_drawdown_pct_across_annual_and_full_period_tests": MAXIMUM_EVALUATION_DRAWDOWN_PCT,
        },
    }
    (OUTPUT_DIR / "signal_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
