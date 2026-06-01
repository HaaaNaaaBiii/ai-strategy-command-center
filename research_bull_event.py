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
from smi_lab.rotation import RotationConfig, RotationResult, backtest_rotation


OUTPUT_DIR = Path("outputs/bull_event")
ROTATION_WEIGHTS = (0.50, 0.60, 0.70)
PORTFOLIO_REBALANCE_BARS = (6, 42)
DRAWDOWN_BRAKES = ((10.0, 0.25), (12.0, 0.35), (15.0, 0.35))


def candidate_configs() -> list[RotationConfig]:
    return [
        RotationConfig(
            btc_ema_period=btc_ema,
            asset_ema_period=asset_ema,
            momentum_period=momentum,
            rebalance_bars=42,
            enter_when_flat=True,
            rotate_on_rebalance=False,
            exposure=1.0,
            max_initial_risk_pct=max_risk,
            funding_lookback_bars=42,
            max_cumulative_funding_rate=funding_threshold,
            stop_atr=3.0,
            tp1_r=3.0,
            tp2_r=8.0,
            tp3_r=20.0,
            tp1_fraction=0.10,
            tp2_fraction=0.10,
        )
        for btc_ema, asset_ema, momentum, max_risk, funding_threshold in product(
            (40, 50, 100),
            (50, 100),
            (90, 180),
            (8.0, 12.0),
            (0.003, 0.006),
        )
    ]


def name(config: RotationConfig) -> str:
    return (
        f"btc{config.btc_ema_period}_asset{config.asset_ema_period}_"
        f"mom{config.momentum_period}_risk{config.max_initial_risk_pct:.0f}_"
        f"fund{config.max_cumulative_funding_rate * 10000:.0f}"
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
    rotation_cache: dict[tuple[str, str, str], RotationResult] = {}
    rows: list[dict[str, float | str | bool]] = []
    for config in candidate_configs():
        for scenario, costs in SCENARIOS.items():
            configured = with_costs(config, costs)
            for phase, start, end in phases:
                rotation_cache[(name(config), scenario, phase)] = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
        for weight, rebalance, brake in product(
            ROTATION_WEIGHTS, PORTFOLIO_REBALANCE_BARS, DRAWDOWN_BRAKES
        ):
            brake_pct, reduced_weight = brake
            returns_by_scenario: dict[str, dict[str, float]] = {}
            worst_drawdown = 0.0
            maximum_risk = 0.0
            for scenario, (fee, slippage) in SCENARIOS.items():
                returns: dict[str, float] = {}
                for phase, _, _ in phases:
                    rotation = rotation_cache[(name(config), scenario, phase)]
                    metrics = combine_with_brake(
                        mature_cache[(scenario, phase)].equity,
                        rotation.equity,
                        weight,
                        rebalance,
                        brake_pct,
                        reduced_weight,
                        fee,
                        slippage,
                    )
                    returns[phase] = metrics["return_pct"]
                    worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                    maximum_risk = max(
                        maximum_risk, maximum_initial_risk(rotation, weight)
                    )
                returns_by_scenario[scenario] = returns
            minimum_excess = min(
                float(
                    (
                        pd.Series(returns_by_scenario[scenario]).loc[
                            bull_benchmark.index
                        ]
                        - bull_benchmark
                    ).min()
                )
                for scenario in SCENARIOS
            )
            target = all(
                bool(
                    (
                        pd.Series(returns_by_scenario[scenario]).loc[
                            bull_benchmark.index
                        ]
                        >= 50.0
                    ).all()
                )
                for scenario in SCENARIOS
            ) and minimum_excess >= MINIMUM_BENCHMARK_EXCESS_PCT
            risk_pass = bool(
                maximum_risk <= MAXIMUM_INITIAL_STOP_RISK_PCT
                and worst_drawdown >= -MAXIMUM_EVALUATION_DRAWDOWN_PCT
            )
            rows.append(
                {
                    "candidate": name(config),
                    **asdict(config),
                    "rotation_weight": weight,
                    "portfolio_rebalance_bars": rebalance,
                    "drawdown_brake_pct": brake_pct,
                    "reduced_rotation_weight": reduced_weight,
                    "bull_target_all_cost_scenarios": target,
                    "risk_pass_all_evaluations": risk_pass,
                    "qualified": target and risk_pass,
                    "minimum_excess_vs_btc_pct": minimum_excess,
                    "normal_full_return_pct": returns_by_scenario["normal"][
                        "full_period"
                    ],
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
    screen = pd.DataFrame(rows).sort_values(
        [
            "qualified",
            "bull_target_all_cost_scenarios",
            "worst_evaluation_drawdown_pct",
            "minimum_excess_vs_btc_pct",
        ],
        ascending=[False, False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_row = screen.iloc[0]
    selected = next(
        config for config in candidate_configs() if name(config) == selected_row["candidate"]
    )
    weight = float(selected_row["rotation_weight"])
    rebalance = int(selected_row["portfolio_rebalance_bars"])
    brake_pct = float(selected_row["drawdown_brake_pct"])
    reduced_weight = float(selected_row["reduced_rotation_weight"])
    annual_rows: list[dict[str, float | str]] = []
    full_rows: list[dict[str, float | str]] = []
    for scenario, (fee, slippage) in SCENARIOS.items():
        configured = with_costs(selected, (fee, slippage))
        for phase, start, end in phase_ranges():
            rotation = backtest_rotation(
                universe, configured, start_ratio=start, end_ratio=end
            )
            metrics = combine_with_brake(
                mature_cache[(scenario, phase)].equity,
                rotation.equity,
                weight,
                rebalance,
                brake_pct,
                reduced_weight,
                fee,
                slippage,
            )
            annual_rows.append({"scenario": scenario, "phase": phase, **metrics})
        rotation = backtest_rotation(universe, configured)
        metrics = combine_with_brake(
            mature_cache[(scenario, "full_period")].equity,
            rotation.equity,
            weight,
            rebalance,
            brake_pct,
            reduced_weight,
            fee,
            slippage,
        )
        full_rows.append({"scenario": scenario, **metrics})
        rotation.trades.to_csv(
            OUTPUT_DIR / f"{scenario}_selected_rotation_trades.csv", index=False
        )
    annual = pd.DataFrame(annual_rows)
    full = pd.DataFrame(full_rows)
    annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    full.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "role": "event-driven, time-phase-independent shadow bull overlay",
        "benchmark": (
            "BTCUSDT buy and hold with standard entry and exit costs, kept "
            "unchanged while strategy costs are stressed"
        ),
        "candidate_count": len(screen),
        "qualified_count": int(screen["qualified"].sum()),
        "selected_config": asdict(selected),
        "selected_portfolio_rule": {
            "rotation_weight": weight,
            "portfolio_rebalance_bars": rebalance,
            "drawdown_brake_pct": brake_pct,
            "reduced_rotation_weight": reduced_weight,
        },
        "qualification_thresholds": {
            "minimum_excess_vs_btc_in_each_bull_year_pct": MINIMUM_BENCHMARK_EXCESS_PCT,
            "maximum_initial_stop_risk_pct": MAXIMUM_INITIAL_STOP_RISK_PCT,
            "maximum_drawdown_pct_across_annual_and_full_period_tests": MAXIMUM_EVALUATION_DRAWDOWN_PCT,
        },
        "activation_ready": False,
        "blocker": (
            "Any selected configuration remains in-sample and portfolio capital "
            "sweeps require locked forward execution verification."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(screen.head(12).to_string(index=False))
    print(
        annual[["scenario", "phase", "return_pct", "sharpe", "max_drawdown_pct"]]
        .to_string(index=False)
    )
    print(full[["scenario", "return_pct", "sharpe", "max_drawdown_pct"]].to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
