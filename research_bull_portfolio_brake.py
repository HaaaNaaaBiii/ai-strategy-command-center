from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bull_combined import ACTIVE_MAX_RISK_PCT, mature_sleeves
from research_bull_offense import benchmark_rows, phase_ranges
from research_bull_rotation import SCENARIOS, with_costs
from smi_lab.backtest import _metrics, backtest_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.regime import attach_btc_momentum_regime
from smi_lab.rotation import (
    RotationConfig,
    RotationResult,
    _rebalance_schedule,
    backtest_rotation,
)


OUTPUT_DIR = Path("outputs/bull_portfolio_brake")
MINIMUM_BENCHMARK_EXCESS_PCT = 5.0
MAXIMUM_INITIAL_STOP_RISK_PCT = 15.0
MAXIMUM_EVALUATION_DRAWDOWN_PCT = 30.0
ROTATION_WEIGHTS = (0.70, 0.71, 0.72, 0.73)
PORTFOLIO_REBALANCE_BARS = (18, 42, 90)
DRAWDOWN_BRAKES = (
    (0.0, 0.0),
    (8.0, 0.25),
    (10.0, 0.25),
    (10.0, 0.35),
    (12.0, 0.35),
    (15.0, 0.35),
)


def candidate_configs() -> list[RotationConfig]:
    configs: list[RotationConfig] = []
    for max_risk, funding_threshold in product(
        (12.0, 15.0, 18.0),
        (0.0, 0.004, 0.005, 0.006, 0.007, 0.008),
    ):
        configs.append(
            RotationConfig(
                btc_ema_period=40,
                asset_ema_period=100,
                momentum_period=180,
                rebalance_bars=42,
                exposure=1.0,
                max_initial_risk_pct=max_risk,
                funding_lookback_bars=42 if funding_threshold else 0,
                max_cumulative_funding_rate=funding_threshold,
                stop_atr=3.0,
                tp1_r=3.0,
                tp2_r=8.0,
                tp3_r=20.0,
                tp1_fraction=0.10,
                tp2_fraction=0.10,
            )
        )
    return configs


def name(config: RotationConfig) -> str:
    return (
        f"btc{config.btc_ema_period}_asset{config.asset_ema_period}_"
        f"mom{config.momentum_period}_risk{config.max_initial_risk_pct:.0f}_"
        f"fund{config.max_cumulative_funding_rate * 10000:.0f}"
    )


def combine_with_brake(
    mature_equity: pd.Series,
    rotation_equity: pd.Series,
    rotation_weight: float,
    rebalance_bars: int,
    drawdown_brake_pct: float,
    reduced_rotation_weight: float,
    fee_bps: float,
    slippage_bps: float,
    initial_equity: float = 10_000.0,
) -> dict[str, float]:
    """Blend sleeves and cut rotation allocation at rebalances during drawdowns."""
    equity = pd.concat(
        [mature_equity.rename("mature"), rotation_equity.rename("rotation")],
        axis=1,
    ).ffill().dropna()
    if equity.empty:
        return _metrics(pd.Series(dtype=float), pd.DataFrame(), initial_equity)
    if drawdown_brake_pct and not 0 <= reduced_rotation_weight < rotation_weight:
        raise ValueError("Reduced rotation weight must be below its target weight.")
    returns = equity.pct_change().fillna(0.0).to_numpy(dtype=float)
    scheduled_rebalances = _rebalance_schedule(equity.index, rebalance_bars).to_numpy()
    mature_value = initial_equity * (1.0 - rotation_weight)
    rotation_value = initial_equity * rotation_weight
    rebalance_cost_rate = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    peak = initial_equity
    brake_rebalances = 0
    curve = np.empty(len(equity), dtype=float)
    for index, row in enumerate(returns):
        mature_value *= 1.0 + row[0]
        rotation_value *= 1.0 + row[1]
        total = mature_value + rotation_value
        peak = max(peak, total)
        drawdown_pct = (total / peak - 1.0) * 100.0
        active_weight = rotation_weight
        if drawdown_brake_pct and drawdown_pct <= -drawdown_brake_pct:
            active_weight = reduced_rotation_weight
        if index and scheduled_rebalances[index]:
            target_rotation = total * active_weight
            cost = abs(target_rotation - rotation_value) * rebalance_cost_rate
            total -= cost
            mature_value = total * (1.0 - active_weight)
            rotation_value = total * active_weight
            if active_weight < rotation_weight:
                brake_rebalances += 1
        curve[index] = total
    metrics = _metrics(
        pd.Series(curve, index=equity.index, dtype=float),
        pd.DataFrame(),
        initial_equity,
    )
    metrics["brake_rebalance_count"] = float(brake_rebalances)
    return metrics


def maximum_initial_risk(rotation: RotationResult, rotation_weight: float) -> float:
    rotation_risk = (
        float(rotation.trades["initial_risk_pct"].max())
        if not rotation.trades.empty
        else 0.0
    )
    return rotation_weight * rotation_risk + (1.0 - rotation_weight) * ACTIVE_MAX_RISK_PCT


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
    scenarios = tuple(SCENARIOS)
    mature_cache = {
        (scenario, phase): backtest_portfolio(
            mature_universe,
            mature_sleeves(scenario),
            start_ratio=start,
            end_ratio=end,
        )
        for scenario in scenarios
        for phase, start, end in phases
    }
    rotation_cache: dict[tuple[str, str, str], RotationResult] = {}
    rows: list[dict[str, float | str | bool]] = []
    for config in candidate_configs():
        for scenario in scenarios:
            configured = with_costs(config, SCENARIOS[scenario])
            for phase, start, end in phases:
                rotation_cache[(name(config), scenario, phase)] = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
        for weight, portfolio_rebalance, brake in product(
            ROTATION_WEIGHTS,
            PORTFOLIO_REBALANCE_BARS,
            DRAWDOWN_BRAKES,
        ):
            drawdown_brake_pct, reduced_weight = brake
            scenario_returns: dict[str, dict[str, float]] = {}
            worst_drawdown = 0.0
            maximum_risk = 0.0
            for scenario in scenarios:
                fee, slippage = SCENARIOS[scenario]
                returns: dict[str, float] = {}
                for phase, _, _ in phases:
                    rotation = rotation_cache[(name(config), scenario, phase)]
                    metrics = combine_with_brake(
                        mature_cache[(scenario, phase)].equity,
                        rotation.equity,
                        weight,
                        portfolio_rebalance,
                        drawdown_brake_pct,
                        reduced_weight,
                        fee,
                        slippage,
                    )
                    returns[phase] = metrics["return_pct"]
                    worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                    maximum_risk = max(
                        maximum_risk, maximum_initial_risk(rotation, weight)
                    )
                scenario_returns[scenario] = returns
            minimum_excess = min(
                float(
                    (
                        pd.Series(scenario_returns[scenario]).loc[bull_benchmark.index]
                        - bull_benchmark
                    ).min()
                )
                for scenario in scenarios
            )
            target = all(
                bool(
                    (
                        (
                            pd.Series(scenario_returns[scenario]).loc[bull_benchmark.index]
                            >= 50.0
                        )
                        & (
                            pd.Series(scenario_returns[scenario]).loc[bull_benchmark.index]
                            >= bull_benchmark + MINIMUM_BENCHMARK_EXCESS_PCT
                        )
                    ).all()
                )
                for scenario in scenarios
            )
            risk_pass = bool(
                maximum_risk <= MAXIMUM_INITIAL_STOP_RISK_PCT
                and worst_drawdown >= -MAXIMUM_EVALUATION_DRAWDOWN_PCT
            )
            rows.append(
                {
                    "candidate": name(config),
                    **asdict(config),
                    "rotation_weight": weight,
                    "portfolio_rebalance_bars": portfolio_rebalance,
                    "drawdown_brake_pct": drawdown_brake_pct,
                    "reduced_rotation_weight": reduced_weight,
                    "bull_target_all_cost_scenarios": target,
                    "risk_pass_all_evaluations": risk_pass,
                    "qualified": target and risk_pass,
                    "minimum_excess_vs_btc_pct": minimum_excess,
                    "normal_full_return_pct": scenario_returns["normal"]["full_period"],
                    "double_full_return_pct": scenario_returns["double_cost"][
                        "full_period"
                    ],
                    "triple_full_return_pct": scenario_returns["triple_cost"][
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
            annual_rows.append(
                {
                    "scenario": scenario,
                    "phase": phase,
                    **metrics,
                    "estimated_maximum_initial_risk_pct": maximum_initial_risk(
                        rotation, weight
                    ),
                }
            )
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
        full_rows.append(
            {
                "scenario": scenario,
                **metrics,
                "estimated_maximum_initial_risk_pct": maximum_initial_risk(
                    rotation, weight
                ),
            }
        )
        rotation.trades.to_csv(
            OUTPUT_DIR / f"{scenario}_selected_rotation_trades.csv", index=False
        )
    annual = pd.DataFrame(annual_rows)
    full = pd.DataFrame(full_rows)
    annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    full.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "role": "portfolio-drawdown-braked shadow bull overlay",
        "benchmark": (
            "BTCUSDT buy and hold with standard entry and exit costs, kept "
            "unchanged while strategy costs are stressed"
        ),
        "bull_market_definition": "annual segment where BTC benchmark return is positive",
        "selection_scenarios": list(SCENARIOS),
        "candidate_count": len(screen),
        "qualified_count": int(screen["qualified"].sum()),
        "selected_candidate": name(selected),
        "selected_config": asdict(selected),
        "selected_rotation_weight": weight,
        "selected_portfolio_rebalance_bars": rebalance,
        "selected_drawdown_brake_pct": brake_pct,
        "selected_reduced_rotation_weight": reduced_weight,
        "risk_caps": {
            "minimum_excess_vs_btc_in_each_bull_year_pct": MINIMUM_BENCHMARK_EXCESS_PCT,
            "maximum_initial_stop_risk_pct": MAXIMUM_INITIAL_STOP_RISK_PCT,
            "maximum_drawdown_pct_across_annual_and_full_period_tests": MAXIMUM_EVALUATION_DRAWDOWN_PCT,
        },
        "rebalance_assumption": (
            "At scheduled closes, sleeve capital is swept to the applicable target "
            "weight with a conservative two-sided fee and slippage charge."
        ),
        "activation_ready": False,
        "blocker": (
            "This rule was selected on the observed five-year sample; capital "
            "sweeps and cost stress still require locked forward verification."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(screen.head(12).to_string(index=False))
    print(
        annual[
            [
                "scenario",
                "phase",
                "return_pct",
                "sharpe",
                "max_drawdown_pct",
                "brake_rebalance_count",
                "estimated_maximum_initial_risk_pct",
            ]
        ].to_string(index=False)
    )
    print(
        full[
            [
                "scenario",
                "return_pct",
                "sharpe",
                "max_drawdown_pct",
                "brake_rebalance_count",
                "estimated_maximum_initial_risk_pct",
            ]
        ].to_string(index=False)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
