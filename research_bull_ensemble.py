from __future__ import annotations

from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bull_combined import ACTIVE_MAX_RISK_PCT, mature_sleeves
from research_bull_offense import benchmark_rows, phase_ranges
from research_bull_portfolio_brake import (
    MAXIMUM_EVALUATION_DRAWDOWN_PCT,
    MAXIMUM_INITIAL_STOP_RISK_PCT,
    MINIMUM_BENCHMARK_EXCESS_PCT,
)
from research_bull_portfolio_sensitivity import candidate_configs, name
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


OUTPUT_DIR = Path("outputs/bull_ensemble")
TOTAL_ROTATION_WEIGHT = 0.72
PORTFOLIO_REBALANCE_BARS = 42
DRAWDOWN_BRAKE_PCT = 12.0
REDUCED_ROTATION_WEIGHT = 0.35
COMPONENT_A_WEIGHTS = (0.25, 0.50, 0.75)


def combine_ensemble(
    mature_equity: pd.Series,
    rotation_a_equity: pd.Series,
    rotation_b_equity: pd.Series,
    component_a_weight: float,
    fee_bps: float,
    slippage_bps: float,
    initial_equity: float = 10_000.0,
) -> dict[str, float]:
    equity = pd.concat(
        [
            mature_equity.rename("mature"),
            rotation_a_equity.rename("rotation_a"),
            rotation_b_equity.rename("rotation_b"),
        ],
        axis=1,
    ).ffill().dropna()
    if equity.empty:
        return _metrics(pd.Series(dtype=float), pd.DataFrame(), initial_equity)
    returns = equity.pct_change().fillna(0.0).to_numpy(dtype=float)
    scheduled_rebalances = _rebalance_schedule(
        equity.index, PORTFOLIO_REBALANCE_BARS
    ).to_numpy()
    mature_value = initial_equity * (1.0 - TOTAL_ROTATION_WEIGHT)
    rotation_a_value = initial_equity * TOTAL_ROTATION_WEIGHT * component_a_weight
    rotation_b_value = (
        initial_equity * TOTAL_ROTATION_WEIGHT * (1.0 - component_a_weight)
    )
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    peak = initial_equity
    brake_rebalances = 0
    curve = np.empty(len(equity), dtype=float)
    for index, row in enumerate(returns):
        mature_value *= 1.0 + row[0]
        rotation_a_value *= 1.0 + row[1]
        rotation_b_value *= 1.0 + row[2]
        total = mature_value + rotation_a_value + rotation_b_value
        peak = max(peak, total)
        drawdown_pct = (total / peak - 1.0) * 100.0
        active_rotation_weight = TOTAL_ROTATION_WEIGHT
        if drawdown_pct <= -DRAWDOWN_BRAKE_PCT:
            active_rotation_weight = REDUCED_ROTATION_WEIGHT
        if index and scheduled_rebalances[index]:
            target_mature = total * (1.0 - active_rotation_weight)
            target_a = total * active_rotation_weight * component_a_weight
            target_b = total * active_rotation_weight * (1.0 - component_a_weight)
            turnover = (
                abs(target_mature - mature_value)
                + abs(target_a - rotation_a_value)
                + abs(target_b - rotation_b_value)
            )
            total -= turnover * cost_rate
            mature_value = total * (1.0 - active_rotation_weight)
            rotation_a_value = total * active_rotation_weight * component_a_weight
            rotation_b_value = (
                total * active_rotation_weight * (1.0 - component_a_weight)
            )
            if active_rotation_weight < TOTAL_ROTATION_WEIGHT:
                brake_rebalances += 1
        curve[index] = total
    metrics = _metrics(
        pd.Series(curve, index=equity.index, dtype=float),
        pd.DataFrame(),
        initial_equity,
    )
    metrics["brake_rebalance_count"] = float(brake_rebalances)
    return metrics


def maximum_initial_risk(
    rotation_a: RotationResult,
    rotation_b: RotationResult,
    component_a_weight: float,
) -> float:
    def risk(result: RotationResult) -> float:
        if result.trades.empty:
            return 0.0
        return float(result.trades["initial_risk_pct"].max())

    rotation_risk = component_a_weight * risk(rotation_a) + (
        1.0 - component_a_weight
    ) * risk(rotation_b)
    return (
        TOTAL_ROTATION_WEIGHT * rotation_risk
        + (1.0 - TOTAL_ROTATION_WEIGHT) * ACTIVE_MAX_RISK_PCT
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
    configs = candidate_configs()
    config_by_name = {name(config): config for config in configs}
    rotation_cache: dict[tuple[str, str, str], RotationResult] = {}
    for config in configs:
        for scenario, costs in SCENARIOS.items():
            configured = with_costs(config, costs)
            for phase, start, end in phases:
                rotation_cache[(name(config), scenario, phase)] = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
    rows: list[dict[str, float | str | bool]] = []
    for config_a, config_b in combinations(configs, 2):
        for component_a_weight in COMPONENT_A_WEIGHTS:
            returns_by_scenario: dict[str, dict[str, float]] = {}
            worst_drawdown = 0.0
            maximum_risk = 0.0
            for scenario, (fee, slippage) in SCENARIOS.items():
                returns: dict[str, float] = {}
                for phase, _, _ in phases:
                    rotation_a = rotation_cache[(name(config_a), scenario, phase)]
                    rotation_b = rotation_cache[(name(config_b), scenario, phase)]
                    metrics = combine_ensemble(
                        mature_cache[(scenario, phase)].equity,
                        rotation_a.equity,
                        rotation_b.equity,
                        component_a_weight,
                        fee,
                        slippage,
                    )
                    returns[phase] = metrics["return_pct"]
                    worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                    maximum_risk = max(
                        maximum_risk,
                        maximum_initial_risk(
                            rotation_a, rotation_b, component_a_weight
                        ),
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
                    and minimum_excess >= MINIMUM_BENCHMARK_EXCESS_PCT
                )
                for scenario in SCENARIOS
            )
            risk_pass = bool(
                maximum_risk <= MAXIMUM_INITIAL_STOP_RISK_PCT
                and worst_drawdown >= -MAXIMUM_EVALUATION_DRAWDOWN_PCT
            )
            rows.append(
                {
                    "component_a": name(config_a),
                    "component_b": name(config_b),
                    "component_a_weight": component_a_weight,
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
    config_a = config_by_name[str(selected_row["component_a"])]
    config_b = config_by_name[str(selected_row["component_b"])]
    component_a_weight = float(selected_row["component_a_weight"])
    annual_rows: list[dict[str, float | str]] = []
    full_rows: list[dict[str, float | str]] = []
    for scenario, (fee, slippage) in SCENARIOS.items():
        for phase, start, end in phase_ranges():
            rotation_a = rotation_cache[(name(config_a), scenario, phase)]
            rotation_b = rotation_cache[(name(config_b), scenario, phase)]
            metrics = combine_ensemble(
                mature_cache[(scenario, phase)].equity,
                rotation_a.equity,
                rotation_b.equity,
                component_a_weight,
                fee,
                slippage,
            )
            annual_rows.append(
                {
                    "scenario": scenario,
                    "phase": phase,
                    **metrics,
                    "estimated_maximum_initial_risk_pct": maximum_initial_risk(
                        rotation_a, rotation_b, component_a_weight
                    ),
                }
            )
        rotation_a = rotation_cache[(name(config_a), scenario, "full_period")]
        rotation_b = rotation_cache[(name(config_b), scenario, "full_period")]
        metrics = combine_ensemble(
            mature_cache[(scenario, "full_period")].equity,
            rotation_a.equity,
            rotation_b.equity,
            component_a_weight,
            fee,
            slippage,
        )
        full_rows.append(
            {
                "scenario": scenario,
                **metrics,
                "estimated_maximum_initial_risk_pct": maximum_initial_risk(
                    rotation_a, rotation_b, component_a_weight
                ),
            }
        )
    annual = pd.DataFrame(annual_rows)
    full = pd.DataFrame(full_rows)
    annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    full.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "role": "two-model parameter-diversified shadow bull overlay",
        "benchmark": (
            "BTCUSDT buy and hold with standard entry and exit costs, kept "
            "unchanged while strategy costs are stressed"
        ),
        "candidate_count": len(screen),
        "qualified_count": int(screen["qualified"].sum()),
        "selected_components": {
            "a": asdict(config_a),
            "b": asdict(config_b),
            "a_share_of_rotation": component_a_weight,
        },
        "portfolio_rule": {
            "total_rotation_weight": TOTAL_ROTATION_WEIGHT,
            "portfolio_rebalance_bars": PORTFOLIO_REBALANCE_BARS,
            "drawdown_brake_pct": DRAWDOWN_BRAKE_PCT,
            "reduced_rotation_weight": REDUCED_ROTATION_WEIGHT,
        },
        "qualification_thresholds": {
            "minimum_excess_vs_btc_in_each_bull_year_pct": MINIMUM_BENCHMARK_EXCESS_PCT,
            "maximum_initial_stop_risk_pct": MAXIMUM_INITIAL_STOP_RISK_PCT,
            "maximum_drawdown_pct_across_annual_and_full_period_tests": MAXIMUM_EVALUATION_DRAWDOWN_PCT,
        },
        "activation_ready": False,
        "blocker": (
            "Both component selection and capital sweeps remain in-sample and "
            "require locked forward verification before notification activation."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(screen.head(15).to_string(index=False))
    print(
        annual[
            ["scenario", "phase", "return_pct", "sharpe", "max_drawdown_pct"]
        ].to_string(index=False)
    )
    print(
        full[["scenario", "return_pct", "sharpe", "max_drawdown_pct"]].to_string(
            index=False
        )
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
