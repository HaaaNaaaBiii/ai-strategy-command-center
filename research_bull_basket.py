from __future__ import annotations

from itertools import product
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


OUTPUT_DIR = Path("outputs/bull_basket")
TOTAL_ROTATION_WEIGHT = 0.72
PORTFOLIO_REBALANCE_BARS = 42
DRAWDOWN_BRAKE_PCT = 12.0
REDUCED_ROTATION_WEIGHT = 0.35


def signal_config(btc_ema: int, asset_ema: int, momentum: int) -> RotationConfig:
    return RotationConfig(
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


def baskets() -> dict[str, list[RotationConfig]]:
    return {
        "core_6": [
            signal_config(40, asset_ema, momentum)
            for asset_ema, momentum in product((90, 100, 110), (165, 180))
        ],
        "expanded_9": [
            signal_config(40, asset_ema, momentum)
            for asset_ema, momentum in product((90, 100, 110), (165, 180, 195))
        ],
        "broad_18": [
            signal_config(btc_ema, asset_ema, momentum)
            for btc_ema, asset_ema, momentum in product(
                (35, 40, 45), (90, 100, 110), (165, 180)
            )
        ],
    }


def name(config: RotationConfig) -> str:
    return (
        f"btc{config.btc_ema_period}_asset{config.asset_ema_period}_"
        f"mom{config.momentum_period}"
    )


def combine_basket(
    mature_equity: pd.Series,
    rotation_equities: list[pd.Series],
    fee_bps: float,
    slippage_bps: float,
    initial_equity: float = 10_000.0,
) -> dict[str, float]:
    frames = [mature_equity.rename("mature")]
    frames.extend(
        equity.rename(f"rotation_{index}")
        for index, equity in enumerate(rotation_equities)
    )
    equity = pd.concat(frames, axis=1).ffill().dropna()
    if equity.empty:
        return _metrics(pd.Series(dtype=float), pd.DataFrame(), initial_equity)
    returns = equity.pct_change().fillna(0.0).to_numpy(dtype=float)
    scheduled_rebalances = _rebalance_schedule(
        equity.index, PORTFOLIO_REBALANCE_BARS
    ).to_numpy()
    values = np.empty(len(rotation_equities) + 1, dtype=float)
    values[0] = initial_equity * (1.0 - TOTAL_ROTATION_WEIGHT)
    values[1:] = initial_equity * TOTAL_ROTATION_WEIGHT / len(rotation_equities)
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    peak = initial_equity
    brake_rebalances = 0
    curve = np.empty(len(equity), dtype=float)
    for index, row in enumerate(returns):
        values *= 1.0 + row
        total = float(values.sum())
        peak = max(peak, total)
        drawdown_pct = (total / peak - 1.0) * 100.0
        active_rotation_weight = TOTAL_ROTATION_WEIGHT
        if drawdown_pct <= -DRAWDOWN_BRAKE_PCT:
            active_rotation_weight = REDUCED_ROTATION_WEIGHT
        if index and scheduled_rebalances[index]:
            targets = np.empty_like(values)
            targets[0] = total * (1.0 - active_rotation_weight)
            targets[1:] = (
                total * active_rotation_weight / len(rotation_equities)
            )
            total -= float(np.abs(targets - values).sum()) * cost_rate
            targets[0] = total * (1.0 - active_rotation_weight)
            targets[1:] = (
                total * active_rotation_weight / len(rotation_equities)
            )
            values = targets
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


def maximum_initial_risk(results: list[RotationResult]) -> float:
    risks = [
        float(result.trades["initial_risk_pct"].max())
        if not result.trades.empty
        else 0.0
        for result in results
    ]
    rotation_risk = float(np.mean(risks))
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
    basket_definitions = baskets()
    unique_configs = {
        name(config): config
        for configurations in basket_definitions.values()
        for config in configurations
    }
    rotation_cache: dict[tuple[str, str, str], RotationResult] = {}
    for config_name, config in unique_configs.items():
        for scenario, costs in SCENARIOS.items():
            configured = with_costs(config, costs)
            for phase, start, end in phases:
                rotation_cache[(config_name, scenario, phase)] = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
    summary_rows: list[dict[str, float | str | bool]] = []
    annual_rows: list[dict[str, float | str]] = []
    full_rows: list[dict[str, float | str]] = []
    for basket_name, configurations in basket_definitions.items():
        returns_by_scenario: dict[str, dict[str, float]] = {}
        worst_drawdown = 0.0
        maximum_risk = 0.0
        for scenario, (fee, slippage) in SCENARIOS.items():
            scenario_returns: dict[str, float] = {}
            for phase, _, _ in phases:
                rotations = [
                    rotation_cache[(name(config), scenario, phase)]
                    for config in configurations
                ]
                metrics = combine_basket(
                    mature_cache[(scenario, phase)].equity,
                    [rotation.equity for rotation in rotations],
                    fee,
                    slippage,
                )
                scenario_returns[phase] = metrics["return_pct"]
                worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                maximum_risk = max(maximum_risk, maximum_initial_risk(rotations))
                row = {
                    "basket": basket_name,
                    "scenario": scenario,
                    "phase": phase,
                    **metrics,
                    "estimated_maximum_initial_risk_pct": maximum_initial_risk(
                        rotations
                    ),
                }
                if phase == "full_period":
                    full_rows.append(row)
                else:
                    annual_rows.append(row)
            returns_by_scenario[scenario] = scenario_returns
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
        summary_rows.append(
            {
                "basket": basket_name,
                "model_count": len(configurations),
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
    summary = pd.DataFrame(summary_rows).sort_values(
        ["qualified", "worst_evaluation_drawdown_pct", "minimum_excess_vs_btc_pct"],
        ascending=[False, False, False],
    )
    annual = pd.DataFrame(annual_rows)
    full = pd.DataFrame(full_rows)
    summary.to_csv(OUTPUT_DIR / "basket_summary.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "annual_metrics.csv", index=False)
    full.to_csv(OUTPUT_DIR / "full_period_metrics.csv", index=False)
    selected_name = str(summary.iloc[0]["basket"])
    selected_annual = annual[annual["basket"] == selected_name]
    selected_full = full[full["basket"] == selected_name]
    selected_annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    selected_full.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "role": "equal-weight parameter-basket shadow bull overlay",
        "benchmark": (
            "BTCUSDT buy and hold with standard entry and exit costs, kept "
            "unchanged while strategy costs are stressed"
        ),
        "basket_definitions": {
            key: [name(config) for config in configurations]
            for key, configurations in basket_definitions.items()
        },
        "selected_basket": selected_name,
        "qualification_thresholds": {
            "minimum_excess_vs_btc_in_each_bull_year_pct": MINIMUM_BENCHMARK_EXCESS_PCT,
            "maximum_initial_stop_risk_pct": MAXIMUM_INITIAL_STOP_RISK_PCT,
            "maximum_drawdown_pct_across_annual_and_full_period_tests": MAXIMUM_EVALUATION_DRAWDOWN_PCT,
        },
        "portfolio_rule": {
            "total_rotation_weight": TOTAL_ROTATION_WEIGHT,
            "portfolio_rebalance_bars": PORTFOLIO_REBALANCE_BARS,
            "drawdown_brake_pct": DRAWDOWN_BRAKE_PCT,
            "reduced_rotation_weight": REDUCED_ROTATION_WEIGHT,
        },
        "activation_ready": False,
        "blocker": (
            "Basket definitions and capital-sweep implementation require locked "
            "forward observation before notification activation."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(
        selected_annual[
            ["scenario", "phase", "return_pct", "sharpe", "max_drawdown_pct"]
        ].to_string(index=False)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
