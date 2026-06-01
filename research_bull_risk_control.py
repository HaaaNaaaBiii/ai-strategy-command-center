from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import pandas as pd

from research_bull_combined import (
    ACTIVE_MAX_RISK_PCT,
    mature_sleeves,
)
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


OUTPUT_DIR = Path("outputs/bull_risk_control")
ROTATION_WEIGHTS = (0.60, 0.65, 0.70)
PORTFOLIO_REBALANCE_BARS = (42, 90)


def candidate_configs() -> list[RotationConfig]:
    configs: list[RotationConfig] = []
    for (
        btc_ema,
        asset_ema,
        momentum,
        max_risk,
        funding_threshold,
        drawdown_control,
    ) in product(
        (40, 50),
        (50, 100),
        (120, 180),
        (0.0, 15.0, 20.0),
        (0.0, 0.003, 0.006),
        (False, True),
    ):
        configs.append(
            RotationConfig(
                btc_ema_period=btc_ema,
                asset_ema_period=asset_ema,
                momentum_period=momentum,
                rebalance_bars=42,
                exposure=1.0,
                max_initial_risk_pct=max_risk,
                drawdown_reduce_at_pct=10.0 if drawdown_control else 0.0,
                drawdown_exposure_multiplier=0.5 if drawdown_control else 1.0,
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
        f"fund{config.max_cumulative_funding_rate * 10000:.0f}_"
        f"dd{config.drawdown_reduce_at_pct:.0f}"
    )


def combine_rebalanced(
    mature_equity: pd.Series,
    rotation_equity: pd.Series,
    rotation_weight: float,
    rebalance_bars: int,
    fee_bps: float,
    slippage_bps: float,
    initial_equity: float = 10_000.0,
) -> dict[str, float]:
    """Blend sleeves with periodic capital sweeps and conservative sweep costs."""
    equity = pd.concat(
        [mature_equity.rename("mature"), rotation_equity.rename("rotation")],
        axis=1,
    ).ffill().dropna()
    if equity.empty:
        return _metrics(pd.Series(dtype=float), pd.DataFrame(), initial_equity)
    returns = equity.pct_change().fillna(0.0)
    scheduled_rebalances = _rebalance_schedule(equity.index, rebalance_bars)
    mature_value = initial_equity * (1.0 - rotation_weight)
    rotation_value = initial_equity * rotation_weight
    rebalance_cost_rate = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    curve: dict[pd.Timestamp, float] = {}
    for index, (timestamp, row) in enumerate(returns.iterrows()):
        mature_value *= 1.0 + float(row["mature"])
        rotation_value *= 1.0 + float(row["rotation"])
        total = mature_value + rotation_value
        if index and bool(scheduled_rebalances.iloc[index]):
            target_rotation = total * rotation_weight
            cost = abs(target_rotation - rotation_value) * rebalance_cost_rate
            total -= cost
            mature_value = total * (1.0 - rotation_weight)
            rotation_value = total * rotation_weight
        curve[timestamp] = total
    return _metrics(pd.Series(curve, dtype=float), pd.DataFrame(), initial_equity)


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
    bull_benchmark = benchmark[benchmark["btc_bull_market"]].set_index("phase")["BTCUSDT"]
    mature_universe = attach_btc_momentum_regime(
        universe, btc_ema_period=100, momentum_period=240, top_n=1
    )
    phase_list = phase_ranges() + [("full_period", 0.0, 1.0)]
    scenarios = ("normal", "double_cost")
    mature_cache = {
        (scenario, phase): backtest_portfolio(
            mature_universe,
            mature_sleeves(scenario),
            start_ratio=start,
            end_ratio=end,
        )
        for scenario in scenarios
        for phase, start, end in phase_list
    }
    rotation_cache: dict[tuple[str, str, str], RotationResult] = {}
    rows: list[dict[str, float | str | bool]] = []
    for config in candidate_configs():
        for scenario in scenarios:
            configured = with_costs(config, SCENARIOS[scenario])
            for phase, start, end in phase_list:
                rotation_cache[(name(config), scenario, phase)] = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
        for weight in ROTATION_WEIGHTS:
            for portfolio_rebalance in PORTFOLIO_REBALANCE_BARS:
                returns: dict[str, dict[str, float]] = {}
                worst_full_drawdown = 0.0
                worst_evaluation_drawdown = 0.0
                max_risk = 0.0
                for scenario in scenarios:
                    fee, slippage = SCENARIOS[scenario]
                    scenario_returns: dict[str, float] = {}
                    for phase, _, _ in phase_list:
                        mature = mature_cache[(scenario, phase)]
                        rotation = rotation_cache[(name(config), scenario, phase)]
                        metrics = combine_rebalanced(
                            mature.equity,
                            rotation.equity,
                            weight,
                            portfolio_rebalance,
                            fee,
                            slippage,
                        )
                        scenario_returns[phase] = metrics["return_pct"]
                        worst_evaluation_drawdown = min(
                            worst_evaluation_drawdown, metrics["max_drawdown_pct"]
                        )
                        max_risk = max(max_risk, maximum_initial_risk(rotation, weight))
                        if phase == "full_period":
                            worst_full_drawdown = min(
                                worst_full_drawdown, metrics["max_drawdown_pct"]
                            )
                    returns[scenario] = scenario_returns
                normal_bull = pd.Series(returns["normal"]).loc[bull_benchmark.index]
                double_bull = pd.Series(returns["double_cost"]).loc[bull_benchmark.index]
                target = bool(
                    ((normal_bull >= 50.0) & (normal_bull > bull_benchmark)).all()
                    and ((double_bull >= 50.0) & (double_bull > bull_benchmark)).all()
                )
                risk_pass = bool(
                    max_risk <= 25.0 and worst_evaluation_drawdown >= -40.0
                )
                rows.append(
                    {
                        "candidate": name(config),
                        **asdict(config),
                        "rotation_weight": weight,
                        "portfolio_rebalance_bars": portfolio_rebalance,
                        "bull_target": target,
                        "risk_pass": risk_pass,
                        "qualified": target and risk_pass,
                        "minimum_double_bull_return_pct": float(double_bull.min()),
                        "minimum_double_excess_vs_btc_pct": float(
                            (double_bull - bull_benchmark).min()
                        ),
                        "normal_full_return_pct": returns["normal"]["full_period"],
                        "double_full_return_pct": returns["double_cost"]["full_period"],
                        "worst_full_period_drawdown_pct": worst_full_drawdown,
                        "worst_evaluation_drawdown_pct": worst_evaluation_drawdown,
                        "estimated_maximum_initial_risk_pct": max_risk,
                    }
                )
    screen = pd.DataFrame(rows).sort_values(
        [
            "qualified",
            "bull_target",
            "worst_evaluation_drawdown_pct",
            "minimum_double_excess_vs_btc_pct",
        ],
        ascending=[False, False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_row = screen.iloc[0]
    selected = next(
        config for config in candidate_configs() if name(config) == selected_row["candidate"]
    )
    selected_weight = float(selected_row["rotation_weight"])
    selected_rebalance = int(selected_row["portfolio_rebalance_bars"])
    annual_rows: list[dict[str, float | str]] = []
    full_rows: list[dict[str, float | str]] = []
    for scenario, (fee, slippage) in SCENARIOS.items():
        configured = with_costs(selected, (fee, slippage))
        for phase, start, end in phase_ranges():
            mature = backtest_portfolio(
                mature_universe, mature_sleeves(scenario), start_ratio=start, end_ratio=end
            )
            rotation = backtest_rotation(
                universe, configured, start_ratio=start, end_ratio=end
            )
            annual_rows.append(
                {
                    "scenario": scenario,
                    "phase": phase,
                    **combine_rebalanced(
                        mature.equity,
                        rotation.equity,
                        selected_weight,
                        selected_rebalance,
                        fee,
                        slippage,
                    ),
                    "estimated_maximum_initial_risk_pct": maximum_initial_risk(
                        rotation, selected_weight
                    ),
                }
            )
        mature = backtest_portfolio(mature_universe, mature_sleeves(scenario))
        rotation = backtest_rotation(universe, configured)
        full_rows.append(
            {
                "scenario": scenario,
                **combine_rebalanced(
                    mature.equity,
                    rotation.equity,
                    selected_weight,
                    selected_rebalance,
                    fee,
                    slippage,
                ),
                "estimated_maximum_initial_risk_pct": maximum_initial_risk(
                    rotation, selected_weight
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
    triple = annual[annual["scenario"] == "triple_cost"].set_index("phase")
    triple_full = full[full["scenario"] == "triple_cost"].iloc[0]
    triple_worst_drawdown = min(
        float(triple["max_drawdown_pct"].min()),
        float(triple_full["max_drawdown_pct"]),
    )
    triple_max_risk = max(
        float(triple["estimated_maximum_initial_risk_pct"].max()),
        float(triple_full["estimated_maximum_initial_risk_pct"]),
    )
    checks = {
        "qualified_candidate_exists": bool(screen["qualified"].any()),
        "triple_cost_bull_target": bool(
            (
                (triple.loc[bull_benchmark.index, "return_pct"] >= 50.0)
                & (triple.loc[bull_benchmark.index, "return_pct"] > bull_benchmark)
            ).all()
        ),
        "triple_cost_risk_pass": bool(
            triple_worst_drawdown >= -40.0 and triple_max_risk <= 25.0
        ),
    }
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "role": "risk-managed shadow bull overlay with periodically rebalanced sleeve capital",
        "benchmark": (
            "BTCUSDT buy and hold with standard entry and exit costs, kept "
            "unchanged while strategy costs are stressed"
        ),
        "bull_market_definition": "annual segment where BTC benchmark return is positive",
        "selection_scenarios": ["normal", "double_cost"],
        "candidate_count": len(screen),
        "selected_candidate": name(selected),
        "selected_config": asdict(selected),
        "selected_rotation_weight": selected_weight,
        "selected_portfolio_rebalance_bars": selected_rebalance,
        "qualified_count": int(screen["qualified"].sum()),
        "risk_caps": {
            "maximum_initial_stop_risk_pct": 25.0,
            "maximum_drawdown_pct_across_annual_and_full_period_tests": 40.0,
        },
        "rebalance_assumption": (
            "Sleeve capital is returned to target weights on schedule with a "
            "conservative two-sided fee and slippage charge on transferred value."
        ),
        "historical_checks": checks,
        "activation_ready": False,
        "blocker": (
            "Candidate selection uses observed bull markets and periodic sleeve "
            "rebalancing must be verified with locked forward fills."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(screen.head(12).to_string(index=False))
    print(annual[["scenario", "phase", "return_pct", "sharpe", "max_drawdown_pct", "estimated_maximum_initial_risk_pct"]].to_string(index=False))
    print(full[["scenario", "return_pct", "sharpe", "max_drawdown_pct", "estimated_maximum_initial_risk_pct"]].to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
