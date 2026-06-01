from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from research_bull_offense import benchmark_rows, phase_ranges
from research_bull_rotation import SCENARIOS, config_name, configurations, with_costs
from smi_lab.backtest import PortfolioBacktestResult, _metrics, backtest_portfolio
from smi_lab.config import StrategyConfig, load_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.regime import attach_btc_momentum_regime
from smi_lab.rotation import RotationConfig, RotationResult, backtest_rotation


OUTPUT_DIR = Path("outputs/bull_combined")
ROTATION_WEIGHTS = (0.40, 0.50, 0.60, 0.70)
ACTIVE_MAX_RISK_PCT = 2.0


def mature_sleeves(
    scenario: str,
) -> list[tuple[str, float, StrategyConfig]]:
    fee, slippage = SCENARIOS[scenario]
    return [
        (name, weight, replace(config, fee_bps=fee, slippage_bps=slippage).validate())
        for name, weight, config in load_portfolio(
            "outputs/maturity_candidate/paper_portfolio.json"
        )
    ]


def combine_results(
    mature: PortfolioBacktestResult,
    rotation: RotationResult,
    rotation_weight: float,
    initial_equity: float = 10_000.0,
) -> dict[str, float]:
    equity = pd.concat(
        [
            mature.equity.rename("mature"),
            rotation.equity.rename("rotation"),
        ],
        axis=1,
    ).ffill().dropna()
    combined_equity = (
        equity["mature"] * (1.0 - rotation_weight)
        + equity["rotation"] * rotation_weight
    )
    trades: list[pd.DataFrame] = []
    if not mature.trades.empty:
        mature_trades = mature.trades.copy()
        mature_trades["pnl"] = mature_trades["pnl"] * (1.0 - rotation_weight)
        if "funding_pnl" in mature_trades:
            mature_trades["funding_pnl"] = (
                mature_trades["funding_pnl"] * (1.0 - rotation_weight)
            )
        trades.append(mature_trades)
    if not rotation.trades.empty:
        rotation_trades = rotation.trades.copy()
        rotation_trades["pnl"] = rotation_trades["pnl"] * rotation_weight
        if "funding_pnl" in rotation_trades:
            rotation_trades["funding_pnl"] = (
                rotation_trades["funding_pnl"] * rotation_weight
            )
        trades.append(rotation_trades)
    combined_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return _metrics(combined_equity, combined_trades, initial_equity)


def initial_risk_estimate(
    rotation: RotationResult, rotation_weight: float
) -> float:
    rotation_risk = (
        float(rotation.trades["initial_risk_pct"].max())
        if not rotation.trades.empty
        else 0.0
    )
    return (
        rotation_weight * rotation_risk
        + (1.0 - rotation_weight) * ACTIVE_MAX_RISK_PCT
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
    bull_benchmark = benchmark[benchmark["btc_bull_market"]].set_index("phase")["BTCUSDT"]
    mature_universe = attach_btc_momentum_regime(
        universe, btc_ema_period=100, momentum_period=240, top_n=1
    )
    phases = phase_ranges()
    mature_cache: dict[tuple[str, str], PortfolioBacktestResult] = {}
    for scenario in ("normal", "double_cost"):
        for phase, start, end in phases:
            mature_cache[(scenario, phase)] = backtest_portfolio(
                mature_universe,
                mature_sleeves(scenario),
                start_ratio=start,
                end_ratio=end,
            )
    rotation_cache: dict[tuple[str, str, str], RotationResult] = {}
    screen_rows: list[dict[str, float | str | bool]] = []
    for rotation_config in configurations():
        for scenario in ("normal", "double_cost"):
            configured = with_costs(rotation_config, SCENARIOS[scenario])
            for phase, start, end in phases:
                rotation_cache[(config_name(rotation_config), scenario, phase)] = (
                    backtest_rotation(
                        universe, configured, start_ratio=start, end_ratio=end
                    )
                )
        for weight in ROTATION_WEIGHTS:
            scenario_values: dict[str, dict[str, float]] = {}
            maximum_risk = 0.0
            worst_drawdown = 0.0
            for scenario in ("normal", "double_cost"):
                phase_values: dict[str, float] = {}
                for phase, _, _ in phases:
                    rotation = rotation_cache[(config_name(rotation_config), scenario, phase)]
                    metrics = combine_results(
                        mature_cache[(scenario, phase)], rotation, weight
                    )
                    phase_values[phase] = metrics["return_pct"]
                    worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                    maximum_risk = max(
                        maximum_risk, initial_risk_estimate(rotation, weight)
                    )
                scenario_values[scenario] = phase_values
            normal_bull = pd.Series(scenario_values["normal"]).loc[bull_benchmark.index]
            double_bull = pd.Series(scenario_values["double_cost"]).loc[bull_benchmark.index]
            normal_target = bool(
                ((normal_bull >= 50.0) & (normal_bull > bull_benchmark)).all()
            )
            double_target = bool(
                ((double_bull >= 50.0) & (double_bull > bull_benchmark)).all()
            )
            strict_risk = maximum_risk <= 10.0 and worst_drawdown >= -20.0
            aggressive_risk = maximum_risk <= 25.0 and worst_drawdown >= -40.0
            screen_rows.append(
                {
                    "rotation_candidate": config_name(rotation_config),
                    **asdict(rotation_config),
                    "rotation_weight": weight,
                    "normal_target": normal_target,
                    "double_cost_target": double_target,
                    "strict_risk_pass": strict_risk,
                    "aggressive_risk_pass": aggressive_risk,
                    "strict_qualified": normal_target and double_target and strict_risk,
                    "aggressive_qualified": normal_target
                    and double_target
                    and aggressive_risk,
                    "minimum_double_bull_excess_vs_btc_pct": float(
                        (double_bull - bull_benchmark).min()
                    ),
                    "minimum_double_bull_return_pct": float(double_bull.min()),
                    "estimated_maximum_initial_risk_pct": maximum_risk,
                    "worst_annual_drawdown_pct": worst_drawdown,
                }
            )
    screen = pd.DataFrame(screen_rows).sort_values(
        [
            "strict_qualified",
            "aggressive_qualified",
            "minimum_double_bull_excess_vs_btc_pct",
            "worst_annual_drawdown_pct",
        ],
        ascending=[False, False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_row = screen.iloc[0]
    selected_rotation = next(
        config
        for config in configurations()
        if config_name(config) == str(selected_row["rotation_candidate"])
    )
    selected_weight = float(selected_row["rotation_weight"])
    annual_rows: list[dict[str, float | str]] = []
    full_rows: list[dict[str, float | str]] = []
    for scenario, costs in SCENARIOS.items():
        configured = with_costs(selected_rotation, costs)
        sleeves = mature_sleeves(scenario)
        for phase, start, end in phases:
            mature = backtest_portfolio(
                mature_universe, sleeves, start_ratio=start, end_ratio=end
            )
            rotation = backtest_rotation(
                universe, configured, start_ratio=start, end_ratio=end
            )
            metrics = combine_results(mature, rotation, selected_weight)
            annual_rows.append(
                {
                    "scenario": scenario,
                    "phase": phase,
                    "rotation_weight": selected_weight,
                    **metrics,
                    "estimated_maximum_initial_risk_pct": initial_risk_estimate(
                        rotation, selected_weight
                    ),
                }
            )
        mature_full = backtest_portfolio(mature_universe, sleeves)
        rotation_full = backtest_rotation(universe, configured)
        full_metrics = combine_results(mature_full, rotation_full, selected_weight)
        full_rows.append(
            {
                "scenario": scenario,
                "rotation_weight": selected_weight,
                **full_metrics,
                "estimated_maximum_initial_risk_pct": initial_risk_estimate(
                    rotation_full, selected_weight
                ),
            }
        )
    annual = pd.DataFrame(annual_rows)
    full = pd.DataFrame(full_rows)
    annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    full.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)
    checks: dict[str, bool] = {}
    for scenario in SCENARIOS:
        tested = annual[annual["scenario"] == scenario].set_index("phase")
        returns = tested.loc[bull_benchmark.index, "return_pct"]
        checks[f"{scenario}_bull_target"] = bool(
            ((returns >= 50.0) & (returns > bull_benchmark)).all()
        )
    metadata = {
        "market": "Binance USD-M perpetual",
        "data_window": data_window(universe["BTCUSDT"]),
        "role": "research-only blend of active SMI portfolio and bull rotation overlay",
        "benchmark": "BTCUSDT buy and hold with standard entry and exit cost",
        "bull_market_definition": "annual segment where BTC benchmark return is positive",
        "rotation_candidate_count": len(configurations()),
        "rotation_weight_count": len(ROTATION_WEIGHTS),
        "selected_rotation": config_name(selected_rotation),
        "selected_rotation_config": asdict(selected_rotation),
        "selected_rotation_weight": selected_weight,
        "strict_caps": {
            "maximum_initial_stop_risk_pct": 10.0,
            "maximum_annual_drawdown_pct": 20.0,
            "qualified_count": int(screen["strict_qualified"].sum()),
        },
        "aggressive_caps": {
            "maximum_initial_stop_risk_pct": 25.0,
            "maximum_annual_drawdown_pct": 40.0,
            "qualified_count": int(screen["aggressive_qualified"].sum()),
        },
        "historical_target_checks": checks,
        "activation_ready": False,
        "blocker": (
            "The bull target is fitted to observed bull years and any qualifying blend "
            "carries far higher drawdown and per-position risk than the active strategy."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(screen.head(10).to_string(index=False))
    print(annual[["scenario", "phase", "return_pct", "max_drawdown_pct", "estimated_maximum_initial_risk_pct"]].to_string(index=False))
    print(full[["scenario", "return_pct", "max_drawdown_pct", "estimated_maximum_initial_risk_pct"]].to_string(index=False))
    print(json.dumps(metadata["strict_caps"], indent=2))
    print(json.dumps(metadata["aggressive_caps"], indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
