from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import pandas as pd

from research_bull_combined import (
    ACTIVE_MAX_RISK_PCT,
    combine_results,
    initial_risk_estimate,
    mature_sleeves,
)
from research_bull_offense import benchmark_rows, phase_ranges
from research_bull_rotation import SCENARIOS, with_costs
from smi_lab.backtest import backtest_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.regime import attach_btc_momentum_regime
from smi_lab.rotation import RotationConfig, backtest_rotation


OUTPUT_DIR = Path("outputs/bull_filtered")
WEIGHTS = (0.60, 0.625, 0.65, 0.675, 0.70)


def candidates() -> list[RotationConfig]:
    return [
        RotationConfig(
            btc_ema_period=btc_ema,
            asset_ema_period=asset_ema,
            momentum_period=momentum,
            rebalance_bars=42,
            exposure=1.0,
            stop_atr=stop_atr,
            tp1_r=3.0,
            tp2_r=8.0,
            tp3_r=20.0,
            tp1_fraction=0.10,
            tp2_fraction=0.10,
        )
        for btc_ema, asset_ema, momentum, stop_atr in product(
            (50, 100, 200),
            (0, 50, 100),
            (90, 120, 180),
            (2.0, 3.0, 4.0),
        )
    ]


def name(config: RotationConfig) -> str:
    return (
        f"btcema{config.btc_ema_period}_assetema{config.asset_ema_period}_"
        f"mom{config.momentum_period}_stop{config.stop_atr:.1f}"
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
    mature: dict[tuple[str, str], object] = {}
    for scenario in ("normal", "double_cost"):
        for phase, start, end in phases:
            mature[(scenario, phase)] = backtest_portfolio(
                mature_universe,
                mature_sleeves(scenario),
                start_ratio=start,
                end_ratio=end,
            )
    rows: list[dict[str, float | str | bool]] = []
    rotation_cache: dict[tuple[str, str, str], object] = {}
    for config in candidates():
        for scenario in ("normal", "double_cost"):
            configured = with_costs(config, SCENARIOS[scenario])
            for phase, start, end in phases:
                rotation_cache[(name(config), scenario, phase)] = backtest_rotation(
                    universe, configured, start_ratio=start, end_ratio=end
                )
        for weight in WEIGHTS:
            returns: dict[str, pd.Series] = {}
            worst_drawdown = 0.0
            maximum_risk = 0.0
            for scenario in ("normal", "double_cost"):
                scenario_returns = {}
                for phase, _, _ in phases:
                    rotation = rotation_cache[(name(config), scenario, phase)]
                    metrics = combine_results(mature[(scenario, phase)], rotation, weight)
                    scenario_returns[phase] = metrics["return_pct"]
                    worst_drawdown = min(worst_drawdown, metrics["max_drawdown_pct"])
                    maximum_risk = max(
                        maximum_risk, initial_risk_estimate(rotation, weight)
                    )
                returns[scenario] = pd.Series(scenario_returns)
            normal_bull = returns["normal"].loc[bull_benchmark.index]
            double_bull = returns["double_cost"].loc[bull_benchmark.index]
            target = bool(
                ((normal_bull >= 50) & (normal_bull > bull_benchmark)).all()
                and ((double_bull >= 50) & (double_bull > bull_benchmark)).all()
            )
            rows.append(
                {
                    "candidate": name(config),
                    **asdict(config),
                    "rotation_weight": weight,
                    "bull_target": target,
                    "risk_cap": bool(
                        maximum_risk <= 25.0 and worst_drawdown >= -40.0
                    ),
                    "qualified": bool(
                        target and maximum_risk <= 25.0 and worst_drawdown >= -40.0
                    ),
                    "minimum_double_bull_return_pct": float(double_bull.min()),
                    "minimum_double_excess_vs_btc_pct": float(
                        (double_bull - bull_benchmark).min()
                    ),
                    "estimated_maximum_initial_risk_pct": maximum_risk,
                    "worst_annual_drawdown_pct": worst_drawdown,
                }
            )
    screen = pd.DataFrame(rows)
    screen["worst_full_period_drawdown_pct"] = float("nan")
    screen["full_period_risk_cap"] = False
    candidate_map = {name(config): config for config in candidates()}
    mature_full = {
        scenario: backtest_portfolio(mature_universe, mature_sleeves(scenario))
        for scenario in ("normal", "double_cost")
    }
    for row_index, row in screen[screen["qualified"]].iterrows():
        config = candidate_map[str(row["candidate"])]
        weight = float(row["rotation_weight"])
        full_drawdown = 0.0
        full_risk = 0.0
        for scenario in ("normal", "double_cost"):
            overlay = backtest_rotation(
                universe, with_costs(config, SCENARIOS[scenario])
            )
            metrics = combine_results(mature_full[scenario], overlay, weight)
            full_drawdown = min(full_drawdown, metrics["max_drawdown_pct"])
            full_risk = max(full_risk, initial_risk_estimate(overlay, weight))
        screen.at[row_index, "worst_full_period_drawdown_pct"] = full_drawdown
        screen.at[row_index, "full_period_risk_cap"] = bool(
            full_drawdown >= -40.0 and full_risk <= 25.0
        )
    screen["fully_qualified"] = screen["qualified"] & screen["full_period_risk_cap"]
    screen = screen.sort_values(
        [
            "fully_qualified",
            "qualified",
            "bull_target",
            "minimum_double_excess_vs_btc_pct",
            "worst_annual_drawdown_pct",
        ],
        ascending=[False, False, False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_row = screen.iloc[0]
    selected = next(config for config in candidates() if name(config) == selected_row["candidate"])
    weight = float(selected_row["rotation_weight"])
    annual_rows = []
    full_rows = []
    for scenario, costs in SCENARIOS.items():
        configured = with_costs(selected, costs)
        for phase, start, end in phases:
            base = backtest_portfolio(
                mature_universe, mature_sleeves(scenario), start_ratio=start, end_ratio=end
            )
            overlay = backtest_rotation(universe, configured, start_ratio=start, end_ratio=end)
            annual_rows.append(
                {
                    "scenario": scenario,
                    "phase": phase,
                    **combine_results(base, overlay, weight),
                    "estimated_maximum_initial_risk_pct": initial_risk_estimate(overlay, weight),
                }
            )
        base = backtest_portfolio(mature_universe, mature_sleeves(scenario))
        overlay = backtest_rotation(universe, configured)
        full_rows.append(
            {
                "scenario": scenario,
                **combine_results(base, overlay, weight),
                "estimated_maximum_initial_risk_pct": initial_risk_estimate(overlay, weight),
            }
        )
    annual = pd.DataFrame(annual_rows)
    full = pd.DataFrame(full_rows)
    annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    full.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)
    metadata = {
        "data_window": data_window(universe["BTCUSDT"]),
        "candidate_count": len(candidates()) * len(WEIGHTS),
        "selected_candidate": name(selected),
        "selected_config": asdict(selected),
        "selected_rotation_weight": weight,
        "annual_qualified_count": int(screen["qualified"].sum()),
        "fully_qualified_count": int(screen["fully_qualified"].sum()),
        "risk_caps": {
            "maximum_initial_stop_risk_pct": 25.0,
            "maximum_annual_drawdown_pct": 40.0,
        },
        "activation_ready": False,
        "blocker": (
            "Selection uses observed bull years; even a risk-capped result requires "
            "locked forward validation before notifications or capital allocation."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(screen.head(12).to_string(index=False))
    print(annual[["scenario", "phase", "return_pct", "max_drawdown_pct", "estimated_maximum_initial_risk_pct"]].to_string(index=False))
    print(full[["scenario", "return_pct", "max_drawdown_pct", "estimated_maximum_initial_risk_pct"]].to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
