from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import pandas as pd

from research_bull_offense import benchmark_rows, phase_ranges
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.rotation import RotationConfig, backtest_rotation


OUTPUT_DIR = Path("outputs/bull_rotation")
SCENARIOS = {
    "normal": (10.0, 5.0),
    "double_cost": (20.0, 10.0),
    "triple_cost": (30.0, 15.0),
}


def configurations() -> list[RotationConfig]:
    result = []
    for momentum, rebalance, exposure, stop_atr, profile in product(
        (90, 120, 180, 240),
        (18, 42),
        (0.50, 0.75, 1.0),
        (1.5, 2.0, 3.0, 4.0),
        ("runner", "wide_runner"),
    ):
        if profile == "runner":
            target = (2.0, 5.0, 12.0, 0.10, 0.10)
        else:
            target = (3.0, 8.0, 20.0, 0.10, 0.10)
        result.append(
            RotationConfig(
                momentum_period=momentum,
                rebalance_bars=rebalance,
                exposure=exposure,
                stop_atr=stop_atr,
                tp1_r=target[0],
                tp2_r=target[1],
                tp3_r=target[2],
                tp1_fraction=target[3],
                tp2_fraction=target[4],
            )
        )
    return result


def config_name(config: RotationConfig) -> str:
    return (
        f"mom{config.momentum_period}_reb{config.rebalance_bars}_"
        f"exp{config.exposure:.2f}_"
        f"stop{config.stop_atr:.1f}_tp{config.tp1_r:.0f}-"
        f"{config.tp2_r:.0f}-{config.tp3_r:.0f}"
    )


def with_costs(config: RotationConfig, costs: tuple[float, float]) -> RotationConfig:
    return RotationConfig(**{**asdict(config), "fee_bps": costs[0], "slippage_bps": costs[1]})


def result_row(
    config: RotationConfig, scenario: str, phase: str, result: object
) -> dict[str, float | str]:
    trades = result.trades
    max_risk = (
        float(trades["initial_risk_pct"].max())
        if not trades.empty and "initial_risk_pct" in trades
        else 0.0
    )
    return {
        "candidate": config_name(config),
        **asdict(config),
        "scenario": scenario,
        "phase": phase,
        **result.metrics,
        "max_initial_risk_pct": max_risk,
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
    bull_benchmark = benchmark[benchmark["btc_bull_market"]].set_index("phase")["BTCUSDT"]
    screen_rows: list[dict[str, float | str | bool]] = []
    for config in configurations():
        annual_rows = []
        for phase, start, end in phase_ranges():
            result = backtest_rotation(universe, config, start_ratio=start, end_ratio=end)
            annual_rows.append(result_row(config, "normal", phase, result))
        annual = pd.DataFrame(annual_rows).set_index("phase")
        bull_returns = annual.loc[bull_benchmark.index, "return_pct"]
        excess = bull_returns - bull_benchmark
        acceptable_risk = bool(
            annual["max_initial_risk_pct"].max() <= 20.0
            and annual["max_drawdown_pct"].min() >= -35.0
        )
        target_met = bool(((bull_returns >= 50.0) & (excess > 0.0)).all())
        screen_rows.append(
            {
                "candidate": config_name(config),
                **asdict(config),
                "qualifies_bull_target": target_met,
                "acceptable_risk": acceptable_risk,
                "qualified_with_risk_cap": target_met and acceptable_risk,
                "minimum_bull_return_pct": float(bull_returns.min()),
                "minimum_bull_excess_vs_btc_pct": float(excess.min()),
                "average_bull_excess_vs_btc_pct": float(excess.mean()),
                "worst_annual_return_pct": float(annual["return_pct"].min()),
                "worst_annual_drawdown_pct": float(annual["max_drawdown_pct"].min()),
                "maximum_initial_risk_pct": float(annual["max_initial_risk_pct"].max()),
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        [
            "qualified_with_risk_cap",
            "qualifies_bull_target",
            "minimum_bull_excess_vs_btc_pct",
            "worst_annual_drawdown_pct",
        ],
        ascending=[False, False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_row = screen.iloc[0]
    selected = next(
        config
        for config in configurations()
        if config_name(config) == str(selected_row["candidate"])
    )
    annual_rows: list[dict[str, float | str]] = []
    full_rows: list[dict[str, float | str]] = []
    for scenario, costs in SCENARIOS.items():
        config = with_costs(selected, costs)
        for phase, start, end in phase_ranges():
            result = backtest_rotation(universe, config, start_ratio=start, end_ratio=end)
            annual_rows.append(result_row(config, scenario, phase, result))
        full = backtest_rotation(universe, config)
        full_rows.append(result_row(config, scenario, "full_period", full))
        full.trades.to_csv(OUTPUT_DIR / f"{scenario}_full_period_trades.csv", index=False)
    annual = pd.DataFrame(annual_rows)
    full = pd.DataFrame(full_rows)
    annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    full.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)
    checks: dict[str, bool] = {}
    for scenario in SCENARIOS:
        evaluated = annual[annual["scenario"] == scenario].set_index("phase")
        returns = evaluated.loc[bull_benchmark.index, "return_pct"]
        checks[f"{scenario}_beats_btc_and_returns_at_least_50_in_bull_years"] = bool(
            ((returns >= 50.0) & (returns > bull_benchmark)).all()
        )
    metadata = {
        "market": "Binance USD-M perpetual",
        "data_window": data_window(universe["BTCUSDT"]),
        "strategy_role": "shadow bull-market rotation overlay; not active notifications",
        "benchmark": "BTCUSDT buy and hold with standard entry and exit cost",
        "bull_market_definition": "annual segment where BTC benchmark return is positive",
        "candidate_count": len(configurations()),
        "selected_candidate": config_name(selected),
        "selected_config": asdict(selected),
        "historical_target_checks": checks,
        "selection_risk_caps": {
            "maximum_initial_stop_risk_pct": 20.0,
            "maximum_annual_drawdown_pct": 35.0,
            "selected_passed": bool(selected_row["acceptable_risk"]),
        },
        "selection_warning": (
            "Bull years were used in selecting this overlay. Results are target-fit "
            "research, not an untouched forward validation."
        ),
        "deployment_ready": False,
        "deployment_blocker": (
            "A fully invested rotation overlay has materially higher risk than the "
            "active SMI portfolio and requires new forward execution evidence."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(benchmark.to_string(index=False))
    print(screen.head(10).to_string(index=False))
    print(f"Selected rotation: {config_name(selected)}")
    print(annual[["scenario", "phase", "return_pct", "sharpe", "max_drawdown_pct", "trades", "max_initial_risk_pct"]].to_string(index=False))
    print(full[["scenario", "return_pct", "sharpe", "max_drawdown_pct", "trades", "max_initial_risk_pct"]].to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
