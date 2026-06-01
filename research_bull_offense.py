from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path

import pandas as pd

from smi_lab.backtest import BacktestResult, backtest_ranked_long
from smi_lab.config import StrategyConfig, load_config, save_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.regime import attach_btc_momentum_regime


OUTPUT_DIR = Path("outputs/bull_offense")
SCENARIOS = {
    "normal": (10.0, 5.0),
    "double_cost": (20.0, 10.0),
    "triple_cost": (30.0, 15.0),
}


@dataclass(frozen=True)
class OffensiveSpec:
    momentum_period: int
    breakout_period: int
    trend_ema: int
    risk_per_trade: float
    max_leverage: float
    profile: str

    @property
    def name(self) -> str:
        return (
            f"mom{self.momentum_period}_break{self.breakout_period}_"
            f"ema{self.trend_ema}_risk{self.risk_per_trade:.2f}_"
            f"lev{self.max_leverage:.0f}_{self.profile}"
        )


def specifications() -> list[OffensiveSpec]:
    return [
        OffensiveSpec(momentum, breakout, trend, risk, leverage, profile)
        for momentum, breakout, trend, risk, leverage, profile in product(
            (120, 180, 240),
            (20, 40),
            (100, 200),
            (0.03, 0.05),
            (3.0, 5.0),
            ("runner", "wide_runner"),
        )
    ]


def build_config(
    base: StrategyConfig,
    spec: OffensiveSpec,
    fee_bps: float,
    slippage_bps: float,
) -> StrategyConfig:
    if spec.profile == "runner":
        stop_atr, tp1_r, tp2_r, tp3_r, tp1_fraction, tp2_fraction = (
            3.0,
            2.0,
            6.0,
            20.0,
            0.15,
            0.15,
        )
    else:
        stop_atr, tp1_r, tp2_r, tp3_r, tp1_fraction, tp2_fraction = (
            4.0,
            2.0,
            8.0,
            30.0,
            0.10,
            0.10,
        )
    return replace(
        base,
        entry_mode="breakout",
        breakout_period=spec.breakout_period,
        smi_period=20,
        smooth_k=3,
        smooth_d=5,
        signal_period=7,
        trend_ema=spec.trend_ema,
        adx_min=12.0,
        stop_atr=stop_atr,
        tp1_r=tp1_r,
        tp2_r=tp2_r,
        tp3_r=tp3_r,
        tp1_fraction=tp1_fraction,
        tp2_fraction=tp2_fraction,
        use_longs=True,
        use_shorts=False,
        regime_mode="avoid_risk_off_longs",
        regime_source="btc_momentum",
        btc_ema_period=100,
        momentum_period=spec.momentum_period,
        momentum_top_n=1,
        risk_per_trade=spec.risk_per_trade,
        max_leverage=spec.max_leverage,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    ).validate()


def phase_ranges() -> list[tuple[str, float, float]]:
    return [(f"year_{year + 1}", year / 5, (year + 1) / 5) for year in range(5)]


def benchmark_rows(
    universe: dict[str, pd.DataFrame],
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    single_side_cost = (fee_bps + slippage_bps) / 10_000.0
    length = len(universe["BTCUSDT"])
    rows: list[dict[str, float | str | bool]] = []
    for phase, start_ratio, end_ratio in phase_ranges():
        start = min(int(length * start_ratio), length - 2)
        end = min(int(length * end_ratio) - 1, length - 1)
        returns: dict[str, float] = {}
        for symbol, frame in universe.items():
            entry = float(frame["open"].iloc[start]) * (1.0 + single_side_cost)
            exit_price = float(frame["close"].iloc[end]) * (1.0 - single_side_cost)
            returns[symbol] = (exit_price / entry - 1.0) * 100.0
        rows.append(
            {
                "phase": phase,
                **returns,
                "equal_weight_return_pct": sum(returns.values()) / len(returns),
                "btc_bull_market": returns["BTCUSDT"] > 0.0,
            }
        )
    return pd.DataFrame(rows)


def result_row(
    spec: OffensiveSpec,
    scenario: str,
    phase: str,
    result: BacktestResult,
) -> dict[str, float | str]:
    return {
        "candidate": spec.name,
        **asdict(spec),
        "scenario": scenario,
        "phase": phase,
        **result.metrics,
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
    base = load_config("outputs/baseline_strategy.json")
    specs = specifications()
    regime_universes = {
        momentum: attach_btc_momentum_regime(
            universe, btc_ema_period=100, momentum_period=momentum, top_n=1
        )
        for momentum in sorted({spec.momentum_period for spec in specs})
    }

    screen_rows: list[dict[str, float | str | bool]] = []
    candidate_annual: dict[str, pd.DataFrame] = {}
    for spec in specs:
        rows: list[dict[str, float | str]] = []
        config = build_config(base, spec, *SCENARIOS["normal"])
        for phase, start, end in phase_ranges():
            result = backtest_ranked_long(
                regime_universes[spec.momentum_period],
                config,
                start_ratio=start,
                end_ratio=end,
            )
            rows.append(result_row(spec, "normal", phase, result))
        annual = pd.DataFrame(rows).set_index("phase")
        candidate_annual[spec.name] = annual
        bull_returns = annual.loc[bull_benchmark.index, "return_pct"]
        excess = bull_returns - bull_benchmark
        qualifies = bool(((bull_returns >= 50.0) & (excess > 0.0)).all())
        screen_rows.append(
            {
                "candidate": spec.name,
                **asdict(spec),
                "qualifies_bull_target": qualifies,
                "minimum_bull_return_pct": float(bull_returns.min()),
                "minimum_bull_excess_vs_btc_pct": float(excess.min()),
                "average_bull_excess_vs_btc_pct": float(excess.mean()),
                "worst_annual_return_pct": float(annual["return_pct"].min()),
                "worst_annual_drawdown_pct": float(annual["max_drawdown_pct"].min()),
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        [
            "qualifies_bull_target",
            "minimum_bull_excess_vs_btc_pct",
            "minimum_bull_return_pct",
            "worst_annual_drawdown_pct",
        ],
        ascending=[False, False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "candidate_screen.csv", index=False)
    selected_name = str(screen.iloc[0]["candidate"])
    selected = next(spec for spec in specs if spec.name == selected_name)

    annual_rows: list[dict[str, float | str]] = []
    full_rows: list[dict[str, float | str]] = []
    for scenario, costs in SCENARIOS.items():
        config = build_config(base, selected, *costs)
        for phase, start, end in phase_ranges():
            result = backtest_ranked_long(
                regime_universes[selected.momentum_period],
                config,
                start_ratio=start,
                end_ratio=end,
            )
            annual_rows.append(result_row(selected, scenario, phase, result))
        full = backtest_ranked_long(regime_universes[selected.momentum_period], config)
        full_rows.append(result_row(selected, scenario, "full_period", full))
        full.trades.to_csv(OUTPUT_DIR / f"{scenario}_full_period_trades.csv", index=False)
    annual = pd.DataFrame(annual_rows)
    full_period = pd.DataFrame(full_rows)
    annual.to_csv(OUTPUT_DIR / "selected_annual_metrics.csv", index=False)
    full_period.to_csv(OUTPUT_DIR / "selected_full_period_metrics.csv", index=False)

    normal_annual = annual[annual["scenario"] == "normal"].set_index("phase")
    normal_bull = normal_annual.loc[bull_benchmark.index, "return_pct"]
    normal_excess = normal_bull - bull_benchmark
    target_pass = bool(((normal_bull >= 50.0) & (normal_excess > 0.0)).all())
    double_annual = annual[annual["scenario"] == "double_cost"].set_index("phase")
    double_bull = double_annual.loc[bull_benchmark.index, "return_pct"]
    cost_stress_pass = bool((double_bull >= 50.0).all())
    metadata = {
        "market": "Binance USD-M perpetual",
        "data_window": data_window(universe["BTCUSDT"]),
        "benchmark": "BTCUSDT buy and hold with standard one-entry/one-exit trading cost",
        "bull_market_definition": "annual test segment where BTC benchmark return is positive",
        "selection_warning": (
            "This is a target-search over already observed bull years and is not "
            "independent out-of-sample validation."
        ),
        "candidate_count": len(specs),
        "selected_candidate": selected.name,
        "selected_spec": asdict(selected),
        "bull_target": (
            "At least 50 percent return and strictly greater than the BTC benchmark "
            "in every BTC-positive annual segment."
        ),
        "normal_cost_bull_target_pass": target_pass,
        "double_cost_bull_return_at_least_50_pass": cost_stress_pass,
        "paper_notification_candidate": False,
        "funded_trading_ready": False,
        "deployment_blocker": (
            "High leverage offensive search must pass independent forward validation "
            "and liquidation/margin modeling before any activation."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if target_pass and cost_stress_pass:
        save_portfolio(
            [(
                "bull_offense_ranked_long",
                1.0,
                build_config(base, selected, *SCENARIOS["normal"]),
            )],
            OUTPUT_DIR / "shadow_portfolio.json",
        )
    print(benchmark.to_string(index=False))
    print(screen.head(10).to_string(index=False))
    print(f"Selected candidate: {selected.name}")
    print(annual[["scenario", "phase", "return_pct", "sharpe", "max_drawdown_pct", "trades"]].to_string(index=False))
    print(full_period[["scenario", "return_pct", "sharpe", "max_drawdown_pct", "trades"]].to_string(index=False))
    print(f"Normal bull target passed: {target_pass}; double-cost 50 percent bull test passed: {cost_stress_pass}")


if __name__ == "__main__":
    main()
