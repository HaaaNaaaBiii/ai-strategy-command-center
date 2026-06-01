from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from smi_lab.backtest import PortfolioBacktestResult, backtest_portfolio
from smi_lab.config import StrategyConfig, load_config, load_portfolio, save_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, load_universe
from smi_lab.regime import attach_btc_momentum_regime


OUTPUT_DIR = Path("outputs/practical_candidate")
BASE_RISK = 0.02
MAX_LEVERAGE = 3.0


def breakout_long_config(base: StrategyConfig, fee_bps: float, slippage_bps: float) -> StrategyConfig:
    return replace(
        base,
        entry_mode="breakout",
        breakout_period=60,
        smi_period=20,
        smooth_k=3,
        smooth_d=5,
        signal_period=7,
        trend_ema=200,
        adx_min=18.0,
        stop_atr=3.0,
        tp1_r=1.5,
        tp2_r=3.0,
        tp3_r=6.0,
        use_longs=True,
        use_shorts=False,
        regime_mode="avoid_risk_off_longs",
        regime_source="btc_momentum",
        btc_ema_period=100,
        momentum_period=180,
        momentum_top_n=1,
        risk_per_trade=BASE_RISK,
        max_leverage=MAX_LEVERAGE,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    ).validate()


def candidate_sleeves(
    pullback: list[tuple[str, float, StrategyConfig]],
    base: StrategyConfig,
    breakout_weight: float,
    fee_bps: float,
    slippage_bps: float,
) -> list[tuple[str, float, StrategyConfig]]:
    pullback_weight = 1.0 - breakout_weight
    sleeves = [
        (
            name,
            weight * pullback_weight,
            replace(
                config,
                regime_source="none",
                risk_per_trade=BASE_RISK,
                max_leverage=MAX_LEVERAGE,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            ).validate(),
        )
        for name, weight, config in pullback
    ]
    sleeves.append(
        (
            "riskon_ranked_breakout_long",
            breakout_weight,
            breakout_long_config(base, fee_bps, slippage_bps),
        )
    )
    return sleeves


def metrics_row(
    candidate: str, scenario: str, phase: str, result: PortfolioBacktestResult
) -> dict[str, str | float]:
    return {"candidate": candidate, "scenario": scenario, "phase": phase, **result.metrics}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_universe = load_universe(
        DEFAULT_SYMBOLS,
        interval="4h",
        bars=bars_for_years("4h", 5),
        market="perpetual",
        include_funding=True,
    )
    base = load_config("outputs/baseline_strategy.json")
    pullback = load_portfolio("outputs/futures_regime/paper_portfolio.json")
    screen_rows: list[dict[str, float]] = []
    candidates: dict[str, tuple[dict[str, pd.DataFrame], list[tuple[str, float, StrategyConfig]]]] = {}
    for momentum_period in (180, 360, 540):
        for top_n in (1, 2):
            universe = attach_btc_momentum_regime(
                raw_universe, btc_ema_period=100, momentum_period=momentum_period, top_n=top_n
            )
            for breakout_weight in (0.20, 0.30):
                name = f"mom{momentum_period}_top{top_n}_break{breakout_weight:.2f}"
                normal = candidate_sleeves(pullback, base, breakout_weight, 10.0, 5.0)
                stress = candidate_sleeves(pullback, base, breakout_weight, 20.0, 10.0)
                dev_normal = backtest_portfolio(universe, normal, start_ratio=0.0, end_ratio=0.8)
                dev_stress = backtest_portfolio(universe, stress, start_ratio=0.0, end_ratio=0.8)
                score = min(dev_normal.metrics["sharpe"], dev_stress.metrics["sharpe"])
                screen_rows.append(
                    {
                        "candidate": name,
                        "momentum_period": momentum_period,
                        "top_n": top_n,
                        "breakout_weight": breakout_weight,
                        "development_score": score,
                        "normal_return_pct": dev_normal.metrics["return_pct"],
                        "normal_sharpe": dev_normal.metrics["sharpe"],
                        "normal_max_drawdown_pct": dev_normal.metrics["max_drawdown_pct"],
                        "stress_return_pct": dev_stress.metrics["return_pct"],
                        "stress_sharpe": dev_stress.metrics["sharpe"],
                        "stress_max_drawdown_pct": dev_stress.metrics["max_drawdown_pct"],
                    }
                )
                candidates[name] = (universe, normal)
    screen = pd.DataFrame(screen_rows).sort_values("development_score", ascending=False)
    screen.to_csv(OUTPUT_DIR / "development_screen.csv", index=False)
    selected = str(screen.iloc[0]["candidate"])
    selected_period = int(screen.iloc[0]["momentum_period"])
    selected_top_n = int(screen.iloc[0]["top_n"])
    selected_weight = float(screen.iloc[0]["breakout_weight"])
    universe = attach_btc_momentum_regime(
        raw_universe, btc_ema_period=100, momentum_period=selected_period, top_n=selected_top_n
    )
    scenarios = {
        "normal": candidate_sleeves(pullback, base, selected_weight, 10.0, 5.0),
        "double_cost": candidate_sleeves(pullback, base, selected_weight, 20.0, 10.0),
    }
    phase_rows: list[dict[str, str | float]] = []
    annual_rows: list[dict[str, str | float]] = []
    for scenario, sleeves in scenarios.items():
        for phase, start, end in [
            ("development", 0.0, 0.8),
            ("holdout", 0.8, 1.0),
            ("full_period", 0.0, 1.0),
        ]:
            result = backtest_portfolio(universe, sleeves, start_ratio=start, end_ratio=end)
            phase_rows.append(metrics_row(selected, scenario, phase, result))
            if scenario == "normal" and phase in {"holdout", "full_period"}:
                result.by_symbol.to_csv(OUTPUT_DIR / f"{phase}_by_symbol.csv", index=False)
                result.trades.to_csv(OUTPUT_DIR / f"{phase}_trades.csv", index=False)
        for year in range(5):
            result = backtest_portfolio(
                universe, sleeves, start_ratio=year / 5, end_ratio=(year + 1) / 5
            )
            annual_rows.append(metrics_row(selected, scenario, f"year_{year + 1}", result))
    phases = pd.DataFrame(phase_rows)
    annual = pd.DataFrame(annual_rows)
    phases.to_csv(OUTPUT_DIR / "phase_metrics.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "annual_metrics.csv", index=False)
    normal_holdout = phases[(phases["scenario"] == "normal") & (phases["phase"] == "holdout")].iloc[0]
    stress_holdout = phases[
        (phases["scenario"] == "double_cost") & (phases["phase"] == "holdout")
    ].iloc[0]
    normal_full = phases[(phases["scenario"] == "normal") & (phases["phase"] == "full_period")].iloc[0]
    stress_full = phases[
        (phases["scenario"] == "double_cost") & (phases["phase"] == "full_period")
    ].iloc[0]
    paper_candidate = (
        normal_holdout["return_pct"] > 0.0
        and stress_holdout["return_pct"] > 0.0
        and normal_full["sharpe"] >= 0.8
        and stress_full["sharpe"] >= 0.6
        and normal_full["max_drawdown_pct"] > -8.0
        and stress_full["max_drawdown_pct"] > -8.0
    )
    if paper_candidate:
        save_portfolio(scenarios["normal"], OUTPUT_DIR / "paper_portfolio.json")
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "selected_on": "first 80 percent normal and double-cost development Sharpe only",
                "candidate": selected,
                "btc_ema_period": 100,
                "momentum_period": selected_period,
                "top_n": selected_top_n,
                "breakout_weight": selected_weight,
                "maximum_simultaneous_stop_risk_approx_pct": BASE_RISK * 100.0,
                "paper_notification_candidate": bool(paper_candidate),
                "funded_trading_ready": False,
                "funded_trading_blocker": (
                    "Historical design iterations consumed the test window; forward paper "
                    "validation and execution-cost confirmation are still required."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.head(5).to_string(index=False))
    print(phases.to_string(index=False))
    print(f"Paper notification candidate: {paper_candidate}; funded trading ready: False")


if __name__ == "__main__":
    main()
