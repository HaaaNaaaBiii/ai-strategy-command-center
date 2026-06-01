from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from smi_lab.backtest import PortfolioBacktestResult, backtest_portfolio
from smi_lab.config import StrategyConfig, load_config, save_config, save_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, load_universe
from smi_lab.evolution import robustness_score


PARAMETER_COLUMNS = (
    "smi_period",
    "smooth_k",
    "smooth_d",
    "signal_period",
    "trend_ema",
    "adx_min",
    "oversold",
    "overbought",
    "stop_atr",
    "tp1_r",
    "tp2_r",
    "tp3_r",
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Select a diversified perpetual-market paper portfolio."
    )
    command.add_argument("--research-dir", default="outputs/futures_regime")
    command.add_argument("--baseline-config", default="outputs/baseline_strategy.json")
    command.add_argument("--interval", default="4h")
    command.add_argument("--years", type=int, default=5)
    return command


def short_candidate_from_leaderboard(
    leaderboard: pd.DataFrame, baseline: StrategyConfig
) -> StrategyConfig:
    candidates = leaderboard[
        (leaderboard["stage"] == "validation")
        & (leaderboard["regime_mode"] == "none")
        & (~leaderboard["use_longs"].astype(str).str.lower().eq("true"))
        & (leaderboard["use_shorts"].astype(str).str.lower().eq("true"))
    ].sort_values("score", ascending=False)
    if candidates.empty:
        raise RuntimeError("No no-regime short candidate is available in the leaderboard.")
    selected = candidates.iloc[0]
    changes: dict[str, float | int | bool | str] = {
        column: selected[column] for column in PARAMETER_COLUMNS
    }
    for column in {"smi_period", "smooth_k", "smooth_d", "signal_period", "trend_ema"}:
        changes[column] = int(changes[column])
    changes.update({"use_longs": False, "use_shorts": True, "regime_mode": "none"})
    return replace(baseline, **changes).validate()


def save_result(destination: Path, prefix: str, result: PortfolioBacktestResult) -> None:
    result.by_symbol.to_csv(destination / f"{prefix}_by_symbol.csv", index=False)
    result.trades.to_csv(destination / f"{prefix}_trades.csv", index=False)


def main() -> None:
    args = parser().parse_args()
    destination = Path(args.research_dir)
    destination.mkdir(parents=True, exist_ok=True)
    baseline = load_config(args.baseline_config)
    leaderboard = pd.read_csv(destination / "leaderboard.csv")
    defensive_short = short_candidate_from_leaderboard(leaderboard, baseline)
    save_config(defensive_short, destination / "no_regime_short_strategy.json")
    universe = load_universe(
        DEFAULT_SYMBOLS,
        interval=args.interval,
        bars=bars_for_years(args.interval, args.years),
        market="perpetual",
        include_funding=True,
    )
    baseline_result = backtest_portfolio(universe, [("trend_core", 1.0, baseline)])
    baseline_holdout = backtest_portfolio(
        universe, [("trend_core", 1.0, baseline)], start_ratio=0.8, end_ratio=1.0
    )
    screens: list[dict[str, float]] = []
    sleeve_candidates: list[tuple[float, list[tuple[str, float, StrategyConfig]]]] = []
    for core_weight in (0.25, 0.50, 0.75):
        sleeves = [
            ("trend_core", core_weight, baseline),
            ("defensive_short", 1.0 - core_weight, defensive_short),
        ]
        development = backtest_portfolio(universe, sleeves, start_ratio=0.0, end_ratio=0.8)
        screens.append(
            {
                "trend_core_weight": core_weight,
                "defensive_short_weight": 1.0 - core_weight,
                "development_score": robustness_score(development.metrics),
                **development.metrics,
            }
        )
        sleeve_candidates.append((core_weight, sleeves))
    screen = pd.DataFrame(screens).sort_values("development_score", ascending=False)
    screen.to_csv(destination / "portfolio_weight_screen.csv", index=False)
    selected_weight = float(screen.iloc[0]["trend_core_weight"])
    selected_sleeves = next(
        sleeves for weight, sleeves in sleeve_candidates if weight == selected_weight
    )
    save_portfolio(selected_sleeves, destination / "portfolio_candidate_v2.json")
    phase_rows = []
    phase_results: dict[str, PortfolioBacktestResult] = {}
    for name, start, end in [
        ("development", 0.0, 0.8),
        ("holdout", 0.8, 1.0),
        ("full_period", 0.0, 1.0),
    ]:
        phase_results[name] = backtest_portfolio(
            universe, selected_sleeves, start_ratio=start, end_ratio=end
        )
        phase_rows.append({"phase": name, **phase_results[name].metrics})
    pd.DataFrame(phase_rows).to_csv(destination / "paper_phase_metrics.csv", index=False)
    annual_rows = []
    for index in range(5):
        result = backtest_portfolio(
            universe,
            selected_sleeves,
            start_ratio=index / 5,
            end_ratio=(index + 1) / 5,
        )
        annual_rows.append({"phase": f"year_{index + 1}", **result.metrics})
    pd.DataFrame(annual_rows).to_csv(destination / "paper_annual_metrics.csv", index=False)
    save_result(destination, "paper_full_period", phase_results["full_period"])
    save_result(destination, "paper_holdout", phase_results["holdout"])
    full = phase_results["full_period"].metrics
    holdout = phase_results["holdout"].metrics
    promoted = (
        holdout["return_pct"] > 0.0
        and holdout["max_drawdown_pct"] > baseline_holdout.metrics["max_drawdown_pct"]
        and holdout["profitable_symbols_pct"] == 100.0
        and full["sharpe"] > baseline_result.metrics["sharpe"]
        and full["max_drawdown_pct"] > baseline_result.metrics["max_drawdown_pct"]
        and full["worst_symbol_return_pct"] > -1.0
    )
    if promoted:
        save_portfolio(selected_sleeves, destination / "paper_portfolio.json")
    pd.DataFrame(
        [
            {"strategy": "perpetual_baseline", **baseline_result.metrics},
            {"strategy": "paper_portfolio", **full},
        ]
    ).to_csv(destination / "paper_strategy_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": "perpetual_baseline_holdout", **baseline_holdout.metrics},
            {"strategy": "paper_portfolio_holdout", **holdout},
        ]
    ).to_csv(destination / "paper_holdout_comparison.csv", index=False)
    (destination / "paper_metadata.json").write_text(
        json.dumps(
            {
                "selection_window": "first_80_percent_only",
                "selected_trend_core_weight": selected_weight,
                "market": "Binance USD-M perpetual",
                "funding_included": True,
                "cboe_candidate_rejected": True,
                "paper_tracking_promoted": promoted,
                "promotion_rule": (
                    "positive holdout return; holdout drawdown improves on baseline; "
                    "all holdout symbols positive; full-period Sharpe and drawdown improve "
                    "on baseline; full-period worst symbol above -1 percent"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Selected trend_core weight: {selected_weight:.0%}")
    print(screen.to_string(index=False))
    print(pd.DataFrame(phase_rows).to_string(index=False))
    print(f"Paper tracking promoted: {promoted}")


if __name__ == "__main__":
    main()
