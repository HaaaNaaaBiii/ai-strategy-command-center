from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import PortfolioBacktestResult, backtest_portfolio
from .config import StrategyConfig, save_config, save_portfolio
from .evolution import Evaluation, ResearchResult, annual_evaluations, evaluate_full_period


def _phase_row(name: str, evaluation: Evaluation) -> dict[str, float | str]:
    return {"phase": name, "score": evaluation.score, **evaluation.metrics}


def _save_evaluation_details(
    destination: Path, prefix: str, evaluation: Evaluation
) -> None:
    trades = [
        backtest.trades
        for backtest in evaluation.results.values()
        if not backtest.trades.empty
    ]
    combined = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    combined.to_csv(destination / f"{prefix}_trades.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": symbol, **backtest.metrics}
            for symbol, backtest in evaluation.results.items()
        ]
    ).to_csv(destination / f"{prefix}_by_symbol.csv", index=False)


def _save_annual_outputs(
    destination: Path,
    filename_prefix: str,
    universe: dict[str, pd.DataFrame],
    config: StrategyConfig,
) -> pd.DataFrame:
    rows = []
    symbol_rows = []
    for name, evaluation in annual_evaluations(universe, config):
        rows.append(_phase_row(name, evaluation))
        symbol_rows.extend(
            {"phase": name, "symbol": symbol, **test.metrics}
            for symbol, test in evaluation.results.items()
        )
    annual = pd.DataFrame(rows)
    annual.to_csv(destination / f"{filename_prefix}_annual_metrics.csv", index=False)
    pd.DataFrame(symbol_rows).to_csv(
        destination / f"{filename_prefix}_annual_by_symbol.csv", index=False
    )
    return annual


def _save_portfolio_result(
    destination: Path, prefix: str, result: PortfolioBacktestResult
) -> None:
    result.trades.to_csv(destination / f"{prefix}_trades.csv", index=False)
    result.by_symbol.to_csv(destination / f"{prefix}_by_symbol.csv", index=False)


def _save_deployed_portfolio(
    destination: Path,
    universe: dict[str, pd.DataFrame],
    baseline_config: StrategyConfig,
    challenger_config: StrategyConfig,
    baseline_full: Evaluation,
    challenger_full: Evaluation,
) -> tuple[PortfolioBacktestResult, bool]:
    sleeves = [
        ("trend_core", 0.50, baseline_config),
        ("defensive_short", 0.50, challenger_config),
    ]
    save_portfolio(sleeves, destination / "portfolio_candidate.json")
    full = backtest_portfolio(universe, sleeves)
    holdout = backtest_portfolio(universe, sleeves, start_ratio=0.8, end_ratio=1.0)
    _save_portfolio_result(destination, "deployed_full_period", full)
    _save_portfolio_result(destination, "deployed_holdout", holdout)
    annual_rows = []
    for index in range(5):
        annual = backtest_portfolio(
            universe, sleeves, start_ratio=index / 5, end_ratio=(index + 1) / 5
        )
        annual_rows.append({"phase": f"year_{index + 1}", **annual.metrics})
    pd.DataFrame(annual_rows).to_csv(
        destination / "deployed_annual_metrics.csv", index=False
    )
    phase_rows = []
    for name, start, end in [
        ("training", 0.0, 0.6),
        ("validation", 0.6, 0.8),
        ("holdout", 0.8, 1.0),
        ("full_period", 0.0, 1.0),
    ]:
        phase = backtest_portfolio(universe, sleeves, start_ratio=start, end_ratio=end)
        phase_rows.append({"phase": name, **phase.metrics})
    pd.DataFrame(phase_rows).to_csv(
        destination / "deployed_phase_metrics.csv", index=False
    )
    promoted = (
        full.metrics["sharpe"] > baseline_full.metrics["sharpe"]
        and full.metrics["max_drawdown_pct"] > baseline_full.metrics["max_drawdown_pct"]
        and full.metrics["profitable_symbols_pct"] == 100.0
        and holdout.metrics["return_pct"] > 0.0
    )
    if promoted:
        save_portfolio(sleeves, destination / "deployed_portfolio.json")
    else:
        (destination / "deployed_portfolio.json").unlink(missing_ok=True)
    pd.DataFrame(
        [
            {"strategy": "baseline", **baseline_full.metrics},
            {"strategy": "defensive_challenger", **challenger_full.metrics},
            {"strategy": "deployed_ensemble", **full.metrics},
        ]
    ).to_csv(destination / "strategy_comparison.csv", index=False)
    pd.DataFrame(
        [{"phase": "holdout", **holdout.metrics, "promoted": promoted}]
    ).to_csv(destination / "deployed_holdout_metrics.csv", index=False)
    return full, promoted


def save_research_outputs(
    result: ResearchResult,
    output_dir: str | Path = "outputs",
    universe: dict[str, pd.DataFrame] | None = None,
    baseline_config: StrategyConfig | None = None,
    research_context: dict[str, object] | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_config(result.best_config, destination / "best_strategy.json")
    pd.DataFrame(
        [
            _phase_row("training", result.training),
            _phase_row("validation", result.validation),
            _phase_row("holdout", result.holdout),
            _phase_row("full_period", result.full_period),
        ]
    ).to_csv(destination / "phase_metrics.csv", index=False)
    result.leaderboard.to_csv(destination / "leaderboard.csv", index=False)
    for phase_name, evaluation in {
        "holdout": result.holdout,
        "full_period": result.full_period,
    }.items():
        _save_evaluation_details(destination, phase_name, evaluation)
    if universe is not None:
        optimized_annual = _save_annual_outputs(
            destination, "optimized", universe, result.best_config
        )
        if baseline_config is not None:
            save_config(baseline_config, destination / "baseline_strategy.json")
            baseline_full = evaluate_full_period(universe, baseline_config)
            _save_evaluation_details(destination, "baseline_full_period", baseline_full)
            baseline_annual = _save_annual_outputs(
                destination, "baseline", universe, baseline_config
            )
            deployed, promoted = _save_deployed_portfolio(
                destination,
                universe,
                baseline_config,
                result.best_config,
                baseline_full,
                result.full_period,
            )
            pd.concat(
                [
                    baseline_annual.assign(strategy="baseline"),
                    optimized_annual.assign(strategy="optimized"),
                ],
                ignore_index=True,
            ).to_csv(destination / "annual_comparison.csv", index=False)
    (destination / "research_metadata.json").write_text(
        json.dumps(
            {
                "boundaries": result.boundaries,
                "deployed_portfolio_promoted": (
                    promoted if universe is not None and baseline_config is not None else False
                ),
                "research_context": research_context or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
