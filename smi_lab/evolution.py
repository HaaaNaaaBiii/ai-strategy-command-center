from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import random
import statistics

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest, combine_results
from .config import StrategyConfig


@dataclass
class Evaluation:
    config: StrategyConfig
    metrics: dict[str, float]
    score: float
    results: dict[str, BacktestResult]


@dataclass
class ResearchResult:
    best_config: StrategyConfig
    training: Evaluation
    validation: Evaluation
    holdout: Evaluation
    full_period: Evaluation
    leaderboard: pd.DataFrame
    boundaries: dict[str, str]


def robustness_score(metrics: dict[str, float]) -> float:
    trades = metrics["trades"]
    profit_factor = min(metrics["profit_factor"], 3.0)
    score = (
        metrics["sharpe"]
        + metrics["cagr_pct"] / 30.0
        + profit_factor * 0.15
        + metrics["max_drawdown_pct"] / 18.0
    )
    if trades < 12:
        score -= (12 - trades) * 0.35
    return float(score)


def _evaluate(
    universe: dict[str, pd.DataFrame],
    config: StrategyConfig,
    start_ratio: float,
    end_ratio: float,
) -> Evaluation:
    results: dict[str, BacktestResult] = {}
    for symbol, frame in universe.items():
        start_index = min(int(len(frame) * start_ratio), len(frame) - 2)
        end_index = min(int(len(frame) * end_ratio) - 1, len(frame) - 1)
        start = frame.index[start_index]
        end = frame.index[end_index]
        results[symbol] = backtest(
            frame.loc[:end],
            config,
            symbol=symbol,
            trade_start=start,
            trade_end=end,
        )
    _, _, metrics = combine_results(results)
    asset_returns = [result.metrics["return_pct"] for result in results.values()]
    metrics["profitable_symbols_pct"] = float(
        sum(value > 0 for value in asset_returns) / len(asset_returns) * 100.0
    )
    metrics["worst_symbol_return_pct"] = float(min(asset_returns))
    metrics["median_symbol_return_pct"] = float(np.median(asset_returns))
    return Evaluation(config, metrics, robustness_score(metrics), results)


def _candidate_configs(
    base: StrategyConfig,
    count: int,
    seed: int,
    regime_modes: tuple[str, ...] = ("none",),
) -> list[StrategyConfig]:
    generator = random.Random(seed)
    settings = {
        "smi_period": (14, 20, 28, 36),
        "smooth_k": (3, 5, 7),
        "smooth_d": (3, 5),
        "signal_period": (3, 5, 7),
        "trend_ema": (60, 100, 150, 200),
        "adx_min": (12.0, 18.0, 24.0),
        "oversold": (-20.0, -30.0, -40.0, -50.0),
        "stop_atr": (1.3, 1.6, 1.9, 2.2, 2.6),
    }
    targets = ((1.0, 2.0, 3.0), (1.0, 2.2, 3.5), (1.2, 2.4, 3.6))
    # Long-horizon baseline analysis found that short setups carry the positive
    # expectancy; optimization may omit longs but may not remove the short leg.
    directions = ((True, True), (False, True))
    configs = [base.validate()]
    seen = {tuple(sorted(base.to_dict().items()))}
    while len(configs) < count:
        tp1, tp2, tp3 = generator.choice(targets)
        use_longs, use_shorts = generator.choice(directions)
        regime_mode = generator.choice(regime_modes)
        candidate = replace(
            base,
            **{key: generator.choice(values) for key, values in settings.items()},
            overbought=-generator.choice(settings["oversold"]),
            tp1_r=tp1,
            tp2_r=tp2,
            tp3_r=tp3,
            use_longs=use_longs,
            use_shorts=use_shorts,
            regime_mode=regime_mode,
            regime_source="cboe_stress" if regime_mode != "none" else "none",
        ).validate()
        key = tuple(sorted(candidate.to_dict().items()))
        if key not in seen:
            seen.add(key)
            configs.append(candidate)
    return configs


def _row(stage: str, rank: int, evaluation: Evaluation) -> dict[str, float | int | str]:
    config = evaluation.config
    return {
        "stage": stage,
        "rank": rank,
        "score": evaluation.score,
        **evaluation.metrics,
        "smi_period": config.smi_period,
        "smooth_k": config.smooth_k,
        "smooth_d": config.smooth_d,
        "signal_period": config.signal_period,
        "trend_ema": config.trend_ema,
        "adx_min": config.adx_min,
        "oversold": config.oversold,
        "overbought": config.overbought,
        "stop_atr": config.stop_atr,
        "tp1_r": config.tp1_r,
        "tp2_r": config.tp2_r,
        "tp3_r": config.tp3_r,
        "use_longs": config.use_longs,
        "use_shorts": config.use_shorts,
        "regime_mode": config.regime_mode,
    }


def annual_evaluations(
    universe: dict[str, pd.DataFrame], config: StrategyConfig, periods: int = 5
) -> list[tuple[str, Evaluation]]:
    """Evaluate equal chronological windows for long-horizon stability reporting."""
    return [
        (
            f"year_{index + 1}",
            _evaluate(universe, config, index / periods, (index + 1) / periods),
        )
        for index in range(periods)
    ]


def evaluate_full_period(
    universe: dict[str, pd.DataFrame], config: StrategyConfig
) -> Evaluation:
    return _evaluate(universe, config, 0.0, 1.0)


def evolve_strategy(
    universe: dict[str, pd.DataFrame],
    base: StrategyConfig | None = None,
    candidate_count: int = 192,
    shortlist: int = 24,
    seed: int = 42,
    regime_modes: tuple[str, ...] = ("none",),
) -> ResearchResult:
    """
    Select a common portfolio strategy using chronological 60/20/20 splits.

    Candidates are screened over the first three years. The shortlist is ranked
    using yearly stability across the four-year development sample. Per-asset
    metrics are reported for diagnosis but selection targets diversified portfolio
    performance. The final year is not referenced during selection.
    """
    if not universe:
        raise ValueError("A market-data universe is required.")
    base = (base or StrategyConfig()).validate()
    candidates = _candidate_configs(
        base, max(candidate_count, shortlist), seed, regime_modes=regime_modes
    )
    training_evaluations = sorted(
        (_evaluate(universe, item, 0.0, 0.6) for item in candidates),
        key=lambda item: item.score,
        reverse=True,
    )
    validation_evaluations: list[Evaluation] = []
    for training_candidate in training_evaluations[:shortlist]:
        validation_candidate = _evaluate(universe, training_candidate.config, 0.6, 0.8)
        development_candidate = _evaluate(universe, training_candidate.config, 0.0, 0.8)
        development_years = [
            _evaluate(
                universe,
                training_candidate.config,
                index / 5,
                (index + 1) / 5,
            )
            for index in range(4)
        ]
        yearly_scores = [item.score for item in development_years]
        losing_years = sum(item.metrics["return_pct"] <= 0.0 for item in development_years)
        missing_asset_fraction = (
            100.0 - development_candidate.metrics["profitable_symbols_pct"]
        ) / 100.0
        asset_loss = max(-development_candidate.metrics["worst_symbol_return_pct"], 0.0)
        stability_penalty = (
            statistics.pstdev(yearly_scores) * 0.25
            + losing_years * 0.20
            + missing_asset_fraction * 0.50
            + asset_loss * 0.25
        )
        validation_candidate.score = (
            validation_candidate.score * 0.35
            + training_candidate.score * 0.25
            + statistics.fmean(yearly_scores) * 0.40
            + development_candidate.metrics["profitable_symbols_pct"] / 400.0
            - stability_penalty
        )
        validation_evaluations.append(validation_candidate)
    validation_evaluations.sort(key=lambda item: item.score, reverse=True)
    selected = validation_evaluations[0].config
    training = next(item for item in training_evaluations if item.config == selected)
    validation = _evaluate(universe, selected, 0.6, 0.8)
    holdout = _evaluate(universe, selected, 0.8, 1.0)
    full_period = _evaluate(universe, selected, 0.0, 1.0)

    rows = [
        _row("training", rank, evaluation)
        for rank, evaluation in enumerate(training_evaluations[:shortlist], start=1)
    ]
    rows.extend(
        _row("validation", rank, evaluation)
        for rank, evaluation in enumerate(validation_evaluations, start=1)
    )
    leaderboard = pd.DataFrame(rows)
    first = next(iter(universe.values()))
    boundaries = {
        "training": f"{first.index[0]} to {first.index[int(len(first) * 0.6) - 1]}",
        "validation": f"{first.index[int(len(first) * 0.6)]} to {first.index[int(len(first) * 0.8) - 1]}",
        "holdout": f"{first.index[int(len(first) * 0.8)]} to {first.index[-1]}",
    }
    return ResearchResult(
        selected,
        training,
        validation,
        holdout,
        full_period,
        leaderboard,
        boundaries,
    )
