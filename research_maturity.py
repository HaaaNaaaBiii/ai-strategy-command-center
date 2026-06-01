from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path

import pandas as pd

from smi_lab.backtest import BacktestResult, PortfolioBacktestResult, _metrics, backtest
from smi_lab.config import StrategyConfig, load_config, load_portfolio, save_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, load_universe
from smi_lab.regime import attach_btc_momentum_regime


OUTPUT_DIR = Path("outputs/maturity_candidate")
BASE_RISK = 0.02
MAX_LEVERAGE = 3.0
SCENARIOS = {
    "normal": (10.0, 5.0),
    "double_cost": (20.0, 10.0),
    "triple_cost": (30.0, 15.0),
}
RESULT_CACHE: dict[tuple[str, pd.Timestamp, pd.Timestamp, StrategyConfig], BacktestResult] = {}


@dataclass(frozen=True)
class CandidateSpec:
    btc_ema_period: int
    momentum_period: int
    top_n: int
    breakout_period: int
    breakout_weight: float

    @property
    def name(self) -> str:
        return (
            f"ema{self.btc_ema_period}_mom{self.momentum_period}_"
            f"top{self.top_n}_break{self.breakout_period}_"
            f"weight{self.breakout_weight:.2f}"
        )


def candidate_grid() -> list[CandidateSpec]:
    """A deliberately small neighborhood fixed before evaluating the final fifth."""
    return [
        CandidateSpec(100, momentum, top_n, breakout, weight)
        for momentum, top_n, breakout, weight in product(
            (120, 180, 240),
            (1, 2),
            (40, 60, 80),
            (0.20, 0.30, 0.40),
        )
    ]


def build_sleeves(
    pullback: list[tuple[str, float, StrategyConfig]],
    base: StrategyConfig,
    spec: CandidateSpec,
    fee_bps: float,
    slippage_bps: float,
) -> list[tuple[str, float, StrategyConfig]]:
    remaining_weight = 1.0 - spec.breakout_weight
    sleeves = [
        (
            name,
            weight * remaining_weight,
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
    breakout = replace(
        base,
        entry_mode="breakout",
        breakout_period=spec.breakout_period,
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
        btc_ema_period=spec.btc_ema_period,
        momentum_period=spec.momentum_period,
        momentum_top_n=spec.top_n,
        risk_per_trade=BASE_RISK,
        max_leverage=MAX_LEVERAGE,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    ).validate()
    sleeves.append(("riskon_ranked_breakout_long", spec.breakout_weight, breakout))
    return sleeves


def result_row(
    spec: CandidateSpec,
    scenario: str,
    phase: str,
    result: PortfolioBacktestResult,
) -> dict[str, str | float | int]:
    return {
        "candidate": spec.name,
        **asdict(spec),
        "scenario": scenario,
        "phase": phase,
        **result.metrics,
    }


def cached_portfolio(
    universe: dict[str, pd.DataFrame],
    sleeves: list[tuple[str, float, StrategyConfig]],
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    initial_equity: float = 10_000.0,
) -> PortfolioBacktestResult:
    """Blend cached single-sleeve paths; weights do not change their trade path."""
    symbol_curves: dict[str, pd.Series] = {}
    symbol_metrics: list[dict[str, str | float]] = []
    portfolio_trades: list[pd.DataFrame] = []
    symbol_count = len(universe)
    for symbol, frame in universe.items():
        start = frame.index[min(int(len(frame) * start_ratio), len(frame) - 2)]
        end = frame.index[min(int(len(frame) * end_ratio) - 1, len(frame) - 1)]
        curves: list[pd.Series] = []
        symbol_trades: list[pd.DataFrame] = []
        for sleeve_name, weight, config in sleeves:
            key = (symbol, start, end, config)
            if key not in RESULT_CACHE:
                RESULT_CACHE[key] = backtest(
                    frame.loc[:end],
                    config,
                    symbol=symbol,
                    initial_equity=initial_equity,
                    trade_start=start,
                    trade_end=end,
                )
            result = RESULT_CACHE[key]
            curves.append(result.equity / initial_equity * weight)
            if not result.trades.empty:
                trades = result.trades.copy()
                trades.insert(0, "sleeve", sleeve_name)
                trades["pnl"] = trades["pnl"] * weight
                if "funding_pnl" in trades:
                    trades["funding_pnl"] = trades["funding_pnl"] * weight
                symbol_trades.append(trades)
        symbol_equity = pd.concat(curves, axis=1).ffill().sum(axis=1) * initial_equity
        combined_symbol_trades = (
            pd.concat(symbol_trades, ignore_index=True) if symbol_trades else pd.DataFrame()
        )
        symbol_curves[symbol] = symbol_equity
        symbol_metrics.append(
            {"symbol": symbol, **_metrics(symbol_equity, combined_symbol_trades, initial_equity)}
        )
        if not combined_symbol_trades.empty:
            scaled = combined_symbol_trades.copy()
            scaled["pnl"] = scaled["pnl"] / symbol_count
            if "funding_pnl" in scaled:
                scaled["funding_pnl"] = scaled["funding_pnl"] / symbol_count
            portfolio_trades.append(scaled)
    equity = pd.concat(symbol_curves, axis=1).ffill().mean(axis=1)
    trades = pd.concat(portfolio_trades, ignore_index=True) if portfolio_trades else pd.DataFrame()
    metrics = _metrics(equity, trades, initial_equity)
    by_symbol = pd.DataFrame(symbol_metrics)
    asset_returns = by_symbol["return_pct"]
    metrics["profitable_symbols_pct"] = float((asset_returns > 0).mean() * 100.0)
    metrics["worst_symbol_return_pct"] = float(asset_returns.min())
    metrics["median_symbol_return_pct"] = float(asset_returns.median())
    return PortfolioBacktestResult(equity, trades, metrics, by_symbol)


def combine_forward_results(results: list[PortfolioBacktestResult]) -> dict[str, float]:
    wealth = 10_000.0
    curves: list[pd.Series] = []
    trades: list[pd.DataFrame] = []
    for result in results:
        if result.equity.empty:
            continue
        scale = wealth / 10_000.0
        curve = result.equity * scale
        curves.append(curve)
        if not result.trades.empty:
            scaled = result.trades.copy()
            scaled["pnl"] = scaled["pnl"] * scale
            if "funding_pnl" in scaled:
                scaled["funding_pnl"] = scaled["funding_pnl"] * scale
            trades.append(scaled)
        wealth = float(curve.iloc[-1])
    equity = pd.concat(curves).sort_index() if curves else pd.Series(dtype=float)
    combined_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return _metrics(equity, combined_trades, 10_000.0)


def select_on_period(
    specs: list[CandidateSpec],
    regime_universes: dict[tuple[int, int, int], dict[str, pd.DataFrame]],
    pullback: list[tuple[str, float, StrategyConfig]],
    base: StrategyConfig,
    start: float,
    end: float,
) -> tuple[CandidateSpec, list[dict[str, str | float | int]]]:
    rows: list[dict[str, str | float | int]] = []
    for spec in specs:
        universe = regime_universes[
            (spec.btc_ema_period, spec.momentum_period, spec.top_n)
        ]
        normal = cached_portfolio(
            universe,
            build_sleeves(pullback, base, spec, *SCENARIOS["normal"]),
            start_ratio=start,
            end_ratio=end,
        )
        stress = cached_portfolio(
            universe,
            build_sleeves(pullback, base, spec, *SCENARIOS["double_cost"]),
            start_ratio=start,
            end_ratio=end,
        )
        rows.append(
            {
                "candidate": spec.name,
                **asdict(spec),
                "normal_return_pct": normal.metrics["return_pct"],
                "normal_sharpe": normal.metrics["sharpe"],
                "normal_max_drawdown_pct": normal.metrics["max_drawdown_pct"],
                "double_return_pct": stress.metrics["return_pct"],
                "double_sharpe": stress.metrics["sharpe"],
                "double_max_drawdown_pct": stress.metrics["max_drawdown_pct"],
                "qualified": bool(
                    normal.metrics["return_pct"] > 0.0
                    and stress.metrics["return_pct"] > 0.0
                    and stress.metrics["max_drawdown_pct"] > -8.0
                ),
                "selection_score": min(
                    normal.metrics["sharpe"], stress.metrics["sharpe"]
                ),
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["qualified", "selection_score", "double_return_pct"],
        ascending=[False, False, False],
    )
    choice = ranked.iloc[0]["candidate"]
    selected = next(spec for spec in specs if spec.name == choice)
    return selected, rows


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
    specs = candidate_grid()
    regime_universes = {
        key: attach_btc_momentum_regime(
            raw_universe,
            btc_ema_period=key[0],
            momentum_period=key[1],
            top_n=key[2],
        )
        for key in {
            (spec.btc_ema_period, spec.momentum_period, spec.top_n) for spec in specs
        }
    }

    calibration_rows: list[dict[str, str | float | int]] = []
    screen_rows: list[dict[str, str | float | int]] = []
    for spec in specs:
        universe = regime_universes[
            (spec.btc_ema_period, spec.momentum_period, spec.top_n)
        ]
        values: dict[str, float] = {}
        for phase, start, end in (("calibration", 0.0, 0.6), ("validation", 0.6, 0.8)):
            for scenario in ("normal", "double_cost"):
                result = cached_portfolio(
                    universe,
                    build_sleeves(pullback, base, spec, *SCENARIOS[scenario]),
                    start_ratio=start,
                    end_ratio=end,
                )
                calibration_rows.append(result_row(spec, scenario, phase, result))
                values[f"{phase}_{scenario}_return_pct"] = result.metrics["return_pct"]
                values[f"{phase}_{scenario}_sharpe"] = result.metrics["sharpe"]
                values[f"{phase}_{scenario}_max_drawdown_pct"] = result.metrics[
                    "max_drawdown_pct"
                ]
        selection_score = min(
            values["calibration_normal_sharpe"],
            values["calibration_double_cost_sharpe"],
            values["validation_normal_sharpe"],
            values["validation_double_cost_sharpe"],
        )
        screen_rows.append(
            {
                "candidate": spec.name,
                **asdict(spec),
                **values,
                "qualified": bool(
                    values["calibration_double_cost_return_pct"] > 0.0
                    and values["validation_double_cost_return_pct"] > 0.0
                    and values["calibration_double_cost_max_drawdown_pct"] > -8.0
                    and values["validation_double_cost_max_drawdown_pct"] > -8.0
                ),
                "selection_score": selection_score,
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        ["qualified", "selection_score", "validation_double_cost_return_pct"],
        ascending=[False, False, False],
    )
    screen.to_csv(OUTPUT_DIR / "early_period_screen.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUTPUT_DIR / "early_phase_metrics.csv", index=False)
    selected = next(spec for spec in specs if spec.name == str(screen.iloc[0]["candidate"]))
    selected_universe = regime_universes[
        (selected.btc_ema_period, selected.momentum_period, selected.top_n)
    ]

    phase_rows: list[dict[str, str | float | int]] = []
    annual_rows: list[dict[str, str | float | int]] = []
    holdout_sensitivity_rows: list[dict[str, str | float | int]] = []
    for scenario, costs in SCENARIOS.items():
        sleeves = build_sleeves(pullback, base, selected, *costs)
        for phase, start, end in (
            ("calibration", 0.0, 0.6),
            ("validation", 0.6, 0.8),
            ("holdout", 0.8, 1.0),
            ("full_period", 0.0, 1.0),
        ):
            result = cached_portfolio(
                selected_universe, sleeves, start_ratio=start, end_ratio=end
            )
            phase_rows.append(result_row(selected, scenario, phase, result))
            if scenario == "normal" and phase in {"holdout", "full_period"}:
                result.by_symbol.to_csv(OUTPUT_DIR / f"{phase}_by_symbol.csv", index=False)
                result.trades.to_csv(OUTPUT_DIR / f"{phase}_trades.csv", index=False)
        for year in range(5):
            result = cached_portfolio(
                selected_universe,
                sleeves,
                start_ratio=year / 5,
                end_ratio=(year + 1) / 5,
            )
            annual_rows.append(result_row(selected, scenario, f"year_{year + 1}", result))
    phases = pd.DataFrame(phase_rows)
    annual = pd.DataFrame(annual_rows)
    phases.to_csv(OUTPUT_DIR / "phase_metrics.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "annual_metrics.csv", index=False)

    for spec in specs:
        result = cached_portfolio(
            regime_universes[(spec.btc_ema_period, spec.momentum_period, spec.top_n)],
            build_sleeves(pullback, base, spec, *SCENARIOS["double_cost"]),
            start_ratio=0.8,
            end_ratio=1.0,
        )
        holdout_sensitivity_rows.append(result_row(spec, "double_cost", "holdout", result))
    sensitivity = pd.DataFrame(holdout_sensitivity_rows)
    sensitivity.to_csv(OUTPUT_DIR / "holdout_parameter_sensitivity.csv", index=False)

    leaveout_rows: list[dict[str, str | float | int]] = []
    for scenario in ("normal", "double_cost", "triple_cost"):
        for omitted in DEFAULT_SYMBOLS:
            reduced = {
                symbol: frame
                for symbol, frame in selected_universe.items()
                if symbol != omitted
            }
            result = cached_portfolio(
                reduced,
                build_sleeves(pullback, base, selected, *SCENARIOS[scenario]),
            )
            leaveout_rows.append(
                {
                    "scenario": scenario,
                    "omitted_symbol": omitted,
                    **result.metrics,
                }
            )
    leaveout = pd.DataFrame(leaveout_rows)
    leaveout.to_csv(OUTPUT_DIR / "leave_one_symbol_out.csv", index=False)

    forward_rows: list[dict[str, str | float | int]] = []
    forward_results: dict[str, list[PortfolioBacktestResult]] = {
        scenario: [] for scenario in SCENARIOS
    }
    for fold_index, test_start in enumerate((0.4, 0.5, 0.6, 0.7, 0.8, 0.9), start=1):
        trained, train_rows = select_on_period(
            specs, regime_universes, pullback, base, 0.0, test_start
        )
        pd.DataFrame(train_rows).to_csv(
            OUTPUT_DIR / f"walk_forward_fold_{fold_index}_selection.csv", index=False
        )
        universe = regime_universes[
            (trained.btc_ema_period, trained.momentum_period, trained.top_n)
        ]
        for scenario, costs in SCENARIOS.items():
            result = cached_portfolio(
                universe,
                build_sleeves(pullback, base, trained, *costs),
                start_ratio=test_start,
                end_ratio=test_start + 0.1,
            )
            forward_results[scenario].append(result)
            forward_rows.append(
                {
                    "fold": fold_index,
                    "test_start_ratio": test_start,
                    "test_end_ratio": test_start + 0.1,
                    **result_row(trained, scenario, "forward_test", result),
                }
            )
    forward = pd.DataFrame(forward_rows)
    forward.to_csv(OUTPUT_DIR / "walk_forward_folds.csv", index=False)
    aggregate_rows = [
        {"scenario": scenario, **combine_forward_results(results)}
        for scenario, results in forward_results.items()
    ]
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUTPUT_DIR / "walk_forward_aggregate.csv", index=False)

    double_full = phases[
        (phases["scenario"] == "double_cost") & (phases["phase"] == "full_period")
    ].iloc[0]
    triple_full = phases[
        (phases["scenario"] == "triple_cost") & (phases["phase"] == "full_period")
    ].iloc[0]
    double_holdout = phases[
        (phases["scenario"] == "double_cost") & (phases["phase"] == "holdout")
    ].iloc[0]
    double_annual = annual[annual["scenario"] == "double_cost"]
    double_leaveout = leaveout[leaveout["scenario"] == "double_cost"]
    double_forward = aggregate[aggregate["scenario"] == "double_cost"].iloc[0]
    positive_forward_folds = int(
        (
            (forward["scenario"] == "double_cost")
            & (forward["return_pct"] > 0.0)
        ).sum()
    )
    positive_sensitivity_pct = float(
        (sensitivity["return_pct"] > 0.0).mean() * 100.0
    )
    checks = {
        "double_cost_full_sharpe_at_least_0_75": bool(double_full["sharpe"] >= 0.75),
        "triple_cost_full_positive_and_sharpe_at_least_0_50": bool(
            triple_full["return_pct"] > 0.0 and triple_full["sharpe"] >= 0.50
        ),
        "double_cost_holdout_positive": bool(double_holdout["return_pct"] > 0.0),
        "double_cost_at_least_four_positive_years": bool(
            (double_annual["return_pct"] > 0.0).sum() >= 4
        ),
        "double_cost_leave_one_symbol_out_all_positive": bool(
            (double_leaveout["return_pct"] > 0.0).all()
        ),
        "double_cost_walk_forward_positive_sharpe_and_four_folds": bool(
            double_forward["return_pct"] > 0.0
            and double_forward["sharpe"] >= 0.50
            and positive_forward_folds >= 4
        ),
        "double_cost_parameter_neighborhood_majority_positive_on_holdout": bool(
            positive_sensitivity_pct >= 60.0
        ),
    }
    historical_maturity_pass = all(checks.values())
    if historical_maturity_pass:
        save_portfolio(
            build_sleeves(pullback, base, selected, *SCENARIOS["normal"]),
            OUTPUT_DIR / "paper_portfolio.json",
        )
    metadata = {
        "market": "Binance USD-M perpetual",
        "bars": "4h",
        "period": "five years ending at the latest cached candle",
        "candidate_family_locked_for_this_run": [asdict(spec) for spec in specs],
        "selected_using": (
            "first 60 percent calibration and next 20 percent validation only; "
            "candidate ranked by its worst normal/double-cost Sharpe in these periods"
        ),
        "selected_candidate": selected.name,
        "selected_spec": asdict(selected),
        "historical_maturity_checks": checks,
        "historical_maturity_pass": historical_maturity_pass,
        "positive_double_cost_forward_folds": positive_forward_folds,
        "positive_double_cost_holdout_sensitivity_pct": positive_sensitivity_pct,
        "funded_trading_ready": False,
        "funded_trading_blocker": (
            "No retrospective optimization can replace a locked, post-selection "
            "forward period with observed fills and slippage."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Selected early-period candidate: {selected.name}")
    print(phases[["scenario", "phase", "return_pct", "sharpe", "max_drawdown_pct", "trades"]].to_string(index=False))
    print(aggregate[["scenario", "return_pct", "sharpe", "max_drawdown_pct", "trades"]].to_string(index=False))
    print(json.dumps(checks, indent=2))
    print(f"Historical maturity pass: {historical_maturity_pass}; funded trading ready: False")


if __name__ == "__main__":
    main()
