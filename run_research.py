from __future__ import annotations

import argparse
from pathlib import Path

from smi_lab.config import StrategyConfig, load_config
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.evolution import annual_evaluations, evolve_strategy
from smi_lab.regime import attach_cboe_regime, cboe_regime_history
from smi_lab.reporting import save_research_outputs


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Research a robust multi-asset SMI strategy.")
    command.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    command.add_argument("--interval", default="4h", choices=["15m", "1h", "4h", "1d"])
    command.add_argument("--years", type=int, choices=range(1, 6), default=5)
    command.add_argument("--bars", type=int, default=None)
    command.add_argument("--candidates", type=int, default=192)
    command.add_argument("--shortlist", type=int, default=24)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--baseline-config", default=None)
    command.add_argument("--output-dir", default="outputs")
    command.add_argument("--market", choices=["spot", "perpetual"], default="spot")
    command.add_argument("--cboe-regime-search", action="store_true")
    command.add_argument("--refresh", action="store_true")
    command.add_argument("--fee-bps", type=float, default=10.0)
    command.add_argument("--slippage-bps", type=float, default=5.0)
    return command


def main() -> None:
    args = parser().parse_args()
    bars = args.bars or bars_for_years(args.interval, args.years)
    universe = load_universe(
        args.symbols,
        interval=args.interval,
        bars=bars,
        refresh=args.refresh,
        market=args.market,
        include_funding=args.market == "perpetual",
    )
    if args.cboe_regime_search:
        regime = cboe_regime_history(refresh=args.refresh)
        universe = {
            symbol: attach_cboe_regime(frame, regime)
            for symbol, frame in universe.items()
        }
    baseline_path = Path(
        args.baseline_config
        or (
            "outputs/baseline_strategy.json"
            if Path("outputs/baseline_strategy.json").exists()
            else "outputs/best_strategy.json"
        )
    )
    baseline_config = (
        load_config(baseline_path) if baseline_path.exists() else StrategyConfig()
    )
    base = baseline_config.with_costs(args.fee_bps, args.slippage_bps)
    result = evolve_strategy(
        universe,
        base=base,
        candidate_count=args.candidates,
        shortlist=args.shortlist,
        seed=args.seed,
        regime_modes=(
            ("none", "avoid_risk_off_longs", "risk_aligned")
            if args.cboe_regime_search
            else ("none",)
        ),
    )
    save_research_outputs(
        result,
        args.output_dir,
        universe=universe,
        baseline_config=baseline_config,
        research_context={
            "market": args.market,
            "funding_included": args.market == "perpetual",
            "cboe_regime_search": args.cboe_regime_search,
        },
    )
    print(f"Data: {data_window(next(iter(universe.values())))} | bars={bars}")
    print(
        f"Symbols: {', '.join(universe)} | timeframe: {args.interval} "
        f"| market={args.market} | funding={args.market == 'perpetual'} "
        f"| cboe_regime={args.cboe_regime_search}"
    )
    print(f"Split: {result.boundaries}")
    print(f"Best config: {result.best_config.to_dict()}")
    for name, evaluation in [
        ("training", result.training),
        ("validation", result.validation),
        ("holdout", result.holdout),
        ("full_period", result.full_period),
    ]:
        metrics = evaluation.metrics
        print(
            f"{name}: return={metrics['return_pct']:.2f}% "
            f"sharpe={metrics['sharpe']:.2f} dd={metrics['max_drawdown_pct']:.2f}% "
            f"trades={int(metrics['trades'])} pf={metrics['profit_factor']:.2f}"
        )
    print("Annual optimized windows:")
    for name, evaluation in annual_evaluations(universe, result.best_config):
        metrics = evaluation.metrics
        print(
            f"{name}: return={metrics['return_pct']:.2f}% "
            f"sharpe={metrics['sharpe']:.2f} dd={metrics['max_drawdown_pct']:.2f}% "
            f"trades={int(metrics['trades'])} pf={metrics['profit_factor']:.2f}"
        )


if __name__ == "__main__":
    main()
