from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smi_lab.config import StrategyConfig, load_config, load_portfolio
from smi_lab.data import DEFAULT_SYMBOLS, load_universe
from smi_lab.notifier import format_signal, send_discord, send_telegram
from smi_lab.paper import (
    allocation_snapshot,
    format_allocation_report,
    load_allocation_strategy,
    update_forward_tracking,
)
from smi_lab.regime import attach_btc_momentum_regime
from smi_lab.strategy import Signal, latest_signal


MATURITY_PORTFOLIO = Path("outputs/maturity_candidate/paper_portfolio.json")
PRACTICAL_PORTFOLIO = Path("outputs/practical_candidate/paper_portfolio.json")
RESEARCH_PORTFOLIO = Path("outputs/futures_regime/paper_portfolio.json")
LEGACY_PORTFOLIO = Path("outputs/deployed_portfolio.json")
DEFAULT_PORTFOLIO = (
    MATURITY_PORTFOLIO
    if MATURITY_PORTFOLIO.exists()
    else PRACTICAL_PORTFOLIO
    if PRACTICAL_PORTFOLIO.exists()
    else RESEARCH_PORTFOLIO
    if RESEARCH_PORTFOLIO.exists()
    else LEGACY_PORTFOLIO
)
DEFAULT_MARKET = "perpetual" if DEFAULT_PORTFOLIO != LEGACY_PORTFOLIO else "spot"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Scan fresh SMI entry signals and optionally notify.")
    command.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    command.add_argument("--interval", default="4h", choices=["15m", "1h", "4h", "1d"])
    command.add_argument("--bars", type=int, default=500)
    command.add_argument("--config", default="outputs/best_strategy.json")
    command.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO))
    command.add_argument("--market", choices=["spot", "perpetual"], default=DEFAULT_MARKET)
    command.add_argument("--channel", choices=["none", "discord", "telegram"], default="none")
    command.add_argument("--strategy", choices=["smi", "allocation", "both"], default="smi")
    command.add_argument("--refresh", action="store_true")
    command.add_argument("--force", action="store_true")
    return command


def load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def configured_sleeves(args: argparse.Namespace) -> list[tuple[str, float, StrategyConfig]]:
    portfolio = Path(args.portfolio)
    if portfolio.exists():
        return load_portfolio(portfolio)
    return [("single_strategy", 1.0, load_config(args.config))]


def scan(args: argparse.Namespace) -> list[tuple[str, float, Signal]]:
    sleeves = configured_sleeves(args)
    requested_symbols = [symbol.upper() for symbol in args.symbols]
    requires_btc_regime = any(
        config.regime_source == "btc_momentum" for _, _, config in sleeves
    )
    symbols = list(dict.fromkeys(
        list(DEFAULT_SYMBOLS) if requires_btc_regime else requested_symbols
    ))
    universe = load_universe(
        symbols, args.interval, args.bars, refresh=args.refresh, market=args.market
    )
    if requires_btc_regime:
        settings = {
            (config.btc_ema_period, config.momentum_period, config.momentum_top_n)
            for _, _, config in sleeves
            if config.regime_source == "btc_momentum"
        }
        if len(settings) != 1:
            raise ValueError("BTC momentum sleeves must share the same regime settings.")
        ema_period, momentum_period, top_n = settings.pop()
        universe = attach_btc_momentum_regime(
            universe, ema_period, momentum_period, top_n
        )
    return [
        (sleeve_name, weight, signal)
        for sleeve_name, weight, config in sleeves
        for symbol in requested_symbols
        if (signal := latest_signal(
            symbol.upper(),
            universe[symbol],
            config,
        )) is not None
    ]


def allocation_report(args: argparse.Namespace) -> tuple[str, str]:
    config, offsets, _ = load_allocation_strategy()
    universe = load_universe(
        list(DEFAULT_SYMBOLS),
        args.interval,
        max(args.bars, 500),
        refresh=args.refresh,
        market="perpetual",
        include_funding=True,
    )
    update = update_forward_tracking(universe)
    snapshot = allocation_snapshot(universe, config, offsets)
    marker = str(snapshot.attrs.get("latest_candle"))
    return format_allocation_report(snapshot, update), marker


def send_message(channel: str, message: str) -> None:
    if channel == "discord":
        send_discord(message)
    elif channel == "telegram":
        send_telegram(message)
    else:
        raise ValueError(f"Unsupported notification channel: {channel}")


def main() -> None:
    args = parser().parse_args()
    state_file = Path("data/notification_state.json")
    state = load_state(state_file)
    signals: list[tuple[str, float, Signal]] = []
    if args.strategy in {"smi", "both"}:
        signals = scan(args)
    if args.strategy in {"smi", "both"} and not signals:
        print("No new entry setup on the latest closed candle.")
    for sleeve_name, weight, signal in signals:
        message = (
            f"Portfolio sleeve: {sleeve_name} | risk allocation: {weight:.0%}\n"
            f"{format_signal(signal)}"
        )
        key = f"{sleeve_name}:{signal.symbol}:{args.interval}:{signal.side}"
        marker = pd.Timestamp(signal.candle_time).isoformat()
        fresh = args.force or state.get(key) != marker
        print(message)
        print()
        if args.channel != "none" and fresh:
            send_message(args.channel, message)
            state[key] = marker
            print(f"Sent via {args.channel}: {key} {marker}")
        elif args.channel != "none":
            print(f"Skipped duplicate notification: {key} {marker}")
    if args.strategy in {"allocation", "both"}:
        message, marker = allocation_report(args)
        key = f"allocation:market_alpha_staggered:{args.interval}"
        fresh = args.force or state.get(key) != marker
        print(message)
        print()
        if args.channel != "none" and fresh:
            send_message(args.channel, message)
            state[key] = marker
            print(f"Sent via {args.channel}: {key} {marker}")
        elif args.channel != "none":
            print(f"Skipped duplicate notification: {key} {marker}")
    if args.channel != "none":
        save_state(state_file, state)


if __name__ == "__main__":
    main()
