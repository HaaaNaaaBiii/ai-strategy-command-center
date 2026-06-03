from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smi_lab.config import StrategyConfig, load_config, load_portfolio
from smi_lab.crypto_universe import crypto_scan_symbols, load_crypto_scan_universe
from smi_lab.data import DEFAULT_SYMBOLS
from smi_lab.notifier import format_signal, send_discord, send_telegram
from smi_lab.paths import data_path, output_path
from smi_lab.paper import (
    allocation_snapshot,
    format_allocation_report,
    load_allocation_strategy,
    update_forward_tracking,
)
from smi_lab.regime import attach_btc_momentum_regime
from smi_lab.strategy import Signal, latest_signal


MATURITY_PORTFOLIO = output_path("maturity_candidate", "paper_portfolio.json")
PRACTICAL_PORTFOLIO = output_path("practical_candidate", "paper_portfolio.json")
RESEARCH_PORTFOLIO = output_path("futures_regime", "paper_portfolio.json")
LEGACY_PORTFOLIO = output_path("deployed_portfolio.json")
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
    command.add_argument(
        "--crypto-universe",
        choices=["top100", "core", "symbols"],
        default="top100",
        help="Crypto scan universe: CoinGecko market-cap top list, core BTC/ETH/DOGE/SOL, or explicit --symbols.",
    )
    command.add_argument("--crypto-limit", type=int, default=100)
    command.add_argument(
        "--include-funding",
        action="store_true",
        help="Attach funding rates during crypto allocation scans. Disabled by default for broad top100 scans.",
    )
    command.add_argument("--interval", default="4h", choices=["15m", "1h", "4h", "1d"])
    command.add_argument("--bars", type=int, default=500)
    command.add_argument("--config", default=str(output_path("best_strategy.json")))
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


def resolved_crypto_symbols(args: argparse.Namespace) -> list[str]:
    if args.crypto_universe == "core":
        return list(DEFAULT_SYMBOLS)
    if args.crypto_universe == "symbols":
        return [symbol.upper() for symbol in args.symbols]
    return list(
        crypto_scan_symbols(
            limit=args.crypto_limit,
            refresh=args.refresh,
        )
    )


def scan(args: argparse.Namespace) -> list[tuple[str, float, Signal]]:
    sleeves = configured_sleeves(args)
    requested_symbols = resolved_crypto_symbols(args)
    requires_btc_regime = any(
        config.regime_source == "btc_momentum" for _, _, config in sleeves
    )
    symbols = list(dict.fromkeys([*DEFAULT_SYMBOLS, *requested_symbols]))
    universe, _ = load_crypto_scan_universe(
        symbols,
        interval=args.interval,
        bars=args.bars,
        refresh=args.refresh,
        market=args.market,
    )
    if "BTCUSDT" not in universe:
        raise RuntimeError("BTCUSDT is required for the crypto signal risk gate.")
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
        if symbol in universe
        if (signal := latest_signal(
            symbol.upper(),
            universe[symbol],
            config,
        )) is not None
    ]


def allocation_report(args: argparse.Namespace) -> tuple[str, str]:
    config, offsets, _ = load_allocation_strategy()
    symbols = resolved_crypto_symbols(args)
    min_bars = max(config.momentum_period, config.asset_ema_period, config.btc_ema_period) + 3
    universe, failures = load_crypto_scan_universe(
        symbols,
        limit=args.crypto_limit,
        interval=args.interval,
        bars=max(args.bars, 500),
        refresh=args.refresh,
        market="perpetual",
        include_funding=args.include_funding,
        min_bars=min_bars,
    )
    if "BTCUSDT" not in universe:
        raise RuntimeError("BTCUSDT is required for the crypto allocation risk gate.")
    update = update_forward_tracking(universe)
    snapshot = allocation_snapshot(universe, config, offsets)
    marker = str(snapshot.attrs.get("latest_candle"))
    message = format_allocation_report(snapshot, update)
    lines = [
        message,
        "",
        f"Universe mode: {args.crypto_universe}",
        f"Loaded symbols: {len(universe)}",
        f"Failed symbols: {len(failures)}",
    ]
    if not failures.empty:
        failed_symbols = ", ".join(failures["symbol"].astype(str).head(15).tolist())
        suffix = " ..." if len(failures) > 15 else ""
        lines.append(f"Failed sample: {failed_symbols}{suffix}")
    return "\n".join(lines), marker


def send_message(channel: str, message: str) -> None:
    if channel == "discord":
        send_discord(message)
    elif channel == "telegram":
        send_telegram(message)
    else:
        raise ValueError(f"Unsupported notification channel: {channel}")


def main() -> None:
    args = parser().parse_args()
    state_file = data_path("notification_state.json")
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
