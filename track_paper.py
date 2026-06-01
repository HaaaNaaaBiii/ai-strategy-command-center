from __future__ import annotations

import argparse

from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, load_universe
from smi_lab.notifier import send_discord, send_telegram
from smi_lab.paper import (
    allocation_snapshot,
    format_allocation_report,
    load_allocation_strategy,
    update_forward_tracking,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Update forward paper tracking for the market alpha allocation."
    )
    command.add_argument("--interval", default="4h", choices=["1h", "4h", "1d"])
    command.add_argument("--years", type=int, default=2)
    command.add_argument("--refresh", action="store_true")
    command.add_argument("--channel", choices=["none", "discord", "telegram"], default="none")
    command.add_argument("--initial-equity", type=float, default=10_000.0)
    return command


def main() -> None:
    args = parser().parse_args()
    config, offsets, _ = load_allocation_strategy()
    universe = load_universe(
        DEFAULT_SYMBOLS,
        interval=args.interval,
        bars=bars_for_years(args.interval, args.years),
        refresh=args.refresh,
        market="perpetual",
        include_funding=True,
    )
    update = update_forward_tracking(universe, initial_equity=args.initial_equity)
    snapshot = allocation_snapshot(universe, config, offsets)
    message = format_allocation_report(snapshot, update)
    print(message)
    if args.channel == "discord":
        send_discord(message)
    elif args.channel == "telegram":
        send_telegram(message)


if __name__ == "__main__":
    main()
