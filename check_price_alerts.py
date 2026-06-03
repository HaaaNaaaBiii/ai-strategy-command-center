from __future__ import annotations

import argparse

from smi_lab.notifier import resolve_discord_mention, resolve_discord_webhook_url
from smi_lab.paths import output_path
from smi_lab.price_alerts import check_equity_price_alerts


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Check equity scan levels and send Discord alerts.")
    command.add_argument("--recommendations", default=str(output_path("equity_scan", "latest_recommendations.csv")))
    command.add_argument("--state", default=str(output_path("alerts", "equity_price_alerts_state.json")))
    command.add_argument("--webhook-url", default=resolve_discord_webhook_url())
    command.add_argument("--mention", default=resolve_discord_mention())
    command.add_argument("--no-refresh", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument(
        "--legacy-level-alerts",
        action="store_true",
        help="Explicitly enable old entry/stop/TP price-level alerts. Disabled by default.",
    )
    return command


def main() -> None:
    args = parser().parse_args()
    if not args.legacy_level_alerts:
        print(
            "Equity price-level alerts are disabled because entry/stop/TP/RR "
            "are not part of the current backtested live strategy."
        )
        return
    if not args.dry_run and not args.webhook_url:
        print("DISCORD_WEBHOOK_URL is not configured; alert check skipped.")
        return
    events = check_equity_price_alerts(
        recommendations_path=args.recommendations,
        state_path=args.state,
        webhook_url=args.webhook_url or None,
        mention=args.mention,
        refresh=not args.no_refresh,
        notify=not args.dry_run,
        record_state=not args.dry_run,
    )
    if not events:
        print("No price alert triggered.")
        return
    for event in events:
        print(
            f"{event.market} {event.symbol} {event.level} "
            f"target={event.target_price:.4f} last={event.last_price:.4f}"
        )


if __name__ == "__main__":
    main()
