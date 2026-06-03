from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smi_lab.equity_scanner import run_equity_scan
from smi_lab.notifier import resolve_discord_mention, send_discord


OUTPUT_DIR = Path("outputs/equity_scan")


def _read_json_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Scan broad Taiwan/U.S. equity universes and write current strategy recommendations."
    )
    command.add_argument("--market", choices=["tw", "us", "both"], default="both")
    command.add_argument("--interval", default="1d", choices=["1d", "1wk", "1h"])
    command.add_argument("--range", dest="range_", default="2y")
    command.add_argument("--top", type=int, default=3)
    command.add_argument("--refresh", action="store_true")
    command.add_argument("--output-dir", default=str(OUTPUT_DIR))
    command.add_argument("--symbols", nargs="*", help="Optional explicit symbol list for one market.")
    command.add_argument("--channel", choices=["none", "discord"], default="none")
    command.add_argument("--webhook-url", default="")
    command.add_argument("--mention", default=resolve_discord_mention())
    command.add_argument(
        "--session",
        default="",
        choices=["", "premarket", "intraday", "postclose"],
        help="Optional scan session label for notifications.",
    )
    return command


def _format_number(value: object, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _format_equity_scan_message(
    summaries: list[dict[str, object]],
    recommendations: pd.DataFrame,
    mention: str = "",
    session: str = "",
) -> str:
    market_names = {"tw": "台股", "us": "美股"}
    session_names = {
        "premarket": "盤前觀察",
        "intraday": "盤中觀察",
        "postclose": "盤後正式更新",
    }
    markets = [str(item.get("market", "")).lower() for item in summaries]
    title_market = " / ".join(market_names.get(market, market.upper()) for market in markets if market)
    title_market = title_market or "股票"
    session_label = session_names.get(session, "")
    title = f"{title_market}{session_label}策略選股" if session_label else f"{title_market}今日策略選股"
    lines = [
        f"{mention + ' ' if mention else ''}{title}",
        "資料基準：最新可取得日線資料；策略為輪動/再平衡，不含未回測的進出場價位層。",
    ]
    for summary in summaries:
        market = str(summary.get("market", "")).lower()
        label = market_names.get(market, market.upper() or "Market")
        loaded = int(summary.get("loaded_symbols", 0) or 0)
        failed = int(summary.get("failed_symbols", 0) or 0)
        lines.append(f"\n[{label}] loaded={loaded}, failed={failed}")
        market_rows = (
            recommendations[recommendations["market"].astype(str).str.lower() == market]
            if not recommendations.empty and "market" in recommendations
            else pd.DataFrame()
        )
        if market_rows.empty:
            lines.append("- HOLD_CASH：目前沒有符合策略 TopN 的標的。")
            continue
        for row in market_rows.head(5).to_dict("records"):
            symbol = str(row.get("symbol", "-"))
            company = str(row.get("company", ""))
            rank = _format_number(row.get("rank"), digits=0)
            score = _format_number(row.get("score"), digits=2)
            reference = _format_number(row.get("reference_price", row.get("close")), digits=2)
            reason = str(row.get("reason", "")).strip()
            if len(reason) > 180:
                reason = reason[:177] + "..."
            lines.append(
                f"- {symbol} {company} | rank {rank} | score {score} | ref {reference}"
            )
            if reason:
                lines.append(f"  {reason}")
    return "\n".join(lines)


def main() -> None:
    args = parser().parse_args()
    markets = ("tw", "us") if args.market == "both" else (args.market,)
    summaries: list[dict[str, object]] = []
    recommendation_frames: list[pd.DataFrame] = []
    for market in markets:
        symbols = args.symbols if args.symbols and len(markets) == 1 else None
        summary = run_equity_scan(
            market,
            interval=args.interval,
            range_=args.range_,
            refresh=args.refresh,
            top_n=args.top,
            output_dir=args.output_dir,
            symbols=symbols,
        )
        summaries.append(summary)
        path = Path(args.output_dir) / f"{market}_recommendations.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame.insert(1, "market", market)
                recommendation_frames.append(frame)
        print(f"[{market}] {summary['status']} | loaded={summary['loaded_symbols']} failed={summary['failed_symbols']}")
        print(f"recommended: {', '.join(summary['recommended_symbols']) or 'HOLD_CASH'}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latest_summary_path = output / "latest_scan_summary.json"
    if args.market != "both":
        existing = [
            item for item in _read_json_list(latest_summary_path)
            if str(item.get("market")) not in set(markets)
        ]
        summaries = [*existing, *summaries]
    (output / "latest_scan_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest_recommendations_path = output / "latest_recommendations.csv"
    if args.market != "both" and latest_recommendations_path.exists():
        existing_frame = pd.read_csv(latest_recommendations_path)
        if not existing_frame.empty and "market" in existing_frame:
            existing_frame = existing_frame[~existing_frame["market"].astype(str).isin(markets)]
            if not existing_frame.empty:
                recommendation_frames.insert(0, existing_frame)
    if recommendation_frames:
        latest_recommendations = pd.concat(recommendation_frames, ignore_index=True)
        latest_recommendations.to_csv(latest_recommendations_path, index=False)
    else:
        latest_recommendations = pd.DataFrame()
        pd.DataFrame().to_csv(latest_recommendations_path, index=False)
    if args.channel == "discord":
        message = _format_equity_scan_message(
            summaries,
            latest_recommendations,
            mention=args.mention,
            session=args.session,
        )
        send_discord(message, webhook_url=args.webhook_url or None)


if __name__ == "__main__":
    main()
