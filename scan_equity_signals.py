from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smi_lab.equity_scanner import run_equity_scan


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
    return command


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
        pd.concat(recommendation_frames, ignore_index=True).to_csv(
            latest_recommendations_path,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(latest_recommendations_path, index=False)


if __name__ == "__main__":
    main()
