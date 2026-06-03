from __future__ import annotations

import argparse
import json

from smi_lab.attention_strategy import run_attention_research


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Backtest the alternative-data attention strategy."
    )
    command.add_argument("--range", dest="range_", default="2y")
    command.add_argument("--refresh", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    report = run_attention_research(range_=args.range_, refresh=args.refresh)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
