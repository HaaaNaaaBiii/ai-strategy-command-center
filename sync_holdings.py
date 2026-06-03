from __future__ import annotations

import argparse
from pathlib import Path

from smi_lab.broker_import import DEFAULT_IMPORT_DIR, sync_broker_exports
from smi_lab.paths import output_path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Automatically import Firstrade/Cathay position CSV exports."
    )
    command.add_argument("--import-dir", default=str(DEFAULT_IMPORT_DIR))
    command.add_argument("--positions-path", default=str(output_path("accounts", "positions.csv")))
    return command


def main() -> None:
    args = parser().parse_args()
    positions, report = sync_broker_exports(
        import_dir=Path(args.import_dir),
        positions_path=Path(args.positions_path),
    )
    print("[import report]")
    print(report.to_string(index=False) if not report.empty else "No CSV files found.")
    print("[positions]")
    print(positions.tail(50).to_string(index=False) if not positions.empty else "No positions imported.")


if __name__ == "__main__":
    main()
