from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_TARGET = Path("E:/AI_Strategy_Command_Center")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Copy local data/outputs storage to another drive and write local storage config.")
    command.add_argument("--target", default=str(DEFAULT_TARGET))
    command.add_argument("--move", action="store_true", help="Delete source data/outputs after a successful copy.")
    return command


def copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(source, target, dirs_exist_ok=True)


def main() -> None:
    args = parser().parse_args()
    root = Path(args.target).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("data", "outputs"):
        source = Path(name)
        target = root / name
        copy_tree(source, target)
        if args.move and source.exists():
            shutil.rmtree(source)
    Path("storage.local.json").write_text(
        json.dumps({"storage_root": str(root)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"storage_root={root}")
    print("storage.local.json written. Restart Streamlit and scheduled scans to use the new storage root.")


if __name__ == "__main__":
    main()
