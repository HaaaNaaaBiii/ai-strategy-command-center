from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_STORAGE_CONFIG = PROJECT_ROOT / "storage.local.json"


def storage_root() -> Path:
    configured = os.environ.get("AI_STRATEGY_STORAGE_ROOT") or os.environ.get("SMI_LAB_STORAGE_DIR")
    if configured:
        return Path(configured).expanduser()
    if LOCAL_STORAGE_CONFIG.exists():
        try:
            payload = json.loads(LOCAL_STORAGE_CONFIG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        root = payload.get("storage_root") if isinstance(payload, dict) else None
        if root:
            return Path(str(root)).expanduser()
    return PROJECT_ROOT


def data_path(*parts: str) -> Path:
    return storage_root().joinpath("data", *parts)


def output_path(*parts: str) -> Path:
    return storage_root().joinpath("outputs", *parts)
