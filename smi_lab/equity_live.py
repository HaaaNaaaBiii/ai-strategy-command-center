from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


LIVE_STRATEGY_VERSION = "live-desk-v3-expanded-universe-rebalance-2026-06-02"

LIVE_ORDER_COLUMNS = [
    "account_id",
    "broker",
    "market",
    "currency",
    "symbol",
    "company",
    "action",
    "target_weight",
    "target_value",
    "current_value",
    "delta_value",
    "side",
    "status",
    "reference_price",
    "order_quantity",
    "notes",
]


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame:
        return pd.Series([default] * len(frame), index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_live_strategy_memory(
    path: str | Path,
    strategy_version: str = LIVE_STRATEGY_VERSION,
) -> dict[str, object]:
    target = Path(path)
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    if payload.get("strategy_version") == strategy_version:
        payload.setdefault("accounts", {})
        return payload
    return {
        "strategy_version": strategy_version,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "reset_reason": "strategy_version_changed",
        "previous_strategy_version": payload.get("strategy_version"),
        "accounts": {},
    }


def save_live_strategy_memory(path: str | Path, memory: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    memory["updated_at"] = utc_now()
    target.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def remember_live_plan(
    path: str | Path,
    account_id: str,
    market: str,
    plan: pd.DataFrame,
    strategy_version: str = LIVE_STRATEGY_VERSION,
) -> dict[str, object]:
    memory = load_live_strategy_memory(path, strategy_version)
    accounts = memory.setdefault("accounts", {})
    symbols = plan["symbol"].astype(str).tolist() if not plan.empty and "symbol" in plan else []
    sides = plan["side"].astype(str).tolist() if not plan.empty and "side" in plan else []
    accounts[account_id] = {
        "market": market,
        "last_generated_at": utc_now(),
        "symbols": symbols,
        "sides": sides,
        "row_count": int(len(plan)),
    }
    save_live_strategy_memory(path, memory)
    return memory


def strategy_recommendations_for_market(
    recommendations: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame(columns=recommendations.columns)
    frame = recommendations.copy()
    if "market" in frame:
        frame = frame[frame["market"].astype(str) == market]
    selected = _bool_series(frame, "selected", True)
    frame = frame[selected]
    if "action" in frame:
        frame = frame[
            frame["action"].astype(str).isin(["ROTATION_REBALANCE", "WAIT_FOR_BREAKOUT", "BUY", "HOLD"])
        ]
    if "rank" in frame:
        frame = frame.sort_values("rank")
    return frame.reset_index(drop=True)


def _position_values(positions: pd.DataFrame, market: str, account_id: str) -> dict[str, float]:
    if positions.empty:
        return {}
    frame = positions.copy()
    if "market" in frame:
        frame = frame[frame["market"].astype(str) == market]
    if "account_id" in frame:
        frame = frame[frame["account_id"].astype(str) == account_id]
    if frame.empty or "symbol" not in frame:
        return {}
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    market_value = pd.to_numeric(frame.get("market_value", pd.NA), errors="coerce")
    fallback = (
        pd.to_numeric(frame.get("quantity", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(frame.get("current_price", 0.0), errors="coerce").fillna(0.0)
    )
    frame["resolved_value"] = market_value.fillna(fallback).fillna(0.0)
    return frame.groupby("symbol")["resolved_value"].sum().to_dict()


def build_equity_live_order_plan(
    recommendations: pd.DataFrame,
    positions: pd.DataFrame,
    market: str,
    account_id: str,
    broker: str,
    currency: str,
    capital: float,
    min_trade_value: float = 100.0,
) -> pd.DataFrame:
    """Build strategy order intents for the equity live sleeves without submitting broker orders."""
    if capital <= 0:
        return pd.DataFrame(columns=LIVE_ORDER_COLUMNS)
    picks = strategy_recommendations_for_market(recommendations, market)
    current_values = _position_values(positions, market, account_id)
    if picks.empty:
        rows = [
            {
                "account_id": account_id,
                "broker": broker,
                "market": market,
                "currency": currency,
                "symbol": "CASH",
                "company": "Cash",
                "action": "HOLD_CASH",
                "target_weight": 1.0,
                "target_value": capital,
                "current_value": 0.0,
                "delta_value": 0.0,
                "side": "HOLD",
                "status": "NO_STRATEGY_PICK",
                "reference_price": 1.0,
                "order_quantity": 0.0,
                "notes": "No current strategy recommendation; keep the sleeve in cash until the rotation model selects assets.",
            }
        ]
        return pd.DataFrame(rows, columns=LIVE_ORDER_COLUMNS)

    target_weight = 1.0 / len(picks)
    rows: list[dict[str, object]] = []
    target_symbols = set()
    for row in picks.to_dict("records"):
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        target_symbols.add(symbol)
        close = _float(row.get("close"))
        reference = close
        target_value = capital * target_weight
        current_value = float(current_values.get(symbol, 0.0))
        delta = target_value - current_value
        if delta > min_trade_value:
            side = "BUY"
            status = "PLANNED_REBALANCE_BUY"
            quantity = delta / reference if reference > 0 else 0.0
        elif delta < -min_trade_value:
            side = "SELL"
            status = "PLANNED_REDUCE"
            quantity = abs(delta) / reference if reference > 0 else 0.0
        else:
            side = "HOLD"
            status = "IN_BAND"
            quantity = 0.0
        rows.append(
            {
                "account_id": account_id,
                "broker": broker,
                "market": market,
                "currency": currency,
                "symbol": symbol,
                "company": row.get("company", symbol),
                "action": row.get("action", "ROTATION_REBALANCE"),
                "target_weight": target_weight,
                "target_value": target_value,
                "current_value": current_value,
                "delta_value": delta,
                "side": side,
                "status": status,
                "reference_price": reference,
                "order_quantity": quantity,
                "notes": "Generated from the backtested rotation/rebalance model. Reference price is latest scan close, not a separate entry signal.",
            }
        )

    for symbol, current_value in current_values.items():
        if symbol in target_symbols or current_value <= min_trade_value:
            continue
        rows.append(
            {
                "account_id": account_id,
                "broker": broker,
                "market": market,
                "currency": currency,
                "symbol": symbol,
                "company": symbol,
                "action": "EXIT_NOT_SELECTED",
                "target_weight": 0.0,
                "target_value": 0.0,
                "current_value": current_value,
                "delta_value": -current_value,
                "side": "SELL",
                "status": "EXIT_NOT_IN_STRATEGY",
                "reference_price": 0.0,
                "order_quantity": 0.0,
                "notes": "Current holding is not in latest strategy picks.",
            }
        )

    return pd.DataFrame(rows, columns=LIVE_ORDER_COLUMNS)


def save_live_order_plan(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in LIVE_ORDER_COLUMNS:
        if column not in output:
            output[column] = None
    output[LIVE_ORDER_COLUMNS].to_csv(target, index=False)
