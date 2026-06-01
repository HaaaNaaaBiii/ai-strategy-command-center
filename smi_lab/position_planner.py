from __future__ import annotations

from typing import Mapping

import pandas as pd


PLAN_COLUMNS = [
    "account_id",
    "market",
    "symbol",
    "target_weight",
    "target_value",
    "current_value",
    "delta_value",
    "side",
    "reference_price",
    "order_quantity",
    "status",
    "notes",
]


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _target_symbol_column(target_allocations: pd.DataFrame) -> str:
    if "symbol" in target_allocations.columns:
        return "symbol"
    if "asset" in target_allocations.columns:
        return "asset"
    raise ValueError("Target allocations require either a symbol or asset column.")


def _market_value(frame: pd.DataFrame) -> pd.Series:
    if "market_value" in frame.columns:
        values = pd.to_numeric(frame["market_value"], errors="coerce")
    else:
        values = pd.Series([float("nan")] * len(frame), index=frame.index)
    fallback = (
        pd.to_numeric(frame.get("quantity", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(frame.get("current_price", 0.0), errors="coerce").fillna(0.0)
    )
    return values.fillna(fallback).fillna(0.0)


def _resolve_equity(
    accounts: pd.DataFrame,
    positions: pd.DataFrame,
    market: str,
    account_id: str | None,
) -> tuple[str, float, float]:
    account_frame = accounts.copy()
    if not account_frame.empty and "market" in account_frame.columns:
        account_frame = account_frame[account_frame["market"].astype(str) == market]
    if account_id:
        account_frame = account_frame[account_frame["account_id"].astype(str) == account_id]

    resolved_account = account_id or f"{market}-portfolio"
    if not account_frame.empty:
        latest = account_frame.tail(1).iloc[0]
        resolved_account = str(latest.get("account_id", resolved_account))
        equity = _float(latest.get("equity"))
        cash = _float(latest.get("cash"))
        if equity > 0:
            return resolved_account, equity, cash

    position_frame = positions.copy()
    if not position_frame.empty and "market" in position_frame.columns:
        position_frame = position_frame[position_frame["market"].astype(str) == market]
    if account_id:
        position_frame = position_frame[position_frame["account_id"].astype(str) == account_id]
    if not position_frame.empty:
        if "account_id" in position_frame.columns and not account_id:
            resolved_account = str(position_frame["account_id"].dropna().astype(str).iloc[0])
        return resolved_account, float(_market_value(position_frame).sum()), 0.0
    return resolved_account, 0.0, 0.0


def build_rebalance_plan(
    accounts: pd.DataFrame,
    positions: pd.DataFrame,
    target_allocations: pd.DataFrame,
    market: str,
    account_id: str | None = None,
    price_lookup: Mapping[str, float] | None = None,
    min_trade_value: float = 10.0,
) -> pd.DataFrame:
    """Compare tracked holdings with strategy target weights and produce order intents."""
    if target_allocations.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    symbol_column = _target_symbol_column(target_allocations)
    account, equity, cash = _resolve_equity(accounts, positions, market, account_id)
    if equity <= 0:
        return pd.DataFrame(columns=PLAN_COLUMNS)

    price_lookup = {key.upper(): float(value) for key, value in (price_lookup or {}).items()}
    targets: dict[str, float] = {}
    for row in target_allocations.to_dict("records"):
        symbol = str(row.get(symbol_column, "")).upper()
        if not symbol:
            continue
        targets[symbol] = targets.get(symbol, 0.0) + _float(row.get("target_weight"))

    position_frame = positions.copy()
    if not position_frame.empty and "market" in position_frame.columns:
        position_frame = position_frame[position_frame["market"].astype(str) == market]
    if account_id and not position_frame.empty and "account_id" in position_frame.columns:
        position_frame = position_frame[position_frame["account_id"].astype(str) == account_id]

    current_values: dict[str, float] = {}
    current_prices: dict[str, float] = {}
    if not position_frame.empty:
        position_frame = position_frame.copy()
        position_frame["symbol"] = position_frame["symbol"].astype(str).str.upper()
        position_frame["resolved_value"] = _market_value(position_frame)
        grouped = position_frame.groupby("symbol", dropna=False)
        current_values = grouped["resolved_value"].sum().to_dict()
        for symbol, group in grouped:
            quantity = pd.to_numeric(group.get("quantity", 0.0), errors="coerce").fillna(0.0).sum()
            value = float(group["resolved_value"].sum())
            explicit = pd.to_numeric(group.get("current_price", 0.0), errors="coerce").replace(0, pd.NA).dropna()
            if quantity:
                current_prices[str(symbol)] = abs(value / quantity)
            elif not explicit.empty:
                current_prices[str(symbol)] = float(explicit.iloc[-1])

    symbols = sorted((set(targets) | set(current_values)) - {"CASH"})
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        target_weight = max(targets.get(symbol, 0.0), 0.0)
        target_value = equity * target_weight
        current_value = float(current_values.get(symbol, 0.0))
        delta = target_value - current_value
        reference_price = price_lookup.get(symbol) or current_prices.get(symbol) or 0.0
        order_quantity = abs(delta) / reference_price if reference_price > 0 else 0.0
        if delta > min_trade_value:
            side = "BUY"
            status = "PLANNED_ENTRY"
        elif delta < -min_trade_value:
            side = "SELL"
            status = "PLANNED_REDUCE"
        else:
            side = "HOLD"
            status = "IN_BAND"
        rows.append(
            {
                "account_id": account,
                "market": market,
                "symbol": symbol,
                "target_weight": target_weight,
                "target_value": target_value,
                "current_value": current_value,
                "delta_value": delta,
                "side": side,
                "reference_price": reference_price,
                "order_quantity": order_quantity,
                "status": status,
                "notes": "Generated from strategy target allocation.",
            }
        )

    if "CASH" in targets or cash > 0:
        target_value = equity * max(targets.get("CASH", 0.0), 0.0)
        rows.append(
            {
                "account_id": account,
                "market": market,
                "symbol": "CASH",
                "target_weight": max(targets.get("CASH", 0.0), 0.0),
                "target_value": target_value,
                "current_value": cash,
                "delta_value": target_value - cash,
                "side": "HOLD",
                "reference_price": 1.0,
                "order_quantity": 0.0,
                "status": "CASH_TARGET",
                "notes": "Cash target; no broker order generated.",
            }
        )

    return pd.DataFrame(rows, columns=PLAN_COLUMNS).sort_values(
        ["status", "delta_value"], ascending=[True, False]
    )
