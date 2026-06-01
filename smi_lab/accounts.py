from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ACCOUNT_COLUMNS = [
    "account_id",
    "broker",
    "market",
    "currency",
    "cash",
    "equity",
    "updated_at",
    "notes",
]
POSITION_COLUMNS = [
    "account_id",
    "broker",
    "market",
    "symbol",
    "company",
    "quantity",
    "average_price",
    "current_price",
    "market_value",
    "unrealized_pnl",
    "updated_at",
    "notes",
]
ORDER_COLUMNS = [
    "created_at",
    "account_id",
    "broker",
    "market",
    "symbol",
    "company",
    "side",
    "status",
    "quantity",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "strategy",
    "notes",
]


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    broker: str
    market: str
    currency: str
    cash: float
    equity: float
    notes: str = ""
    updated_at: str = ""

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["updated_at"] = row["updated_at"] or utc_now()
        return row


@dataclass(frozen=True)
class PositionSnapshot:
    account_id: str
    broker: str
    market: str
    symbol: str
    company: str
    quantity: float
    average_price: float
    current_price: float
    notes: str = ""
    updated_at: str = ""

    def to_row(self) -> dict[str, object]:
        quantity = float(self.quantity)
        average = float(self.average_price)
        current = float(self.current_price)
        row = asdict(self)
        row["market_value"] = quantity * current
        row["unrealized_pnl"] = quantity * (current - average)
        row["updated_at"] = row["updated_at"] or utc_now()
        return row


@dataclass(frozen=True)
class OrderTracker:
    account_id: str
    broker: str
    market: str
    symbol: str
    company: str
    side: str
    status: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    strategy: str
    notes: str = ""
    created_at: str = ""

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["created_at"] = row["created_at"] or utc_now()
        return row


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_table(path: str | Path, columns: list[str]) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(target)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = None
    return frame[columns]


def save_table(frame: pd.DataFrame, path: str | Path, columns: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = None
    output[columns].to_csv(target, index=False)


def upsert_account(path: str | Path, snapshot: AccountSnapshot) -> pd.DataFrame:
    frame = load_table(path, ACCOUNT_COLUMNS)
    row = snapshot.to_row()
    frame = frame[frame["account_id"] != snapshot.account_id]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    save_table(frame, path, ACCOUNT_COLUMNS)
    return frame


def upsert_position(path: str | Path, snapshot: PositionSnapshot) -> pd.DataFrame:
    frame = load_table(path, POSITION_COLUMNS)
    row = snapshot.to_row()
    mask = ~(
        (frame["account_id"] == snapshot.account_id)
        & (frame["symbol"] == snapshot.symbol)
    )
    frame = pd.concat([frame[mask], pd.DataFrame([row])], ignore_index=True)
    save_table(frame, path, POSITION_COLUMNS)
    return frame


def append_order(path: str | Path, order: OrderTracker) -> pd.DataFrame:
    frame = load_table(path, ORDER_COLUMNS)
    frame = pd.concat([frame, pd.DataFrame([order.to_row()])], ignore_index=True)
    save_table(frame, path, ORDER_COLUMNS)
    return frame
