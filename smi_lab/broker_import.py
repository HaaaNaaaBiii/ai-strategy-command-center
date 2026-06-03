from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .accounts import POSITION_COLUMNS, PositionSnapshot, load_table, save_table
from .equity_signals import company_name
from .paths import data_path, output_path


DEFAULT_IMPORT_DIR = data_path("broker_imports")

ALIASES = {
    "symbol": {
        "symbol",
        "ticker",
        "ticker symbol",
        "stock symbol",
        "security id",
        "證券代號",
        "股票代號",
        "代號",
        "商品代號",
    },
    "company": {
        "name",
        "description",
        "security description",
        "security name",
        "股票名稱",
        "證券名稱",
        "名稱",
        "商品名稱",
    },
    "quantity": {
        "quantity",
        "qty",
        "shares",
        "position",
        "持股",
        "股數",
        "庫存股數",
        "數量",
    },
    "average_price": {
        "average price",
        "avg price",
        "average cost",
        "avg cost",
        "cost basis/share",
        "成本",
        "平均成本",
        "成交均價",
        "均價",
    },
    "current_price": {
        "current price",
        "last price",
        "market price",
        "price",
        "現價",
        "市價",
        "收盤價",
    },
    "market_value": {
        "market value",
        "value",
        "市值",
        "庫存市值",
        "參考市值",
    },
    "unrealized_pnl": {
        "unrealized gain/loss",
        "unrealized pnl",
        "gain/loss",
        "p/l",
        "未實現損益",
        "損益",
        "參考損益",
    },
    "cash": {"cash", "available cash", "cash balance", "現金", "可用餘額"},
    "equity": {"account value", "net liquidation", "total equity", "總資產", "淨值"},
}


@dataclass(frozen=True)
class BrokerImportResult:
    path: str
    broker: str
    market: str
    account_id: str
    imported_positions: int
    skipped_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "broker": self.broker,
            "market": self.market,
            "account_id": self.account_id,
            "imported_positions": self.imported_positions,
            "skipped_rows": self.skipped_rows,
        }


def _normalized_header(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    columns = {_normalized_header(column): str(column) for column in frame.columns}
    mapping: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            normalized = _normalized_header(alias)
            if normalized in columns:
                mapping[target] = columns[normalized]
                break
    return mapping


def detect_broker(path: str | Path, frame: pd.DataFrame) -> tuple[str, str]:
    name = Path(path).name.lower()
    headers = {_normalized_header(column) for column in frame.columns}
    if "firstrade" in name or "firsttrade" in name or "symbol" in headers or "ticker" in headers:
        return "Firstrade", "us"
    if "cathay" in name or "國泰" in name or "證券代號" in headers or "股票代號" in headers:
        return "Cathay Securities", "tw"
    return "Unknown Broker", "unknown"


def _to_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value)
    text = text.replace(",", "").replace("$", "").replace("NT$", "").replace("%", "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_symbol(symbol: object, market: str) -> str:
    value = str(symbol).strip().upper()
    value = re.sub(r"\s+", "", value)
    if market == "tw" and value.isdigit() and "." not in value:
        return f"{value}.TW"
    return value


def _read_csv(path: str | Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def normalize_broker_positions(
    path: str | Path,
    broker: str | None = None,
    market: str | None = None,
    account_id: str | None = None,
) -> tuple[pd.DataFrame, BrokerImportResult]:
    raw = _read_csv(path)
    detected_broker, detected_market = detect_broker(path, raw)
    broker = broker or detected_broker
    market = market or detected_market
    account_id = account_id or f"{broker.lower().replace(' ', '-')}-auto"
    mapping = _column_map(raw)
    if "symbol" not in mapping or "quantity" not in mapping:
        raise ValueError(f"Cannot import {path}: symbol and quantity columns are required.")
    rows: list[dict[str, object]] = []
    skipped = 0
    for _, item in raw.iterrows():
        symbol = _normalize_symbol(item[mapping["symbol"]], market)
        quantity = _to_float(item[mapping["quantity"]])
        if not symbol or quantity == 0.0:
            skipped += 1
            continue
        company = (
            str(item[mapping["company"]]).strip()
            if "company" in mapping and not pd.isna(item[mapping["company"]])
            else company_name(symbol)
        )
        average_price = _to_float(item[mapping["average_price"]]) if "average_price" in mapping else 0.0
        current_price = _to_float(item[mapping["current_price"]]) if "current_price" in mapping else 0.0
        market_value = _to_float(item[mapping["market_value"]]) if "market_value" in mapping else 0.0
        if current_price == 0.0 and market_value and quantity:
            current_price = market_value / quantity
        snapshot = PositionSnapshot(
            account_id=account_id,
            broker=broker,
            market=market,
            symbol=symbol,
            company=company,
            quantity=quantity,
            average_price=average_price,
            current_price=current_price,
            notes=f"auto_import:{Path(path).name}",
        )
        row = snapshot.to_row()
        if market_value:
            row["market_value"] = market_value
        if "unrealized_pnl" in mapping:
            row["unrealized_pnl"] = _to_float(item[mapping["unrealized_pnl"]])
        rows.append(row)
    frame = pd.DataFrame(rows, columns=POSITION_COLUMNS)
    result = BrokerImportResult(
        path=str(path),
        broker=broker,
        market=market,
        account_id=account_id,
        imported_positions=len(frame),
        skipped_rows=skipped,
    )
    return frame, result


def sync_broker_exports(
    import_dir: str | Path = DEFAULT_IMPORT_DIR,
    positions_path: str | Path = output_path("accounts", "positions.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(import_dir)
    existing = load_table(positions_path, POSITION_COLUMNS)
    reports: list[dict[str, object]] = []
    if not source.exists():
        source.mkdir(parents=True, exist_ok=True)
        return existing, pd.DataFrame(columns=BrokerImportResult("", "", "", "", 0, 0).to_dict().keys())
    imported_frames: list[pd.DataFrame] = []
    for path in sorted(source.rglob("*.csv")):
        try:
            frame, result = normalize_broker_positions(path)
        except ValueError as exc:
            reports.append(
                {
                    "path": str(path),
                    "broker": "unknown",
                    "market": "unknown",
                    "account_id": "",
                    "imported_positions": 0,
                    "skipped_rows": 0,
                    "error": str(exc),
                }
            )
            continue
        reports.append(result.to_dict())
        if not frame.empty:
            imported_frames.append(frame)
    if imported_frames:
        imported = pd.concat(imported_frames, ignore_index=True)
        key = ["account_id", "symbol"]
        keep_existing = existing.merge(imported[key], on=key, how="left", indicator=True)
        keep_existing = keep_existing[keep_existing["_merge"] == "left_only"].drop(columns=["_merge"])
        updated = pd.concat([keep_existing, imported], ignore_index=True)
    else:
        updated = existing
    save_table(updated, positions_path, POSITION_COLUMNS)
    return updated, pd.DataFrame(reports)
