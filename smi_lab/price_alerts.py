from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .equity_data import fetch_yahoo_chart
from .equity_live import strategy_recommendations_for_market
from .notifier import resolve_discord_mention, send_discord


LEVEL_COLUMNS = {
    "ENTRY": "entry_price",
    "STOP": "stop_loss",
    "TP1": "take_profit_1",
    "TP2": "take_profit_2",
}


@dataclass(frozen=True)
class AlertEvent:
    market: str
    symbol: str
    company: str
    level: str
    target_price: float
    last_price: float
    triggered_at: str
    risk_reward: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "company": self.company,
            "level": self.level,
            "target_price": self.target_price,
            "last_price": self.last_price,
            "triggered_at": self.triggered_at,
            "risk_reward": self.risk_reward,
        }


def _load_state(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {"triggered": {}}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"triggered": {}}
    payload.setdefault("triggered", {})
    return payload


def _save_state(path: str | Path, state: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest_price(symbol: str, market: str, refresh: bool = True) -> tuple[float, float, float]:
    try:
        frame = fetch_yahoo_chart(
            symbol,
            interval="1m",
            range_="1d",
            refresh=refresh,
            cache_dir="data/alert_prices",
        )
    except Exception:
        frame = fetch_yahoo_chart(
            symbol,
            interval="1d",
            range_="5d",
            refresh=refresh,
            cache_dir="data/alert_prices",
        )
    latest = frame.dropna(subset=["high", "low", "close"]).iloc[-1]
    return float(latest["close"]), float(latest["high"]), float(latest["low"])


def _level_triggered(level: str, target: float, close: float, high: float, low: float) -> bool:
    if level in {"ENTRY", "TP1", "TP2"}:
        return high >= target or close >= target
    if level == "STOP":
        return low <= target or close <= target
    return False


def format_alert_message(event: AlertEvent, mention: str = "") -> str:
    mention_text = f"{mention} " if mention else ""
    rr_text = f" | RR {event.risk_reward:.2f}" if event.risk_reward is not None else ""
    return (
        f"{mention_text}Price alert | {event.market.upper()} {event.symbol} {event.level}\n"
        f"{event.company}\n"
        f"Target: {event.target_price:.4f} | Last: {event.last_price:.4f}{rr_text}\n"
        f"Time UTC: {event.triggered_at}"
    )


def check_equity_price_alerts(
    recommendations_path: str | Path = "outputs/equity_scan/latest_recommendations.csv",
    state_path: str | Path = "outputs/alerts/equity_price_alerts_state.json",
    webhook_url: str | None = None,
    mention: str | None = None,
    refresh: bool = True,
    notify: bool = True,
    record_state: bool = True,
) -> list[AlertEvent]:
    recommendations = pd.read_csv(recommendations_path)
    if recommendations.empty:
        return []
    if "market" in recommendations:
        recommendations = pd.concat(
            [
                strategy_recommendations_for_market(recommendations, "tw"),
                strategy_recommendations_for_market(recommendations, "us"),
            ],
            ignore_index=True,
        )
    else:
        selected = recommendations["selected"] if "selected" in recommendations else True
        if not isinstance(selected, bool):
            recommendations = recommendations[selected.astype(str).str.lower().isin({"true", "1", "yes", "y"})]
    if recommendations.empty:
        return []
    state = _load_state(state_path)
    triggered = state.setdefault("triggered", {})
    if not record_state:
        triggered = dict(triggered)
    events: list[AlertEvent] = []
    mention = mention if mention is not None else resolve_discord_mention()
    for row in recommendations.to_dict("records"):
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        market = str(row.get("market", ""))
        close, high, low = _latest_price(symbol, market, refresh=refresh)
        entry_key = f"{symbol}:ENTRY:{row.get('entry_price')}"
        entry_already_triggered = entry_key in triggered
        for level, column in LEVEL_COLUMNS.items():
            target = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(target) or float(target) <= 0:
                continue
            if level in {"STOP", "TP1", "TP2"} and not entry_already_triggered:
                continue
            key = f"{symbol}:{level}:{float(target):.6f}"
            if key in triggered:
                continue
            if not _level_triggered(level, float(target), close, high, low):
                continue
            rr = None
            if level == "TP1":
                rr = float(row["risk_reward_1"]) if "risk_reward_1" in row and not pd.isna(row["risk_reward_1"]) else None
            elif level == "TP2":
                rr = float(row["risk_reward_2"]) if "risk_reward_2" in row and not pd.isna(row["risk_reward_2"]) else None
            event = AlertEvent(
                market=market,
                symbol=symbol,
                company=str(row.get("company", symbol)),
                level=level,
                target_price=float(target),
                last_price=close,
                triggered_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                risk_reward=rr,
            )
            events.append(event)
            triggered[key] = event.to_dict()
            if level == "ENTRY":
                entry_already_triggered = True
            if notify:
                send_discord(format_alert_message(event, mention), webhook_url=webhook_url)
    if record_state:
        state["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        _save_state(state_path, state)
    return events
