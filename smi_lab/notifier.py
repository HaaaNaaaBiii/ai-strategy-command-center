from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .strategy import Signal


def format_signal(signal: Signal) -> str:
    precision = 6 if signal.entry < 10 else 2
    price = lambda number: f"{number:.{precision}f}"
    return (
        f"SMI Momentum Signal | {signal.symbol} {signal.side}\n"
        f"Candle (UTC): {signal.candle_time:%Y-%m-%d %H:%M}\n"
        f"Entry reference: {price(signal.entry)}\n"
        f"SL: {price(signal.stop_loss)}\n"
        f"TP1: {price(signal.tp1)} (1R, close 40%)\n"
        f"TP2: {price(signal.tp2)} (2R, close 35%)\n"
        f"TP3: {price(signal.tp3)} (3R, close 25%)\n"
        f"SMI: {signal.smi:.2f} | ADX: {signal.adx:.2f}\n"
        "Risk model: after TP1 move stop to break-even; after TP2 trail by ATR."
    )


def _post_json(url: str, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "smi-signal-lab/1.0"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        if response.status >= 300:
            raise RuntimeError(f"Notification endpoint returned HTTP {response.status}.")


def send_discord(message: str, webhook_url: str | None = None) -> None:
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise ValueError("DISCORD_WEBHOOK_URL is not configured.")
    _post_json(url, {"content": message})


def send_telegram(
    message: str, token: str | None = None, chat_id: str | None = None
) -> None:
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    encoded = urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = Request(url, data=encoded, method="POST")
    with urlopen(request, timeout=20) as response:
        if response.status >= 300:
            raise RuntimeError(f"Telegram returned HTTP {response.status}.")


def notify(signal: Signal, channel: str) -> None:
    message = format_signal(signal)
    if channel == "discord":
        send_discord(message)
    elif channel == "telegram":
        send_telegram(message)
    else:
        raise ValueError(f"Unsupported notification channel: {channel}")
