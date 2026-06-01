from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .indicators import adx, ema


ACTIONS = ("Strong Sell", "Sell", "Neutral", "Buy", "Strong Buy")


@dataclass(frozen=True)
class TechnicalSummary:
    symbol: str
    as_of: pd.Timestamp
    close: float
    moving_average_action: str
    indicator_action: str
    summary_action: str
    ma_buy: int
    ma_sell: int
    ma_neutral: int
    indicator_buy: int
    indicator_sell: int
    indicator_neutral: int
    rsi: float
    macd: float
    macd_signal: float
    roc_pct: float
    atr_pct: float
    view: str

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "close": self.close,
            "summary": self.summary_action,
            "moving_averages": self.moving_average_action,
            "indicators": self.indicator_action,
            "ma_buy": self.ma_buy,
            "ma_neutral": self.ma_neutral,
            "ma_sell": self.ma_sell,
            "indicator_buy": self.indicator_buy,
            "indicator_neutral": self.indicator_neutral,
            "indicator_sell": self.indicator_sell,
            "rsi": self.rsi,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "roc_pct": self.roc_pct,
            "atr_pct": self.atr_pct,
            "ai_view": self.view,
        }


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def stochastic_k(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    lowest = frame["low"].rolling(period, min_periods=period).min()
    highest = frame["high"].rolling(period, min_periods=period).max()
    return 100.0 * (frame["close"] - lowest) / (highest - lowest).replace(0.0, np.nan)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    return line, signal


def cci(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    average = typical.rolling(period, min_periods=period).mean()
    deviation = (typical - average).abs().rolling(period, min_periods=period).mean()
    return (typical - average) / (0.015 * deviation.replace(0.0, np.nan))


def williams_r(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    highest = frame["high"].rolling(period, min_periods=period).max()
    lowest = frame["low"].rolling(period, min_periods=period).min()
    return -100.0 * (highest - frame["close"]) / (highest - lowest).replace(0.0, np.nan)


def _action_from_counts(buy: int, sell: int, neutral: int) -> str:
    total = buy + sell + neutral
    if total == 0:
        return "Neutral"
    score = (buy - sell) / total
    if score >= 0.55:
        return "Strong Buy"
    if score >= 0.20:
        return "Buy"
    if score <= -0.55:
        return "Strong Sell"
    if score <= -0.20:
        return "Sell"
    return "Neutral"


def _vote(value: float, buy_when: bool, sell_when: bool) -> str:
    if not math.isfinite(value):
        return "Neutral"
    if buy_when:
        return "Buy"
    if sell_when:
        return "Sell"
    return "Neutral"


def _count(actions: list[str]) -> tuple[int, int, int]:
    buy = sum(action == "Buy" for action in actions)
    sell = sum(action == "Sell" for action in actions)
    neutral = len(actions) - buy - sell
    return buy, sell, neutral


def summarize_technical(symbol: str, frame: pd.DataFrame) -> TechnicalSummary:
    if len(frame) < 60:
        raise ValueError("At least 60 rows are required for a technical summary.")
    data = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    if data.empty:
        raise ValueError("Technical summary requires OHLC data.")
    close = data["close"].astype(float)
    latest_close = float(close.iloc[-1])
    ma_actions: list[str] = []
    for period in (5, 10, 20, 50, 100, 200):
        sma = close.rolling(period, min_periods=period).mean().iloc[-1]
        exp = ema(close, period).iloc[-1]
        for average in (sma, exp):
            ma_actions.append(
                _vote(
                    float(average) if pd.notna(average) else float("nan"),
                    latest_close > average if pd.notna(average) else False,
                    latest_close < average if pd.notna(average) else False,
                )
            )
    ma_buy, ma_sell, ma_neutral = _count(ma_actions)

    rsi_value = float(rsi(close).iloc[-1])
    stoch_value = float(stochastic_k(data).iloc[-1])
    macd_line, macd_signal = macd(close)
    macd_value = float(macd_line.iloc[-1])
    macd_signal_value = float(macd_signal.iloc[-1])
    cci_value = float(cci(data).iloc[-1])
    williams_value = float(williams_r(data).iloc[-1])
    roc_value = float(close.pct_change(12).iloc[-1] * 100.0)
    adx_value = float(adx(data, 14).iloc[-1])
    ema_20 = float(ema(close, 20).iloc[-1])
    ema_50 = float(ema(close, 50).iloc[-1])
    atr_pct = float(
        (
            (data["high"] - data["low"])
            .rolling(14, min_periods=14)
            .mean()
            .iloc[-1]
            / latest_close
            * 100.0
        )
    )

    indicator_actions = [
        _vote(rsi_value, rsi_value < 30.0, rsi_value > 70.0),
        _vote(stoch_value, stoch_value < 20.0, stoch_value > 80.0),
        _vote(macd_value, macd_value > macd_signal_value, macd_value < macd_signal_value),
        _vote(cci_value, cci_value < -100.0, cci_value > 100.0),
        _vote(williams_value, williams_value < -80.0, williams_value > -20.0),
        _vote(roc_value, roc_value > 0.0, roc_value < 0.0),
        _vote(adx_value, adx_value >= 20.0 and ema_20 > ema_50, adx_value >= 20.0 and ema_20 < ema_50),
    ]
    indicator_buy, indicator_sell, indicator_neutral = _count(indicator_actions)
    moving_average_action = _action_from_counts(ma_buy, ma_sell, ma_neutral)
    indicator_action = _action_from_counts(
        indicator_buy, indicator_sell, indicator_neutral
    )
    summary_action = _action_from_counts(
        ma_buy + indicator_buy, ma_sell + indicator_sell, ma_neutral + indicator_neutral
    )
    view = build_ai_view(
        summary_action,
        moving_average_action,
        indicator_action,
        rsi_value,
        roc_value,
        atr_pct,
    )
    return TechnicalSummary(
        symbol=symbol,
        as_of=data.index[-1],
        close=latest_close,
        moving_average_action=moving_average_action,
        indicator_action=indicator_action,
        summary_action=summary_action,
        ma_buy=ma_buy,
        ma_sell=ma_sell,
        ma_neutral=ma_neutral,
        indicator_buy=indicator_buy,
        indicator_sell=indicator_sell,
        indicator_neutral=indicator_neutral,
        rsi=rsi_value,
        macd=macd_value,
        macd_signal=macd_signal_value,
        roc_pct=roc_value,
        atr_pct=atr_pct,
        view=view,
    )


def build_ai_view(
    summary_action: str,
    moving_average_action: str,
    indicator_action: str,
    rsi_value: float,
    roc_pct: float,
    atr_pct: float,
) -> str:
    bias = {
        "Strong Buy": "多方趨勢明確",
        "Buy": "偏多但仍需等回撤或確認",
        "Neutral": "多空分歧，適合降低追價",
        "Sell": "偏弱，應以防守和風控優先",
        "Strong Sell": "空方壓力明顯，避免逆勢加碼",
    }[summary_action]
    rsi_note = (
        "RSI 過熱"
        if rsi_value > 70
        else "RSI 超賣"
        if rsi_value < 30
        else "RSI 中性"
    )
    momentum_note = "12 期動能為正" if roc_pct > 0 else "12 期動能為負"
    volatility_note = (
        "波動偏高，部位應縮小"
        if atr_pct >= 8
        else "波動中等"
        if atr_pct >= 3
        else "波動偏低"
    )
    return (
        f"{bias}；均線為 {moving_average_action}，震盪指標為 {indicator_action}。"
        f"{rsi_note}，{momentum_note}，{volatility_note}。"
    )


def summarize_universe(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [summarize_technical(symbol, frame).to_dict() for symbol, frame in universe.items()]
    return pd.DataFrame(rows)
