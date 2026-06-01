from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import StrategyConfig
from .indicators import adx, atr, ema, smi


@dataclass(frozen=True)
class Signal:
    symbol: str
    candle_time: pd.Timestamp
    side: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    smi: float
    adx: float

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)


def build_feature_frame(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    config.validate()
    result = frame.copy()
    result["atr"] = atr(result, config.atr_period)
    result["adx"] = adx(result, config.adx_period)
    result["trend_ema"] = ema(result["close"], config.trend_ema)
    result["smi"], result["smi_signal"] = smi(
        result,
        config.smi_period,
        config.smooth_k,
        config.smooth_d,
        config.signal_period,
    )
    crossed_up = (result["smi"] > result["smi_signal"]) & (
        result["smi"].shift(1) <= result["smi_signal"].shift(1)
    )
    crossed_down = (result["smi"] < result["smi_signal"]) & (
        result["smi"].shift(1) >= result["smi_signal"].shift(1)
    )
    trending_up = (result["close"] > result["trend_ema"]) & (
        result["trend_ema"] > result["trend_ema"].shift(5)
    )
    trending_down = (result["close"] < result["trend_ema"]) & (
        result["trend_ema"] < result["trend_ema"].shift(5)
    )
    strength = result["adx"] >= config.adx_min
    if config.regime_mode != "none" and "risk_off" not in result:
        raise ValueError("A regime-filtered strategy requires risk_off market features.")
    risk_off = (
        result["risk_off"].fillna(False).astype(bool)
        if "risk_off" in result
        else pd.Series(False, index=result.index)
    )
    long_regime = pd.Series(True, index=result.index)
    short_regime = pd.Series(True, index=result.index)
    if config.regime_mode in {"avoid_risk_off_longs", "risk_aligned"}:
        long_regime = ~risk_off
    if config.regime_mode == "risk_aligned":
        short_regime = risk_off
    if config.entry_mode == "pullback":
        long_trigger = crossed_up & (result["smi"].shift(1) <= config.oversold)
        short_trigger = crossed_down & (
            result["smi"].shift(1) >= config.overbought
        )
    else:
        upper_channel = (
            result["high"].shift(1).rolling(config.breakout_period).max()
        )
        lower_channel = (
            result["low"].shift(1).rolling(config.breakout_period).min()
        )
        long_trigger = (
            (result["close"] > upper_channel)
            & (result["smi"] > result["smi_signal"])
            & (result["smi"] > 0)
        )
        short_trigger = (
            (result["close"] < lower_channel)
            & (result["smi"] < result["smi_signal"])
            & (result["smi"] < 0)
        )
    result["long_signal"] = (
        long_trigger
        & trending_up
        & strength
        & config.use_longs
        & long_regime
    )
    result["short_signal"] = (
        short_trigger
        & trending_down
        & strength
        & config.use_shorts
        & short_regime
    )
    return result


def latest_signal(
    symbol: str, frame: pd.DataFrame, config: StrategyConfig
) -> Signal | None:
    featured = build_feature_frame(frame, config).dropna()
    if featured.empty:
        return None
    bar = featured.iloc[-1]
    if bool(bar["long_signal"]):
        side = "LONG"
        direction = 1.0
    elif bool(bar["short_signal"]):
        side = "SHORT"
        direction = -1.0
    else:
        return None
    entry = float(bar["close"])
    risk = float(bar["atr"] * config.stop_atr)
    return Signal(
        symbol=symbol,
        candle_time=featured.index[-1],
        side=side,
        entry=entry,
        stop_loss=entry - direction * risk,
        tp1=entry + direction * risk * config.tp1_r,
        tp2=entry + direction * risk * config.tp2_r,
        tp3=entry + direction * risk * config.tp3_r,
        smi=float(bar["smi"]),
        adx=float(bar["adx"]),
    )
