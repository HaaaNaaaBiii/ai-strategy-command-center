from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    return wilder(true_range(frame), period)


def adx(frame: pd.DataFrame, period: int) -> pd.Series:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    average_range = atr(frame, period).replace(0.0, np.nan)
    plus_di = 100.0 * wilder(plus_dm, period) / average_range
    minus_di = 100.0 * wilder(minus_dm, period) / average_range
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    directional_index = 100.0 * (plus_di - minus_di).abs() / denominator
    return wilder(directional_index, period)


def smi(
    frame: pd.DataFrame,
    period: int,
    smooth_k: int,
    smooth_d: int,
    signal_period: int,
) -> tuple[pd.Series, pd.Series]:
    highest = frame["high"].rolling(period, min_periods=period).max()
    lowest = frame["low"].rolling(period, min_periods=period).min()
    distance = frame["close"] - (highest + lowest) / 2.0
    half_range = (highest - lowest) / 2.0
    smoothed_distance = ema(ema(distance, smooth_k), smooth_d)
    smoothed_range = ema(ema(half_range, smooth_k), smooth_d).replace(0.0, np.nan)
    value = 100.0 * smoothed_distance / smoothed_range
    signal = ema(value, signal_period)
    return value, signal
