from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Position, _adverse_fill, _gap_price, _metrics, _touched
from .indicators import atr


@dataclass(frozen=True)
class RotationConfig:
    btc_ema_period: int = 100
    asset_ema_period: int = 0
    momentum_period: int = 90
    rebalance_bars: int = 42
    rebalance_offset_bars: int = 0
    enter_when_flat: bool = False
    rotate_on_rebalance: bool = True
    exposure: float = 1.0
    max_initial_risk_pct: float = 0.0
    drawdown_reduce_at_pct: float = 0.0
    drawdown_exposure_multiplier: float = 1.0
    funding_lookback_bars: int = 0
    max_cumulative_funding_rate: float = 0.0
    atr_period: int = 14
    stop_atr: float = 6.0
    tp1_r: float = 2.0
    tp2_r: float = 5.0
    tp3_r: float = 12.0
    tp1_fraction: float = 0.10
    tp2_fraction: float = 0.10
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    def target_fractions(self) -> tuple[float, float, float]:
        last = 1.0 - self.tp1_fraction - self.tp2_fraction
        if last <= 0:
            raise ValueError("Rotation target fractions must leave a final position.")
        return self.tp1_fraction, self.tp2_fraction, last

    def validate(self) -> "RotationConfig":
        if self.btc_ema_period < 5 or self.momentum_period < 5:
            raise ValueError("Rotation trend and momentum periods must be at least five bars.")
        if self.asset_ema_period not in {0} and self.asset_ema_period < 5:
            raise ValueError("The optional asset trend period must be zero or at least five.")
        if self.rebalance_bars < 1 or self.exposure <= 0 or self.stop_atr <= 0:
            raise ValueError("Rotation sizing and rebalance values must be positive.")
        if not 0 <= self.rebalance_offset_bars < self.rebalance_bars:
            raise ValueError("Rotation rebalance offset must be within its interval.")
        if self.max_initial_risk_pct < 0 or self.drawdown_reduce_at_pct < 0:
            raise ValueError("Rotation risk caps must not be negative.")
        if not 0 < self.drawdown_exposure_multiplier <= 1.0:
            raise ValueError("Rotation drawdown exposure multiplier must be in (0, 1].")
        if self.funding_lookback_bars < 0 or self.max_cumulative_funding_rate < 0:
            raise ValueError("Rotation funding filter values must not be negative.")
        if not (0 < self.tp1_r < self.tp2_r < self.tp3_r):
            raise ValueError("Rotation take-profit levels must be increasing.")
        self.target_fractions()
        return self


@dataclass
class RotationResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


def _rebalance_schedule(
    index: pd.DatetimeIndex, rebalance_bars: int, rebalance_offset_bars: int = 0
) -> pd.Series:
    """Return a time-anchored schedule that is invariant to loaded history length."""
    if len(index) < 2:
        return pd.Series(False, index=index, dtype=bool)
    interval = index.to_series().diff().dropna().median()
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        raise ValueError("Rotation candles must have a positive timestamp interval.")
    anchor = pd.Timestamp("1970-01-01", tz=index.tz)
    offsets = ((index - anchor) / interval).astype("int64")
    return pd.Series(
        offsets % rebalance_bars == rebalance_offset_bars, index=index, dtype=bool
    )


def backtest_rotation(
    universe: dict[str, pd.DataFrame],
    config: RotationConfig,
    initial_equity: float = 10_000.0,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> RotationResult:
    """Trade the strongest risk-on asset from one account with stops and targets."""
    config.validate()
    if "BTCUSDT" not in universe:
        raise ValueError("BTCUSDT is required for bull rotation.")
    index = next(iter(universe.values())).index
    for frame in universe.values():
        index = index.intersection(frame.index)
    index = index.sort_values()
    if len(index) < 2:
        empty = pd.Series(dtype=float)
        return RotationResult(empty, pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity))
    scheduled_rebalances = _rebalance_schedule(
        index, config.rebalance_bars, config.rebalance_offset_bars
    )
    data = {
        symbol: frame.reindex(index).assign(atr=atr(frame.reindex(index), config.atr_period))
        for symbol, frame in universe.items()
    }
    closes = pd.concat({symbol: frame["close"] for symbol, frame in data.items()}, axis=1)
    ranks = closes.pct_change(config.momentum_period).rank(
        axis=1, ascending=False, method="min"
    )
    if config.funding_lookback_bars and config.max_cumulative_funding_rate:
        funding = pd.concat(
            {
                symbol: frame.get("funding_rate", pd.Series(0.0, index=index))
                for symbol, frame in data.items()
            },
            axis=1,
        ).fillna(0.0)
        cumulative_funding = funding.rolling(
            config.funding_lookback_bars, min_periods=1
        ).sum()
        ranks = ranks.where(
            cumulative_funding <= config.max_cumulative_funding_rate
        )
    if config.asset_ema_period:
        asset_trending = closes >= closes.ewm(
            span=config.asset_ema_period,
            adjust=False,
            min_periods=config.asset_ema_period,
        ).mean()
        ranks = ranks.where(asset_trending)
    top_symbol = ranks.fillna(np.inf).idxmin(axis=1).where(ranks.notna().any(axis=1))
    btc = closes["BTCUSDT"]
    risk_on = btc >= btc.ewm(
        span=config.btc_ema_period, adjust=False, min_periods=config.btc_ema_period
    ).mean()
    desired = top_symbol.where(risk_on).shift(1)
    start = index[min(int(len(index) * start_ratio), len(index) - 2)]
    end = index[min(int(len(index) * end_ratio) - 1, len(index) - 1)]
    timestamps = index[index <= end]
    fee = config.fee_bps / 10_000.0
    slippage = config.slippage_bps / 10_000.0
    balance = initial_equity
    position: Position | None = None
    held_symbol: str | None = None
    entry_balance = initial_equity
    completed: list[dict[str, object]] = []
    curve: dict[pd.Timestamp, float] = {}
    peak_equity = initial_equity

    def close_quantity(
        current: Position,
        quantity: float,
        exit_price: float,
        label: str,
        exit_time: pd.Timestamp,
    ) -> None:
        nonlocal balance, position, held_symbol
        quantity = min(quantity, current.remaining)
        if quantity <= 1e-12:
            return
        exit_fee = quantity * exit_price * fee
        realized = current.direction * (exit_price - current.entry) * quantity - exit_fee
        current.pnl += realized
        balance += realized
        current.remaining -= quantity
        current.exits.append(label)
        if current.remaining <= current.quantity * 1e-10:
            risk_cash = current.risk_per_unit * current.quantity
            completed.append(
                {
                    "symbol": held_symbol,
                    "side": current.side,
                    "entry_time": current.entry_time,
                    "exit_time": exit_time,
                    "entry": current.entry,
                    "last_exit": exit_price,
                    "quantity": current.quantity,
                    "pnl": current.pnl,
                    "funding_pnl": current.funding_pnl,
                    "r_multiple": current.pnl / risk_cash if risk_cash else 0.0,
                    "initial_risk_pct": risk_cash / entry_balance * 100.0,
                    "initial_exposure": current.quantity * current.entry / entry_balance,
                    "exit_reason": " / ".join(current.exits),
                    "tp_count": current.targets_hit,
                }
            )
            position = None
            held_symbol = None

    for i in range(1, len(timestamps)):
        timestamp = timestamps[i]
        in_window = timestamp >= start
        scheduled = bool(scheduled_rebalances.loc[timestamp])
        target_symbol = desired.loc[timestamp]
        if position is not None and held_symbol is not None:
            row = data[held_symbol].loc[timestamp]
            funding_rate = float(row.get("funding_rate", 0.0))
            if np.isfinite(funding_rate) and funding_rate != 0.0:
                funding_price = row.get("funding_mark_price", row["open"])
                funding_price = (
                    float(funding_price) if pd.notna(funding_price) else float(row["open"])
                )
                funding_cash = (
                    -position.remaining * funding_price * funding_rate
                )
                position.funding_pnl += funding_cash
                position.pnl += funding_cash
                balance += funding_cash
            regime_exit = pd.isna(target_symbol)
            rotation_exit = (
                config.rotate_on_rebalance
                and scheduled
                and target_symbol != held_symbol
            )
            if regime_exit or rotation_exit:
                exit_price = _adverse_fill(
                    float(row["open"]), "LONG", is_entry=False, slippage=slippage
                )
                close_quantity(
                    position,
                    position.remaining,
                    exit_price,
                    "RISK_OFF" if regime_exit else "ROTATE",
                    timestamp,
                )

        if (
            in_window
            and (config.enter_when_flat or scheduled)
            and position is None
            and pd.notna(target_symbol)
            and i < len(timestamps) - 1
        ):
            held_symbol = str(target_symbol)
            row = data[held_symbol].loc[timestamp]
            previous = data[held_symbol].loc[timestamps[i - 1]]
            if pd.notna(previous["atr"]):
                entry = _adverse_fill(
                    float(row["open"]), "LONG", is_entry=True, slippage=slippage
                )
                risk_per_unit = float(previous["atr"]) * config.stop_atr
                entry_balance = balance
                current_drawdown_pct = (balance / peak_equity - 1.0) * 100.0
                exposure = config.exposure
                if (
                    config.drawdown_reduce_at_pct
                    and current_drawdown_pct <= -config.drawdown_reduce_at_pct
                ):
                    exposure *= config.drawdown_exposure_multiplier
                quantity = balance * exposure / entry
                if config.max_initial_risk_pct:
                    quantity = min(
                        quantity,
                        balance * config.max_initial_risk_pct
                        / 100.0
                        / risk_per_unit,
                    )
                entry_fee = quantity * entry * fee
                balance -= entry_fee
                fractions = config.target_fractions()
                position = Position(
                    side="LONG",
                    entry_time=timestamp,
                    entry=entry,
                    quantity=quantity,
                    remaining=quantity,
                    initial_stop=entry - risk_per_unit,
                    stop=entry - risk_per_unit,
                    targets=(
                        entry + risk_per_unit * config.tp1_r,
                        entry + risk_per_unit * config.tp2_r,
                        entry + risk_per_unit * config.tp3_r,
                    ),
                    target_quantities=tuple(quantity * fraction for fraction in fractions),
                    risk_per_unit=risk_per_unit,
                    entry_fee=entry_fee,
                    pnl=-entry_fee,
                )
            else:
                held_symbol = None

        if position is not None and held_symbol is not None:
            current = position
            row = data[held_symbol].loc[timestamp]
            if _touched("LONG", float(row["high"]), float(row["low"]), current.stop, True):
                price = _gap_price(
                    "LONG", float(row["open"]), current.stop, True, slippage
                )
                close_quantity(current, current.remaining, price, "SL", timestamp)
            else:
                for target_index in range(current.targets_hit, 3):
                    target = current.targets[target_index]
                    if not _touched(
                        "LONG", float(row["high"]), float(row["low"]), target, False
                    ):
                        break
                    quantity = (
                        current.target_quantities[target_index]
                        if target_index < 2
                        else current.remaining
                    )
                    price = _gap_price(
                        "LONG", float(row["open"]), target, False, slippage
                    )
                    current.targets_hit += 1
                    close_quantity(
                        current, quantity, price, f"TP{target_index + 1}", timestamp
                    )
                    if position is None:
                        break
                if position is not None and current.targets_hit >= 1:
                    current.stop = max(current.stop, current.entry)
                if position is not None and current.targets_hit >= 2:
                    current.stop = max(
                        current.stop, float(row["close"]) - float(row["atr"]) * 2.0
                    )

        if in_window:
            unrealized = 0.0
            if position is not None and held_symbol is not None:
                row = data[held_symbol].loc[timestamp]
                unrealized = (float(row["close"]) - position.entry) * position.remaining
            marked_equity = balance + unrealized
            curve[timestamp] = marked_equity
            peak_equity = max(peak_equity, marked_equity)

    if position is not None and held_symbol is not None:
        final_time = timestamps[-1]
        price = _adverse_fill(
            float(data[held_symbol].loc[final_time, "close"]),
            "LONG",
            is_entry=False,
            slippage=slippage,
        )
        close_quantity(position, position.remaining, price, "END", final_time)
        if final_time >= start:
            curve[final_time] = balance
    equity = pd.Series(curve, name="rotation_equity", dtype=float).sort_index()
    trades = pd.DataFrame(completed)
    return RotationResult(equity, trades, _metrics(equity, trades, initial_equity))
