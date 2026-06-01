from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .backtest import _metrics
from .rotation import _rebalance_schedule


@dataclass(frozen=True)
class TrendAllocationConfig:
    """Long/cash allocation driven by trend and cross-sectional momentum."""

    momentum_period: int = 180
    asset_ema_period: int = 90
    btc_ema_period: int = 24
    top_n: int = 1
    rebalance_bars: int = 42
    rebalance_offset_bars: int = 0
    gross_exposure: float = 0.40
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    def validate(self) -> "TrendAllocationConfig":
        if min(self.momentum_period, self.asset_ema_period, self.btc_ema_period) < 5:
            raise ValueError("Trend allocation indicator periods must be at least five bars.")
        if self.top_n < 1 or self.rebalance_bars < 1:
            raise ValueError("Trend allocation rank and schedule values must be positive.")
        if not 0 <= self.rebalance_offset_bars < self.rebalance_bars:
            raise ValueError("Trend allocation rebalance offset must be within its interval.")
        if not 0 < self.gross_exposure <= 1.0:
            raise ValueError("Trend allocation exposure must be in (0, 1].")
        if min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("Trend allocation trading costs must not be negative.")
        return self


@dataclass
class AllocationResult:
    equity: pd.Series
    rebalances: pd.DataFrame
    metrics: dict[str, float]


def _aligned_frames(
    universe: dict[str, pd.DataFrame],
) -> tuple[
    list[str],
    pd.DatetimeIndex,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if "BTCUSDT" not in universe:
        raise ValueError("BTCUSDT is required for the risk gate.")
    if not universe:
        raise ValueError("The allocation account requires market data.")
    symbols = list(universe)
    index = next(iter(universe.values())).index
    for frame in universe.values():
        index = index.intersection(frame.index)
    index = index.sort_values()
    closes = pd.DataFrame(
        {symbol: universe[symbol].reindex(index)["close"] for symbol in symbols},
        index=index,
    )
    opens = pd.DataFrame(
        {symbol: universe[symbol].reindex(index)["open"] for symbol in symbols},
        index=index,
    )
    funding = pd.DataFrame(
        {
            symbol: universe[symbol]
            .reindex(index)
            .get("funding_rate", pd.Series(0.0, index=index))
            for symbol in symbols
        },
        index=index,
    ).fillna(0.0)
    funding_marks = pd.DataFrame(
        {
            symbol: universe[symbol]
            .reindex(index)
            .get("funding_mark_price", opens[symbol])
            for symbol in symbols
        },
        index=index,
    ).fillna(opens)
    return symbols, index, closes, opens, funding, funding_marks


def _phase_window(
    index: pd.DatetimeIndex,
    trade_start: pd.Timestamp | None,
    trade_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = trade_start if trade_start is not None else index[1]
    end = trade_end if trade_end is not None else index[-1]
    return max(start, index[1]), min(end, index[-1])


def backtest_trend_allocation(
    universe: dict[str, pd.DataFrame],
    config: TrendAllocationConfig,
    initial_equity: float = 10_000.0,
    trade_start: pd.Timestamp | None = None,
    trade_end: pd.Timestamp | None = None,
) -> AllocationResult:
    """Backtest a fixed-time, next-open long/cash allocation account."""
    config.validate()
    symbols, index, closes, opens, funding, funding_marks = _aligned_frames(universe)
    if len(index) < 2:
        empty = pd.Series(dtype=float)
        return AllocationResult(
            empty, pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity)
        )
    start, end = _phase_window(index, trade_start, trade_end)
    timestamps = index[index <= end]
    if len(timestamps) < 2 or start > end:
        empty = pd.Series(dtype=float)
        return AllocationResult(
            empty, pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity)
        )
    close_array = closes.to_numpy(dtype=float)
    open_array = opens.to_numpy(dtype=float)
    funding_array = funding.to_numpy(dtype=float)
    funding_mark_array = funding_marks.to_numpy(dtype=float)
    momentum = closes.pct_change(config.momentum_period).to_numpy(dtype=float)
    asset_ema = closes.ewm(
        span=config.asset_ema_period,
        adjust=False,
        min_periods=config.asset_ema_period,
    ).mean().to_numpy(dtype=float)
    btc_ema = closes["BTCUSDT"].ewm(
        span=config.btc_ema_period,
        adjust=False,
        min_periods=config.btc_ema_period,
    ).mean().to_numpy(dtype=float)
    btc_index = symbols.index("BTCUSDT")
    scheduled = _rebalance_schedule(
        index, config.rebalance_bars, config.rebalance_offset_bars
    ).to_numpy()
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    cash = initial_equity
    quantities = np.zeros(len(symbols), dtype=float)
    prior_weights = np.zeros(len(symbols), dtype=float)
    total_cost = 0.0
    funding_pnl = 0.0
    curve: dict[pd.Timestamp, float] = {}
    events: list[dict[str, object]] = []

    for i in range(1, len(timestamps)):
        timestamp = timestamps[i]
        if timestamp < start:
            continue
        funding_cash = -float(
            np.sum(quantities * funding_mark_array[i] * funding_array[i])
        )
        cash += funding_cash
        funding_pnl += funding_cash
        if scheduled[i]:
            prior = i - 1
            eligible = (
                (close_array[prior, btc_index] > btc_ema[prior])
                & (close_array[prior] > asset_ema[prior])
                & (momentum[prior] > 0.0)
            )
            scores = np.where(eligible, momentum[prior], -np.inf)
            order = np.argsort(scores)[::-1]
            selected = order[np.isfinite(scores[order])][: config.top_n]
            weights = np.zeros(len(symbols), dtype=float)
            if len(selected):
                weights[selected] = config.gross_exposure / len(selected)
            if np.any(weights != prior_weights):
                open_equity = cash + float(np.sum(quantities * open_array[i]))
                desired_values = open_equity * weights
                current_values = quantities * open_array[i]
                turnover = float(np.abs(desired_values - current_values).sum())
                costs = turnover * cost_rate
                cash = open_equity - float(desired_values.sum()) - costs
                quantities = np.divide(
                    desired_values,
                    open_array[i],
                    out=np.zeros(len(symbols), dtype=float),
                    where=open_array[i] != 0.0,
                )
                total_cost += costs
                events.append(
                    {
                        "timestamp": timestamp,
                        "selected_symbols": ",".join(symbols[j] for j in selected),
                        "gross_exposure": float(weights.sum()),
                        "turnover": turnover,
                        "cost": costs,
                        "equity_before_cost": open_equity,
                    }
                )
                prior_weights = weights
        curve[timestamp] = cash + float(np.sum(quantities * close_array[i]))

    equity = pd.Series(curve, name="trend_allocation_equity", dtype=float).sort_index()
    if not equity.empty:
        liquidation_notional = float(np.sum(quantities * close_array[len(timestamps) - 1]))
        liquidation_cost = liquidation_notional * cost_rate
        total_cost += liquidation_cost
        equity.iloc[-1] -= liquidation_cost
    metrics = _metrics(equity, pd.DataFrame(), initial_equity)
    metrics.update(
        {
            "funding_pnl": funding_pnl,
            "trading_cost": total_cost,
            "rebalances": float(len(events)),
        }
    )
    return AllocationResult(equity, pd.DataFrame(events), metrics)


def backtest_staggered_trend_allocation(
    universe: dict[str, pd.DataFrame],
    config: TrendAllocationConfig,
    offsets: tuple[int, ...],
    initial_equity: float = 10_000.0,
    trade_start: pd.Timestamp | None = None,
    trade_end: pd.Timestamp | None = None,
) -> AllocationResult:
    """Blend independent scheduled sleeves to avoid selecting one rebalance phase."""
    config.validate()
    if not offsets:
        raise ValueError("Staggered allocation requires at least one schedule offset.")
    if len(set(offsets)) != len(offsets):
        raise ValueError("Staggered allocation schedule offsets must be unique.")
    if any(offset < 0 or offset >= config.rebalance_bars for offset in offsets):
        raise ValueError("Staggered allocation offset is outside the rebalance interval.")
    sleeve_equity = initial_equity / len(offsets)
    results = [
        backtest_trend_allocation(
            universe,
            replace(config, rebalance_offset_bars=offset),
            initial_equity=sleeve_equity,
            trade_start=trade_start,
            trade_end=trade_end,
        )
        for offset in offsets
    ]
    equity = pd.concat(
        [result.equity.rename(str(offset)) for offset, result in zip(offsets, results)],
        axis=1,
    ).sum(axis=1)
    event_frames = []
    for offset, result in zip(offsets, results):
        if not result.rebalances.empty:
            frame = result.rebalances.copy()
            frame["rebalance_offset_bars"] = offset
            event_frames.append(frame)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    metrics = _metrics(equity, pd.DataFrame(), initial_equity)
    metrics.update(
        {
            "funding_pnl": sum(result.metrics["funding_pnl"] for result in results),
            "trading_cost": sum(result.metrics["trading_cost"] for result in results),
            "rebalances": sum(result.metrics["rebalances"] for result in results),
            "schedule_sleeves": float(len(offsets)),
        }
    )
    return AllocationResult(equity, events, metrics)


def backtest_buy_and_hold(
    universe: dict[str, pd.DataFrame],
    target_weights: dict[str, float],
    initial_equity: float = 10_000.0,
    trade_start: pd.Timestamp | None = None,
    trade_end: pd.Timestamp | None = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> AllocationResult:
    """Backtest an unlevered spot-like benchmark with one entry and one exit."""
    symbols, index, closes, opens, _, _ = _aligned_frames(universe)
    if abs(sum(target_weights.get(symbol, 0.0) for symbol in symbols) - 1.0) > 1e-9:
        raise ValueError("Buy-and-hold benchmark weights must sum to one.")
    start, end = _phase_window(index, trade_start, trade_end)
    active = index[(index >= start) & (index <= end)]
    if active.empty:
        empty = pd.Series(dtype=float)
        return AllocationResult(
            empty, pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity)
        )
    start_i = index.get_loc(active[0])
    end_i = index.get_loc(active[-1])
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    weights = np.array([target_weights.get(symbol, 0.0) for symbol in symbols])
    invested = initial_equity / (1.0 + cost_rate)
    quantities = invested * weights / opens.iloc[start_i].to_numpy(dtype=float)
    equity = (
        closes.iloc[start_i : end_i + 1].to_numpy(dtype=float) * quantities
    ).sum(axis=1)
    equity[-1] *= 1.0 - cost_rate
    series = pd.Series(equity, index=active, name="benchmark_equity", dtype=float)
    metrics = _metrics(series, pd.DataFrame(), initial_equity)
    metrics.update(
        {
            "funding_pnl": 0.0,
            "trading_cost": initial_equity - invested + float(equity[-1]) * cost_rate,
            "rebalances": 1.0,
        }
    )
    return AllocationResult(series, pd.DataFrame(), metrics)
