from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .strategy import build_feature_frame


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry: float
    quantity: float
    remaining: float
    initial_stop: float
    stop: float
    targets: tuple[float, float, float]
    target_quantities: tuple[float, float, float]
    risk_per_unit: float
    entry_fee: float
    pnl: float
    funding_pnl: float = 0.0
    exits: list[str] = field(default_factory=list)
    targets_hit: int = 0

    @property
    def direction(self) -> float:
        return 1.0 if self.side == "LONG" else -1.0


@dataclass
class BacktestResult:
    symbol: str
    config: StrategyConfig
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class PortfolioBacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    by_symbol: pd.DataFrame


def _adverse_fill(price: float, side: str, is_entry: bool, slippage: float) -> float:
    direction = 1.0 if side == "LONG" else -1.0
    movement = direction if is_entry else -direction
    return price * (1.0 + movement * slippage)


def _touched(side: str, high: float, low: float, level: float, stop: bool) -> bool:
    if side == "LONG":
        return low <= level if stop else high >= level
    return high >= level if stop else low <= level


def _gap_price(
    side: str, open_price: float, level: float, stop: bool, slippage: float
) -> float:
    if side == "LONG":
        raw = min(open_price, level) if stop else max(open_price, level)
    else:
        raw = max(open_price, level) if stop else min(open_price, level)
    return _adverse_fill(raw, side, is_entry=False, slippage=slippage)


def _metrics(
    equity: pd.Series, trades: pd.DataFrame, initial_equity: float
) -> dict[str, float]:
    if equity.empty:
        return {
            "final_equity": initial_equity,
            "return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "funding_pnl": 0.0,
        }
    daily_change = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elapsed_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1.0)
    periods_per_year = 365.0 * max(len(equity) - 1, 1) / elapsed_days
    volatility = float(daily_change.std(ddof=0))
    sharpe = (
        float(daily_change.mean()) / volatility * sqrt(periods_per_year)
        if volatility > 0
        else 0.0
    )
    total_return = float(equity.iloc[-1] / initial_equity - 1.0)
    years = elapsed_days / 365.0
    cagr = float((equity.iloc[-1] / initial_equity) ** (1.0 / years) - 1.0)
    drawdown = equity / equity.cummax() - 1.0
    pnls = trades["pnl"] if not trades.empty else pd.Series(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if not losses.empty
        else (float("inf") if not wins.empty else 0.0)
    )
    return {
        "final_equity": float(equity.iloc[-1]),
        "return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min()) * 100.0,
        "trades": float(len(trades)),
        "win_rate_pct": float((pnls > 0).mean() * 100.0) if len(pnls) else 0.0,
        "profit_factor": profit_factor,
        "avg_r": float(trades["r_multiple"].mean()) if not trades.empty else 0.0,
        "funding_pnl": (
            float(trades["funding_pnl"].sum())
            if not trades.empty and "funding_pnl" in trades
            else 0.0
        ),
    }


def backtest(
    frame: pd.DataFrame,
    config: StrategyConfig,
    symbol: str = "",
    initial_equity: float = 10_000.0,
    trade_start: pd.Timestamp | None = None,
    trade_end: pd.Timestamp | None = None,
) -> BacktestResult:
    """Run a single-position next-bar execution simulation."""
    config.validate()
    featured = build_feature_frame(frame, config)
    if trade_end is not None:
        featured = featured.loc[featured.index <= trade_end]
    if featured.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(symbol, config, empty, pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity))

    fee = config.fee_bps / 10_000.0
    slippage = config.slippage_bps / 10_000.0
    balance = initial_equity
    position: Position | None = None
    completed: list[dict[str, object]] = []
    curve: dict[pd.Timestamp, float] = {}
    next_entry_index = 0
    active_start = trade_start or featured.index[0]

    def close_quantity(
        current: Position, quantity: float, exit_price: float, label: str, exit_time: pd.Timestamp
    ) -> None:
        nonlocal balance, position, next_entry_index
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
                    "symbol": symbol,
                    "side": current.side,
                    "entry_time": current.entry_time,
                    "exit_time": exit_time,
                    "entry": current.entry,
                    "last_exit": exit_price,
                    "quantity": current.quantity,
                    "pnl": current.pnl,
                    "funding_pnl": current.funding_pnl,
                    "r_multiple": current.pnl / risk_cash if risk_cash else 0.0,
                    "exit_reason": " / ".join(current.exits),
                    "tp_count": current.targets_hit,
                }
            )
            position = None

    for i in range(1, len(featured)):
        timestamp = featured.index[i]
        row = featured.iloc[i]
        previous = featured.iloc[i - 1]
        if position is not None:
            funding_rate = float(row.get("funding_rate", 0.0))
            if np.isfinite(funding_rate) and funding_rate != 0.0:
                funding_mark_price = row.get("funding_mark_price", row["open"])
                funding_price = (
                    float(funding_mark_price)
                    if pd.notna(funding_mark_price)
                    else float(row["open"])
                )
                funding_cash = (
                    -position.direction
                    * position.remaining
                    * funding_price
                    * funding_rate
                )
                position.funding_pnl += funding_cash
                position.pnl += funding_cash
                balance += funding_cash
        in_window = timestamp >= active_start
        may_enter = (
            in_window
            and i >= next_entry_index
            and i < len(featured) - 1
            and position is None
            and pd.notna(previous["atr"])
        )
        side: str | None = None
        if may_enter and bool(previous["long_signal"]):
            side = "LONG"
        elif may_enter and bool(previous["short_signal"]):
            side = "SHORT"

        if side is not None:
            entry = _adverse_fill(float(row["open"]), side, is_entry=True, slippage=slippage)
            risk_per_unit = float(previous["atr"] * config.stop_atr)
            if risk_per_unit > 0:
                direction = 1.0 if side == "LONG" else -1.0
                risk_quantity = balance * config.risk_per_trade / risk_per_unit
                capped_quantity = balance * config.max_leverage / entry
                quantity = min(risk_quantity, capped_quantity)
                entry_fee = quantity * entry * fee
                balance -= entry_fee
                targets = (
                    entry + direction * risk_per_unit * config.tp1_r,
                    entry + direction * risk_per_unit * config.tp2_r,
                    entry + direction * risk_per_unit * config.tp3_r,
                )
                fractions = config.target_fractions()
                position = Position(
                    side=side,
                    entry_time=timestamp,
                    entry=entry,
                    quantity=quantity,
                    remaining=quantity,
                    initial_stop=entry - direction * risk_per_unit,
                    stop=entry - direction * risk_per_unit,
                    targets=targets,
                    target_quantities=tuple(quantity * f for f in fractions),
                    risk_per_unit=risk_per_unit,
                    entry_fee=entry_fee,
                    pnl=-entry_fee,
                )

        if position is not None:
            current = position
            high = float(row["high"])
            low = float(row["low"])
            open_price = float(row["open"])
            if _touched(current.side, high, low, current.stop, stop=True):
                price = _gap_price(current.side, open_price, current.stop, True, slippage)
                close_quantity(current, current.remaining, price, "SL", timestamp)
                if position is None:
                    next_entry_index = i + config.cooldown_bars + 1
            else:
                for target_index in range(current.targets_hit, 3):
                    target = current.targets[target_index]
                    if not _touched(current.side, high, low, target, stop=False):
                        break
                    quantity = (
                        current.target_quantities[target_index]
                        if target_index < 2
                        else current.remaining
                    )
                    price = _gap_price(current.side, open_price, target, False, slippage)
                    current.targets_hit += 1
                    close_quantity(current, quantity, price, f"TP{target_index + 1}", timestamp)
                    if position is None:
                        next_entry_index = i + config.cooldown_bars + 1
                        break
                if position is not None and current.targets_hit >= 1:
                    if current.side == "LONG":
                        current.stop = max(current.stop, current.entry)
                    else:
                        current.stop = min(current.stop, current.entry)
                if position is not None and current.targets_hit >= 2 and pd.notna(row["atr"]):
                    trailing = float(row["atr"]) * 1.2
                    if current.side == "LONG":
                        current.stop = max(current.stop, float(row["close"]) - trailing)
                    else:
                        current.stop = min(current.stop, float(row["close"]) + trailing)

        if timestamp >= active_start:
            unrealized = 0.0
            if position is not None:
                unrealized = (
                    position.direction
                    * (float(row["close"]) - position.entry)
                    * position.remaining
                )
            curve[timestamp] = balance + unrealized

    if position is not None:
        final_time = featured.index[-1]
        final_close = float(featured["close"].iloc[-1])
        price = _adverse_fill(final_close, position.side, is_entry=False, slippage=slippage)
        close_quantity(position, position.remaining, price, "END", final_time)
        if final_time >= active_start:
            curve[final_time] = balance

    equity = pd.Series(curve, name=symbol or "equity", dtype=float).sort_index()
    trades = pd.DataFrame(completed)
    metrics = _metrics(equity, trades, initial_equity)
    return BacktestResult(symbol, config, equity, trades, metrics)


def backtest_ranked_long(
    universe: dict[str, pd.DataFrame],
    config: StrategyConfig,
    initial_equity: float = 10_000.0,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> BacktestResult:
    """Trade one selected long asset at a time from a shared portfolio account."""
    config.validate()
    if config.use_shorts:
        raise ValueError("The ranked long account only accepts long-only configurations.")
    if not universe:
        raise ValueError("The ranked long account requires at least one market.")
    featured = {
        symbol: build_feature_frame(frame, config) for symbol, frame in universe.items()
    }
    common_index = next(iter(featured.values())).index
    for frame in featured.values():
        common_index = common_index.intersection(frame.index)
    common_index = common_index.sort_values()
    if len(common_index) < 2:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            "RANKED_LONG", config, empty, pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity)
        )
    start = common_index[min(int(len(common_index) * start_ratio), len(common_index) - 2)]
    end = common_index[min(int(len(common_index) * end_ratio) - 1, len(common_index) - 1)]
    timestamps = common_index[common_index <= end]
    fee = config.fee_bps / 10_000.0
    slippage = config.slippage_bps / 10_000.0
    balance = initial_equity
    position: Position | None = None
    held_symbol: str | None = None
    completed: list[dict[str, object]] = []
    curve: dict[pd.Timestamp, float] = {}
    next_entry_index = 0

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
                    "exit_reason": " / ".join(current.exits),
                    "tp_count": current.targets_hit,
                }
            )
            position = None
            held_symbol = None

    for i in range(1, len(timestamps)):
        timestamp = timestamps[i]
        if position is not None and held_symbol is not None:
            row = featured[held_symbol].loc[timestamp]
            funding_rate = float(row.get("funding_rate", 0.0))
            if np.isfinite(funding_rate) and funding_rate != 0.0:
                funding_mark_price = row.get("funding_mark_price", row["open"])
                funding_price = (
                    float(funding_mark_price)
                    if pd.notna(funding_mark_price)
                    else float(row["open"])
                )
                funding_cash = (
                    -position.direction
                    * position.remaining
                    * funding_price
                    * funding_rate
                )
                position.funding_pnl += funding_cash
                position.pnl += funding_cash
                balance += funding_cash
        may_enter = (
            timestamp >= start
            and i >= next_entry_index
            and i < len(timestamps) - 1
            and position is None
        )
        if may_enter:
            previous_time = timestamps[i - 1]
            candidates = sorted(
                symbol
                for symbol, frame in featured.items()
                if bool(frame.loc[previous_time, "long_signal"])
                and pd.notna(frame.loc[previous_time, "atr"])
            )
            if candidates:
                held_symbol = candidates[0]
                row = featured[held_symbol].loc[timestamp]
                previous = featured[held_symbol].loc[previous_time]
                entry = _adverse_fill(
                    float(row["open"]), "LONG", is_entry=True, slippage=slippage
                )
                risk_per_unit = float(previous["atr"] * config.stop_atr)
                if risk_per_unit > 0:
                    risk_quantity = balance * config.risk_per_trade / risk_per_unit
                    capped_quantity = balance * config.max_leverage / entry
                    quantity = min(risk_quantity, capped_quantity)
                    entry_fee = quantity * entry * fee
                    balance -= entry_fee
                    targets = (
                        entry + risk_per_unit * config.tp1_r,
                        entry + risk_per_unit * config.tp2_r,
                        entry + risk_per_unit * config.tp3_r,
                    )
                    fractions = config.target_fractions()
                    position = Position(
                        side="LONG",
                        entry_time=timestamp,
                        entry=entry,
                        quantity=quantity,
                        remaining=quantity,
                        initial_stop=entry - risk_per_unit,
                        stop=entry - risk_per_unit,
                        targets=targets,
                        target_quantities=tuple(quantity * fraction for fraction in fractions),
                        risk_per_unit=risk_per_unit,
                        entry_fee=entry_fee,
                        pnl=-entry_fee,
                    )
                else:
                    held_symbol = None

        if position is not None and held_symbol is not None:
            current = position
            row = featured[held_symbol].loc[timestamp]
            high = float(row["high"])
            low = float(row["low"])
            open_price = float(row["open"])
            if _touched("LONG", high, low, current.stop, stop=True):
                price = _gap_price("LONG", open_price, current.stop, True, slippage)
                close_quantity(current, current.remaining, price, "SL", timestamp)
                if position is None:
                    next_entry_index = i + config.cooldown_bars + 1
            else:
                for target_index in range(current.targets_hit, 3):
                    target = current.targets[target_index]
                    if not _touched("LONG", high, low, target, stop=False):
                        break
                    quantity = (
                        current.target_quantities[target_index]
                        if target_index < 2
                        else current.remaining
                    )
                    price = _gap_price("LONG", open_price, target, False, slippage)
                    current.targets_hit += 1
                    close_quantity(current, quantity, price, f"TP{target_index + 1}", timestamp)
                    if position is None:
                        next_entry_index = i + config.cooldown_bars + 1
                        break
                if position is not None and current.targets_hit >= 1:
                    current.stop = max(current.stop, current.entry)
                if position is not None and current.targets_hit >= 2 and pd.notna(row["atr"]):
                    current.stop = max(current.stop, float(row["close"]) - float(row["atr"]) * 1.2)

        if timestamp >= start:
            unrealized = 0.0
            if position is not None and held_symbol is not None:
                row = featured[held_symbol].loc[timestamp]
                unrealized = (float(row["close"]) - position.entry) * position.remaining
            curve[timestamp] = balance + unrealized

    if position is not None and held_symbol is not None:
        final_time = timestamps[-1]
        final_close = float(featured[held_symbol].loc[final_time, "close"])
        price = _adverse_fill(final_close, "LONG", is_entry=False, slippage=slippage)
        close_quantity(position, position.remaining, price, "END", final_time)
        if final_time >= start:
            curve[final_time] = balance
    equity = pd.Series(curve, name="ranked_long_equity", dtype=float).sort_index()
    trades = pd.DataFrame(completed)
    return BacktestResult("RANKED_LONG", config, equity, trades, _metrics(equity, trades, initial_equity))


def combine_results(
    results: dict[str, BacktestResult], initial_equity: float = 10_000.0
) -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:
    curves = []
    trades = []
    divisor = max(len(results), 1)
    for result in results.values():
        if not result.equity.empty:
            curves.append(result.equity / initial_equity)
        if not result.trades.empty:
            adjusted = result.trades.copy()
            adjusted["pnl"] = adjusted["pnl"] / divisor
            if "funding_pnl" in adjusted:
                adjusted["funding_pnl"] = adjusted["funding_pnl"] / divisor
            trades.append(adjusted)
    if not curves:
        equity = pd.Series(dtype=float)
    else:
        equity = pd.concat(curves, axis=1).ffill().mean(axis=1) * initial_equity
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return equity, all_trades, _metrics(equity, all_trades, initial_equity)


def backtest_portfolio(
    universe: dict[str, pd.DataFrame],
    sleeves: list[tuple[str, float, StrategyConfig]],
    initial_equity: float = 10_000.0,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> PortfolioBacktestResult:
    """Evaluate weighted strategy sleeves equally allocated across market symbols."""
    if not universe or not sleeves:
        raise ValueError("Portfolio testing requires symbols and strategy sleeves.")
    if abs(sum(weight for _, weight, _ in sleeves) - 1.0) > 1e-9:
        raise ValueError("Portfolio sleeve weights must sum to 1.")
    symbol_curves: dict[str, pd.Series] = {}
    symbol_metrics: list[dict[str, float | str]] = []
    portfolio_trades: list[pd.DataFrame] = []
    symbol_count = len(universe)
    for symbol, frame in universe.items():
        start = frame.index[min(int(len(frame) * start_ratio), len(frame) - 2)]
        end = frame.index[min(int(len(frame) * end_ratio) - 1, len(frame) - 1)]
        curves: list[pd.Series] = []
        symbol_trades: list[pd.DataFrame] = []
        for sleeve_name, weight, config in sleeves:
            result = backtest(
                frame.loc[:end],
                config,
                symbol=symbol,
                initial_equity=initial_equity,
                trade_start=start,
                trade_end=end,
            )
            curves.append(result.equity / initial_equity * weight)
            if not result.trades.empty:
                trades = result.trades.copy()
                trades.insert(0, "sleeve", sleeve_name)
                trades["pnl"] = trades["pnl"] * weight
                if "funding_pnl" in trades:
                    trades["funding_pnl"] = trades["funding_pnl"] * weight
                symbol_trades.append(trades)
        symbol_equity = pd.concat(curves, axis=1).ffill().sum(axis=1) * initial_equity
        combined_symbol_trades = (
            pd.concat(symbol_trades, ignore_index=True) if symbol_trades else pd.DataFrame()
        )
        symbol_curves[symbol] = symbol_equity
        symbol_metrics.append(
            {"symbol": symbol, **_metrics(symbol_equity, combined_symbol_trades, initial_equity)}
        )
        if not combined_symbol_trades.empty:
            scaled = combined_symbol_trades.copy()
            scaled["pnl"] = scaled["pnl"] / symbol_count
            if "funding_pnl" in scaled:
                scaled["funding_pnl"] = scaled["funding_pnl"] / symbol_count
            portfolio_trades.append(scaled)
    equity = pd.concat(symbol_curves, axis=1).ffill().mean(axis=1)
    trades = pd.concat(portfolio_trades, ignore_index=True) if portfolio_trades else pd.DataFrame()
    metrics = _metrics(equity, trades, initial_equity)
    by_symbol = pd.DataFrame(symbol_metrics)
    asset_returns = by_symbol["return_pct"] if not by_symbol.empty else pd.Series(dtype=float)
    metrics["profitable_symbols_pct"] = float((asset_returns > 0).mean() * 100.0)
    metrics["worst_symbol_return_pct"] = float(asset_returns.min())
    metrics["median_symbol_return_pct"] = float(asset_returns.median())
    return PortfolioBacktestResult(equity, trades, metrics, by_symbol)
