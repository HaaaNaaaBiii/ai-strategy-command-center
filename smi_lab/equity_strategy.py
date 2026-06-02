from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .backtest import _metrics
from .indicators import ema


@dataclass(frozen=True)
class EquitySelectionConfig:
    market_symbol: str
    top_n: int = 3
    rebalance_bars: int = 20
    short_momentum_period: int = 63
    long_momentum_period: int = 126
    trend_period: int = 200
    volatility_period: int = 20
    max_volatility_pct: float = 80.0
    gross_exposure: float = 1.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    weighting_method: str = "equal"
    min_position_weight: float = 0.0
    max_position_weight: float = 1.0

    def validate(self) -> "EquitySelectionConfig":
        if self.top_n < 1 or self.rebalance_bars < 1:
            raise ValueError("top_n and rebalance_bars must be positive.")
        if min(self.short_momentum_period, self.long_momentum_period, self.trend_period) < 5:
            raise ValueError("indicator periods must be at least five bars.")
        if not 0 < self.gross_exposure <= 1.0:
            raise ValueError("gross_exposure must be in (0, 1].")
        if min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("trading costs must not be negative.")
        if self.weighting_method not in {"equal", "score", "capped_score"}:
            raise ValueError("weighting_method must be equal, score, or capped_score.")
        if self.min_position_weight < 0 or self.max_position_weight <= 0:
            raise ValueError("position weight caps must be positive.")
        if self.min_position_weight > self.max_position_weight:
            raise ValueError("min_position_weight must not exceed max_position_weight.")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class EquityStrategyResult:
    equity: pd.Series
    rebalances: pd.DataFrame
    metrics: dict[str, float]


def default_equity_config(market: str) -> EquitySelectionConfig:
    if market == "tw":
        return EquitySelectionConfig(
            market_symbol="0050.TW",
            top_n=3,
            rebalance_bars=20,
            short_momentum_period=20,
            long_momentum_period=126,
            trend_period=100,
            fee_bps=14.25,
            slippage_bps=5.0,
        )
    if market == "us":
        return EquitySelectionConfig(
            market_symbol="SPY",
            top_n=3,
            rebalance_bars=40,
            short_momentum_period=63,
            long_momentum_period=126,
            trend_period=200,
            fee_bps=1.0,
            slippage_bps=3.0,
        )
    raise ValueError(f"Unsupported equity market: {market}")


def _aligned_closes(universe: dict[str, pd.DataFrame]) -> tuple[list[str], pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    if not universe:
        raise ValueError("equity strategy requires a non-empty universe.")
    symbols = list(universe)
    index = next(iter(universe.values())).index
    for frame in universe.values():
        index = index.intersection(frame.index)
    index = index.sort_values()
    closes = pd.DataFrame(
        {symbol: universe[symbol].reindex(index)["close"] for symbol in symbols},
        index=index,
    ).astype(float)
    opens = pd.DataFrame(
        {symbol: universe[symbol].reindex(index)["open"] for symbol in symbols},
        index=index,
    ).astype(float)
    return symbols, index, closes, opens


def rank_equities(
    universe: dict[str, pd.DataFrame],
    config: EquitySelectionConfig,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    config.validate()
    if config.market_symbol not in universe:
        raise ValueError("market benchmark symbol is required for the market gate.")
    symbols, index, closes, _ = _aligned_closes(universe)
    timestamp = as_of or index[-1]
    position = index.get_indexer([timestamp], method="pad")[0]
    if position < max(config.long_momentum_period, config.trend_period):
        raise ValueError("not enough history for equity ranking.")
    market_close = closes[config.market_symbol]
    market_trend = ema(market_close, config.trend_period)
    risk_on = bool(market_close.iloc[position] > market_trend.iloc[position])
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        if symbol == config.market_symbol:
            continue
        close = closes[symbol]
        short_momentum = close.pct_change(config.short_momentum_period).iloc[position] * 100.0
        long_momentum = close.pct_change(config.long_momentum_period).iloc[position] * 100.0
        trend = ema(close, config.trend_period).iloc[position]
        volatility = close.pct_change().rolling(
            config.volatility_period,
            min_periods=config.volatility_period,
        ).std().iloc[position] * np.sqrt(252.0) * 100.0
        above_trend = bool(close.iloc[position] > trend)
        eligible = (
            risk_on
            and above_trend
            and pd.notna(short_momentum)
            and pd.notna(long_momentum)
            and short_momentum > 0
            and long_momentum > 0
            and volatility <= config.max_volatility_pct
        )
        score = (
            0.45 * float(long_momentum)
            + 0.30 * float(short_momentum)
            + (15.0 if above_trend else -15.0)
            - 0.10 * float(volatility)
        )
        rows.append(
            {
                "symbol": symbol,
                "as_of": index[position],
                "risk_on": risk_on,
                "eligible": eligible,
                "score": score,
                "short_momentum_pct": float(short_momentum),
                "long_momentum_pct": float(long_momentum),
                "annualized_volatility_pct": float(volatility),
                "above_trend": above_trend,
                "close": float(close.iloc[position]),
            }
        )
    return pd.DataFrame(rows).sort_values(["eligible", "score"], ascending=[False, False])


def _score_weights(selected: pd.DataFrame, gross_exposure: float) -> np.ndarray:
    scores = pd.to_numeric(selected.get("score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    if scores.sum() <= 0:
        return np.full(len(selected), gross_exposure / len(selected), dtype=float)
    return scores.to_numpy(dtype=float) / float(scores.sum()) * gross_exposure


def _apply_weight_caps(
    raw_weights: np.ndarray,
    gross_exposure: float,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    if len(raw_weights) == 0:
        return raw_weights
    min_weight = min(min_weight, gross_exposure / len(raw_weights))
    max_weight = max(max_weight, gross_exposure / len(raw_weights))
    remaining = np.ones(len(raw_weights), dtype=bool)
    final = np.zeros(len(raw_weights), dtype=float)
    remaining_total = gross_exposure
    base = np.maximum(raw_weights, 0.0)
    for _ in range(len(raw_weights) + 1):
        if not remaining.any():
            break
        active = np.where(remaining)[0]
        active_base = base[active]
        if active_base.sum() <= 0:
            scaled = np.full(len(active), remaining_total / len(active), dtype=float)
        else:
            scaled = active_base / float(active_base.sum()) * remaining_total
        low = scaled < min_weight
        high = scaled > max_weight
        if not low.any() and not high.any():
            final[active] = scaled
            remaining[active] = False
            break
        constrained = low | high
        for local_index, constrained_now in enumerate(constrained):
            if not constrained_now:
                continue
            index = active[local_index]
            value = min_weight if low[local_index] else max_weight
            final[index] = value
            remaining[index] = False
            remaining_total -= value
    if remaining.any():
        active = np.where(remaining)[0]
        final[active] = remaining_total / len(active)
    leftover = gross_exposure - float(final.sum())
    for _ in range(len(final) + 1):
        if abs(leftover) < 1e-10:
            break
        if leftover > 0:
            candidates = np.where(final < max_weight - 1e-10)[0]
            if len(candidates) == 0:
                break
            room = max_weight - final[candidates]
            add = np.minimum(room, leftover / len(candidates))
            final[candidates] += add
            leftover = gross_exposure - float(final.sum())
        else:
            candidates = np.where(final > min_weight + 1e-10)[0]
            if len(candidates) == 0:
                break
            room = final[candidates] - min_weight
            remove = np.minimum(room, abs(leftover) / len(candidates))
            final[candidates] -= remove
            leftover = gross_exposure - float(final.sum())
    total = final.sum()
    return final / total * gross_exposure if total > 0 else final


def target_weights_from_ranking(ranking: pd.DataFrame, config: EquitySelectionConfig) -> dict[str, float]:
    selected = ranking[ranking["eligible"]].head(config.top_n)
    if selected.empty:
        return {}
    if config.weighting_method == "equal":
        weights = np.full(len(selected), config.gross_exposure / len(selected), dtype=float)
    else:
        weights = _score_weights(selected, config.gross_exposure)
        if config.weighting_method == "capped_score":
            weights = _apply_weight_caps(
                weights,
                config.gross_exposure,
                config.min_position_weight,
                config.max_position_weight,
            )
    return dict(zip(selected["symbol"].astype(str), weights))


def backtest_equity_selection(
    universe: dict[str, pd.DataFrame],
    config: EquitySelectionConfig,
    initial_equity: float = 10_000.0,
) -> EquityStrategyResult:
    config.validate()
    symbols, index, closes, opens = _aligned_closes(universe)
    if config.market_symbol not in symbols:
        raise ValueError("market benchmark symbol is required for backtesting.")
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    cash = initial_equity
    quantities = np.zeros(len(symbols), dtype=float)
    prior_weights = np.zeros(len(symbols), dtype=float)
    curve: dict[pd.Timestamp, float] = {}
    events: list[dict[str, object]] = []
    start = max(config.long_momentum_period, config.trend_period) + 1
    for i in range(start, len(index)):
        if (i - start) % config.rebalance_bars == 0:
            ranking = rank_equities(universe, config, as_of=index[i - 1])
            target_weights = target_weights_from_ranking(ranking, config)
            selected_symbols = list(target_weights)
            weights = np.zeros(len(symbols), dtype=float)
            for symbol, weight in target_weights.items():
                weights[symbols.index(symbol)] = weight
            if np.any(weights != prior_weights):
                open_equity = cash + float(np.sum(quantities * opens.iloc[i].to_numpy(dtype=float)))
                desired_values = open_equity * weights
                current_values = quantities * opens.iloc[i].to_numpy(dtype=float)
                turnover = float(np.abs(desired_values - current_values).sum())
                cost = turnover * cost_rate
                cash = open_equity - float(desired_values.sum()) - cost
                quantities = np.divide(
                    desired_values,
                    opens.iloc[i].to_numpy(dtype=float),
                    out=np.zeros(len(symbols), dtype=float),
                    where=opens.iloc[i].to_numpy(dtype=float) != 0.0,
                )
                prior_weights = weights
                events.append(
                    {
                        "timestamp": index[i],
                        "selected_symbols": ",".join(selected_symbols) or "CASH",
                        "target_weights": ",".join(
                            f"{symbol}:{target_weights[symbol]:.6f}" for symbol in selected_symbols
                        )
                        or "CASH:0.000000",
                        "turnover": turnover,
                        "turnover_pct": turnover / open_equity * 100.0 if open_equity else 0.0,
                        "cost": cost,
                        "equity_before_cost": open_equity,
                    }
                )
        curve[index[i]] = cash + float(np.sum(quantities * closes.iloc[i].to_numpy(dtype=float)))
    equity = pd.Series(curve, name="equity_selection", dtype=float).sort_index()
    metrics = _metrics(equity, pd.DataFrame(), initial_equity)
    event_frame = pd.DataFrame(events)
    metrics.update(
        {
            "rebalances": float(len(events)),
            "avg_turnover_pct": float(event_frame["turnover_pct"].mean()) if not event_frame.empty else 0.0,
            "total_cost": float(event_frame["cost"].sum()) if not event_frame.empty else 0.0,
        }
    )
    return EquityStrategyResult(equity, event_frame, metrics)


def benchmark_buy_and_hold(
    frame: pd.DataFrame,
    initial_equity: float = 10_000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> EquityStrategyResult:
    if frame.empty:
        empty = pd.Series(dtype=float)
        return EquityStrategyResult(empty, pd.DataFrame(), _metrics(empty, pd.DataFrame(), initial_equity))
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    invested = initial_equity / (1.0 + cost_rate)
    quantity = invested / float(frame["open"].iloc[0])
    equity = frame["close"].astype(float) * quantity
    equity.iloc[-1] *= 1.0 - cost_rate
    equity.name = "benchmark"
    metrics = _metrics(equity, pd.DataFrame(), initial_equity)
    metrics.update({"rebalances": 1.0})
    return EquityStrategyResult(equity, pd.DataFrame(), metrics)
