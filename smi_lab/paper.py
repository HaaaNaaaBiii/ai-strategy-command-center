from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .allocation import (
    TrendAllocationConfig,
    _aligned_frames,
    backtest_buy_and_hold,
    backtest_staggered_trend_allocation,
)
from .data import data_window, utc_timestamp
from .paths import output_path
from .rotation import _rebalance_schedule


DEFAULT_STRATEGY_PATH = output_path("market_alpha_staggered", "paper_strategy.json")
DEFAULT_TRACKING_DIR = output_path("forward_tracking")
DEFAULT_STATE_PATH = DEFAULT_TRACKING_DIR / "market_alpha_staggered_state.json"
EVENT_COLUMNS = [
    "timestamp",
    "selected_symbols",
    "gross_exposure",
    "turnover",
    "cost",
    "equity_before_cost",
    "rebalance_offset_bars",
]


@dataclass(frozen=True)
class PaperUpdate:
    status: str
    started_at: str
    last_updated_at: str
    latest_candle: str
    equity: float
    return_pct: float
    equal_weight_return_pct: float
    btc_return_pct: float
    excess_vs_equal_weight_pct: float
    excess_vs_btc_pct: float
    max_drawdown_pct: float
    forward_days: float
    data_lag_hours: float
    live_ready: bool
    blockers: list[str]
    events: int
    data_window: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_allocation_strategy(
    path: str | Path = DEFAULT_STRATEGY_PATH,
) -> tuple[TrendAllocationConfig, tuple[int, ...], dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = TrendAllocationConfig(**payload["config"]).validate()
    offsets = tuple(int(offset) for offset in payload.get("rebalance_offsets", range(config.rebalance_bars)))
    return config, offsets, payload


def allocation_snapshot(
    universe: dict[str, pd.DataFrame],
    config: TrendAllocationConfig,
    offsets: tuple[int, ...],
) -> pd.DataFrame:
    symbols, index, closes, _, _, _ = _aligned_frames(universe)
    if len(index) < max(config.momentum_period, config.asset_ema_period, config.btc_ema_period) + 2:
        raise ValueError("Not enough bars for the allocation snapshot.")
    close_array = closes.to_numpy(dtype=float)
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
    rows: list[dict[str, object]] = []
    aggregate_weights = {symbol: 0.0 for symbol in symbols}
    sleeve_weight = 1.0 / len(offsets)
    for offset in offsets:
        schedule = _rebalance_schedule(index, config.rebalance_bars, offset)
        scheduled_positions = np.flatnonzero(schedule.to_numpy())
        scheduled_positions = scheduled_positions[scheduled_positions > 0]
        if len(scheduled_positions) == 0:
            continue
        bar_index = int(scheduled_positions[-1])
        prior = bar_index - 1
        eligible = (
            (close_array[prior, btc_index] > btc_ema[prior])
            & (close_array[prior] > asset_ema[prior])
            & (momentum[prior] > 0.0)
        )
        scores = np.where(eligible, momentum[prior], -np.inf)
        order = np.argsort(scores)[::-1]
        selected = order[np.isfinite(scores[order])][: config.top_n]
        selected_symbols = [symbols[item] for item in selected]
        target_weight = config.gross_exposure * sleeve_weight
        if selected_symbols:
            per_symbol = target_weight / len(selected_symbols)
            for symbol in selected_symbols:
                aggregate_weights[symbol] += per_symbol
        rows.append(
            {
                "rebalance_offset_bars": offset,
                "last_rebalance_time": index[bar_index],
                "decision_candle": index[prior],
                "selected_symbols": ",".join(selected_symbols) or "CASH",
                "sleeve_target_weight": target_weight if selected_symbols else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["aggregate_weights"] = aggregate_weights
    frame.attrs["latest_candle"] = index[-1]
    return frame


def aggregate_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    weights = snapshot.attrs.get("aggregate_weights", {})
    rows = [
        {"asset": asset, "target_weight": weight}
        for asset, weight in weights.items()
        if weight > 1e-12
    ]
    cash = max(0.0, 1.0 - sum(weights.values()))
    rows.append({"asset": "CASH", "target_weight": cash})
    return pd.DataFrame(rows).sort_values("target_weight", ascending=False)


def _drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min() * 100.0)


def _latest_close_time(frame: pd.DataFrame) -> pd.Timestamp:
    if "close_time" in frame:
        return pd.Timestamp(frame["close_time"].iloc[-1])
    return pd.Timestamp(frame.index[-1])


def _readiness_blockers(
    forward_days: float,
    data_lag_hours: float,
    return_pct: float,
    equal_weight_return_pct: float,
    max_drawdown_pct: float,
    events: int,
) -> list[str]:
    blockers: list[str] = []
    if forward_days < 30.0:
        blockers.append("forward paper tracking is shorter than 30 days")
    if data_lag_hours > 12.0:
        blockers.append("market data is older than 12 hours")
    if return_pct < equal_weight_return_pct:
        blockers.append("paper strategy has not outperformed the equal-weight crypto benchmark")
    if max_drawdown_pct < -10.0:
        blockers.append("paper drawdown exceeded the -10% live-readiness guardrail")
    if events < 3:
        blockers.append("too few forward rebalance events to validate execution behavior")
    return blockers


def _load_state(path: Path, latest_candle: pd.Timestamp, initial_equity: float) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "started_at": latest_candle.isoformat(),
        "initial_equity": initial_equity,
        "created_at": utc_timestamp(),
        "last_notified_marker": "",
    }


def update_forward_tracking(
    universe: dict[str, pd.DataFrame],
    strategy_path: str | Path = DEFAULT_STRATEGY_PATH,
    output_dir: str | Path = DEFAULT_TRACKING_DIR,
    state_path: str | Path = DEFAULT_STATE_PATH,
    initial_equity: float = 10_000.0,
) -> PaperUpdate:
    config, offsets, _ = load_allocation_strategy(strategy_path)
    output = Path(output_dir)
    state_file = Path(state_path)
    output.mkdir(parents=True, exist_ok=True)
    latest_candle = universe["BTCUSDT"].index[-1]
    state = _load_state(state_file, latest_candle, initial_equity)
    started_at = pd.Timestamp(str(state["started_at"]))
    starting_equity = float(state.get("initial_equity", initial_equity))
    account_equity = starting_equity
    status = "initialized"
    events = pd.DataFrame()
    equity = pd.Series(dtype=float)
    if latest_candle > started_at:
        result = backtest_staggered_trend_allocation(
            universe,
            config,
            offsets,
            initial_equity=account_equity,
            trade_start=started_at,
            trade_end=latest_candle,
        )
        equity = result.equity
        events = result.rebalances
        if not equity.empty:
            equity.reset_index().rename(
                columns={"index": "timestamp", equity.name or 0: "equity"}
            ).to_csv(output / "market_alpha_staggered_equity.csv", index=False)
            account_equity = float(equity.iloc[-1])
        if events.empty:
            events = pd.DataFrame(columns=EVENT_COLUMNS)
        events.to_csv(output / "market_alpha_staggered_events.csv", index=False)
        status = "updated"
    else:
        pd.DataFrame(
            [{"timestamp": latest_candle, "equity": account_equity}]
        ).to_csv(output / "market_alpha_staggered_equity.csv", index=False)
        pd.DataFrame(columns=EVENT_COLUMNS).to_csv(
            output / "market_alpha_staggered_events.csv", index=False
        )
    equal_weights = {symbol: 1.0 / len(universe) for symbol in universe}
    btc_weights = {symbol: (1.0 if symbol == "BTCUSDT" else 0.0) for symbol in universe}
    equal_weight = backtest_buy_and_hold(
        universe,
        equal_weights,
        initial_equity=starting_equity,
        trade_start=started_at,
        trade_end=latest_candle,
    )
    btc = backtest_buy_and_hold(
        universe,
        btc_weights,
        initial_equity=starting_equity,
        trade_start=started_at,
        trade_end=latest_candle,
    )
    pd.DataFrame(
        [
            {"benchmark": "strategy", "return_pct": (account_equity / starting_equity - 1.0) * 100.0, "max_drawdown_pct": _drawdown_pct(equity)},
            {"benchmark": "equal_weight_market", **equal_weight.metrics},
            {"benchmark": "BTCUSDT", **btc.metrics},
        ]
    ).to_csv(output / "market_alpha_staggered_forward_benchmarks.csv", index=False)
    return_pct = (account_equity / starting_equity - 1.0) * 100.0
    equal_weight_return = float(equal_weight.metrics["return_pct"])
    btc_return = float(btc.metrics["return_pct"])
    max_drawdown = _drawdown_pct(equity)
    forward_days = max(
        (latest_candle - started_at).total_seconds() / 86400.0,
        0.0,
    )
    latest_close = _latest_close_time(universe["BTCUSDT"])
    data_lag_hours = max(
        (
            datetime.now(timezone.utc)
            - latest_close.to_pydatetime().astimezone(timezone.utc)
        ).total_seconds()
        / 3600.0,
        0.0,
    )
    blockers = _readiness_blockers(
        forward_days,
        data_lag_hours,
        return_pct,
        equal_weight_return,
        max_drawdown,
        len(events),
    )
    state.update(
        {
            "last_updated_at": utc_timestamp(),
            "latest_candle": latest_candle.isoformat(),
            "last_equity": account_equity,
            "return_pct": return_pct,
        }
    )
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    update = PaperUpdate(
        status=status,
        started_at=str(state["started_at"]),
        last_updated_at=str(state["last_updated_at"]),
        latest_candle=latest_candle.isoformat(),
        equity=account_equity,
        return_pct=return_pct,
        equal_weight_return_pct=equal_weight_return,
        btc_return_pct=btc_return,
        excess_vs_equal_weight_pct=return_pct - equal_weight_return,
        excess_vs_btc_pct=return_pct - btc_return,
        max_drawdown_pct=max_drawdown,
        forward_days=forward_days,
        data_lag_hours=data_lag_hours,
        live_ready=not blockers,
        blockers=blockers,
        events=len(events),
        data_window=data_window(universe["BTCUSDT"]),
    )
    (output / "market_alpha_staggered_status.json").write_text(
        json.dumps(update.to_dict(), indent=2), encoding="utf-8"
    )
    return update


def format_allocation_report(snapshot: pd.DataFrame, update: PaperUpdate | None = None) -> str:
    aggregate = aggregate_snapshot(snapshot)
    latest = snapshot.attrs.get("latest_candle")
    lines = [
        "Market Alpha Allocation Snapshot",
        f"Latest candle UTC: {latest}",
        "Target allocation:",
    ]
    for row in aggregate.to_dict("records"):
        lines.append(f"- {row['asset']}: {row['target_weight']:.2%}")
    if update is not None:
        lines.extend(
            [
                "",
                f"Paper tracking: {update.status}",
                f"Started: {update.started_at}",
                f"Equity: {update.equity:.2f} ({update.return_pct:+.2f}%)",
                f"Equal-weight benchmark: {update.equal_weight_return_pct:+.2f}%",
                f"BTC benchmark: {update.btc_return_pct:+.2f}%",
                f"Live ready: {update.live_ready}",
            ]
        )
        if update.blockers:
            lines.append("Blockers:")
            lines.extend(f"- {blocker}" for blocker in update.blockers)
    return "\n".join(lines)
