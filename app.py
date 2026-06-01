from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smi_lab.accounts import (
    ACCOUNT_COLUMNS,
    ORDER_COLUMNS,
    POSITION_COLUMNS,
    AccountSnapshot,
    OrderTracker,
    PositionSnapshot,
    append_order,
    load_table,
    upsert_account,
    upsert_position,
)
from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.equity_data import (
    DEFAULT_TW_SYMBOLS,
    DEFAULT_US_SYMBOLS,
    load_equity_universe,
)
from smi_lab.equity_signals import (
    add_company_names,
    build_equity_trade_plan,
    company_name,
)
from smi_lab.equity_strategy import (
    backtest_equity_selection,
    benchmark_buy_and_hold,
    default_equity_config,
    rank_equities,
)
from smi_lab.indicators import atr, ema
from smi_lab.notifier import send_discord, send_telegram
from smi_lab.paper import (
    aggregate_snapshot,
    allocation_snapshot,
    format_allocation_report,
    load_allocation_strategy,
    update_forward_tracking,
)
from smi_lab.technical import summarize_universe


OUTPUT_DIR = Path("outputs")
MARKET_ALPHA_DIR = OUTPUT_DIR / "market_alpha_staggered"
TRACKING_DIR = OUTPUT_DIR / "forward_tracking"
EQUITY_SELECTION_DIR = OUTPUT_DIR / "equity_selection"
ACCOUNT_DIR = OUTPUT_DIR / "accounts"
ACCOUNTS_FILE = ACCOUNT_DIR / "accounts.csv"
POSITIONS_FILE = ACCOUNT_DIR / "positions.csv"
ORDERS_FILE = ACCOUNT_DIR / "orders.csv"


st.set_page_config(
    page_title="AI Strategy Command Center",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2.2rem;}
    [data-testid="stSidebar"] {background: #0b1220;}
    .hero {
        padding: 1.0rem 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.14), rgba(15, 23, 42, 0.72));
        margin-bottom: 1.0rem;
    }
    .hero h1 {font-size: 2.0rem; margin-bottom: 0.2rem;}
    .hero p {color: #cbd5e1; margin-bottom: 0;}
    .section-card {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.45);
    }
    .small-muted {color: #94a3b8; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}%"


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def price(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.4f}".rstrip("0").rstrip(".")


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def read_json_or_empty(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_crypto_data(
    symbols: list[str], interval: str, bars: int, refresh: bool
) -> dict[str, pd.DataFrame]:
    return load_universe(
        symbols,
        interval=interval,
        bars=bars,
        refresh=refresh,
        market="perpetual",
        include_funding=True,
    )


def send_report(
    channel: str,
    message: str,
    discord_url: str,
    telegram_token: str,
    telegram_chat: str,
) -> None:
    if channel == "Discord":
        send_discord(message, discord_url)
    elif channel == "Telegram":
        send_telegram(message, telegram_token, telegram_chat)
    else:
        raise ValueError(f"Unsupported channel: {channel}")


def display_technical_table(frame: pd.DataFrame, show_company: bool = False) -> None:
    if frame.empty:
        st.info("No technical data is loaded yet.")
        return
    display = frame.copy()
    if show_company and "symbol" in display:
        display = add_company_names(display)
    for column in ("close", "rsi", "macd", "macd_signal", "roc_pct", "atr_pct"):
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.2f}")
    st.dataframe(display, hide_index=True, width="stretch")


def latest_status() -> dict[str, object]:
    return read_json_or_empty(TRACKING_DIR / "market_alpha_staggered_status.json")


def chart_ohlc(
    frame: pd.DataFrame,
    title: str,
    levels: dict[str, float] | None = None,
    trend_period: int | None = None,
    marker_times: list[pd.Timestamp] | None = None,
) -> go.Figure:
    data = frame.tail(180).dropna(subset=["open", "high", "low", "close"])
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )
    for period, color in ((20, "#38bdf8"), (50, "#f59e0b")):
        if len(data) >= period:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ema(data["close"].astype(float), period),
                    mode="lines",
                    name=f"EMA {period}",
                    line=dict(color=color, width=1.4),
                )
            )
    if trend_period and len(data) >= min(trend_period, len(data)):
        trend = ema(frame["close"].astype(float), trend_period).reindex(data.index)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=trend,
                mode="lines",
                name=f"Strategy Trend EMA {trend_period}",
                line=dict(color="#a78bfa", width=1.8),
            )
        )
    if levels:
        colors = {
            "Entry": "#60a5fa",
            "Strategy Exit": "#f97316",
            "Stop Loss": "#ef4444",
            "TP1": "#22c55e",
            "TP2": "#16a34a",
        }
        for name, value in levels.items():
            if value is None or pd.isna(value):
                continue
            fig.add_hline(
                y=float(value),
                line_dash="dash",
                line_color=colors.get(name, "#94a3b8"),
                annotation_text=f"{name}: {price(value)}",
                annotation_position="top left",
            )
    if marker_times:
        marker_times = [time for time in marker_times if time in data.index]
        if marker_times:
            closes = data.loc[marker_times, "close"]
            fig.add_trace(
                go.Scatter(
                    x=marker_times,
                    y=closes,
                    mode="markers",
                    name="Strategy Rebalance",
                    marker=dict(size=10, color="#eab308", symbol="triangle-up"),
                )
            )
    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def crypto_levels(symbol: str, frame: pd.DataFrame, target_weight: float) -> dict[str, float]:
    close = float(frame["close"].iloc[-1])
    current_atr = float(atr(frame, 14).dropna().iloc[-1])
    if not target_weight:
        return {
            "Entry": close,
            "Strategy Exit": close,
            "Stop Loss": close - 2.0 * current_atr,
            "TP1": close + 2.0 * current_atr,
            "TP2": close + 4.0 * current_atr,
        }
    return {
        "Entry": close,
        "Strategy Exit": close - 2.5 * current_atr,
        "Stop Loss": close - 2.0 * current_atr,
        "TP1": close + 2.0 * current_atr,
        "TP2": close + 4.0 * current_atr,
    }


def selectable_symbol_table(
    frame: pd.DataFrame, key: str, default_symbol: str | None = None
) -> str | None:
    if frame.empty:
        return default_symbol
    display = add_company_names(frame) if "company" not in frame else frame
    event = st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    selected_symbol = default_symbol
    try:
        rows = event.selection.rows
        if rows:
            selected_symbol = str(display.iloc[rows[0]]["symbol"])
    except Exception:
        pass
    choices = display["symbol"].tolist()
    if choices:
        selected_symbol = st.selectbox(
            "Chart symbol",
            choices,
            index=choices.index(selected_symbol) if selected_symbol in choices else 0,
            format_func=lambda symbol: f"{symbol} | {company_name(symbol)}",
            key=f"{key}_selectbox",
        )
    return selected_symbol


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


with st.sidebar:
    st.title("AI Strategy Lab")
    page = st.radio(
        "Workspace",
        [
            "Dashboard",
            "Crypto",
            "Stocks",
            "Accounts",
            "Research",
            "Records",
            "Deployment",
        ],
    )
    st.divider()
    st.caption("Global crypto data controls")
    crypto_symbols = st.multiselect(
        "Crypto universe",
        list(DEFAULT_SYMBOLS),
        default=list(DEFAULT_SYMBOLS),
    )
    crypto_interval = st.selectbox("Crypto interval", ["4h", "1h", "1d", "15m"], index=0)
    crypto_years = st.slider("Crypto history years", 1, 5, 2)
    crypto_bars = bars_for_years(crypto_interval, crypto_years)
    refresh_crypto = st.checkbox("Refresh market data", value=False)


if page == "Dashboard":
    hero(
        "AI Strategy Command Center",
        "Crypto allocation, Taiwan and U.S. stock selection, account tracking, and deployment status in one workspace.",
    )
    status = latest_status()
    metrics = read_csv_or_empty(MARKET_ALPHA_DIR / "selected_metrics.csv")
    tw_metrics = read_csv_or_empty(EQUITY_SELECTION_DIR / "tw_metrics.csv")
    us_metrics = read_csv_or_empty(EQUITY_SELECTION_DIR / "us_metrics.csv")
    cols = st.columns(4)
    cols[0].metric("Crypto live-ready", str(status.get("live_ready", "Unknown")))
    cols[1].metric("Crypto paper return", pct(status.get("return_pct")))
    cols[2].metric("Forward days", money(status.get("forward_days")))
    cols[3].metric("Open blockers", len(status.get("blockers", [])) if status else "-")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Taiwan Strategy")
        st.dataframe(tw_metrics, hide_index=True, width="stretch")
    with cols[1]:
        st.subheader("U.S. Strategy")
        st.dataframe(us_metrics, hide_index=True, width="stretch")
    if not metrics.empty:
        st.subheader("Crypto Research Metrics")
        st.dataframe(metrics.tail(20), hide_index=True, width="stretch")

elif page == "Crypto":
    hero(
        "Crypto Strategy",
        "Signal center, forward paper tracking, and Pionex live-account order tracking. Execution remains manual until API risk controls are explicitly enabled.",
    )
    signal_tab, tracking_tab, chart_tab, notify_tab = st.tabs(
        ["Signal Center", "Forward Tracking", "K-Line & Levels", "Notification"]
    )
    with signal_tab:
        if st.button("Refresh crypto signal", type="primary"):
            try:
                universe = load_crypto_data(
                    crypto_symbols,
                    crypto_interval,
                    max(crypto_bars, 500),
                    refresh_crypto,
                )
                st.session_state["crypto_universe"] = universe
                st.success(f"Loaded {len(universe)} symbols: {data_window(next(iter(universe.values())))}")
            except Exception as exc:
                st.error(f"Crypto data load failed: {exc}")
        universe = st.session_state.get("crypto_universe")
        if universe:
            display_technical_table(summarize_universe(universe))
            try:
                config, offsets, metadata = load_allocation_strategy()
                snapshot = allocation_snapshot(universe, config, offsets)
                allocation = aggregate_snapshot(snapshot)
                st.subheader("Target Allocation")
                st.dataframe(allocation, hide_index=True, width="stretch")
                st.json(metadata, expanded=False)
            except Exception as exc:
                st.warning(f"Allocation snapshot failed: {exc}")
        else:
            st.info("Refresh crypto signal to load technical summaries and allocation.")
    with tracking_tab:
        if st.button("Update forward paper tracking", type="primary"):
            try:
                universe = load_crypto_data(
                    list(DEFAULT_SYMBOLS),
                    "4h",
                    max(bars_for_years("4h", 2), 500),
                    refresh_crypto,
                )
                update = update_forward_tracking(universe)
                config, offsets, _ = load_allocation_strategy()
                snapshot = allocation_snapshot(universe, config, offsets)
                st.session_state["paper_update"] = update
                st.session_state["paper_message"] = format_allocation_report(snapshot, update)
                st.success("Forward tracking updated.")
            except Exception as exc:
                st.error(f"Forward tracking failed: {exc}")
        update = st.session_state.get("paper_update")
        status = latest_status()
        if update:
            status = update.to_dict()
        if status:
            cols = st.columns(5)
            cols[0].metric("Live ready", str(status.get("live_ready")))
            cols[1].metric("Equity", money(status.get("equity")))
            cols[2].metric("Return", pct(status.get("return_pct")))
            cols[3].metric("Equal-weight", pct(status.get("equal_weight_return_pct")))
            cols[4].metric("Max DD", pct(status.get("max_drawdown_pct")))
            blockers = status.get("blockers") or []
            if blockers:
                st.warning("Live-readiness blockers: " + " | ".join(str(item) for item in blockers))
        equity = read_csv_or_empty(TRACKING_DIR / "market_alpha_staggered_equity.csv")
        if not equity.empty and "equity" in equity:
            st.line_chart(equity.set_index("timestamp")["equity"])
        events = read_csv_or_empty(TRACKING_DIR / "market_alpha_staggered_events.csv")
        if not events.empty:
            st.dataframe(events.tail(80), hide_index=True, width="stretch")
    with chart_tab:
        universe = st.session_state.get("crypto_universe")
        if not universe:
            st.info("Load crypto data in Signal Center first.")
        else:
            config, offsets, _ = load_allocation_strategy()
            snapshot = allocation_snapshot(universe, config, offsets)
            allocation = aggregate_snapshot(snapshot)
            weights = {
                str(row["asset"]): float(row["target_weight"])
                for row in allocation.to_dict("records")
            }
            symbol = st.selectbox("Crypto chart", list(universe), key="crypto_chart_symbol")
            target_weight = weights.get(symbol, 0.0)
            levels = crypto_levels(symbol, universe[symbol], target_weight)
            cols = st.columns(5)
            cols[0].metric("Target weight", pct(target_weight * 100.0))
            cols[1].metric("Entry", price(levels["Entry"]))
            cols[2].metric("Stop", price(levels["Stop Loss"]))
            cols[3].metric("TP1", price(levels["TP1"]))
            cols[4].metric("TP2", price(levels["TP2"]))
            st.plotly_chart(
                chart_ohlc(universe[symbol], f"{symbol} strategy levels", levels, 50),
                width="stretch",
            )
    with notify_tab:
        st.subheader("Manual notification")
        channel = st.radio("Channel", ["Discord", "Telegram"], horizontal=True)
        discord_url = st.text_input(
            "Discord webhook URL",
            value=os.getenv("DISCORD_WEBHOOK_URL", ""),
            type="password",
        )
        telegram_token = st.text_input(
            "Telegram bot token",
            value=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            type="password",
        )
        telegram_chat = st.text_input(
            "Telegram chat ID",
            value=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
        if st.button("Send latest allocation report"):
            try:
                universe = st.session_state.get("crypto_universe")
                if not universe:
                    universe = load_crypto_data(
                        list(DEFAULT_SYMBOLS),
                        "4h",
                        max(bars_for_years("4h", 2), 500),
                        False,
                    )
                config, offsets, _ = load_allocation_strategy()
                snapshot = allocation_snapshot(universe, config, offsets)
                message = format_allocation_report(snapshot)
                send_report(channel, message, discord_url, telegram_token, telegram_chat)
                st.success("Notification sent.")
            except Exception as exc:
                st.error(f"Notification failed: {exc}")

elif page == "Stocks":
    hero(
        "Taiwan / U.S. Stock Strategy",
        "Market-adjusted selection, company names, selectable candlestick charts, and explicit entry, exit, stop, and TP levels.",
    )

    def render_equity_page(title: str, market: str, defaults: tuple[str, ...]) -> None:
        config = default_equity_config(market)
        with st.expander(f"{title} data controls", expanded=False):
            symbols_text = st.text_area(
                "Universe",
                value="\n".join(defaults),
                height=130,
                key=f"{market}_symbols",
            )
            col_a, col_b, col_c = st.columns(3)
            interval = col_a.selectbox("Interval", ["1d", "1wk", "1h"], index=0, key=f"{market}_interval")
            range_ = col_b.selectbox("History", ["1y", "2y", "5y", "6mo"], index=1, key=f"{market}_range")
            refresh = col_c.checkbox("Refresh", value=False, key=f"{market}_refresh")
            run = st.button(f"Load {title}", type="primary", key=f"{market}_load")
        if run:
            symbols = [line.strip() for line in symbols_text.splitlines() if line.strip()]
            source_symbols = list(dict.fromkeys([*symbols, config.market_symbol]))
            try:
                universe = load_equity_universe(
                    source_symbols,
                    market=market,
                    interval=interval,
                    range_=range_,
                    refresh=refresh,
                )
                st.session_state[f"{market}_universe"] = universe
                st.success(f"Loaded {len(universe)} symbols.")
            except Exception as exc:
                st.error(f"{title} data load failed: {exc}")
        universe = st.session_state.get(f"{market}_universe")
        if not universe:
            st.info(f"Load {title} data to show strategy ranking and charts.")
            return
        technical = summarize_universe(universe)
        display_technical_table(technical, show_company=True)
        ranking = rank_equities(universe, config)
        ranking = add_company_names(ranking)
        result = backtest_equity_selection(universe, config)
        benchmark = benchmark_buy_and_hold(
            universe[config.market_symbol],
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
        )
        EQUITY_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(EQUITY_SELECTION_DIR / f"{market}_ranking.csv", index=False)
        result.rebalances.to_csv(EQUITY_SELECTION_DIR / f"{market}_rebalances.csv", index=False)
        pd.DataFrame(
            [
                {"strategy": "equity_selection", **result.metrics},
                {"strategy": config.market_symbol, **benchmark.metrics},
            ]
        ).to_csv(EQUITY_SELECTION_DIR / f"{market}_metrics.csv", index=False)
        cols = st.columns(4)
        cols[0].metric("Strategy return", pct(result.metrics["return_pct"]))
        cols[1].metric("Benchmark return", pct(benchmark.metrics["return_pct"]))
        cols[2].metric("Strategy max DD", pct(result.metrics["max_drawdown_pct"]))
        cols[3].metric("Rebalances", int(float(result.metrics["rebalances"])))
        st.subheader("Ranking")
        selected_symbol = selectable_symbol_table(
            ranking,
            key=f"{market}_ranking_table",
            default_symbol=ranking["symbol"].iloc[0] if not ranking.empty else None,
        )
        if not selected_symbol:
            return
        plan = build_equity_trade_plan(selected_symbol, universe, config, ranking)
        st.subheader(f"{selected_symbol} | {plan.company}")
        cols = st.columns(6)
        cols[0].metric("Action", plan.action)
        cols[1].metric("Entry", price(plan.entry_price))
        cols[2].metric("Strategy exit", price(plan.strategy_exit))
        cols[3].metric("Stop loss", price(plan.stop_loss))
        cols[4].metric("TP1", price(plan.take_profit_1))
        cols[5].metric("TP2", price(plan.take_profit_2))
        st.caption(plan.reason)
        markers: list[pd.Timestamp] = []
        if not result.rebalances.empty:
            selected_events = result.rebalances[
                result.rebalances["selected_symbols"].fillna("").str.contains(
                    selected_symbol, regex=False
                )
            ]
            markers = [pd.Timestamp(value) for value in selected_events["timestamp"].tolist()]
        levels = {
            "Entry": plan.entry_price,
            "Strategy Exit": plan.strategy_exit,
            "Stop Loss": plan.stop_loss,
            "TP1": plan.take_profit_1,
            "TP2": plan.take_profit_2,
        }
        st.plotly_chart(
            chart_ohlc(
                universe[selected_symbol],
                f"{selected_symbol} strategy chart",
                levels=levels,
                trend_period=config.trend_period,
                marker_times=markers,
            ),
            width="stretch",
        )
        curves = pd.concat(
            {"strategy": result.equity, config.market_symbol: benchmark.equity},
            axis=1,
            sort=False,
        ).dropna()
        if not curves.empty:
            st.subheader("Strategy vs Benchmark")
            st.line_chart(curves / curves.iloc[0] * 100.0)

    tw_tab, us_tab = st.tabs(["Taiwan Stocks", "U.S. Stocks"])
    with tw_tab:
        render_equity_page("Taiwan Stocks", "tw", DEFAULT_TW_SYMBOLS)
    with us_tab:
        render_equity_page("U.S. Stocks", "us", DEFAULT_US_SYMBOLS)

elif page == "Accounts":
    hero(
        "Account & Order Tracking",
        "Pionex crypto account tracking, Cathay Taiwan securities tracking, and Firstrade U.S. brokerage tracking. This app records state; it does not place live orders yet.",
    )
    account_tab, position_tab, order_tab = st.tabs(
        ["Account Snapshots", "Positions", "Order Tracker"]
    )
    with account_tab:
        st.subheader("Save account snapshot")
        col_a, col_b, col_c = st.columns(3)
        market = col_a.selectbox("Market", ["crypto", "tw", "us"], index=0)
        broker_default = {"crypto": "Pionex", "tw": "Cathay Securities", "us": "Firstrade"}[market]
        broker = col_b.text_input("Broker", value=broker_default)
        account_id = col_c.text_input("Account ID", value=f"{broker_default.lower().replace(' ', '-')}-main")
        col_d, col_e, col_f = st.columns(3)
        currency = col_d.text_input("Currency", value="USDT" if market == "crypto" else "TWD" if market == "tw" else "USD")
        cash = col_e.number_input("Cash", min_value=0.0, value=0.0, step=100.0)
        equity = col_f.number_input("Equity", min_value=0.0, value=0.0, step=100.0)
        notes = st.text_area("Notes", value="")
        if st.button("Save account snapshot", type="primary"):
            frame = upsert_account(
                ACCOUNTS_FILE,
                AccountSnapshot(
                    account_id=account_id,
                    broker=broker,
                    market=market,
                    currency=currency,
                    cash=cash,
                    equity=equity,
                    notes=notes,
                ),
            )
            st.success("Account snapshot saved.")
            st.dataframe(frame, hide_index=True, width="stretch")
        accounts = load_table(ACCOUNTS_FILE, ACCOUNT_COLUMNS)
        if not accounts.empty:
            st.subheader("Current account snapshots")
            st.dataframe(accounts, hide_index=True, width="stretch")
    with position_tab:
        st.subheader("Track manual positions")
        col_a, col_b, col_c = st.columns(3)
        market = col_a.selectbox("Position market", ["crypto", "tw", "us"], index=0)
        broker = col_b.text_input(
            "Position broker",
            value={"crypto": "Pionex", "tw": "Cathay Securities", "us": "Firstrade"}[market],
        )
        account_id = col_c.text_input("Position account ID", value=f"{broker.lower().replace(' ', '-')}-main")
        col_d, col_e = st.columns(2)
        symbol = col_d.text_input("Symbol", value="BTCUSDT" if market == "crypto" else "2330.TW" if market == "tw" else "AAPL")
        company = col_e.text_input("Company", value=company_name(symbol))
        col_f, col_g, col_h = st.columns(3)
        quantity = col_f.number_input("Quantity", value=0.0, step=1.0)
        average_price = col_g.number_input("Average price", min_value=0.0, value=0.0, step=1.0)
        current_price = col_h.number_input("Current price", min_value=0.0, value=0.0, step=1.0)
        notes = st.text_area("Position notes", value="")
        if st.button("Save position", type="primary"):
            frame = upsert_position(
                POSITIONS_FILE,
                PositionSnapshot(
                    account_id=account_id,
                    broker=broker,
                    market=market,
                    symbol=symbol.upper(),
                    company=company,
                    quantity=quantity,
                    average_price=average_price,
                    current_price=current_price,
                    notes=notes,
                ),
            )
            st.success("Position saved.")
            st.dataframe(frame, hide_index=True, width="stretch")
        positions = load_table(POSITIONS_FILE, POSITION_COLUMNS)
        if not positions.empty:
            st.subheader("Tracked positions")
            st.dataframe(positions, hide_index=True, width="stretch")
    with order_tab:
        st.subheader("Crypto order tracker")
        st.info("Pionex live execution is intentionally disabled until API keys, canary limits, and kill-switch rules are configured outside Git.")
        col_a, col_b, col_c = st.columns(3)
        broker = col_a.text_input("Order broker", value="Pionex")
        account_id = col_b.text_input("Order account ID", value="pionex-live-main")
        market = col_c.selectbox("Order market", ["crypto", "tw", "us"], index=0)
        col_d, col_e = st.columns(2)
        symbol = col_d.text_input("Order symbol", value="BTCUSDT" if market == "crypto" else "AAPL")
        company = col_e.text_input("Order company", value=company_name(symbol))
        col_f, col_g, col_h = st.columns(3)
        side = col_f.selectbox("Side", ["BUY", "SELL"], index=0)
        status = col_g.selectbox("Status", ["PLANNED", "SUBMITTED", "FILLED", "PARTIAL", "CANCELLED"], index=0)
        quantity = col_h.number_input("Order quantity", min_value=0.0, value=0.0, step=0.001, format="%.6f")
        col_i, col_j, col_k, col_l = st.columns(4)
        entry_price = col_i.number_input("Entry price", min_value=0.0, value=0.0, step=1.0)
        stop_loss = col_j.number_input("Stop loss", min_value=0.0, value=0.0, step=1.0)
        take_profit_1 = col_k.number_input("TP1", min_value=0.0, value=0.0, step=1.0)
        take_profit_2 = col_l.number_input("TP2", min_value=0.0, value=0.0, step=1.0)
        strategy = st.text_input("Strategy", value="market_alpha_staggered")
        notes = st.text_area("Order notes", value="")
        if st.button("Append order tracker row", type="primary"):
            frame = append_order(
                ORDERS_FILE,
                OrderTracker(
                    account_id=account_id,
                    broker=broker,
                    market=market,
                    symbol=symbol.upper(),
                    company=company,
                    side=side,
                    status=status,
                    quantity=quantity,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    strategy=strategy,
                    notes=notes,
                ),
            )
            st.success("Order tracker row saved.")
            st.dataframe(frame.tail(50), hide_index=True, width="stretch")
        orders = load_table(ORDERS_FILE, ORDER_COLUMNS)
        if not orders.empty:
            st.subheader("Tracked orders")
            st.dataframe(orders.tail(100), hide_index=True, width="stretch")

elif page == "Research":
    hero(
        "Strategy Research",
        "Optimization status, current limits, and next research targets for crypto, Taiwan stocks, and U.S. stocks.",
    )
    st.subheader("Current optimization stance")
    st.markdown(
        """
        - Crypto: still in paper-forward mode. It needs at least 30 forward days, more rebalance events, real slippage checks, and Pionex canary limits before funded execution.
        - Taiwan stocks: current two-year test beats 0050.TW, but the universe is too narrow. Next optimization is liquidity filters, sector caps, and broader stock coverage.
        - U.S. stocks: current two-year test beats SPY, but it is concentrated in mega-cap momentum. Next optimization is sector caps, earnings blackout rules, and QQQ/SPY dual benchmark validation.
        - All markets: walk-forward and out-of-sample testing are required before increasing capital.
        """
    )
    st.subheader("Latest generated research files")
    for path in [
        MARKET_ALPHA_DIR / "selected_metrics.csv",
        EQUITY_SELECTION_DIR / "tw_metrics.csv",
        EQUITY_SELECTION_DIR / "us_metrics.csv",
        TRACKING_DIR / "market_alpha_staggered_forward_benchmarks.csv",
    ]:
        st.caption(str(path))
        st.dataframe(read_csv_or_empty(path), hide_index=True, width="stretch")

elif page == "Records":
    hero("Records", "Generated research, paper tracking, account, and order files.")
    files = [
        MARKET_ALPHA_DIR / "metadata.json",
        MARKET_ALPHA_DIR / "selected_metrics.csv",
        MARKET_ALPHA_DIR / "selected_benchmark_comparison.csv",
        MARKET_ALPHA_DIR / "candidate_screen.csv",
        TRACKING_DIR / "market_alpha_staggered_status.json",
        TRACKING_DIR / "market_alpha_staggered_equity.csv",
        TRACKING_DIR / "market_alpha_staggered_events.csv",
        TRACKING_DIR / "market_alpha_staggered_forward_benchmarks.csv",
        EQUITY_SELECTION_DIR / "tw_ranking.csv",
        EQUITY_SELECTION_DIR / "tw_metrics.csv",
        EQUITY_SELECTION_DIR / "tw_strategy_metadata.json",
        EQUITY_SELECTION_DIR / "tw_rebalances.csv",
        EQUITY_SELECTION_DIR / "tw_equity.csv",
        EQUITY_SELECTION_DIR / "us_ranking.csv",
        EQUITY_SELECTION_DIR / "us_metrics.csv",
        EQUITY_SELECTION_DIR / "us_strategy_metadata.json",
        EQUITY_SELECTION_DIR / "us_rebalances.csv",
        EQUITY_SELECTION_DIR / "us_equity.csv",
        ACCOUNTS_FILE,
        POSITIONS_FILE,
        ORDERS_FILE,
    ]
    for path in files:
        st.caption(str(path))
        if path.suffix == ".json":
            st.json(read_json_or_empty(path), expanded=False)
        else:
            st.dataframe(read_csv_or_empty(path), hide_index=True, width="stretch")

elif page == "Deployment":
    hero(
        "Deployment",
        "Local mobile access is active; cloud deployment requires a GitHub remote repository connected to Streamlit Community Cloud.",
    )
    ip = local_ip()
    st.subheader("Local URLs")
    st.code(f"http://localhost:8501\nhttp://{ip}:8501")
    st.subheader("GitHub / Streamlit Cloud")
    st.markdown(
        """
        Required cloud settings:

        - Repository: this project pushed to GitHub.
        - Branch: `master`.
        - Main file: `app.py`.
        - Runtime: `python-3.12`.
        - Dependencies: `requirements.txt`.
        """
    )
    st.link_button("Open GitHub new repository", "https://github.com/new")
    st.link_button("Open Streamlit Community Cloud", "https://share.streamlit.io")
