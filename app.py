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
    .nav-shell {
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.62);
    }
    .section-card, .metric-card {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.45);
    }
    .metric-card {min-height: 116px;}
    .metric-label {color: #94a3b8; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em;}
    .metric-value {color: #f8fafc; font-size: 1.65rem; font-weight: 650; margin-top: 0.24rem;}
    .metric-note {color: #cbd5e1; font-size: 0.86rem; margin-top: 0.35rem;}
    .action-banner {
        border-radius: 18px;
        border: 1px solid rgba(56, 189, 248, 0.26);
        background: rgba(14, 165, 233, 0.10);
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .action-banner strong {font-size: 1.1rem;}
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


def metric_card(label: str, value: object, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def plot_metric_comparison(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if not frame.empty and {"strategy", "return_pct", "max_drawdown_pct"}.issubset(frame.columns):
        labels = frame["strategy"].astype(str)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=frame["return_pct"].astype(float),
                name="Return %",
                marker_color="#22c55e",
            )
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=frame["max_drawdown_pct"].astype(float),
                name="Max DD %",
                marker_color="#ef4444",
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        barmode="group",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def plot_allocation(frame: pd.DataFrame, title: str = "Target Allocation") -> go.Figure:
    labels = frame["asset"].astype(str).tolist() if not frame.empty else ["CASH"]
    values = frame["target_weight"].astype(float).tolist() if not frame.empty else [1.0]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            textinfo="label+percent",
            marker=dict(colors=["#38bdf8", "#22c55e", "#f59e0b", "#ef4444", "#64748b"]),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=330,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
    )
    return fig


def plot_readiness(status: dict[str, object]) -> go.Figure:
    ready = bool(status.get("live_ready", False))
    blockers = len(status.get("blockers", []) or [])
    score = 100 if ready else max(0, 70 - blockers * 18)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Live Readiness"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e" if ready else "#f59e0b"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.22)"},
                    {"range": [40, 75], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [75, 100], "color": "rgba(34, 197, 94, 0.22)"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def action_score(value: object) -> int:
    return {
        "Strong Sell": -2,
        "Sell": -1,
        "Neutral": 0,
        "Buy": 1,
        "Strong Buy": 2,
    }.get(str(value), 0)


def plot_signal_scores(frame: pd.DataFrame, title: str) -> go.Figure:
    data = frame.copy()
    if data.empty:
        data = pd.DataFrame({"symbol": [], "summary": []})
    data["score"] = data.get("summary", pd.Series(dtype=object)).map(action_score)
    colors = data["score"].map(lambda value: "#22c55e" if value > 0 else "#ef4444" if value < 0 else "#94a3b8")
    fig = go.Figure(
        go.Bar(
            x=data["symbol"].astype(str),
            y=data["score"],
            marker_color=colors,
            text=data.get("summary", pd.Series(dtype=object)),
            textposition="outside",
        )
    )
    fig.update_yaxes(range=[-2.4, 2.4], tickvals=[-2, -1, 0, 1, 2])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def plot_ranking(frame: pd.DataFrame, title: str) -> go.Figure:
    data = frame.copy().head(12)
    labels = [
        f"{row.symbol} | {row.company}" if "company" in frame.columns else str(row.symbol)
        for row in data.itertuples()
    ]
    colors = data["eligible"].map(lambda value: "#22c55e" if bool(value) else "#64748b")
    fig = go.Figure(
        go.Bar(
            x=data["score"].astype(float),
            y=labels,
            orientation="h",
            marker_color=colors,
            text=data["score"].astype(float).round(1),
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=390,
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_account_equity(accounts: pd.DataFrame) -> go.Figure:
    data = accounts.copy()
    if data.empty:
        data = pd.DataFrame({"broker": [], "equity": []})
    fig = go.Figure(
        go.Bar(
            x=data["broker"].astype(str),
            y=pd.to_numeric(data["equity"], errors="coerce").fillna(0.0),
            marker_color="#38bdf8",
            text=data["currency"].astype(str) if "currency" in data else None,
        )
    )
    fig.update_layout(
        title="Account Equity by Broker",
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def plot_positions(positions: pd.DataFrame) -> go.Figure:
    data = positions.copy() if not positions.empty else pd.DataFrame()
    if not data.empty:
        data["market_value"] = pd.to_numeric(data["market_value"], errors="coerce").fillna(0.0)
        data = data.sort_values("market_value", ascending=True)
    labels = data["symbol"].astype(str) if not data.empty else []
    values = pd.to_numeric(data["market_value"], errors="coerce").fillna(0.0) if not data.empty else []
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#22c55e"))
    fig.update_layout(
        title="Position Market Value",
        template="plotly_dark",
        height=330,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


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
            "Watch Entry": "#60a5fa",
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


def crypto_strategy_plan(symbol: str, frame: pd.DataFrame, target_weight: float) -> dict[str, object]:
    close = float(frame["close"].iloc[-1])
    atr_values = atr(frame, 14).dropna()
    current_atr = float(atr_values.iloc[-1]) if not atr_values.empty else max(close * 0.03, 0.01)
    trend_values = ema(frame["close"].astype(float), 50).dropna()
    trend = float(trend_values.iloc[-1]) if not trend_values.empty else close
    breakout_values = frame["high"].astype(float).shift(1).rolling(20, min_periods=5).max().dropna()
    breakout = float(breakout_values.iloc[-1]) if not breakout_values.empty else close
    entry = max(breakout + 0.10 * current_atr, trend + 0.10 * current_atr)
    levels: dict[str, float | None] = {
        "Watch Entry" if target_weight <= 0 else "Entry": entry,
        "Strategy Exit": max(entry - 2.5 * current_atr, trend),
    }
    action = "HOLD_CASH"
    reason = "No active allocation. Wait for the strategy to rotate back in before placing a live order."
    if target_weight > 0:
        action = "WAIT_FOR_BREAKOUT"
        reason = "Active target weight exists, but entry still waits for the strategy breakout trigger."
        levels.update(
            {
                "Stop Loss": max(entry - 2.0 * current_atr, 0.0),
                "TP1": entry + 2.0 * current_atr,
                "TP2": entry + 4.0 * current_atr,
            }
        )
    return {
        "symbol": symbol,
        "action": action,
        "reason": reason,
        "close": close,
        "target_weight": target_weight,
        "entry_trigger": entry,
        "atr": current_atr,
        "trend": trend,
        "levels": levels,
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


NAV_OPTIONS = [
    "Dashboard",
    "Crypto",
    "Stocks",
    "Accounts",
    "Research",
    "Records",
    "Deployment",
]

st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
page = st.pills("Workspace", NAV_OPTIONS, default="Dashboard", label_visibility="collapsed") or "Dashboard"
st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Control Panel")
    st.caption("Data controls only. Main navigation is on the top of the page.")
    st.divider()
    st.caption("Crypto data")
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
    with cols[0]:
        metric_card("Crypto live-ready", str(status.get("live_ready", "Unknown")), "Forward gate status")
    with cols[1]:
        metric_card("Paper return", pct(status.get("return_pct")), "Crypto strategy")
    with cols[2]:
        metric_card("Forward days", money(status.get("forward_days")), "Minimum target: 30")
    with cols[3]:
        blockers = len(status.get("blockers", []) or []) if status else "-"
        metric_card("Open blockers", blockers, "Must be zero before live")

    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(plot_readiness(status), width="stretch")
    with right:
        comparison_rows = []
        for market_name, frame in (("Taiwan", tw_metrics), ("U.S.", us_metrics)):
            if not frame.empty and {"strategy", "return_pct"}.issubset(frame.columns):
                for row in frame.to_dict("records"):
                    comparison_rows.append(
                        {
                            "market": market_name,
                            "strategy": row["strategy"],
                            "return_pct": safe_float(row.get("return_pct")),
                        }
                    )
        comparison = pd.DataFrame(comparison_rows)
        fig = go.Figure()
        if not comparison.empty:
            for strategy in comparison["strategy"].unique():
                subset = comparison[comparison["strategy"] == strategy]
                fig.add_trace(
                    go.Bar(
                        x=subset["market"],
                        y=subset["return_pct"],
                        name=str(strategy),
                    )
                )
        fig.update_layout(
            title="Stock Strategy vs Benchmark Return",
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=45, b=10),
            barmode="group",
        )
        st.plotly_chart(fig, width="stretch")

    stock_left, stock_right = st.columns(2)
    with stock_left:
        st.plotly_chart(plot_metric_comparison(tw_metrics, "Taiwan Metrics"), width="stretch")
    with stock_right:
        st.plotly_chart(plot_metric_comparison(us_metrics, "U.S. Metrics"), width="stretch")
    with st.expander("Raw dashboard data"):
        st.caption("Taiwan")
        st.dataframe(tw_metrics, hide_index=True, width="stretch")
        st.caption("U.S.")
        st.dataframe(us_metrics, hide_index=True, width="stretch")
        st.caption("Crypto metrics")
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
            summary = summarize_universe(universe)
            st.plotly_chart(plot_signal_scores(summary, "Crypto Technical Bias"), width="stretch")
            try:
                config, offsets, metadata = load_allocation_strategy()
                snapshot = allocation_snapshot(universe, config, offsets)
                allocation = aggregate_snapshot(snapshot)
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.plotly_chart(plot_allocation(allocation), width="stretch")
                with col_b:
                    st.markdown(
                        """
                        <div class="action-banner">
                            <strong>Signal rule</strong><br/>
                            Active allocation means the strategy has selected a sleeve, but order entry still waits for the breakout trigger. If allocation is cash, the correct action is no trade.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.json(metadata, expanded=False)
                with st.expander("Technical and allocation details"):
                    display_technical_table(summary)
                    st.dataframe(allocation, hide_index=True, width="stretch")
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
            curve = equity.set_index("timestamp")["equity"]
            fig = go.Figure(go.Scatter(x=curve.index, y=curve, mode="lines", line=dict(color="#38bdf8")))
            fig.update_layout(
                title="Forward Paper Equity",
                template="plotly_dark",
                height=330,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        events = read_csv_or_empty(TRACKING_DIR / "market_alpha_staggered_events.csv")
        if not events.empty:
            with st.expander("Forward rebalance events"):
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
            plan = crypto_strategy_plan(symbol, universe[symbol], target_weight)
            levels = plan["levels"]
            cols = st.columns(5)
            cols[0].metric("Action", str(plan["action"]))
            cols[1].metric("Target weight", pct(target_weight * 100.0))
            cols[2].metric("Entry trigger", price(plan["entry_trigger"]))
            cols[3].metric("ATR", price(plan["atr"]))
            cols[4].metric("Close", price(plan["close"]))
            st.caption(str(plan["reason"]))
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
        summary_metrics = pd.DataFrame(
            [
                {"strategy": "equity_selection", **result.metrics},
                {"strategy": config.market_symbol, **benchmark.metrics},
            ]
        )
        cols = st.columns(4)
        with cols[0]:
            metric_card("Strategy return", pct(result.metrics["return_pct"]), "Backtest")
        with cols[1]:
            metric_card("Benchmark return", pct(benchmark.metrics["return_pct"]), config.market_symbol)
        with cols[2]:
            metric_card("Strategy max DD", pct(result.metrics["max_drawdown_pct"]), "Lower is better")
        with cols[3]:
            metric_card("Rebalances", int(float(result.metrics["rebalances"])), "Strategy events")

        chart_a, chart_b = st.columns([1, 1])
        with chart_a:
            st.plotly_chart(plot_metric_comparison(summary_metrics, f"{title}: Strategy vs Benchmark"), width="stretch")
        with chart_b:
            st.plotly_chart(plot_signal_scores(technical, f"{title}: Technical Bias"), width="stretch")

        st.subheader("Strategy Ranking")
        st.plotly_chart(plot_ranking(ranking, f"{title}: Momentum / Trend Score"), width="stretch")
        choices = ranking["symbol"].tolist()
        selected_symbol = (
            st.pills(
                "Select stock",
                choices,
                default=choices[0] if choices else None,
                format_func=lambda symbol: f"{symbol} | {company_name(symbol)}",
                key=f"{market}_symbol_pills",
            )
            if choices
            else None
        )
        with st.expander("Raw ranking and technical details"):
            display_technical_table(technical, show_company=True)
            st.dataframe(ranking, hide_index=True, width="stretch")
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
            "Entry" if plan.entry_price is not None else "Watch Entry": plan.entry_price,
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
            normalized = curves / curves.iloc[0] * 100.0
            fig = go.Figure()
            for column in normalized.columns:
                fig.add_trace(go.Scatter(x=normalized.index, y=normalized[column], mode="lines", name=str(column)))
            fig.update_layout(
                template="plotly_dark",
                height=330,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Indexed equity",
            )
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(plot_account_equity(accounts), width="stretch")
            total_equity = pd.to_numeric(accounts["equity"], errors="coerce").fillna(0.0).sum()
            total_cash = pd.to_numeric(accounts["cash"], errors="coerce").fillna(0.0).sum()
            cols = st.columns(2)
            with cols[0]:
                metric_card("Tracked equity", money(total_equity), "Manual snapshots")
            with cols[1]:
                metric_card("Tracked cash", money(total_cash), "Manual snapshots")
            with st.expander("Account table"):
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
            st.plotly_chart(plot_positions(positions), width="stretch")
            with st.expander("Position table"):
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
            status_counts = orders["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig = go.Figure(
                go.Pie(
                    labels=status_counts["status"],
                    values=status_counts["count"],
                    hole=0.55,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                title="Order Status Mix",
                template="plotly_dark",
                height=320,
                margin=dict(l=10, r=10, t=45, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
            with st.expander("Order table"):
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
    research_files = [
        MARKET_ALPHA_DIR / "selected_metrics.csv",
        EQUITY_SELECTION_DIR / "tw_metrics.csv",
        EQUITY_SELECTION_DIR / "us_metrics.csv",
        TRACKING_DIR / "market_alpha_staggered_forward_benchmarks.csv",
    ]
    tw_metrics = read_csv_or_empty(EQUITY_SELECTION_DIR / "tw_metrics.csv")
    us_metrics = read_csv_or_empty(EQUITY_SELECTION_DIR / "us_metrics.csv")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_metric_comparison(tw_metrics, "Taiwan Optimization Snapshot"), width="stretch")
    with col_b:
        st.plotly_chart(plot_metric_comparison(us_metrics, "U.S. Optimization Snapshot"), width="stretch")
    forward = read_csv_or_empty(TRACKING_DIR / "market_alpha_staggered_forward_benchmarks.csv")
    if not forward.empty and {"benchmark", "return_pct"}.issubset(forward.columns):
        fig = go.Figure(
            go.Bar(
                x=forward["benchmark"].astype(str),
                y=forward["return_pct"].astype(float),
                marker_color=["#38bdf8", "#22c55e", "#f59e0b"][: len(forward)],
            )
        )
        fig.update_layout(
            title="Crypto Forward Benchmark Return",
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, width="stretch")
    with st.expander("Latest generated research files"):
        for path in research_files:
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
