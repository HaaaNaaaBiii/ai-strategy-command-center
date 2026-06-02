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
from smi_lab.equity_universe import equity_scan_symbols
from smi_lab.indicators import atr, ema
from smi_lab.market_info import (
    cached_crypto_snapshots,
    cached_equity_snapshots,
    fetch_market_news,
)
from smi_lab.broker_import import DEFAULT_IMPORT_DIR, sync_broker_exports
from smi_lab.notifier import (
    resolve_discord_mention,
    resolve_discord_webhook_url,
    send_discord,
    send_telegram,
)
from smi_lab.paper import (
    aggregate_snapshot,
    allocation_snapshot,
    format_allocation_report,
    load_allocation_strategy,
    update_forward_tracking,
)
from smi_lab.position_planner import build_rebalance_plan
from smi_lab.price_alerts import check_equity_price_alerts
from smi_lab.technical import summarize_universe


OUTPUT_DIR = Path("outputs")
MARKET_ALPHA_DIR = OUTPUT_DIR / "market_alpha_staggered"
TRACKING_DIR = OUTPUT_DIR / "forward_tracking"
EQUITY_SELECTION_DIR = OUTPUT_DIR / "equity_selection"
EQUITY_SCAN_DIR = OUTPUT_DIR / "equity_scan"
ACCOUNT_DIR = OUTPUT_DIR / "accounts"
ACCOUNTS_FILE = ACCOUNT_DIR / "accounts.csv"
POSITIONS_FILE = ACCOUNT_DIR / "positions.csv"
ORDERS_FILE = ACCOUNT_DIR / "orders.csv"
NEWS_FILE = OUTPUT_DIR / "news" / "market_news.json"
NEWS_FILES = {
    "crypto": OUTPUT_DIR / "news" / "crypto_news.json",
    "tw": OUTPUT_DIR / "news" / "tw_news.json",
    "us": OUTPUT_DIR / "news" / "us_news.json",
}
ALERT_STATE_FILE = OUTPUT_DIR / "alerts" / "equity_price_alerts_state.json"
BROKER_IMPORT_DIR = DEFAULT_IMPORT_DIR


I18N = {
    "zh": {
        "workspace": "工作區",
        "control_panel": "控制面板",
        "data_controls": "資料控制",
        "language": "語言",
        "dashboard": "儀錶板",
        "crypto": "加密貨幣",
        "stocks": "股票",
        "accounts": "帳戶",
        "research": "研究",
        "records": "紀錄",
        "deployment": "部署",
        "dashboard_title": "策略儀錶板",
        "dashboard_subtitle": "目前掃盤推薦、資料健康度、帳戶狀態與市場新聞。回測放在研究分頁。",
        "crypto_mode": "加密策略模式",
        "equity_scan": "股票掃盤",
        "tracked_equity": "追蹤資產",
        "tracked_positions": "追蹤持倉",
        "tw_picks": "台股掃盤推薦",
        "us_picks": "美股掃盤推薦",
        "market_news": "市場新聞",
        "refresh_news": "更新市場新聞",
        "news_crypto": "加密",
        "news_tw": "台股",
        "news_us": "美股",
        "stocks_title": "台股 / 美股策略",
        "stocks_subtitle": "依市場調整的策略掃盤、K 線圖、進出場、停損、TP 與 RR。",
        "latest_scan": "最新策略掃盤",
        "price_alerts": "價格提醒",
        "check_alerts": "檢查價格提醒",
    },
    "en": {
        "workspace": "Workspace",
        "control_panel": "Control Panel",
        "data_controls": "Data controls",
        "language": "Language",
        "dashboard": "Dashboard",
        "crypto": "Crypto",
        "stocks": "Stocks",
        "accounts": "Accounts",
        "research": "Research",
        "records": "Records",
        "deployment": "Deployment",
        "dashboard_title": "Strategy Dashboard",
        "dashboard_subtitle": "Current scan recommendations, data health, account status, and market news. Backtests are kept in Research.",
        "crypto_mode": "Crypto mode",
        "equity_scan": "Equity scan",
        "tracked_equity": "Tracked equity",
        "tracked_positions": "Tracked positions",
        "tw_picks": "Taiwan Scan Picks",
        "us_picks": "U.S. Scan Picks",
        "market_news": "Market News",
        "refresh_news": "Refresh market news",
        "news_crypto": "Crypto",
        "news_tw": "Taiwan Stocks",
        "news_us": "U.S. Stocks",
        "stocks_title": "Taiwan / U.S. Stock Strategy",
        "stocks_subtitle": "Market-adjusted scans, charts, entries, stops, take-profits, and RR.",
        "latest_scan": "Latest Strategy Scan",
        "price_alerts": "Price Alerts",
        "check_alerts": "Check price alerts",
    },
}


st.set_page_config(
    page_title="AI Strategy Command Center",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


def tr(key: str) -> str:
    lang = st.session_state.get("lang", "zh")
    return I18N.get(lang, I18N["zh"]).get(key, I18N["en"].get(key, key))


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
    with st.container(border=True):
        st.title(title)
        st.caption(subtitle)


def metric_card(label: str, value: object, note: str = "") -> None:
    with st.container(border=True):
        st.caption(label.upper())
        st.metric(label=" ", value=value)
        if note:
            st.caption(note)


def recommendation_cards(frame: pd.DataFrame, title: str, limit: int = 3) -> None:
    st.subheader(title)
    if frame.empty:
        st.info("No scan recommendations yet.")
        return
    for row in frame.head(limit).to_dict("records"):
        with st.container(border=True):
            st.markdown(f"**{row.get('symbol', '-')}**")
            st.caption(str(row.get("company", "")))
            st.caption(str(row.get("action", "-")))
            level_cols = st.columns(2)
            level_cols[0].metric("Entry", price(row.get("entry_price")))
            level_cols[1].metric("SL", price(row.get("stop_loss")))
            level_cols[0].metric("TP1", price(row.get("take_profit_1")))
            level_cols[1].metric("TP2", price(row.get("take_profit_2")))
            if "risk_reward_2" in row and not pd.isna(row.get("risk_reward_2")):
                st.metric("RR to TP2", f"{float(row.get('risk_reward_2')):.2f}")


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


def readiness_label(status: dict[str, object]) -> tuple[str, str]:
    ready = bool(status.get("live_ready", False))
    blockers = status.get("blockers", []) or []
    if ready:
        return "Ready", "Forward gate passed"
    if blockers:
        return "Paper Mode", f"{len(blockers)} blocker(s)"
    return "Paper Mode", "Forward tracking required"


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


def plot_rebalance_plan(plan: pd.DataFrame) -> go.Figure:
    data = plan.copy() if not plan.empty else pd.DataFrame()
    if data.empty:
        data = pd.DataFrame({"symbol": [], "current_value": [], "target_value": []})
    data = data[data["symbol"].astype(str) != "CASH"]
    data["current_value"] = pd.to_numeric(data.get("current_value", 0.0), errors="coerce").fillna(0.0)
    data["target_value"] = pd.to_numeric(data.get("target_value", 0.0), errors="coerce").fillna(0.0)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data["symbol"].astype(str),
            y=data["current_value"],
            name="Current",
            marker_color="#64748b",
        )
    )
    fig.add_trace(
        go.Bar(
            x=data["symbol"].astype(str),
            y=data["target_value"],
            name="Target",
            marker_color="#38bdf8",
        )
    )
    fig.update_layout(
        title="Current vs Strategy Target Value",
        barmode="group",
        template="plotly_dark",
        height=340,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def plot_market_snapshot(frame: pd.DataFrame, title: str) -> go.Figure:
    data = frame.copy()
    if data.empty:
        data = pd.DataFrame({"symbol": [], "change_pct": []})
    data["change_pct"] = pd.to_numeric(data.get("change_pct", 0.0), errors="coerce").fillna(0.0)
    colors = data["change_pct"].map(lambda value: "#22c55e" if value >= 0 else "#ef4444")
    fig = go.Figure(
        go.Bar(
            x=data["symbol"].astype(str),
            y=data["change_pct"],
            marker_color=colors,
            text=data["change_pct"].round(2).astype(str) + "%",
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=270,
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis_title="Change %",
    )
    return fig


def plot_compact_market_snapshot(frame: pd.DataFrame, title: str, limit: int = 6) -> go.Figure:
    data = frame.copy().head(limit)
    if data.empty:
        data = pd.DataFrame({"symbol": [], "change_pct": []})
    data["change_pct"] = pd.to_numeric(data.get("change_pct", 0.0), errors="coerce").fillna(0.0)
    colors = data["change_pct"].map(lambda value: "#22c55e" if value >= 0 else "#ef4444")
    fig = go.Figure(
        go.Bar(
            x=data["change_pct"],
            y=data["symbol"].astype(str),
            orientation="h",
            marker_color=colors,
            text=data["change_pct"].round(2).astype(str) + "%",
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=250,
        margin=dict(l=8, r=8, t=42, b=8),
        xaxis_title="Change %",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_news_cards(items: list[object]) -> None:
    if not items:
        st.info("No market news is cached yet. Use Refresh news when network access is available.")
        return
    for item in items:
        title = str(getattr(item, "title", ""))
        source = str(getattr(item, "source", ""))
        published_at = str(getattr(item, "published_at", ""))
        link = str(getattr(item, "link", ""))
        with st.container(border=True):
            st.caption(source)
            st.markdown(f"**{title}**")
            st.caption(published_at)
            if link:
                st.link_button("Open source", link)


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

with st.sidebar:
    language_choice = st.selectbox("Language / 語言", ["繁體中文", "English"], index=0)
    st.session_state["lang"] = "zh" if language_choice == "繁體中文" else "en"

NAV_LABELS = {
    "Dashboard": tr("dashboard"),
    "Crypto": tr("crypto"),
    "Stocks": tr("stocks"),
    "Accounts": tr("accounts"),
    "Research": tr("research"),
    "Records": tr("records"),
    "Deployment": tr("deployment"),
}
label_to_page = {label: page for page, label in NAV_LABELS.items()}
selected_label = st.pills(
    tr("workspace"),
    list(NAV_LABELS.values()),
    default=NAV_LABELS["Dashboard"],
    label_visibility="collapsed",
)
page = label_to_page.get(selected_label or NAV_LABELS["Dashboard"], "Dashboard")

with st.sidebar:
    st.title(tr("control_panel"))
    st.caption(tr("data_controls"))
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
        tr("dashboard_title"),
        tr("dashboard_subtitle"),
    )
    status = latest_status()
    crypto_snapshot = cached_crypto_snapshots(DEFAULT_SYMBOLS, crypto_interval)
    tw_scan = read_csv_or_empty(EQUITY_SCAN_DIR / "tw_recommendations.csv")
    us_scan = read_csv_or_empty(EQUITY_SCAN_DIR / "us_recommendations.csv")
    latest_scan_summary = read_json_or_empty(EQUITY_SCAN_DIR / "latest_scan_summary.json")
    tw_snapshot = (
        tw_scan[["symbol", "company", "close"]].assign(change_pct=tw_scan.get("score", 0.0))
        if not tw_scan.empty and {"symbol", "company", "close"}.issubset(tw_scan.columns)
        else cached_equity_snapshots(DEFAULT_TW_SYMBOLS, "tw")
    )
    us_snapshot = (
        us_scan[["symbol", "company", "close"]].assign(change_pct=us_scan.get("score", 0.0))
        if not us_scan.empty and {"symbol", "company", "close"}.issubset(us_scan.columns)
        else cached_equity_snapshots(DEFAULT_US_SYMBOLS, "us")
    )
    accounts = load_table(ACCOUNTS_FILE, ACCOUNT_COLUMNS)
    positions = load_table(POSITIONS_FILE, POSITION_COLUMNS)
    readiness, readiness_note = readiness_label(status)
    scan_items = latest_scan_summary if isinstance(latest_scan_summary, list) else []
    loaded_symbols = sum(int(item.get("loaded_symbols", 0)) for item in scan_items if isinstance(item, dict))
    failed_symbols = sum(int(item.get("failed_symbols", 0)) for item in scan_items if isinstance(item, dict))
    cols = st.columns(4)
    with cols[0]:
        metric_card(tr("crypto_mode"), readiness, readiness_note)
    with cols[1]:
        metric_card(tr("equity_scan"), f"{loaded_symbols} loaded", f"{failed_symbols} failed")
    with cols[2]:
        tracked_equity = pd.to_numeric(accounts.get("equity", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        metric_card(tr("tracked_equity"), money(tracked_equity), "Manual broker snapshots")
    with cols[3]:
        tracked_positions = len(positions) if not positions.empty else 0
        metric_card(tr("tracked_positions"), tracked_positions, "FT / Cathay / Pionex")

    rec_left, rec_right = st.columns(2)
    with rec_left:
        recommendation_cards(tw_scan, tr("tw_picks"))
    with rec_right:
        recommendation_cards(us_scan, tr("us_picks"))

    market_a, market_b, market_c = st.columns(3)
    with market_a:
        st.plotly_chart(plot_compact_market_snapshot(crypto_snapshot, "Crypto 24h"), width="stretch")
    with market_b:
        st.plotly_chart(plot_compact_market_snapshot(tw_snapshot, "Taiwan Scan Scores"), width="stretch")
    with market_c:
        st.plotly_chart(plot_compact_market_snapshot(us_snapshot, "U.S. Scan Scores"), width="stretch")

    if not accounts.empty or not positions.empty:
        account_col, position_col = st.columns(2)
        with account_col:
            st.plotly_chart(plot_account_equity(accounts), width="stretch")
        with position_col:
            st.plotly_chart(plot_positions(positions), width="stretch")

    refresh_news = st.button(tr("refresh_news"))
    st.subheader(tr("market_news"))
    news_tabs = st.tabs([tr("news_crypto"), tr("news_tw"), tr("news_us")])
    for tab, category in zip(news_tabs, ("crypto", "tw", "us")):
        with tab:
            news = fetch_market_news(
                NEWS_FILES[category],
                refresh=refresh_news,
                max_items=6,
                category=category,
            )
            news_cols = st.columns(3)
            for idx, item in enumerate(news):
                with news_cols[idx % 3]:
                    render_news_cards([item])

    with st.expander("Market snapshot data"):
        st.caption("Crypto")
        st.dataframe(crypto_snapshot, hide_index=True, width="stretch")
        st.caption("Taiwan")
        st.dataframe(tw_snapshot, hide_index=True, width="stretch")
        st.caption("U.S.")
        st.dataframe(us_snapshot, hide_index=True, width="stretch")

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
                    st.info(
                        "Signal rule: active allocation means the strategy has selected a sleeve, "
                        "but order entry still waits for the breakout trigger. If allocation is cash, "
                        "the correct action is no trade."
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
        tr("stocks_title"),
        tr("stocks_subtitle"),
    )

    def render_equity_page(title: str, market: str, defaults: tuple[str, ...]) -> None:
        config = default_equity_config(market)
        scan_recommendations = read_csv_or_empty(EQUITY_SCAN_DIR / f"{market}_recommendations.csv")
        scan_ranking = read_csv_or_empty(EQUITY_SCAN_DIR / f"{market}_scan_ranking.csv")
        scan_summary = read_json_or_empty(EQUITY_SCAN_DIR / f"{market}_scan_summary.json")
        st.subheader(tr("latest_scan"))
        if scan_summary:
            cols = st.columns(4)
            cols[0].metric("Scan status", str(scan_summary.get("status", "Unknown")))
            cols[1].metric("Loaded", str(scan_summary.get("loaded_symbols", "-")))
            cols[2].metric("Eligible", str(scan_summary.get("eligible_symbols", "-")))
            cols[3].metric("Updated UTC", str(scan_summary.get("scan_time_utc", "-"))[:19])
        if not scan_recommendations.empty:
            st.plotly_chart(
                plot_ranking(scan_recommendations, f"{title}: Current Recommended Entries"),
                width="stretch",
            )
            display_cols = [
                column
                for column in (
                    "symbol",
                    "company",
                    "action",
                    "rank",
                    "score",
                    "close",
                    "entry_price",
                    "stop_loss",
                    "take_profit_1",
                    "take_profit_2",
                    "risk_reward_1",
                    "risk_reward_2",
                    "reason",
                )
                if column in scan_recommendations
            ]
            st.dataframe(scan_recommendations[display_cols], hide_index=True, width="stretch")
        else:
            st.info("No hourly scan output yet. Run `scan_equity_signals.py` or wait for the hourly automation.")
        with st.expander("Latest full scan ranking", expanded=False):
            if scan_ranking.empty:
                st.caption("No scan ranking file exists yet.")
            else:
                st.dataframe(scan_ranking.head(80), hide_index=True, width="stretch")
        with st.expander(f"{title} data controls", expanded=False):
            symbols_text = st.text_area(
                "Universe",
                value="\n".join(defaults),
                height=180,
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
        cols = st.columns(8)
        cols[0].metric("Action", plan.action)
        cols[1].metric("Entry", price(plan.entry_price))
        cols[2].metric("Strategy exit", price(plan.strategy_exit))
        cols[3].metric("Stop loss", price(plan.stop_loss))
        cols[4].metric("TP1", price(plan.take_profit_1))
        cols[5].metric("TP2", price(plan.take_profit_2))
        cols[6].metric("RR1", "-" if plan.risk_reward_1 is None else f"{plan.risk_reward_1:.2f}")
        cols[7].metric("RR2", "-" if plan.risk_reward_2 is None else f"{plan.risk_reward_2:.2f}")
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

    tw_tab, us_tab, alerts_tab = st.tabs(["Taiwan Stocks", "U.S. Stocks", tr("price_alerts")])
    with tw_tab:
        render_equity_page("Taiwan Stocks", "tw", equity_scan_symbols("tw"))
    with us_tab:
        render_equity_page("U.S. Stocks", "us", equity_scan_symbols("us"))
    with alerts_tab:
        st.subheader(tr("price_alerts"))
        st.caption("Uses latest scan recommendations. Configure DISCORD_WEBHOOK_URL and DISCORD_MENTION for scheduled alerts.")
        webhook = st.text_input("Discord webhook URL", value=resolve_discord_webhook_url(), type="password")
        mention = st.text_input("Discord mention/tag", value=resolve_discord_mention(), placeholder="<@USER_ID>")
        dry_run = st.checkbox("Dry run", value=True)
        if st.button(tr("check_alerts"), type="primary"):
            try:
                events = check_equity_price_alerts(
                    webhook_url=webhook or None,
                    mention=mention,
                    notify=not dry_run,
                    record_state=not dry_run,
                )
                if not events:
                    st.info("No price alert triggered.")
                else:
                    st.success(f"{len(events)} alert(s) triggered.")
                    st.dataframe(pd.DataFrame([event.to_dict() for event in events]), hide_index=True, width="stretch")
            except Exception as exc:
                st.error(f"Price alert check failed: {exc}")
        state = read_json_or_empty(ALERT_STATE_FILE)
        if state:
            with st.expander("Alert state"):
                st.json(state, expanded=False)

elif page == "Accounts":
    hero(
        "Account & Order Tracking",
        "Pionex crypto account tracking, Cathay Taiwan securities tracking, and Firstrade U.S. brokerage tracking. This app records state; it does not place live orders yet.",
    )
    account_tab, position_tab, planner_tab, order_tab, integration_tab = st.tabs(
        ["Account Snapshots", "Positions", "Position Planner", "Order Tracker", "Broker Integration Plan"]
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
        st.subheader("Automated position sync")
        st.caption(
            f"Drop Firstrade or Cathay CSV exports into `{BROKER_IMPORT_DIR}`. "
            "The importer auto-detects broker columns and updates tracked positions."
        )
        sync_cols = st.columns([1, 2])
        with sync_cols[0]:
            if st.button("Sync broker exports", type="primary"):
                positions, report = sync_broker_exports(
                    import_dir=BROKER_IMPORT_DIR,
                    positions_path=POSITIONS_FILE,
                )
                st.session_state["broker_import_report"] = report
                st.success(f"Synced {len(positions)} tracked positions.")
        report = st.session_state.get("broker_import_report")
        if report is not None and not report.empty:
            st.dataframe(report, hide_index=True, width="stretch")
        st.divider()
        st.subheader("Manual position override")
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
    with planner_tab:
        st.subheader("Automated position planner")
        st.caption(
            "Generate target orders from the current crypto allocation strategy and tracked holdings. "
            "The app writes planned orders only; live execution remains disabled."
        )
        accounts = load_table(ACCOUNTS_FILE, ACCOUNT_COLUMNS)
        positions = load_table(POSITIONS_FILE, POSITION_COLUMNS)
        crypto_accounts = accounts[accounts["market"].astype(str) == "crypto"] if not accounts.empty else pd.DataFrame()
        account_options = crypto_accounts["account_id"].dropna().astype(str).tolist() if not crypto_accounts.empty else []
        selected_account = st.selectbox(
            "Planning account",
            account_options or ["pionex-live-main"],
            index=0,
        )
        planner_cols = st.columns(2)
        equity_override = planner_cols[0].number_input(
            "Portfolio equity override",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Use this when no account snapshot is saved yet.",
        )
        min_trade_value = planner_cols[1].number_input("Minimum trade value", min_value=0.0, value=25.0, step=5.0)
        if st.button("Generate crypto position plan", type="primary"):
            try:
                planning_accounts = accounts
                if equity_override > 0:
                    override_row = pd.DataFrame(
                        [
                            {
                                "account_id": selected_account,
                                "broker": "Pionex",
                                "market": "crypto",
                                "currency": "USDT",
                                "cash": 0.0,
                                "equity": equity_override,
                                "updated_at": "planner_override",
                                "notes": "Temporary planner equity override.",
                            }
                        ]
                    )
                    planning_accounts = pd.concat([accounts, override_row], ignore_index=True)
                universe = load_crypto_data(crypto_symbols, crypto_interval, crypto_bars, refresh_crypto)
                config, offsets, _ = load_allocation_strategy()
                snapshot = allocation_snapshot(universe, config, offsets)
                allocation = aggregate_snapshot(snapshot)
                price_lookup = {
                    symbol: float(frame["close"].iloc[-1])
                    for symbol, frame in universe.items()
                    if not frame.empty
                }
                plan = build_rebalance_plan(
                    planning_accounts,
                    positions,
                    allocation,
                    "crypto",
                    account_id=selected_account,
                    price_lookup=price_lookup,
                    min_trade_value=min_trade_value,
                )
                enriched = plan.copy()
                if not enriched.empty:
                    enriched["entry_trigger"] = None
                    enriched["stop_loss"] = None
                    enriched["take_profit_1"] = None
                    enriched["take_profit_2"] = None
                    for idx, row in enriched.iterrows():
                        symbol = str(row["symbol"])
                        if symbol in universe and str(row["side"]) == "BUY":
                            trade_plan = crypto_strategy_plan(
                                symbol,
                                universe[symbol],
                                float(row["target_weight"]),
                            )
                            levels = trade_plan["levels"]
                            enriched.at[idx, "entry_trigger"] = trade_plan["entry_trigger"]
                            enriched.at[idx, "stop_loss"] = levels.get("Stop Loss")
                            enriched.at[idx, "take_profit_1"] = levels.get("TP1")
                            enriched.at[idx, "take_profit_2"] = levels.get("TP2")
                st.session_state["position_plan"] = enriched
                st.success("Position plan generated.")
            except Exception as exc:  # pragma: no cover - Streamlit displays runtime data issues.
                st.error(f"Position plan failed: {exc}")
        plan = st.session_state.get("position_plan")
        if isinstance(plan, pd.DataFrame) and not plan.empty:
            st.plotly_chart(plot_rebalance_plan(plan), width="stretch")
            cols = st.columns(3)
            actionable = plan[plan["side"].isin(["BUY", "SELL"])]
            actionable_delta = pd.to_numeric(actionable["delta_value"], errors="coerce").fillna(0.0)
            cols[0].metric("Actionable orders", len(actionable))
            cols[1].metric("Buy value", money(actionable_delta[actionable["side"] == "BUY"].sum()))
            cols[2].metric("Sell value", money(abs(actionable_delta[actionable["side"] == "SELL"].sum())))
            st.dataframe(plan, hide_index=True, width="stretch")
            if st.button("Append planned orders to tracker"):
                appended = 0
                for row in actionable.to_dict("records"):
                    symbol = str(row["symbol"])
                    if symbol == "CASH":
                        continue
                    quantity_value = safe_float(row.get("order_quantity"))
                    if quantity_value <= 0:
                        continue
                    entry = safe_float(row.get("entry_trigger"), safe_float(row.get("reference_price")))
                    orders = append_order(
                        ORDERS_FILE,
                        OrderTracker(
                            account_id=str(row["account_id"]),
                            broker="Pionex",
                            market="crypto",
                            symbol=symbol,
                            company=company_name(symbol),
                            side=str(row["side"]),
                            status="PLANNED",
                            quantity=quantity_value,
                            entry_price=entry,
                            stop_loss=safe_float(row.get("stop_loss")),
                            take_profit_1=safe_float(row.get("take_profit_1")),
                            take_profit_2=safe_float(row.get("take_profit_2")),
                            strategy="market_alpha_staggered_position_plan",
                            notes=str(row.get("notes", "")),
                        ),
                    )
                    appended = len(orders)
                st.success(f"Order tracker updated. Total rows: {appended}.")
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
    with integration_tab:
        st.subheader("How to track FT and Cathay holdings")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            metric_card("Firstrade", "Auto CSV sync", "Drop exports into data/broker_imports.")
            st.markdown(
                """
                - Export positions / account value from Firstrade as CSV.
                - Place the file under `data/broker_imports/firstrade/`.
                - Click `Sync broker exports`; the app maps common Firstrade columns automatically.
                - Next phase: add browser-assisted download after you confirm the exported file format.
                """
            )
        with col_b:
            metric_card("Cathay", "Auto CSV sync", "Taiwan symbols are normalized to `.TW`.")
            st.markdown(
                """
                - Export holdings from Cathay Securities as CSV or a spreadsheet saved as CSV.
                - Place the file under `data/broker_imports/cathay/`.
                - Numeric symbols such as `2330` become `2330.TW` automatically.
                - Stock orders remain manual; the app handles position tracking and reconciliation.
                """
            )
        with col_c:
            metric_card("Pionex", "API later", "Start with tracking, then canary-size live execution.")
            st.markdown(
                """
                - Record live account equity and planned crypto orders here.
                - Enable API only after keys, max order size, daily loss limit, and kill switch are configured.
                - API secrets must stay outside Git and Streamlit public settings.
                """
            )
        st.info(
            "Current workflow: sync broker exports, generate a strategy position plan, then append planned orders for manual review or later API execution."
        )

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
        EQUITY_SCAN_DIR / "latest_scan_summary.json",
        EQUITY_SCAN_DIR / "latest_recommendations.csv",
        EQUITY_SCAN_DIR / "tw_scan_ranking.csv",
        EQUITY_SCAN_DIR / "tw_recommendations.csv",
        EQUITY_SCAN_DIR / "tw_scan_metrics.csv",
        EQUITY_SCAN_DIR / "tw_scan_failures.csv",
        EQUITY_SCAN_DIR / "us_scan_ranking.csv",
        EQUITY_SCAN_DIR / "us_recommendations.csv",
        EQUITY_SCAN_DIR / "us_scan_metrics.csv",
        EQUITY_SCAN_DIR / "us_scan_failures.csv",
        OUTPUT_DIR / "equity_optimization" / "latest_optimization_report.json",
        OUTPUT_DIR / "equity_optimization" / "tw_top_candidates.csv",
        OUTPUT_DIR / "equity_optimization" / "us_top_candidates.csv",
        NEWS_FILES["crypto"],
        NEWS_FILES["tw"],
        NEWS_FILES["us"],
        ALERT_STATE_FILE,
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
