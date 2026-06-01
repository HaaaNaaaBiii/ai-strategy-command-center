from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from smi_lab.data import DEFAULT_SYMBOLS, bars_for_years, data_window, load_universe
from smi_lab.equity_data import (
    DEFAULT_TW_SYMBOLS,
    DEFAULT_US_SYMBOLS,
    load_equity_universe,
)
from smi_lab.equity_strategy import (
    backtest_equity_selection,
    benchmark_buy_and_hold,
    default_equity_config,
    rank_equities,
)
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


st.set_page_config(page_title="Crypto / Stocks AI Strategy Lab", layout="wide")
st.title("Crypto / Stocks AI Strategy Lab")
st.caption(
    "加密貨幣策略檢討、前瞻紙上追蹤、台股與美股技術摘要。"
    "所有訊號僅供研究與通知，不連接交易帳戶。"
)


def pct(value: float) -> str:
    return f"{value:.2f}%"


def money(value: float) -> str:
    return f"{value:,.2f}"


def display_technical_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("沒有可顯示的技術摘要。")
        return
    display = frame.copy()
    for column in ("close", "rsi", "macd", "macd_signal", "roc_pct", "atr_pct"):
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.2f}")
    display = display.rename(
        columns={
            "symbol": "標的",
            "as_of": "資料時間",
            "close": "收盤",
            "summary": "Summary",
            "moving_averages": "Moving Averages",
            "indicators": "Indicators",
            "ma_buy": "MA Buy",
            "ma_neutral": "MA Neutral",
            "ma_sell": "MA Sell",
            "indicator_buy": "Ind Buy",
            "indicator_neutral": "Ind Neutral",
            "indicator_sell": "Ind Sell",
            "rsi": "RSI",
            "roc_pct": "ROC %",
            "atr_pct": "ATR %",
            "ai_view": "AI 技術看法",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def send_report(channel: str, message: str, discord_url: str, telegram_token: str, telegram_chat: str) -> None:
    if channel == "Discord":
        send_discord(message, discord_url)
    elif channel == "Telegram":
        send_telegram(message, telegram_token, telegram_chat)
    else:
        raise ValueError(f"Unsupported channel: {channel}")


with st.sidebar:
    st.header("資料設定")
    crypto_symbols = st.multiselect(
        "加密貨幣",
        list(DEFAULT_SYMBOLS),
        default=list(DEFAULT_SYMBOLS),
    )
    crypto_interval = st.selectbox("加密 K 線週期", ["4h", "1h", "1d", "15m"], index=0)
    crypto_years = st.slider("加密資料年數", 1, 5, 2)
    crypto_bars = bars_for_years(crypto_interval, crypto_years)
    refresh_crypto = st.checkbox("更新加密資料", value=False)
    st.divider()
    st.caption("台股代碼未輸入副檔名時會自動補 `.TW`。")


crypto_tab, tracker_tab, tw_tab, us_tab, architecture_tab, records_tab = st.tabs(
    ["加密策略", "前瞻紙上追蹤", "台股", "美股", "策略架構", "紀錄"]
)


with crypto_tab:
    st.subheader("加密策略檢討與當前 AI 技術看法")
    metadata_path = MARKET_ALPHA_DIR / "metadata.json"
    comparison_path = MARKET_ALPHA_DIR / "selected_benchmark_comparison.csv"
    metrics_path = MARKET_ALPHA_DIR / "selected_metrics.csv"
    if metadata_path.exists():
        st.json(pd.read_json(metadata_path, typ="series").to_dict(), expanded=False)
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        stressed = comparison[comparison["scenario"] == "triple_cost"]
        st.caption("市場超額配置候選：三倍交易成本下與四幣等權大盤、BTC 買入持有比較。")
        st.dataframe(stressed, hide_index=True, width="stretch")
    if metrics_path.exists():
        selected = pd.read_csv(metrics_path)
        full = selected[
            (selected["scenario"] == "triple_cost")
            & (selected["phase"] == "full_period")
        ]
        if not full.empty:
            row = full.iloc[0]
            cols = st.columns(4)
            cols[0].metric("五年回測", pct(float(row["return_pct"])))
            cols[1].metric("最大回撤", pct(float(row["max_drawdown_pct"])))
            cols[2].metric("交易成本", money(float(row["trading_cost"])))
            cols[3].metric("再平衡次數", f"{int(float(row['rebalances']))}")

    if st.button("更新加密 AI 技術摘要", type="primary"):
        try:
            universe = load_crypto_data(
                crypto_symbols,
                crypto_interval,
                max(crypto_bars, 500),
                refresh_crypto,
            )
            st.session_state["crypto_universe"] = universe
            st.success(f"資料區間：{data_window(next(iter(universe.values())))}")
        except Exception as exc:
            st.error(f"加密資料讀取失敗：{exc}")

    universe = st.session_state.get("crypto_universe")
    if universe:
        display_technical_table(summarize_universe(universe))
        try:
            config, offsets, _ = load_allocation_strategy()
            snapshot = allocation_snapshot(universe, config, offsets)
            st.caption("市場超額配置目前目標權重")
            st.dataframe(
                aggregate_snapshot(snapshot),
                hide_index=True,
                width="stretch",
            )
            with st.expander("發送配置摘要通知"):
                channel = st.radio("通知方式", ["Discord", "Telegram"], horizontal=True)
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
                if st.button("發送配置摘要"):
                    message = format_allocation_report(snapshot)
                    try:
                        send_report(channel, message, discord_url, telegram_token, telegram_chat)
                        st.success("已發送配置摘要。")
                    except Exception as exc:
                        st.error(f"通知失敗：{exc}")
        except Exception as exc:
            st.warning(f"配置摘要無法計算：{exc}")
    else:
        st.info("按下更新後，會以 Investing.com 類似的 Moving Averages / Indicators / Summary 模式產生每個幣種的技術看法。")


with tracker_tab:
    st.subheader("前瞻紙上追蹤")
    st.write(
        "紙上追蹤從第一次按下更新或執行 `track_paper.py` 的最新 K 線開始，"
        "之後只統計啟用日以後的前瞻績效。"
    )
    if st.button("更新紙上帳戶", type="primary"):
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
            st.success("紙上帳戶已更新。")
        except Exception as exc:
            st.error(f"紙上追蹤更新失敗：{exc}")

    update = st.session_state.get("paper_update")
    if update:
        cols = st.columns(4)
        cols[0].metric("狀態", update.status)
        cols[1].metric("紙上權益", money(update.equity))
        cols[2].metric("前瞻報酬", pct(update.return_pct))
        cols[3].metric("Live Ready", str(update.live_ready))
        cols = st.columns(4)
        cols[0].metric("四幣等權", pct(update.equal_weight_return_pct))
        cols[1].metric("超額等權", pct(update.excess_vs_equal_weight_pct))
        cols[2].metric("紙上回撤", pct(update.max_drawdown_pct))
        cols[3].metric("事件數", str(update.events))
        if update.blockers:
            st.warning("尚未達實盤門檻：" + "；".join(update.blockers))
        st.code(st.session_state.get("paper_message", ""))

    equity_file = TRACKING_DIR / "market_alpha_staggered_equity.csv"
    events_file = TRACKING_DIR / "market_alpha_staggered_events.csv"
    status_file = TRACKING_DIR / "market_alpha_staggered_status.json"
    if status_file.exists():
        st.caption("最新紙上追蹤狀態")
        st.json(pd.read_json(status_file, typ="series").to_dict(), expanded=False)
    if equity_file.exists():
        equity = pd.read_csv(equity_file)
        if not equity.empty and "equity" in equity:
            st.line_chart(equity.set_index("timestamp")["equity"])
    if events_file.exists():
        events = read_csv_or_empty(events_file)
        if not events.empty:
            st.caption("紙上再平衡事件")
            st.dataframe(events.tail(100), hide_index=True, width="stretch")


def equity_page(title: str, market: str, defaults: tuple[str, ...]) -> None:
    st.subheader(title)
    st.write(
        "目前先完成資料、技術摘要與策略架構層；台股與美股尚未啟用交易通知，"
        "必須先累積紙上追蹤。"
    )
    symbols_text = st.text_area(
        f"{title} 代碼",
        value="\n".join(defaults),
        height=140,
        key=f"{market}_symbols",
    )
    interval = st.selectbox(
        f"{title} 週期",
        ["1d", "1wk", "1h"],
        index=0,
        key=f"{market}_interval",
    )
    range_ = st.selectbox(
        f"{title} 資料範圍",
        ["1y", "2y", "5y", "6mo"],
        index=0,
        key=f"{market}_range",
    )
    refresh = st.checkbox(f"更新 {title} 資料", value=False, key=f"{market}_refresh")
    if st.button(f"更新 {title} AI 技術摘要", key=f"{market}_button"):
        config = default_equity_config(market)
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
            st.success(f"已載入 {len(universe)} 個標的。")
        except Exception as exc:
            st.error(f"{title} 資料讀取失敗：{exc}")
    universe = st.session_state.get(f"{market}_universe")
    if universe:
        display_technical_table(summarize_universe(universe))
        try:
            config = default_equity_config(market)
            ranking = rank_equities(universe, config)
            result = backtest_equity_selection(universe, config)
            benchmark = benchmark_buy_and_hold(
                universe[config.market_symbol],
                fee_bps=config.fee_bps,
                slippage_bps=config.slippage_bps,
            )
            EQUITY_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
            ranking.to_csv(EQUITY_SELECTION_DIR / f"{market}_ranking.csv", index=False)
            result.rebalances.to_csv(
                EQUITY_SELECTION_DIR / f"{market}_rebalances.csv", index=False
            )
            pd.DataFrame(
                [
                    {"strategy": "equity_selection", **result.metrics},
                    {"strategy": config.market_symbol, **benchmark.metrics},
                ]
            ).to_csv(EQUITY_SELECTION_DIR / f"{market}_metrics.csv", index=False)
            st.caption("市場別策略設定")
            st.json(
                {
                    "benchmark": config.market_symbol,
                    "top_n": config.top_n,
                    "rebalance_bars": config.rebalance_bars,
                    "short_momentum_period": config.short_momentum_period,
                    "long_momentum_period": config.long_momentum_period,
                    "trend_period": config.trend_period,
                    "cost_bps": config.fee_bps + config.slippage_bps,
                    "status": "research_only",
                },
                expanded=False,
            )
            st.caption("選股策略排名")
            st.dataframe(ranking, hide_index=True, width="stretch")
            cols = st.columns(4)
            cols[0].metric("策略報酬", pct(float(result.metrics["return_pct"])))
            cols[1].metric("基準報酬", pct(float(benchmark.metrics["return_pct"])))
            cols[2].metric("策略回撤", pct(float(result.metrics["max_drawdown_pct"])))
            cols[3].metric("再平衡", f"{int(float(result.metrics['rebalances']))}")
            curves = pd.concat(
                {
                    "strategy": result.equity,
                    config.market_symbol: benchmark.equity,
                },
                axis=1,
                sort=False,
            ).dropna()
            if not curves.empty:
                st.line_chart(curves / curves.iloc[0] * 100.0)
        except Exception as exc:
            st.warning(f"{title} 選股策略暫時無法計算：{exc}")
        first_symbol = next(iter(universe))
        st.caption(f"{first_symbol} 價格走勢")
        st.line_chart(universe[first_symbol]["close"])


with tw_tab:
    equity_page("台股", "tw", DEFAULT_TW_SYMBOLS)


with us_tab:
    equity_page("美股", "us", DEFAULT_US_SYMBOLS)


with architecture_tab:
    st.subheader("策略架構設計")
    st.markdown(
        """
### 1. 加密貨幣

- 核心候選：`market_alpha_staggered_trend_allocation`，不再依賴 SMI。
- 主要基準：BTC / ETH / DOGE / SOL 四幣等權買入持有。
- 決策：BTC EMA100 風險閘門 + 幣種 EMA42 趨勢過濾 + 180 根相對動能排名。
- 部位：42 個等額週期 sleeve，總曝險上限 35%，其餘現金。
- 風控：無槓桿、含資金費率、含交易成本；先紙上追蹤，再評估是否取代既有 SMI 通知。
- 實盤門檻：至少 30 天前瞻追蹤、勝過四幣等權、回撤不低於 -10%、資料延遲小於 12 小時、至少 3 次前瞻再平衡。

### 2. AI 技術看法

- 參考 Investing.com 的摘要模式：分成 Moving Averages、Indicators、Summary。
- Moving Averages：SMA/EMA 5、10、20、50、100、200。
- Indicators：RSI、Stochastic、MACD、CCI、Williams %R、ROC、ADX 趨勢確認。
- 輸出不是保證式預測，而是規則化的技術面偏多/偏空解讀。

### 3. 台股 / 美股

- 第一階段：資料擷取、技術摘要、觀察清單與架構設計。
- 第二階段：台股以 0050.TW、美股以 SPY 作大盤閘門。
- 第三階段：相對強弱排序、趨勢過濾、波動懲罰、top-N 再平衡回測。
- 第四階段：擴大股票池，加入流動性、財報/除權息事件與前瞻紙上追蹤。
- 第五階段：只有在前瞻追蹤超越各自大盤後，才開啟通知器。
"""
    )


with records_tab:
    st.subheader("輸出紀錄")
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
    ]
    for path in files:
        if not path.exists():
            st.info(f"尚未產生：{path}")
            continue
        st.caption(str(path))
        if path.suffix == ".json":
            st.json(pd.read_json(path, typ="series").to_dict(), expanded=False)
        else:
            st.dataframe(read_csv_or_empty(path), hide_index=True, width="stretch")
