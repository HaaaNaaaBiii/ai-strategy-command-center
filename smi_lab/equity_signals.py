from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .equity_strategy import EquitySelectionConfig, rank_equities
from .indicators import atr, ema


COMPANY_NAMES = {
    "0050.TW": "Yuanta Taiwan 50 ETF",
    "1101.TW": "Taiwan Cement",
    "1216.TW": "Uni-President",
    "1301.TW": "Formosa Plastics",
    "1303.TW": "Nan Ya Plastics",
    "1326.TW": "Formosa Chemicals",
    "1590.TW": "Airtac",
    "2002.TW": "China Steel",
    "2207.TW": "Hotai Motor",
    "2301.TW": "Lite-On Technology",
    "2303.TW": "United Microelectronics",
    "2330.TW": "Taiwan Semiconductor",
    "2317.TW": "Hon Hai Precision",
    "2308.TW": "Delta Electronics",
    "2327.TW": "Yageo",
    "2357.TW": "ASUS",
    "2379.TW": "Realtek",
    "2382.TW": "Quanta Computer",
    "2395.TW": "Advantech",
    "2408.TW": "Nanya Technology",
    "2412.TW": "Chunghwa Telecom",
    "2454.TW": "MediaTek",
    "2603.TW": "Evergreen Marine",
    "2609.TW": "Yang Ming Marine",
    "2615.TW": "Wan Hai Lines",
    "2880.TW": "Hua Nan Financial",
    "2881.TW": "Fubon Financial",
    "2882.TW": "Cathay Financial",
    "2883.TW": "China Development Financial",
    "2884.TW": "E.Sun Financial",
    "2885.TW": "Yuanta Financial",
    "2886.TW": "Mega Financial",
    "2887.TW": "Taishin Financial",
    "2890.TW": "SinoPac Financial",
    "2891.TW": "CTBC Financial",
    "2892.TW": "First Financial",
    "2912.TW": "President Chain Store",
    "3008.TW": "Largan Precision",
    "3034.TW": "Novatek",
    "3037.TW": "Unimicron",
    "3045.TW": "Taiwan Mobile",
    "3231.TW": "Wistron",
    "3711.TW": "ASE Technology",
    "4904.TW": "Far EasTone",
    "4938.TW": "Pegatron",
    "5871.TW": "Chailease",
    "5876.TW": "Shanghai Commercial & Savings Bank",
    "5880.TW": "Taiwan Cooperative Financial",
    "6415.TW": "Silergy",
    "6505.TW": "Formosa Petrochemical",
    "6669.TW": "Wiwynn",
    "6770.TW": "Powerchip Semiconductor",
    "8046.TW": "Nan Ya PCB",
    "8454.TW": "momo.com",
    "SPY": "SPDR S&P 500 ETF",
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "ABT": "Abbott Laboratories",
    "ADBE": "Adobe",
    "AMAT": "Applied Materials",
    "AMD": "Advanced Micro Devices",
    "AMGN": "Amgen",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "AVGO": "Broadcom",
    "BA": "Boeing",
    "BAC": "Bank of America",
    "BKNG": "Booking Holdings",
    "CAT": "Caterpillar",
    "CMCSA": "Comcast",
    "COP": "ConocoPhillips",
    "COST": "Costco",
    "CRM": "Salesforce",
    "CSCO": "Cisco",
    "CVX": "Chevron",
    "DHR": "Danaher",
    "DIS": "Disney",
    "GE": "GE Aerospace",
    "GS": "Goldman Sachs",
    "HD": "Home Depot",
    "HON": "Honeywell",
    "IBM": "IBM",
    "INTU": "Intuit",
    "ISRG": "Intuitive Surgical",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "LIN": "Linde",
    "LLY": "Eli Lilly",
    "LOW": "Lowe's",
    "MA": "Mastercard",
    "MCD": "McDonald's",
    "MRK": "Merck",
    "MU": "Micron",
    "NFLX": "Netflix",
    "NKE": "Nike",
    "NOW": "ServiceNow",
    "ORCL": "Oracle",
    "PEP": "PepsiCo",
    "PG": "Procter & Gamble",
    "PM": "Philip Morris",
    "QCOM": "Qualcomm",
    "RTX": "RTX",
    "SBUX": "Starbucks",
    "SPGI": "S&P Global",
    "TMO": "Thermo Fisher Scientific",
    "TXN": "Texas Instruments",
    "UNH": "UnitedHealth",
    "V": "Visa",
    "VZ": "Verizon",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
}


@dataclass(frozen=True)
class EquityTradePlan:
    symbol: str
    company: str
    action: str
    selected: bool
    eligible: bool
    rank: int | None
    close: float
    entry_price: float | None
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    strategy_exit: float | None
    atr: float
    trend_level: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def company_name(symbol: str) -> str:
    return COMPANY_NAMES.get(symbol.upper(), symbol.upper())


def add_company_names(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame:
        return frame
    result = frame.copy()
    result.insert(1, "company", result["symbol"].map(company_name))
    return result


def _latest_value(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[-1])


def build_equity_trade_plan(
    symbol: str,
    universe: dict[str, pd.DataFrame],
    config: EquitySelectionConfig,
    ranking: pd.DataFrame | None = None,
) -> EquityTradePlan:
    if symbol not in universe:
        raise ValueError(f"Symbol is not loaded: {symbol}")
    ranking = ranking if ranking is not None else rank_equities(universe, config)
    ranked_symbols = ranking["symbol"].tolist() if "symbol" in ranking else []
    selected_symbols = (
        ranking[ranking["eligible"]].head(config.top_n)["symbol"].tolist()
        if "eligible" in ranking
        else []
    )
    frame = universe[symbol].dropna(subset=["open", "high", "low", "close"])
    if len(frame) < max(20, config.trend_period):
        raise ValueError("Not enough history to build a trade plan.")
    close = float(frame["close"].iloc[-1])
    current_atr = _latest_value(atr(frame, 14))
    if pd.isna(current_atr) or current_atr <= 0:
        current_atr = max(close * 0.03, 0.01)
    trend = _latest_value(ema(frame["close"].astype(float), config.trend_period))
    if pd.isna(trend) or trend <= 0:
        trend = close
    breakout = _latest_value(
        frame["high"].astype(float).shift(1).rolling(20, min_periods=5).max()
    )
    if pd.isna(breakout) or breakout <= 0:
        breakout = close
    selected = symbol in selected_symbols
    eligible = bool(
        not ranking.empty
        and symbol in set(ranking.loc[ranking["eligible"], "symbol"].tolist())
    )
    rank = ranked_symbols.index(symbol) + 1 if symbol in ranked_symbols else None
    if selected:
        entry = max(breakout + 0.10 * current_atr, trend + 0.10 * current_atr)
        stop = min(entry - 2.0 * current_atr, trend * 0.98)
        take_profit_1 = entry + 2.0 * current_atr
        take_profit_2 = entry + 4.0 * current_atr
        strategy_exit = max(entry - 2.5 * current_atr, trend)
        action = "WAIT_FOR_BREAKOUT"
        reason = (
            "Eligible and inside the current top-N sleeve. Entry waits for the "
            "strategy breakout trigger instead of chasing the latest close."
        )
    elif eligible:
        entry = None
        stop = None
        take_profit_1 = None
        take_profit_2 = None
        strategy_exit = trend
        action = "HOLD_CASH"
        reason = "Eligible but not inside the current top-N sleeve; keep cash unless it rotates in."
    else:
        entry = None
        stop = None
        take_profit_1 = None
        take_profit_2 = None
        strategy_exit = trend
        action = "HOLD_CASH_OR_EXIT"
        reason = "Fails at least one market, trend, momentum, or volatility filter."
    return EquityTradePlan(
        symbol=symbol,
        company=company_name(symbol),
        action=action,
        selected=selected,
        eligible=eligible,
        rank=rank,
        close=close,
        entry_price=entry,
        stop_loss=max(stop, 0.0) if stop is not None else None,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        strategy_exit=max(strategy_exit, 0.0) if strategy_exit is not None else None,
        atr=current_atr,
        trend_level=trend,
        reason=reason,
    )
