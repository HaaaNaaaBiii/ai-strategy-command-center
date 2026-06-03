from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import pandas as pd

from .equity_strategy import EquitySelectionConfig, rank_equities
from .indicators import atr, ema


COMPANY_NAMES = {
    "0050.TW": "Yuanta Taiwan 50 ETF",
    "1101.TW": "Taiwan Cement",
    "1102.TW": "Asia Cement",
    "1216.TW": "Uni-President",
    "1229.TW": "Lien Hwa Industrial",
    "1301.TW": "Formosa Plastics",
    "1303.TW": "Nan Ya Plastics",
    "1326.TW": "Formosa Chemicals",
    "1402.TW": "Far Eastern New Century",
    "1476.TW": "Eclat Textile",
    "1504.TW": "TECO Electric & Machinery",
    "1513.TW": "Chung-Hsin Electric",
    "1519.TW": "Fortune Electric",
    "1590.TW": "Airtac",
    "1605.TW": "Walsin Lihwa",
    "1722.TW": "Taiwan Fertilizer",
    "1802.TW": "Taiwan Glass",
    "2002.TW": "China Steel",
    "2049.TW": "Hiwin Technologies",
    "2105.TW": "Cheng Shin Rubber",
    "2201.TW": "Yulon Motor",
    "2207.TW": "Hotai Motor",
    "2227.TW": "Yulon Nissan Motor",
    "2301.TW": "Lite-On Technology",
    "2303.TW": "United Microelectronics",
    "2313.TW": "Compeq Manufacturing",
    "2330.TW": "Taiwan Semiconductor",
    "2317.TW": "Hon Hai Precision",
    "2324.TW": "Compal Electronics",
    "2308.TW": "Delta Electronics",
    "2327.TW": "Yageo",
    "2344.TW": "Winbond Electronics",
    "2353.TW": "Acer",
    "2354.TW": "Foxconn Technology",
    "2356.TW": "Inventec",
    "2357.TW": "ASUS",
    "2360.TW": "Chroma ATE",
    "2362.TW": "Clevo",
    "2368.TW": "Gold Circuit Electronics",
    "2376.TW": "Gigabyte Technology",
    "2377.TW": "Micro-Star International",
    "2379.TW": "Realtek",
    "2382.TW": "Quanta Computer",
    "2383.TW": "Elite Material",
    "2385.TW": "Chicony Electronics",
    "2395.TW": "Advantech",
    "2404.TW": "United Integrated Services",
    "2408.TW": "Nanya Technology",
    "2409.TW": "AUO",
    "2412.TW": "Chunghwa Telecom",
    "2441.TW": "Greatek Electronics",
    "2449.TW": "King Yuan Electronics",
    "2454.TW": "MediaTek",
    "2455.TW": "Visual Photonics Epitaxy",
    "2458.TW": "Elan Microelectronics",
    "2474.TW": "Catcher Technology",
    "2498.TW": "HTC",
    "2542.TW": "Highwealth Construction",
    "2548.TW": "Huaku Development",
    "2603.TW": "Evergreen Marine",
    "2606.TW": "U-Ming Marine",
    "2609.TW": "Yang Ming Marine",
    "2610.TW": "China Airlines",
    "2615.TW": "Wan Hai Lines",
    "2618.TW": "EVA Airways",
    "2633.TW": "Taiwan High Speed Rail",
    "2801.TW": "Chang Hwa Bank",
    "2809.TW": "King's Town Bank",
    "2812.TW": "Taichung Bank",
    "2834.TW": "Taiwan Business Bank",
    "2845.TW": "Far Eastern International Bank",
    "2880.TW": "Hua Nan Financial",
    "2881.TW": "Fubon Financial",
    "2882.TW": "Cathay Financial",
    "2883.TW": "China Development Financial",
    "2884.TW": "E.Sun Financial",
    "2885.TW": "Yuanta Financial",
    "2886.TW": "Mega Financial",
    "2887.TW": "Taishin Financial",
    "2888.TW": "Shin Kong Financial",
    "2890.TW": "SinoPac Financial",
    "2891.TW": "CTBC Financial",
    "2892.TW": "First Financial",
    "2897.TW": "O-Bank",
    "2912.TW": "President Chain Store",
    "3005.TW": "Getac",
    "3008.TW": "Largan Precision",
    "3014.TW": "ITE Tech",
    "3017.TW": "Asia Vital Components",
    "3019.TW": "Asia Optical",
    "3023.TW": "Sinbon Electronics",
    "3034.TW": "Novatek",
    "3035.TW": "Faraday Technology",
    "3036.TW": "WT Microelectronics",
    "3037.TW": "Unimicron",
    "3044.TW": "Tripod Technology",
    "3045.TW": "Taiwan Mobile",
    "3189.TW": "Kinsus Interconnect",
    "3231.TW": "Wistron",
    "3443.TW": "Global Unichip",
    "3481.TW": "Innolux",
    "3533.TW": "LotES",
    "3653.TW": "Jentech Precision",
    "3661.TW": "Alchip Technologies",
    "3702.TW": "WPG Holdings",
    "3706.TW": "Mitac Holdings",
    "3711.TW": "ASE Technology",
    "4904.TW": "Far EasTone",
    "4938.TW": "Pegatron",
    "4958.TW": "Zhen Ding Technology",
    "5269.TW": "ASMedia Technology",
    "5871.TW": "Chailease",
    "5876.TW": "Shanghai Commercial & Savings Bank",
    "5880.TW": "Taiwan Cooperative Financial",
    "6239.TW": "Powertech Technology",
    "6269.TW": "Flexium Interconnect",
    "6409.TW": "Voltronic Power",
    "6415.TW": "Silergy",
    "6446.TW": "PharmaEssentia",
    "6505.TW": "Formosa Petrochemical",
    "6531.TW": "ApMemory Technology",
    "6669.TW": "Wiwynn",
    "6770.TW": "Powerchip Semiconductor",
    "6789.TW": "VisEra Technologies",
    "8046.TW": "Nan Ya PCB",
    "8112.TW": "Supreme Electronics",
    "8150.TW": "ChipMOS Technologies",
    "8210.TW": "Chenbro Micom",
    "8454.TW": "momo.com",
    "8464.TW": "Nidec Chaun-Choung Technology",
    "9910.TW": "Feng Tay Enterprises",
    "9914.TW": "Merida Industry",
    "9917.TW": "Taiwan Secom",
    "9921.TW": "Giant Manufacturing",
    "9933.TW": "China Steel Chemical",
    "9939.TW": "Hota Industrial",
    "9958.TW": "Century Steel",
    "SPY": "SPDR S&P 500 ETF",
    "ABNB": "Airbnb",
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "ABT": "Abbott Laboratories",
    "ACN": "Accenture",
    "ADBE": "Adobe",
    "ADI": "Analog Devices",
    "AIG": "American International Group",
    "ALL": "Allstate",
    "AMAT": "Applied Materials",
    "AMD": "Advanced Micro Devices",
    "AMGN": "Amgen",
    "AMT": "American Tower",
    "ANET": "Arista Networks",
    "APD": "Air Products and Chemicals",
    "APP": "AppLovin",
    "ARM": "Arm Holdings",
    "ASML": "ASML Holding",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "AVGO": "Broadcom",
    "AXP": "American Express",
    "BA": "Boeing",
    "BAC": "Bank of America",
    "BKNG": "Booking Holdings",
    "BLK": "BlackRock",
    "BMY": "Bristol Myers Squibb",
    "BSX": "Boston Scientific",
    "BX": "Blackstone",
    "C": "Citigroup",
    "CARR": "Carrier Global",
    "CAT": "Caterpillar",
    "CB": "Chubb",
    "CDNS": "Cadence Design Systems",
    "CHTR": "Charter Communications",
    "CI": "Cigna",
    "CL": "Colgate-Palmolive",
    "CME": "CME Group",
    "CMG": "Chipotle Mexican Grill",
    "CMCSA": "Comcast",
    "COF": "Capital One Financial",
    "COIN": "Coinbase",
    "COP": "ConocoPhillips",
    "COST": "Costco",
    "CRM": "Salesforce",
    "CRWD": "CrowdStrike",
    "CSCO": "Cisco",
    "CVX": "Chevron",
    "DDOG": "Datadog",
    "DE": "Deere",
    "DELL": "Dell Technologies",
    "DECK": "Deckers Outdoor",
    "DHR": "Danaher",
    "DIS": "Disney",
    "CAVA": "Cava Group",
    "CELH": "Celsius Holdings",
    "COTY": "Coty",
    "CROX": "Crocs",
    "EL": "Estee Lauder",
    "ELF": "e.l.f. Beauty",
    "ELV": "Elevance Health",
    "EOG": "EOG Resources",
    "EQIX": "Equinix",
    "ETN": "Eaton",
    "FCX": "Freeport-McMoRan",
    "FI": "Fiserv",
    "GE": "GE Aerospace",
    "GEHC": "GE HealthCare",
    "GILD": "Gilead Sciences",
    "GM": "General Motors",
    "GS": "Goldman Sachs",
    "HCA": "HCA Healthcare",
    "HD": "Home Depot",
    "HON": "Honeywell",
    "HPE": "Hewlett Packard Enterprise",
    "HUM": "Humana",
    "IBM": "IBM",
    "ICE": "Intercontinental Exchange",
    "INTU": "Intuit",
    "ISRG": "Intuitive Surgical",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KKR": "KKR",
    "KLAC": "KLA",
    "KO": "Coca-Cola",
    "LIN": "Linde",
    "LLY": "Eli Lilly",
    "LMT": "Lockheed Martin",
    "LOW": "Lowe's",
    "LRCX": "Lam Research",
    "LULU": "Lululemon",
    "MA": "Mastercard",
    "MAR": "Marriott International",
    "MCO": "Moody's",
    "MCD": "McDonald's",
    "MDLZ": "Mondelez International",
    "MDT": "Medtronic",
    "MELI": "MercadoLibre",
    "MMM": "3M",
    "MRK": "Merck",
    "MMC": "Marsh & McLennan",
    "MO": "Altria",
    "MPC": "Marathon Petroleum",
    "MRVL": "Marvell Technology",
    "MS": "Morgan Stanley",
    "MSI": "Motorola Solutions",
    "MU": "Micron",
    "NEE": "NextEra Energy",
    "NFLX": "Netflix",
    "NKE": "Nike",
    "NET": "Cloudflare",
    "NOC": "Northrop Grumman",
    "NOW": "ServiceNow",
    "NXPI": "NXP Semiconductors",
    "ORCL": "Oracle",
    "PANW": "Palo Alto Networks",
    "PEP": "PepsiCo",
    "PFE": "Pfizer",
    "PH": "Parker-Hannifin",
    "PLD": "Prologis",
    "PLTR": "Palantir",
    "PNC": "PNC Financial Services",
    "PG": "Procter & Gamble",
    "PM": "Philip Morris",
    "PYPL": "PayPal",
    "QCOM": "Qualcomm",
    "RCL": "Royal Caribbean",
    "REGN": "Regeneron Pharmaceuticals",
    "RBLX": "Roblox",
    "ROK": "Rockwell Automation",
    "RTX": "RTX",
    "SBUX": "Starbucks",
    "SCHW": "Charles Schwab",
    "SHOP": "Shopify",
    "SHW": "Sherwin-Williams",
    "SMCI": "Super Micro Computer",
    "SNOW": "Snowflake",
    "SO": "Southern Company",
    "SPGI": "S&P Global",
    "SPOT": "Spotify",
    "SYK": "Stryker",
    "T": "AT&T",
    "TEAM": "Atlassian",
    "TJX": "TJX Companies",
    "TMO": "Thermo Fisher Scientific",
    "TSM": "Taiwan Semiconductor ADR",
    "TXN": "Texas Instruments",
    "UBER": "Uber Technologies",
    "ULTA": "Ulta Beauty",
    "UNH": "UnitedHealth",
    "UPS": "UPS",
    "USB": "U.S. Bancorp",
    "V": "Visa",
    "VRTX": "Vertex Pharmaceuticals",
    "VZ": "Verizon",
    "WELL": "Welltower",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "ZTS": "Zoetis",
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
    risk_reward_1: float | None
    risk_reward_2: float | None
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


def _metric(row: Mapping[str, object], column: str) -> float | None:
    value = row.get(column)
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag(row: Mapping[str, object], column: str) -> bool:
    value = row.get(column)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def explain_equity_selection_reason(
    row: Mapping[str, object],
    config: EquitySelectionConfig,
    selected: bool,
    rank: int | None = None,
) -> str:
    score = _metric(row, "score")
    short_momentum = _metric(row, "short_momentum_pct")
    long_momentum = _metric(row, "long_momentum_pct")
    volatility = _metric(row, "annualized_volatility_pct")
    risk_on = _flag(row, "risk_on")
    above_trend = _flag(row, "above_trend")
    eligible = _flag(row, "eligible")
    resolved_rank = rank or int(_metric(row, "scan_rank") or _metric(row, "rank") or 0) or None
    score_text = "-" if score is None else f"{score:.2f}"
    short_text = "-" if short_momentum is None else f"{short_momentum:.2f}%"
    long_text = "-" if long_momentum is None else f"{long_momentum:.2f}%"
    vol_text = "-" if volatility is None else f"{volatility:.2f}%"
    rank_text = "-" if resolved_rank is None else str(resolved_rank)
    context = (
        f"rank {rank_text}, score {score_text}, short momentum {short_text}, "
        f"long momentum {long_text}, volatility {vol_text}."
    )
    if selected:
        return (
            f"Selected inside Top {config.top_n}: {context} "
            f"Market gate is {'risk-on' if risk_on else 'risk-off'}, price is "
            f"{'above' if above_trend else 'below'} the trend line. Live action is rotation/rebalance; "
            "latest close is only the planning reference price."
        )
    if eligible:
        return (
            f"Eligible but outside Top {config.top_n}: {context} "
            "It remains on the watchlist but is not allocated until it rotates into the selected sleeve."
        )

    blockers: list[str] = []
    if not risk_on:
        blockers.append("market gate is risk-off")
    if not above_trend:
        blockers.append("price is below trend")
    if short_momentum is None or short_momentum <= 0:
        blockers.append("short momentum is not positive")
    if long_momentum is None or long_momentum <= 0:
        blockers.append("long momentum is not positive")
    if volatility is not None and volatility > config.max_volatility_pct:
        blockers.append(f"volatility {volatility:.2f}% exceeds max {config.max_volatility_pct:.2f}%")
    blocker_text = "; ".join(blockers) if blockers else "fails one or more eligibility filters"
    return f"Excluded from Top {config.top_n}: {blocker_text}. {context}"


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
    selected = symbol in selected_symbols
    eligible = bool(
        not ranking.empty
        and symbol in set(ranking.loc[ranking["eligible"], "symbol"].tolist())
    )
    rank = ranked_symbols.index(symbol) + 1 if symbol in ranked_symbols else None
    ranked = ranking[ranking["symbol"].astype(str) == symbol].head(1)
    ranked_row = ranked.iloc[0].to_dict() if not ranked.empty else {}
    entry = None
    stop = None
    take_profit_1 = None
    take_profit_2 = None
    risk_reward_1 = None
    risk_reward_2 = None
    strategy_exit = None
    if selected:
        action = "ROTATION_REBALANCE"
    elif eligible:
        action = "HOLD_CASH"
    else:
        action = "HOLD_CASH_OR_EXIT"
    reason = explain_equity_selection_reason(ranked_row, config, selected=selected, rank=rank)
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
        risk_reward_1=risk_reward_1,
        risk_reward_2=risk_reward_2,
        strategy_exit=max(strategy_exit, 0.0) if strategy_exit is not None else None,
        atr=current_atr,
        trend_level=trend,
        reason=reason,
    )
