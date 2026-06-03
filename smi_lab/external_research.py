from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import data_path, output_path


DEFAULT_INVESTING_IMPORT_DIR = data_path("investing")
DEFAULT_INVESTING_MONITOR_DIR = output_path("external_research")

OUTPUT_COLUMNS = [
    "market",
    "symbol",
    "company",
    "source",
    "as_of",
    "rating",
    "fair_value",
    "analyst_target",
    "upside_pct",
    "technical_summary",
    "fundamental_summary",
    "risk_summary",
    "notes",
    "url",
]


def _normalize_market(value: pd.Series) -> pd.Series:
    result = value.astype(str).str.lower().str.strip()
    return result.mask(result.isin({"", "nan", "none"}), "")


def _ensure_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in OUTPUT_COLUMNS:
        if column not in result:
            result[column] = ""
    return result[OUTPUT_COLUMNS]


def load_investing_research(import_dir: str | Path = DEFAULT_INVESTING_IMPORT_DIR) -> pd.DataFrame:
    directory = Path(import_dir)
    if not directory.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "symbol" not in frame:
            continue
        frame = frame.copy()
        if "source" in frame:
            frame["source"] = frame["source"].fillna("investing.com").replace("", "investing.com")
        else:
            frame["source"] = "investing.com"
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        if "market" in frame:
            frame["market"] = _normalize_market(frame["market"])
        else:
            frame["market"] = ""
        frames.append(_ensure_output_columns(frame))
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["market", "symbol", "source", "as_of"],
        keep="last",
    )


def attach_external_research(
    recommendations: pd.DataFrame,
    market: str,
    import_dir: str | Path = DEFAULT_INVESTING_IMPORT_DIR,
) -> pd.DataFrame:
    market_key = market.lower().strip()
    if recommendations.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    research = load_investing_research(import_dir)
    if research.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    selected = recommendations.copy()
    if "market" not in selected:
        selected["market"] = market_key
    if "company" not in selected:
        selected["company"] = ""
    selected["market"] = _normalize_market(selected["market"])
    selected.loc[selected["market"].eq(""), "market"] = market_key
    selected["symbol"] = selected["symbol"].astype(str).str.upper().str.strip()
    scoped = research[
        (_normalize_market(research["market"]).isin({market_key, ""}))
        & (research["symbol"].astype(str).str.upper().isin(set(selected["symbol"])))
    ].copy()
    if scoped.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    scoped["market"] = _normalize_market(scoped["market"])
    scoped.loc[scoped["market"].eq(""), "market"] = market_key
    merged = selected[["market", "symbol", "company"]].merge(
        scoped.drop(columns=["company"], errors="ignore"),
        on=["market", "symbol"],
        how="inner",
    )
    return _ensure_output_columns(merged)


def write_external_research_monitor(
    recommendations: pd.DataFrame,
    market: str,
    import_dir: str | Path = DEFAULT_INVESTING_IMPORT_DIR,
    output_dir: str | Path = DEFAULT_INVESTING_MONITOR_DIR,
) -> pd.DataFrame:
    monitor = attach_external_research(recommendations, market, import_dir=import_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    monitor.to_csv(output / f"{market}_investing_monitor.csv", index=False)
    return monitor
