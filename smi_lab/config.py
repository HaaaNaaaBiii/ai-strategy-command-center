from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    """All tunable trading assumptions used by signal generation and testing."""

    smi_period: int = 20
    smooth_k: int = 5
    smooth_d: int = 3
    signal_period: int = 5
    trend_ema: int = 100
    atr_period: int = 14
    adx_period: int = 14
    adx_min: float = 18.0
    oversold: float = -30.0
    overbought: float = 30.0
    stop_atr: float = 1.8
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    tp3_r: float = 3.0
    tp1_fraction: float = 0.40
    tp2_fraction: float = 0.35
    risk_per_trade: float = 0.01
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    max_leverage: float = 1.0
    cooldown_bars: int = 2
    use_longs: bool = True
    use_shorts: bool = True
    regime_mode: str = "none"
    regime_source: str = "none"
    entry_mode: str = "pullback"
    breakout_period: int = 40
    btc_ema_period: int = 100
    momentum_period: int = 180
    momentum_top_n: int = 1

    def target_fractions(self) -> tuple[float, float, float]:
        third = 1.0 - self.tp1_fraction - self.tp2_fraction
        if third <= 0:
            raise ValueError("TP fractions must leave a positive final position.")
        return self.tp1_fraction, self.tp2_fraction, third

    def validate(self) -> "StrategyConfig":
        if not (0 < self.risk_per_trade <= 0.05):
            raise ValueError("risk_per_trade must be in (0, 0.05].")
        if not (0 < self.tp1_r < self.tp2_r < self.tp3_r):
            raise ValueError("Take-profit levels must be strictly increasing.")
        if self.stop_atr <= 0 or self.max_leverage <= 0:
            raise ValueError("stop_atr and max_leverage must be positive.")
        if not self.use_longs and not self.use_shorts:
            raise ValueError("At least one trade direction must be enabled.")
        if self.regime_mode not in {
            "none",
            "avoid_risk_off_longs",
            "risk_aligned",
        }:
            raise ValueError(f"Unsupported regime mode: {self.regime_mode}")
        if self.regime_source not in {"none", "cboe_stress", "btc_momentum"}:
            raise ValueError(f"Unsupported regime source: {self.regime_source}")
        if self.regime_mode != "none" and self.regime_source == "none":
            raise ValueError("A filtered strategy must specify its regime source.")
        if self.entry_mode not in {"pullback", "breakout"}:
            raise ValueError(f"Unsupported entry mode: {self.entry_mode}")
        if self.breakout_period < 5:
            raise ValueError("breakout_period must be at least 5.")
        if (
            self.btc_ema_period < 5
            or self.momentum_period < 5
            or self.momentum_top_n < 1
        ):
            raise ValueError("BTC momentum regime periods and rank must be positive.")
        self.target_fractions()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_costs(self, fee_bps: float, slippage_bps: float) -> "StrategyConfig":
        return replace(self, fee_bps=fee_bps, slippage_bps=slippage_bps).validate()


def save_config(config: StrategyConfig, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(config.validate().to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_config(path: str | Path, default: StrategyConfig | None = None) -> StrategyConfig:
    source = Path(path)
    if not source.exists():
        return (default or StrategyConfig()).validate()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return StrategyConfig(**payload).validate()


def save_portfolio(
    sleeves: list[tuple[str, float, StrategyConfig]], path: str | Path
) -> None:
    if abs(sum(weight for _, weight, _ in sleeves) - 1.0) > 1e-9:
        raise ValueError("Portfolio sleeve weights must sum to 1.")
    payload = {
        "name": "balanced_smi_ensemble",
        "sleeves": [
            {"name": name, "weight": weight, "config": config.validate().to_dict()}
            for name, weight, config in sleeves
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_portfolio(path: str | Path) -> list[tuple[str, float, StrategyConfig]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sleeves = [
        (item["name"], float(item["weight"]), StrategyConfig(**item["config"]).validate())
        for item in payload["sleeves"]
    ]
    if abs(sum(weight for _, weight, _ in sleeves) - 1.0) > 1e-9:
        raise ValueError("Portfolio sleeve weights must sum to 1.")
    return sleeves
