from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from smi_lab.backtest import backtest, backtest_portfolio, backtest_ranked_long
from smi_lab.allocation import (
    TrendAllocationConfig,
    backtest_staggered_trend_allocation,
    backtest_trend_allocation,
)
from smi_lab.config import StrategyConfig, load_portfolio, save_portfolio
from smi_lab.data import (
    attach_funding_rates,
    bars_for_years,
    fetch_bybit_linear_klines,
    fetch_klines,
    get_klines,
)
from smi_lab.equity_data import fetch_yahoo_chart
from smi_lab.evolution import _candidate_configs
from smi_lab.accounts import (
    ACCOUNT_COLUMNS,
    ORDER_COLUMNS,
    AccountSnapshot,
    OrderTracker,
    append_order,
    load_table,
    upsert_account,
)
from smi_lab.broker_import import normalize_broker_positions, sync_broker_exports
from smi_lab.crypto_universe import crypto_scan_symbols, load_crypto_scan_universe
from smi_lab.attention_strategy import AttentionConfig, attention_features, backtest_attention_strategy
from smi_lab.equity_live import build_equity_live_order_plan, load_live_strategy_memory, remember_live_plan
from smi_lab.equity_signals import add_company_names, build_equity_trade_plan, explain_equity_selection_reason
from smi_lab.equity_scanner import build_scan_recommendations
from smi_lab.equity_strategy import (
    EquitySelectionConfig,
    backtest_equity_selection,
    benchmark_buy_and_hold,
    rank_equities,
    target_weights_from_ranking,
)
from smi_lab.equity_universe import equity_scan_symbols
from smi_lab.market_info import cached_crypto_snapshots, fetch_equity_symbol_news
from smi_lab.paper import aggregate_snapshot, allocation_snapshot, update_forward_tracking
from smi_lab.position_planner import build_rebalance_plan
from smi_lab.price_alerts import check_equity_price_alerts, format_alert_message, AlertEvent
from smi_lab.regime import attach_btc_momentum_regime, attach_cboe_regime
from smi_lab.rotation import RotationConfig, _rebalance_schedule, backtest_rotation
from smi_lab.strategy import build_feature_frame
from smi_lab.technical import summarize_technical


def fake_frame(entry_bar_low: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=7, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100, 100],
            "high": [100, 100, 100.5, 101.2, 102.2, 103.2, 100],
            "low": [100, 100, entry_bar_low, 100, 100.5, 101.5, 100],
            "close": [100, 100, 100, 101, 102, 103, 100],
            "volume": [1] * 7,
        },
        index=index,
    )


def signaled_features(frame: pd.DataFrame, _: StrategyConfig) -> pd.DataFrame:
    result = frame.copy()
    result["atr"] = 1.0
    result["long_signal"] = False
    result["short_signal"] = False
    result.iloc[1, result.columns.get_loc("long_signal")] = True
    return result


def cached_candles(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="4h", tz="UTC", name="open_time")
    result = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "close_time": index + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1),
        },
        index=index,
    )
    return result


def rising_frame(periods: int = 220) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="4h", tz="UTC")
    close = pd.Series(range(100, 100 + periods), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
            "close_time": index + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1),
            "funding_rate": 0.0,
        },
        index=index,
    )


class StrategyTests(unittest.TestCase):
    def test_five_year_four_hour_history_has_required_size(self) -> None:
        self.assertEqual(bars_for_years("4h", 5), 10958)
        with self.assertRaises(ValueError):
            bars_for_years("4h", 6)

    @patch("smi_lab.data._request_url_json", side_effect=RuntimeError("Bybit unavailable"))
    @patch("smi_lab.data._request_json")
    def test_perpetual_klines_fall_back_to_spot_when_futures_blocked(
        self, request_json: object, _: object
    ) -> None:
        rows = [
            [
                1_700_000_000_000 + i * 14_400_000,
                "100",
                "101",
                "99",
                "100",
                "1",
                1_700_000_000_000 + (i + 1) * 14_400_000 - 1,
                "1",
                1,
                "1",
                "1",
                "0",
            ]
            for i in range(70)
        ]
        request_json.side_effect = [RuntimeError("HTTP Error 451"), rows]

        frame = fetch_klines(
            "BTCUSDT",
            interval="4h",
            bars=50,
            end_time_ms=1_700_000_000_000 + 80 * 14_400_000,
            market="perpetual",
        )

        self.assertFalse(frame.empty)
        self.assertEqual(frame.attrs["data_source"], "binance_spot")
        self.assertEqual(request_json.call_count, 2)

    @patch("smi_lab.data._request_url_json")
    @patch("smi_lab.data._request_json", side_effect=RuntimeError("HTTP Error 451"))
    def test_perpetual_klines_use_bybit_before_spot_when_available(
        self, _: object, request_url_json: object
    ) -> None:
        rows = [
            [
                str(1_700_000_000_000 + i * 14_400_000),
                "100",
                "101",
                "99",
                "100",
                "1",
                "1",
            ]
            for i in range(70)
        ]
        request_url_json.return_value = {"retCode": 0, "result": {"list": rows[::-1]}}

        frame = fetch_klines(
            "BTCUSDT",
            interval="4h",
            bars=50,
            end_time_ms=1_700_000_000_000 + 80 * 14_400_000,
            market="perpetual",
        )

        self.assertFalse(frame.empty)
        self.assertEqual(frame.attrs["data_source"], "bybit_linear")

    @patch("smi_lab.data._request_url_json")
    def test_bybit_klines_page_backwards_for_long_history(self, request_url_json: object) -> None:
        interval_ms = 14_400_000
        base = 1_700_000_000_000

        def row(i: int) -> list[str]:
            return [
                str(base + i * interval_ms),
                "100",
                "101",
                "99",
                "100",
                "1",
                "1",
            ]

        request_url_json.side_effect = [
            {"retCode": 0, "result": {"list": [row(i) for i in range(205, 1205)][::-1]}},
            {"retCode": 0, "result": {"list": [row(i) for i in range(95, 205)][::-1]}},
        ]

        frame = fetch_bybit_linear_klines(
            "BTCUSDT",
            interval="4h",
            bars=1100,
            end_time_ms=base + 1205 * interval_ms,
        )

        self.assertEqual(len(frame), 1100)
        self.assertEqual(request_url_json.call_count, 2)
        self.assertEqual(frame.index[0], pd.to_datetime(base + 105 * interval_ms, unit="ms", utc=True))
        self.assertEqual(frame.index[-1], pd.to_datetime(base + 1204 * interval_ms, unit="ms", utc=True))

    @patch("smi_lab.equity_data._fetch_stooq_daily")
    @patch("smi_lab.equity_data._fetch_yahoo_payload", side_effect=RuntimeError("HTTP Error 429"))
    def test_equity_fetch_uses_stooq_when_yahoo_is_rate_limited(
        self, _: object, stooq_daily: object
    ) -> None:
        index = pd.date_range("2025-01-01", periods=3, freq="1D", tz="UTC", name="open_time")
        stooq_daily.return_value = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1.0, 1.0, 1.0],
                "close_time": index,
            },
            index=index,
        )
        with TemporaryDirectory() as directory:
            frame = fetch_yahoo_chart("AAPL", interval="1d", range_="1y", cache_dir=directory, refresh=True)

        self.assertFalse(frame.empty)
        self.assertEqual(float(frame["close"].iloc[-1]), 102.5)

    def test_research_candidates_keep_short_leg_enabled(self) -> None:
        candidates = _candidate_configs(StrategyConfig(), count=30, seed=42)
        self.assertTrue(all(candidate.use_shorts for candidate in candidates))

    def test_regime_candidates_use_allowed_modes(self) -> None:
        candidates = _candidate_configs(
            StrategyConfig(),
            count=30,
            seed=42,
            regime_modes=("none", "risk_aligned"),
        )
        self.assertTrue(
            all(candidate.regime_mode in {"none", "risk_aligned"} for candidate in candidates)
        )

    def test_portfolio_config_round_trip(self) -> None:
        sleeves = [
            ("trend_core", 0.5, StrategyConfig()),
            ("defensive_short", 0.5, StrategyConfig(use_longs=False)),
        ]
        with TemporaryDirectory() as directory:
            target = f"{directory}/portfolio.json"
            save_portfolio(sleeves, target)
            loaded = load_portfolio(target)
        self.assertEqual([(name, weight) for name, weight, _ in loaded], [
            ("trend_core", 0.5),
            ("defensive_short", 0.5),
        ])
        self.assertFalse(loaded[1][2].use_longs)

    def test_indicators_create_boolean_signals(self) -> None:
        index = pd.date_range("2025-01-01", periods=260, freq="4h", tz="UTC")
        closes = pd.Series([100 + i * 0.05 + (i % 12) for i in range(260)], index=index)
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 1,
                "low": closes - 1,
                "close": closes,
                "volume": 1.0,
            }
        )
        output = build_feature_frame(frame, StrategyConfig())
        self.assertIn("smi", output.columns)
        self.assertEqual(output["long_signal"].dtype, bool)
        self.assertEqual(output["short_signal"].dtype, bool)

    def test_regime_filtered_strategy_requires_known_regime(self) -> None:
        with self.assertRaises(ValueError):
            build_feature_frame(
                fake_frame(),
                StrategyConfig(
                    regime_mode="avoid_risk_off_longs", regime_source="cboe_stress"
                ),
            )

    def test_breakout_entry_mode_creates_boolean_signals(self) -> None:
        output = build_feature_frame(fake_frame(), StrategyConfig(entry_mode="breakout", breakout_period=5))
        self.assertEqual(output["long_signal"].dtype, bool)
        self.assertEqual(output["short_signal"].dtype, bool)

    def test_cboe_regime_only_applies_after_publication_time(self) -> None:
        frame = fake_frame()
        regime = pd.DataFrame(
            {
                "available_time": [
                    pd.Timestamp("2025-01-01 08:00:00", tz="UTC"),
                    pd.Timestamp("2025-01-01 16:00:00", tz="UTC"),
                ],
                "risk_off": [False, True],
            }
        )
        output = attach_cboe_regime(frame, regime)
        self.assertFalse(bool(output.loc[pd.Timestamp("2025-01-01 04:00", tz="UTC"), "risk_off"]))
        self.assertTrue(bool(output.loc[pd.Timestamp("2025-01-01 16:00", tz="UTC"), "risk_off"]))

    def test_btc_momentum_regime_only_allows_ranked_risk_on_asset(self) -> None:
        index = pd.date_range("2025-01-01", periods=20, freq="4h", tz="UTC")
        universe = {
            "BTCUSDT": pd.DataFrame({"close": range(100, 120)}, index=index),
            "ETHUSDT": pd.DataFrame({"close": range(100, 140, 2)}, index=index),
        }
        output = attach_btc_momentum_regime(
            universe, btc_ema_period=5, momentum_period=5, top_n=1
        )
        self.assertTrue(bool(output["BTCUSDT"]["risk_off"].iloc[-1]))
        self.assertFalse(bool(output["ETHUSDT"]["risk_off"].iloc[-1]))

    def test_funding_timestamp_is_aligned_to_candle_boundary(self) -> None:
        frame = cached_candles("2025-01-01", 3)
        funding = pd.DataFrame(
            {"funding_rate": [0.001], "mark_price": [101.0]},
            index=[pd.Timestamp("2025-01-01 04:00:00.022", tz="UTC")],
        )
        output = attach_funding_rates(frame, funding)
        self.assertEqual(
            output.loc[pd.Timestamp("2025-01-01 04:00", tz="UTC"), "funding_rate"],
            0.001,
        )
        self.assertEqual(
            output.loc[pd.Timestamp("2025-01-01 04:00", tz="UTC"), "funding_mark_price"],
            101.0,
        )

    @patch("smi_lab.backtest.build_feature_frame", side_effect=signaled_features)
    def test_trade_has_three_profit_targets(self, _: object) -> None:
        config = StrategyConfig(
            stop_atr=1.0, fee_bps=0.0, slippage_bps=0.0, cooldown_bars=0
        )
        result = backtest(fake_frame(), config, initial_equity=10_000)
        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["tp_count"], 3)
        self.assertIn("TP1 / TP2 / TP3", trade["exit_reason"])
        self.assertGreater(trade["pnl"], 0)

    @patch("smi_lab.backtest.build_feature_frame", side_effect=signaled_features)
    def test_long_position_pays_positive_funding_rate(self, _: object) -> None:
        frame = fake_frame()
        frame["funding_rate"] = 0.0
        frame["funding_mark_price"] = 200.0
        frame.iloc[3, frame.columns.get_loc("funding_rate")] = 0.001
        config = StrategyConfig(
            stop_atr=1.0, fee_bps=0.0, slippage_bps=0.0, cooldown_bars=0
        )
        result = backtest(frame, config, initial_equity=10_000)
        self.assertLess(result.trades.iloc[0]["funding_pnl"], 0.0)
        self.assertLess(result.metrics["funding_pnl"], 0.0)
        self.assertAlmostEqual(result.trades.iloc[0]["funding_pnl"], -20.0)

    @patch("smi_lab.backtest.build_feature_frame", side_effect=signaled_features)
    def test_stop_has_priority_on_ambiguous_entry_candle(self, _: object) -> None:
        config = StrategyConfig(
            stop_atr=1.0, fee_bps=0.0, slippage_bps=0.0, cooldown_bars=0
        )
        frame = fake_frame(entry_bar_low=98.0)
        frame.iloc[2, frame.columns.get_loc("high")] = 104.0
        result = backtest(frame, config, initial_equity=10_000)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["exit_reason"], "SL")
        self.assertEqual(trade["tp_count"], 0)
        self.assertLess(trade["pnl"], 0)

    @patch("smi_lab.backtest.build_feature_frame", side_effect=signaled_features)
    def test_portfolio_backtest_blends_two_sleeves(self, _: object) -> None:
        config = StrategyConfig(
            stop_atr=1.0, fee_bps=0.0, slippage_bps=0.0, cooldown_bars=0
        )
        result = backtest_portfolio(
            {"BTCUSDT": fake_frame()},
            [("a", 0.5, config), ("b", 0.5, config)],
        )
        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.metrics["profitable_symbols_pct"], 100.0)
        self.assertGreater(result.metrics["return_pct"], 0.0)

    @patch("smi_lab.backtest.build_feature_frame", side_effect=signaled_features)
    def test_ranked_long_uses_shared_account_for_single_selected_asset(self, _: object) -> None:
        config = StrategyConfig(
            stop_atr=1.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            cooldown_bars=0,
            use_shorts=False,
        )
        result = backtest_ranked_long({"BTCUSDT": fake_frame()}, config)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["symbol"], "BTCUSDT")
        self.assertGreater(result.metrics["return_pct"], 0.0)

    @patch("smi_lab.data.fetch_klines")
    def test_short_refresh_preserves_longer_history_cache(self, download: object) -> None:
        download.side_effect = [
            cached_candles("2025-01-01", 20),
            cached_candles("2025-01-04", 5),
        ]
        with TemporaryDirectory() as directory:
            get_klines("BTCUSDT", bars=20, cache_dir=directory, refresh=True)
            recent = get_klines("BTCUSDT", bars=5, cache_dir=directory, refresh=True)
            saved = pd.read_csv(f"{directory}/BTCUSDT_4h.csv")
        self.assertEqual(len(recent), 5)
        self.assertGreater(len(saved), 20)

    @patch("smi_lab.data.fetch_klines")
    def test_mixed_timestamp_cache_loads_after_refresh(self, download: object) -> None:
        download.return_value = cached_candles("2025-01-02", 5)
        with TemporaryDirectory() as directory:
            target = f"{directory}/BTCUSDT_4h.csv"
            prior = cached_candles("2025-01-01", 6).reset_index()
            prior.loc[0, "close_time"] = "2025-01-01 03:59:59+00:00"
            prior.to_csv(target, index=False)
            refreshed = get_klines("BTCUSDT", bars=5, cache_dir=directory, refresh=True)
        self.assertEqual(len(refreshed), 5)

    def test_rotation_trade_records_initial_risk_and_exit_targets(self) -> None:
        index = pd.date_range("2025-01-01", periods=20, freq="4h", tz="UTC")
        btc_close = pd.Series(range(100, 120), index=index, dtype=float)
        eth_close = pd.Series(range(100, 140, 2), index=index, dtype=float)
        universe = {
            "BTCUSDT": pd.DataFrame(
                {
                    "open": btc_close,
                    "high": btc_close + 1,
                    "low": btc_close - 1,
                    "close": btc_close,
                    "volume": 1.0,
                }
            ),
            "ETHUSDT": pd.DataFrame(
                {
                    "open": eth_close,
                    "high": eth_close + 2,
                    "low": eth_close - 1,
                    "close": eth_close,
                    "volume": 1.0,
                }
            ),
        }
        result = backtest_rotation(
            universe,
            RotationConfig(
                btc_ema_period=5,
                momentum_period=5,
                rebalance_bars=1,
                atr_period=2,
                stop_atr=2.0,
                fee_bps=0.0,
                slippage_bps=0.0,
            ),
        )
        self.assertFalse(result.trades.empty)
        self.assertIn("initial_risk_pct", result.trades.columns)
        self.assertGreater(result.trades["initial_risk_pct"].iloc[0], 0.0)

    def test_rotation_rebalance_schedule_is_invariant_to_history_window(self) -> None:
        index = pd.date_range("2025-01-01", periods=80, freq="4h", tz="UTC").as_unit(
            "us"
        )
        full_schedule = _rebalance_schedule(index, 42, 17)
        shortened_schedule = _rebalance_schedule(index[13:], 42, 17)
        self.assertGreater(int(full_schedule.sum()), 0)
        pd.testing.assert_series_equal(
            full_schedule.loc[index[13:]],
            shortened_schedule,
        )

    def test_rotation_caps_initial_stop_risk(self) -> None:
        index = pd.date_range("2025-01-01", periods=30, freq="4h", tz="UTC")
        closes = pd.Series(range(100, 130), index=index, dtype=float)
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 5,
                "low": closes - 5,
                "close": closes,
                "volume": 1.0,
            }
        )
        result = backtest_rotation(
            {"BTCUSDT": frame},
            RotationConfig(
                btc_ema_period=5,
                momentum_period=5,
                rebalance_bars=1,
                atr_period=2,
                stop_atr=3.0,
                max_initial_risk_pct=5.0,
                fee_bps=0.0,
                slippage_bps=0.0,
            ),
        )
        self.assertFalse(result.trades.empty)
        self.assertLessEqual(result.trades["initial_risk_pct"].max(), 5.0 + 1e-9)

    def test_event_rotation_enters_without_waiting_for_rebalance_offset(self) -> None:
        index = pd.date_range("2025-01-01", periods=30, freq="4h", tz="UTC")
        closes = pd.Series(range(100, 130), index=index, dtype=float)
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 1,
                "low": closes - 1,
                "close": closes,
                "volume": 1.0,
            }
        )
        result = backtest_rotation(
            {"BTCUSDT": frame},
            RotationConfig(
                btc_ema_period=5,
                momentum_period=5,
                rebalance_bars=1_000,
                rebalance_offset_bars=999,
                enter_when_flat=True,
                rotate_on_rebalance=False,
                atr_period=2,
                fee_bps=0.0,
                slippage_bps=0.0,
            ),
        )
        self.assertFalse(result.trades.empty)

    def test_trend_allocation_selects_strongest_asset_at_capped_exposure(self) -> None:
        index = pd.date_range("2025-01-01", periods=30, freq="4h", tz="UTC")
        btc = pd.Series(range(100, 130), index=index, dtype=float)
        eth = pd.Series(range(100, 160, 2), index=index, dtype=float)
        universe = {
            "BTCUSDT": pd.DataFrame(
                {"open": btc, "high": btc + 1, "low": btc - 1, "close": btc}
            ),
            "ETHUSDT": pd.DataFrame(
                {"open": eth, "high": eth + 1, "low": eth - 1, "close": eth}
            ),
        }
        result = backtest_trend_allocation(
            universe,
            TrendAllocationConfig(
                momentum_period=5,
                asset_ema_period=5,
                btc_ema_period=5,
                top_n=1,
                rebalance_bars=1,
                gross_exposure=0.40,
                fee_bps=0.0,
                slippage_bps=0.0,
            ),
        )
        self.assertFalse(result.rebalances.empty)
        self.assertEqual(result.rebalances.iloc[0]["selected_symbols"], "ETHUSDT")
        self.assertEqual(result.rebalances.iloc[0]["gross_exposure"], 0.40)
        self.assertGreater(result.metrics["return_pct"], 0.0)

    def test_trend_allocation_long_position_pays_positive_funding(self) -> None:
        index = pd.date_range("2025-01-01", periods=30, freq="4h", tz="UTC")
        closes = pd.Series(range(100, 130), index=index, dtype=float)
        plain = pd.DataFrame(
            {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes}
        )
        charged = plain.copy()
        charged["funding_rate"] = 0.001
        config = TrendAllocationConfig(
            momentum_period=5,
            asset_ema_period=5,
            btc_ema_period=5,
            rebalance_bars=1,
            gross_exposure=0.40,
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        no_funding = backtest_trend_allocation({"BTCUSDT": plain}, config)
        with_funding = backtest_trend_allocation({"BTCUSDT": charged}, config)
        self.assertLess(with_funding.metrics["funding_pnl"], 0.0)
        self.assertLess(
            with_funding.metrics["return_pct"], no_funding.metrics["return_pct"]
        )

    def test_staggered_trend_allocation_blends_rebalance_offsets(self) -> None:
        index = pd.date_range("2025-01-01", periods=80, freq="4h", tz="UTC")
        price = pd.Series(range(100, 180), index=index, dtype=float)
        frame = pd.DataFrame(
            {
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "funding_rate": 0.0,
            }
        )
        config = TrendAllocationConfig(
            momentum_period=5,
            asset_ema_period=5,
            btc_ema_period=5,
            rebalance_bars=6,
            gross_exposure=0.40,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        result = backtest_staggered_trend_allocation({"BTCUSDT": frame}, config, (0, 1, 2))

        self.assertFalse(result.rebalances.empty)
        self.assertEqual(set(result.rebalances["rebalance_offset_bars"]), {0, 1, 2})
        self.assertEqual(result.metrics["schedule_sleeves"], 3.0)

    def test_technical_summary_returns_investing_style_labels(self) -> None:
        summary = summarize_technical("BTCUSDT", rising_frame())

        self.assertIn(summary.summary_action, {"Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"})
        self.assertGreater(summary.ma_buy, summary.ma_sell)
        self.assertTrue(summary.view)

    def test_allocation_snapshot_and_forward_tracking_initialize(self) -> None:
        universe = {"BTCUSDT": rising_frame(220), "ETHUSDT": rising_frame(220)}
        config = TrendAllocationConfig(
            momentum_period=5,
            asset_ema_period=5,
            btc_ema_period=5,
            rebalance_bars=6,
            gross_exposure=0.30,
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        snapshot = allocation_snapshot(universe, config, (0, 1, 2))
        aggregate = aggregate_snapshot(snapshot)

        self.assertFalse(snapshot.empty)
        self.assertIn("CASH", set(aggregate["asset"]))
        with TemporaryDirectory() as directory:
            strategy_path = f"{directory}/strategy.json"
            state_path = f"{directory}/state.json"
            with open(strategy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "name": "test",
                        "status": "shadow_only",
                        "config": config.__dict__,
                        "rebalance_offsets": [0, 1, 2],
                    },
                    handle,
                )
            update = update_forward_tracking(
                universe,
                strategy_path=strategy_path,
                output_dir=directory,
                state_path=state_path,
            )
        self.assertEqual(update.status, "initialized")
        self.assertEqual(update.return_pct, 0.0)

    def test_equity_selection_ranks_stronger_trending_stock(self) -> None:
        index = pd.date_range("2025-01-01", periods=260, freq="1D", tz="UTC")
        market = pd.Series(np.linspace(100, 140, len(index)), index=index)
        strong = pd.Series(np.linspace(100, 220, len(index)), index=index)
        weak = pd.Series(np.linspace(100, 110, len(index)), index=index)
        universe = {
            "SPY": pd.DataFrame({"open": market, "high": market + 1, "low": market - 1, "close": market}, index=index),
            "STRONG": pd.DataFrame({"open": strong, "high": strong + 1, "low": strong - 1, "close": strong}, index=index),
            "WEAK": pd.DataFrame({"open": weak, "high": weak + 1, "low": weak - 1, "close": weak}, index=index),
        }
        config = EquitySelectionConfig(
            market_symbol="SPY",
            top_n=1,
            rebalance_bars=20,
            short_momentum_period=20,
            long_momentum_period=60,
            trend_period=80,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        ranking = rank_equities(universe, config)
        result = backtest_equity_selection(universe, config)
        benchmark = benchmark_buy_and_hold(universe["SPY"], fee_bps=0.0, slippage_bps=0.0)

        self.assertEqual(ranking.iloc[0]["symbol"], "STRONG")
        self.assertGreater(result.metrics["return_pct"], benchmark.metrics["return_pct"])
        self.assertFalse(result.rebalances.empty)

    def test_attention_features_detect_nonfinancial_attention_spike(self) -> None:
        dates = pd.date_range("2025-01-01", periods=120, freq="1D", tz="UTC")
        mentions = (1.0 + (np.arange(len(dates)) % 3)).astype(float)
        mentions[-7:] = 18.0
        attention = pd.DataFrame(
            {
                "date": dates,
                "symbol": "ELF",
                "company": "e.l.f. Beauty",
                "category": "beauty",
                "source": "test",
                "mentions": mentions,
                "norm": np.nan,
            }
        )
        config = AttentionConfig(
            spike_lookback_days=30,
            recent_days=7,
            min_recent_mentions=3.0,
            min_spike_z=1.0,
        )

        features = attention_features(attention, config)
        latest = features[features["symbol"] == "ELF"].iloc[-1]

        self.assertGreater(float(latest["recent_mentions"]), 100.0)
        self.assertGreater(float(latest["spike_z"]), 1.0)
        self.assertGreater(float(latest["attention_score"]), 1.0)

    def test_attention_backtest_rotates_into_prior_day_top_signal(self) -> None:
        index = pd.date_range("2025-01-01", periods=80, freq="1D", tz="UTC")

        def daily_frame(values: np.ndarray) -> pd.DataFrame:
            series = pd.Series(values, index=index)
            return pd.DataFrame(
                {
                    "open": series,
                    "high": series + 1.0,
                    "low": series - 1.0,
                    "close": series,
                    "volume": 1000.0,
                },
                index=index,
            )

        universe = {
            "SPY": daily_frame(np.linspace(100, 100, len(index))),
            "ELF": daily_frame(np.linspace(100, 150, len(index))),
            "ULTA": daily_frame(np.linspace(100, 95, len(index))),
        }
        features = pd.DataFrame(
            [
                {
                    "date": date,
                    "symbol": symbol,
                    "company": "e.l.f. Beauty" if symbol == "ELF" else "Ulta Beauty",
                    "category": "beauty",
                    "recent_mentions": 25.0 if symbol == "ELF" else 2.0,
                    "baseline_mentions": 5.0,
                    "spike_z": 3.0 if symbol == "ELF" else 0.2,
                    "attention_growth_pct": 400.0 if symbol == "ELF" else 0.0,
                    "attention_score": 4.4 if symbol == "ELF" else 0.2,
                }
                for date in index
                for symbol in ["ELF", "ULTA"]
            ]
        )
        config = AttentionConfig(
            top_n=1,
            rebalance_days=5,
            spike_lookback_days=10,
            recent_days=3,
            min_recent_mentions=3.0,
            min_spike_z=1.0,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        result = backtest_attention_strategy(universe, features, config)

        self.assertFalse(result.events.empty)
        self.assertIn("ELF", result.events.iloc[0]["selected_symbols"])
        self.assertGreater(result.metrics["return_pct"], 0.0)

    def test_score_weighting_allocates_more_to_higher_score(self) -> None:
        ranking = pd.DataFrame(
            [
                {"symbol": "LEADER", "eligible": True, "score": 90.0},
                {"symbol": "RUNNER", "eligible": True, "score": 45.0},
                {"symbol": "LAGGARD", "eligible": True, "score": 15.0},
            ]
        )
        config = EquitySelectionConfig(
            market_symbol="SPY",
            top_n=3,
            weighting_method="score",
        )
        weights = target_weights_from_ranking(ranking, config)

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["LEADER"], weights["RUNNER"])
        self.assertGreater(weights["RUNNER"], weights["LAGGARD"])

    def test_capped_score_weighting_limits_concentration(self) -> None:
        ranking = pd.DataFrame(
            [
                {"symbol": "LEADER", "eligible": True, "score": 900.0},
                {"symbol": "RUNNER", "eligible": True, "score": 50.0},
                {"symbol": "LAGGARD", "eligible": True, "score": 10.0},
            ]
        )
        config = EquitySelectionConfig(
            market_symbol="SPY",
            top_n=3,
            weighting_method="capped_score",
            min_position_weight=0.20,
            max_position_weight=0.40,
        )
        weights = target_weights_from_ranking(ranking, config)

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertLessEqual(weights["LEADER"], 0.40)
        self.assertGreaterEqual(weights["LAGGARD"], 0.20)

    def test_equity_trade_plan_adds_company_without_unbacktested_levels(self) -> None:
        index = pd.date_range("2025-01-01", periods=260, freq="1D", tz="UTC")
        market = pd.Series(np.linspace(100, 140, len(index)), index=index)
        strong = pd.Series(np.linspace(100, 220, len(index)), index=index)
        universe = {
            "SPY": pd.DataFrame({"open": market, "high": market + 1, "low": market - 1, "close": market}, index=index),
            "AAPL": pd.DataFrame({"open": strong, "high": strong + 1, "low": strong - 1, "close": strong}, index=index),
        }
        config = EquitySelectionConfig(
            market_symbol="SPY",
            top_n=1,
            rebalance_bars=20,
            short_momentum_period=20,
            long_momentum_period=60,
            trend_period=80,
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        ranking = add_company_names(rank_equities(universe, config))
        plan = build_equity_trade_plan("AAPL", universe, config, ranking)

        self.assertIn("company", ranking.columns)
        self.assertEqual(plan.action, "ROTATION_REBALANCE")
        self.assertIsNone(plan.entry_price)
        self.assertIsNone(plan.stop_loss)
        self.assertIsNone(plan.take_profit_1)
        self.assertIsNone(plan.take_profit_2)
        self.assertIsNone(plan.risk_reward_1)
        self.assertIsNone(plan.risk_reward_2)
        self.assertIsNone(plan.strategy_exit)
        self.assertGreater(plan.close, 0)
        self.assertIn("score", plan.reason)
        self.assertIn("rank 1", plan.reason)

    def test_equity_scanner_builds_strategy_recommendations(self) -> None:
        index = pd.date_range("2025-01-01", periods=260, freq="1D", tz="UTC")
        market = pd.Series(np.linspace(100, 140, len(index)), index=index)
        strong = pd.Series(np.linspace(100, 230, len(index)), index=index)
        second = pd.Series(np.linspace(100, 190, len(index)), index=index)
        weak = pd.Series(np.linspace(100, 105, len(index)), index=index)
        universe = {
            "SPY": pd.DataFrame({"open": market, "high": market + 1, "low": market - 1, "close": market}, index=index),
            "AAPL": pd.DataFrame({"open": strong, "high": strong + 1, "low": strong - 1, "close": strong}, index=index),
            "MSFT": pd.DataFrame({"open": second, "high": second + 1, "low": second - 1, "close": second}, index=index),
            "WEAK": pd.DataFrame({"open": weak, "high": weak + 1, "low": weak - 1, "close": weak}, index=index),
        }
        config = EquitySelectionConfig(
            market_symbol="SPY",
            top_n=2,
            rebalance_bars=20,
            short_momentum_period=20,
            long_momentum_period=60,
            trend_period=80,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        ranking, recommendations, metrics = build_scan_recommendations(universe, config)

        self.assertEqual(recommendations["symbol"].tolist(), ["AAPL", "MSFT"])
        self.assertTrue((recommendations["action"] == "ROTATION_REBALANCE").all())
        self.assertIn("score", recommendations.columns)
        self.assertIn("reference_price", recommendations.columns)
        self.assertNotIn("risk_reward_2", recommendations.columns)
        self.assertNotIn("entry_price", recommendations.columns)
        self.assertIn("reason", ranking.columns)
        self.assertEqual(len(set(recommendations["reason"].tolist())), 2)
        self.assertIn("rank 1", recommendations.iloc[0]["reason"])
        self.assertIn("rank 2", recommendations.iloc[1]["reason"])
        self.assertIn("equity_selection_scan", set(metrics["strategy"]))
        self.assertIn("2330.TW", equity_scan_symbols("tw"))
        self.assertGreater(len(equity_scan_symbols("tw")), 100)
        self.assertGreater(len(equity_scan_symbols("us")), 140)
        self.assertIn("3661.TW", equity_scan_symbols("tw"))
        self.assertIn("PANW", equity_scan_symbols("us"))

    def test_equity_reason_explains_volatility_filter(self) -> None:
        reason = explain_equity_selection_reason(
            {
                "symbol": "2327.TW",
                "eligible": False,
                "risk_on": True,
                "above_trend": True,
                "score": 153.63,
                "short_momentum_pct": 122.38,
                "long_momentum_pct": 244.44,
                "annualized_volatility_pct": 80.81,
                "scan_rank": 35,
            },
            EquitySelectionConfig(market_symbol="0050.TW", max_volatility_pct=80.0),
            selected=False,
            rank=35,
        )

        self.assertIn("volatility 80.81% exceeds max 80.00%", reason)
        self.assertIn("rank 35", reason)

    def test_account_tables_upsert_and_append(self) -> None:
        with TemporaryDirectory() as directory:
            account_path = f"{directory}/accounts.csv"
            order_path = f"{directory}/orders.csv"
            accounts = upsert_account(
                account_path,
                AccountSnapshot(
                    account_id="pionex-live-main",
                    broker="Pionex",
                    market="crypto",
                    currency="USDT",
                    cash=1000.0,
                    equity=1000.0,
                ),
            )
            orders = append_order(
                order_path,
                OrderTracker(
                    account_id="pionex-live-main",
                    broker="Pionex",
                    market="crypto",
                    symbol="BTCUSDT",
                    company="Bitcoin",
                    side="BUY",
                    status="PLANNED",
                    quantity=0.01,
                    entry_price=100.0,
                    stop_loss=90.0,
                    take_profit_1=110.0,
                    take_profit_2=120.0,
                    strategy="test",
                ),
            )

            self.assertEqual(accounts.iloc[0]["broker"], "Pionex")
            self.assertEqual(orders.iloc[0]["status"], "PLANNED")
            self.assertEqual(list(load_table(account_path, ACCOUNT_COLUMNS).columns), ACCOUNT_COLUMNS)
            self.assertEqual(list(load_table(order_path, ORDER_COLUMNS).columns), ORDER_COLUMNS)

    def test_broker_import_normalizes_firstrade_csv(self) -> None:
        with TemporaryDirectory() as directory:
            path = f"{directory}/firstrade_positions.csv"
            pd.DataFrame(
                [
                    {
                        "Symbol": "AAPL",
                        "Description": "Apple",
                        "Quantity": "2",
                        "Average Cost": "100",
                        "Last Price": "110",
                    }
                ]
            ).to_csv(path, index=False)
            frame, report = normalize_broker_positions(path)

        self.assertEqual(report.broker, "Firstrade")
        self.assertEqual(frame.iloc[0]["symbol"], "AAPL")
        self.assertEqual(float(frame.iloc[0]["market_value"]), 220.0)

    def test_broker_sync_normalizes_cathay_csv(self) -> None:
        with TemporaryDirectory() as directory:
            import_dir = f"{directory}/imports/cathay"
            Path(import_dir).mkdir(parents=True)
            path = f"{import_dir}/國泰庫存.csv"
            pd.DataFrame(
                [
                    {
                        "股票代號": "2330",
                        "股票名稱": "台積電",
                        "庫存股數": "10",
                        "平均成本": "500",
                        "現價": "550",
                    }
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")
            positions_path = f"{directory}/positions.csv"
            positions, report = sync_broker_exports(f"{directory}/imports", positions_path)

        self.assertEqual(report.iloc[0]["broker"], "Cathay Securities")
        self.assertEqual(positions.iloc[0]["symbol"], "2330.TW")
        self.assertEqual(float(positions.iloc[0]["market_value"]), 5500.0)

    def test_rebalance_plan_generates_buy_and_sell_intents(self) -> None:
        accounts = pd.DataFrame(
            [
                {
                    "account_id": "pionex-main",
                    "broker": "Pionex",
                    "market": "crypto",
                    "currency": "USDT",
                    "cash": 8000.0,
                    "equity": 10000.0,
                }
            ]
        )
        positions = pd.DataFrame(
            [
                {
                    "account_id": "pionex-main",
                    "market": "crypto",
                    "symbol": "BTCUSDT",
                    "quantity": 0.02,
                    "current_price": 50000.0,
                    "market_value": 1000.0,
                },
                {
                    "account_id": "pionex-main",
                    "market": "crypto",
                    "symbol": "DOGEUSDT",
                    "quantity": 1000.0,
                    "current_price": 0.2,
                    "market_value": 200.0,
                },
            ]
        )
        targets = pd.DataFrame(
            [
                {"asset": "BTCUSDT", "target_weight": 0.35},
                {"asset": "CASH", "target_weight": 0.65},
            ]
        )

        plan = build_rebalance_plan(
            accounts,
            positions,
            targets,
            "crypto",
            account_id="pionex-main",
            price_lookup={"BTCUSDT": 50000.0, "DOGEUSDT": 0.2},
            min_trade_value=10.0,
        )

        btc = plan[plan["symbol"] == "BTCUSDT"].iloc[0]
        doge = plan[plan["symbol"] == "DOGEUSDT"].iloc[0]
        self.assertEqual(btc["side"], "BUY")
        self.assertAlmostEqual(float(btc["delta_value"]), 2500.0)
        self.assertAlmostEqual(float(btc["order_quantity"]), 0.05)
        self.assertEqual(doge["side"], "SELL")
        self.assertAlmostEqual(float(doge["order_quantity"]), 1000.0)

    @patch("smi_lab.price_alerts.send_discord")
    @patch("smi_lab.price_alerts._latest_price", return_value=(105.0, 106.0, 104.0))
    def test_price_alerts_trigger_entry_with_mention(self, _: object, send_discord: object) -> None:
        with TemporaryDirectory() as directory:
            recommendations = Path(directory) / "recommendations.csv"
            state = Path(directory) / "state.json"
            pd.DataFrame(
                [
                    {
                        "market": "us",
                        "symbol": "AAPL",
                        "company": "Apple",
                        "entry_price": 105.0,
                        "stop_loss": 95.0,
                        "take_profit_1": 115.0,
                        "take_profit_2": 125.0,
                        "risk_reward_1": 1.0,
                        "risk_reward_2": 2.0,
                    }
                ]
            ).to_csv(recommendations, index=False)

            events = check_equity_price_alerts(
                recommendations,
                state,
                webhook_url="https://example.invalid/webhook",
                mention="<@123>",
                notify=True,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].level, "ENTRY")
        self.assertTrue(send_discord.called)
        self.assertIn("<@123>", send_discord.call_args.args[0])

    @patch("smi_lab.price_alerts.send_discord")
    @patch("smi_lab.price_alerts._latest_price", return_value=(105.0, 106.0, 104.0))
    def test_price_alerts_dry_run_does_not_record_state(self, _: object, send_discord: object) -> None:
        with TemporaryDirectory() as directory:
            recommendations = Path(directory) / "recommendations.csv"
            state = Path(directory) / "state.json"
            pd.DataFrame(
                [
                    {
                        "market": "us",
                        "symbol": "AAPL",
                        "company": "Apple",
                        "entry_price": 105.0,
                        "stop_loss": 95.0,
                        "take_profit_1": 115.0,
                        "take_profit_2": 125.0,
                    }
                ]
            ).to_csv(recommendations, index=False)

            events = check_equity_price_alerts(
                recommendations,
                state,
                notify=False,
                record_state=False,
            )

        self.assertEqual(len(events), 1)
        self.assertFalse(state.exists())
        self.assertFalse(send_discord.called)

    @patch("smi_lab.price_alerts.send_discord")
    @patch("smi_lab.price_alerts._latest_price", return_value=(105.0, 106.0, 104.0))
    def test_price_alerts_ignore_unselected_strategy_rows(self, _: object, send_discord: object) -> None:
        with TemporaryDirectory() as directory:
            recommendations = Path(directory) / "recommendations.csv"
            state = Path(directory) / "state.json"
            pd.DataFrame(
                [
                    {
                        "market": "us",
                        "symbol": "AAPL",
                        "company": "Apple",
                        "selected": False,
                        "action": "HOLD_CASH",
                        "entry_price": 105.0,
                        "stop_loss": 95.0,
                        "take_profit_1": 115.0,
                        "take_profit_2": 125.0,
                    }
                ]
            ).to_csv(recommendations, index=False)

            events = check_equity_price_alerts(recommendations, state, notify=True)

        self.assertEqual(events, [])
        self.assertFalse(send_discord.called)

    def test_equity_live_plan_uses_close_reference_and_equal_weights(self) -> None:
        recommendations = pd.DataFrame(
            [
                {
                    "market": "us",
                    "symbol": "AAPL",
                    "company": "Apple",
                    "selected": True,
                    "action": "ROTATION_REBALANCE",
                    "close": 108.0,
                    "rank": 1,
                },
                {
                    "market": "us",
                    "symbol": "MSFT",
                    "company": "Microsoft",
                    "selected": True,
                    "action": "ROTATION_REBALANCE",
                    "close": 216.0,
                    "rank": 2,
                },
            ]
        )

        plan = build_equity_live_order_plan(
            recommendations=recommendations,
            positions=pd.DataFrame(),
            market="us",
            account_id="strategy-us-10000",
            broker="Firstrade",
            currency="USD",
            capital=10_000.0,
            min_trade_value=100.0,
        )

        self.assertEqual(plan["symbol"].tolist(), ["AAPL", "MSFT"])
        self.assertTrue((plan["side"] == "BUY").all())
        self.assertAlmostEqual(float(plan.iloc[0]["target_weight"]), 0.5)
        self.assertAlmostEqual(float(plan.iloc[0]["target_value"]), 5000.0)
        self.assertAlmostEqual(float(plan.iloc[0]["reference_price"]), 108.0)
        self.assertAlmostEqual(float(plan.iloc[0]["order_quantity"]), 5000.0 / 108.0)
        self.assertNotIn("risk_reward_2", plan.columns)

    def test_live_strategy_memory_persists_until_version_changes(self) -> None:
        with TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            plan = pd.DataFrame(
                [
                    {"symbol": "AAPL", "side": "BUY"},
                    {"symbol": "MSFT", "side": "HOLD"},
                ]
            )
            remember_live_plan(memory_path, "strategy-us-10000", "us", plan, strategy_version="v1")
            current = load_live_strategy_memory(memory_path, strategy_version="v1")
            reset = load_live_strategy_memory(memory_path, strategy_version="v2")

        self.assertEqual(current["strategy_version"], "v1")
        self.assertIn("strategy-us-10000", current["accounts"])
        self.assertEqual(reset["strategy_version"], "v2")
        self.assertEqual(reset["previous_strategy_version"], "v1")
        self.assertEqual(reset["accounts"], {})

    def test_alert_message_includes_rr_when_available(self) -> None:
        message = format_alert_message(
            AlertEvent(
                market="us",
                symbol="AAPL",
                company="Apple",
                level="TP2",
                target_price=120.0,
                last_price=121.0,
                triggered_at="2026-01-01T00:00:00+00:00",
                risk_reward=2.0,
            ),
            mention="<@123>",
        )

        self.assertIn("RR 2.00", message)
        self.assertIn("<@123>", message)

    def test_cached_market_snapshot_uses_local_crypto_cache(self) -> None:
        index = pd.date_range("2025-01-01", periods=8, freq="4h", tz="UTC", name="open_time")
        frame = pd.DataFrame(
            {
                "open_time": index,
                "open": range(100, 108),
                "high": range(101, 109),
                "low": range(99, 107),
                "close": range(100, 108),
                "volume": 1.0,
                "close_time": index + pd.Timedelta(hours=4),
            }
        )
        with TemporaryDirectory() as directory:
            target = f"{directory}/BTCUSDT_4h_perpetual.csv"
            frame.to_csv(target, index=False)
            snapshot = cached_crypto_snapshots(["BTCUSDT"], cache_dir=directory)

        self.assertEqual(snapshot.iloc[0]["symbol"], "BTCUSDT")
        self.assertGreater(float(snapshot.iloc[0]["change_pct"]), 0.0)

    @patch("smi_lab.crypto_universe._fetch_coingecko_markets")
    def test_crypto_market_cap_universe_filters_stables_and_maps_usdt(self, fetch_markets: object) -> None:
        fetch_markets.return_value = [
            {"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1},
            {"id": "tether", "symbol": "usdt", "market_cap_rank": 3},
            {"id": "wrapped-bitcoin", "symbol": "wbtc", "market_cap_rank": 14},
            {"id": "solana", "symbol": "sol", "market_cap_rank": 6},
        ]
        with TemporaryDirectory() as directory:
            symbols = crypto_scan_symbols(
                limit=10,
                refresh=True,
                cache_path=Path(directory) / "top.json",
            )

        self.assertIn("BTCUSDT", symbols)
        self.assertIn("SOLUSDT", symbols)
        self.assertNotIn("USDTUSDT", symbols)
        self.assertNotIn("WBTCUSDT", symbols)

    @patch("smi_lab.crypto_universe.get_klines")
    def test_crypto_scan_universe_records_symbol_failures(self, get_klines_mock: object) -> None:
        def loader(symbol: str, **_: object) -> pd.DataFrame:
            if symbol == "BADUSDT":
                raise RuntimeError("missing pair")
            return cached_candles("2025-01-01", 80)

        get_klines_mock.side_effect = loader
        universe, failures = load_crypto_scan_universe(
            ["BTCUSDT", "BADUSDT"],
            bars=80,
            min_bars=50,
            request_pause=0.0,
        )

        self.assertIn("BTCUSDT", universe)
        self.assertEqual(failures.iloc[0]["symbol"], "BADUSDT")

    def test_equity_symbol_news_uses_cache_only_without_network(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "us_AAPL.json"
            target.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "title": "Apple product news",
                                "source": "Cached Feed",
                                "link": "https://example.com/aapl",
                                "published_at": "2026-01-01T00:00:00+00:00",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch("smi_lab.market_info.urlopen") as mocked_urlopen:
                items = fetch_equity_symbol_news(
                    "AAPL",
                    "us",
                    cache_dir=directory,
                    cache_only=True,
                )

        self.assertEqual(items[0].title, "Apple product news")
        mocked_urlopen.assert_not_called()

    def test_equity_symbol_news_refresh_parses_rss(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return (
                    b"<?xml version='1.0'?><rss><channel><title>Mock Feed</title>"
                    b"<item><title>TXN earnings update</title>"
                    b"<link>https://example.com/txn</link>"
                    b"<pubDate>Mon, 01 Jun 2026 12:00:00 GMT</pubDate></item>"
                    b"</channel></rss>"
                )

        with TemporaryDirectory() as directory:
            with patch("smi_lab.market_info.urlopen", return_value=Response()):
                items = fetch_equity_symbol_news(
                    "TXN",
                    "us",
                    company="Texas Instruments",
                    cache_dir=directory,
                    refresh=True,
                    max_items=1,
                )
            cached = json.loads((Path(directory) / "us_TXN.json").read_text(encoding="utf-8"))

        self.assertEqual(items[0].source, "Mock Feed")
        self.assertEqual(cached["symbol"], "TXN")
        self.assertEqual(cached["items"][0]["title"], "TXN earnings update")

    def test_rotation_avoids_asset_with_excessive_recent_funding(self) -> None:
        index = pd.date_range("2025-01-01", periods=30, freq="4h", tz="UTC")
        closes = pd.Series(range(100, 130), index=index, dtype=float)
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 1,
                "low": closes - 1,
                "close": closes,
                "volume": 1.0,
                "funding_rate": 0.01,
            }
        )
        result = backtest_rotation(
            {"BTCUSDT": frame},
            RotationConfig(
                btc_ema_period=5,
                momentum_period=5,
                rebalance_bars=1,
                atr_period=2,
                funding_lookback_bars=6,
                max_cumulative_funding_rate=0.001,
                fee_bps=0.0,
                slippage_bps=0.0,
            ),
        )
        self.assertTrue(result.trades.empty)


if __name__ == "__main__":
    unittest.main()
