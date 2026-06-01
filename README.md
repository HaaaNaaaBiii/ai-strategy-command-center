# Crypto Signal Lab

BTC、ETH、DOGE、SOL 的量化策略研究與通知工具。目前前瞻通知候選仍為 SMI 組合；另研究不受 SMI 限制的大盤超額配置影子候選。資料使用 Binance USD-M 永續合約 K 線與歷史資金費率；程式不連接交易帳戶、不下單。

## 五年研究範圍

- 市場：`BTCUSDT`、`ETHUSDT`、`DOGEUSDT`、`SOLUSDT`
- 週期：`4h`
- 根數：每個市場 `10,958` 根已收盤 K 線
- 日期：`2021-05-26 16:00` 至 `2026-05-26 23:59 UTC`
- 市場資料：Binance USD-M 永續合約 K 線；持倉跨越結算時間時納入歷史資金費率
- 成本：單邊手續費 `10 bps`、單邊滑價 `5 bps`
- 執行：收盤確認訊號，下一根 K 線開盤成交；同根 K 線若同時觸及停損及止盈，保守判定停損

## 分析與優化結論

先前的低風險雙袖組合雖控制回撤，但報酬不足以作為實戰候選。本輪保留 USD-M 永續價格與資金費率模型，並搜尋 54 組受限的多方突破配置。候選只以最初 `60%` 校準期及其後 `20%` 驗證期的標準/雙倍成本最差 Sharpe 排序，最後 `20%` 僅供歷史穩健度評估。選出的突破袖只在 `BTC > EMA100` 且幣種近 `240` 根 `4h` 相對動能排名第一時啟用。

| 策略 / 成本情境 | 五年報酬 | Sharpe | 最大回撤 | 交易數 | Profit Factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 永續雙向基線 | 4.07% | 0.35 | -5.26% | 229 | 1.15 |
| 舊紙上組合 | 4.83% | 0.60 | -2.17% | 311 | 1.33 |
| 前版通知候選，標準成本 | 12.65% | 1.01 | -2.31% | 397 | 1.45 |
| 成熟度候選，標準成本 | **14.68%** | **1.23** | **-2.06%** | 406 | **1.54** |
| 成熟度候選，費用及滑價加倍 | **11.73%** | **1.00** | **-2.31%** | 405 | **1.42** |
| 成熟度候選，費用及滑價三倍 | **7.87%** | **0.69** | **-2.55%** | 406 | **1.27** |

Cboe `SPX`/`VIX` 壓力濾網候選先前在第五年為 `-1.14%`、Sharpe `-0.68`，維持拒絕狀態。新候選不使用美股濾網，而使用加密市場本身的可即時觀測狀態。

歷史成熟度通過的前瞻通知候選由三袖組成，使用約 `2%` 的總同時初始停損風險上限：

- `trend_core`：SMI 回撤雙向核心策略，配額 `15%`。
- `defensive_short`：SMI 回撤空方策略，配額 `45%`。
- `riskon_ranked_breakout_long`：BTC 上升趨勢且相對動能第一名時才啟用的 SMI 多方突破策略，配額 `40%`。

新候選在標準成本下的完整五年各市場結果：

| 市場 | 報酬 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: |
| BTCUSDT | 10.40% | 0.57 | -6.10% |
| ETHUSDT | 6.78% | 0.34 | -6.73% |
| DOGEUSDT | 26.91% | 1.46 | -4.72% |
| SOLUSDT | 14.62% | 0.76 | -5.12% |

標準成本下的年度視窗報酬為 `+2.10%`、`+1.18%`、`+3.70%`、`+4.95%`、`+2.35%`。雙倍成本下五個年度視窗仍全數為正，完整五年為 `+11.73%`；三倍成本完整五年仍為 `+7.87%`。雙倍成本下，移除任一單一幣種後完整五年報酬仍為正。

逐段前推式重選測試使用六個後續半年度區段；雙倍成本下六段皆正，合計為 `+7.94%`、Sharpe `1.12`、最大回撤 `-2.50%`。54 組參數鄰域在最後一年雙倍成本下亦全部為正。因此此版本通過「歷史成熟度」門檻，可用於鎖定規則後的前瞻訊號通知；它不是已證明的資金實盤策略。研發過程已查看全部五年資料，`2026-05-26` 後的新訊號必須完成前瞻追蹤與成交成本驗證，才可考慮極小資金試運行。

## 牛市進攻衛星研究（未通過）

為研究「BTC 牛市年度至少 `+50%` 且超越 BTC 買入持有」的額外目標，另建相對動能 rotation 影子策略。它不取代目前的 SMI 通知組合，也不啟用通知。

- 基準：含單次標準進出成本的 `BTCUSDT` 買入持有；BTC 報酬為正的年度切片視為牛市。
- 兩個 BTC 牛市年度基準分別為 `+158.61%` 與 `+58.07%`。
- 合格門檻：標準、雙倍、三倍成本下的每個 BTC 牛市年度皆須至少 `+50%`，且至少領先 BTC `5` 個百分點；所有年度與完整五年的最大回撤不得低於 `-30%`，估算初始止損風險不得超過 `15%`。
- 執行校正：rotation 交易與組合資金再平衡均以固定 UTC 時點錨定，不可依載入資料窗的第一列改變排程。校正前 `outputs/bull_filtered/` 的結果已被取代，不得作策略判斷。
- 每筆 rotation 交易仍包含 `3 ATR` 止損及 `3R / 8R / 20R` 三段止盈。

| 校正後研究 | 候選數 | 合格數 | 三倍成本牛市年 1 | 三倍成本牛市年 2 | 三倍成本五年報酬 | 最差最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定排程 + 組合風控 | 1,296 | 0 | - | - | - | - |
| 兩模型參數分散，最佳但不合格 | 1,053 | 0 | +116.73% | +49.65% | +148.80% | -27.92% |
| 預定等權參數籃子，最佳但不合格 | 3 | 0 | +113.50% | +54.35% | +160.13% | -30.24% |
| 事件驅動 rotation，最佳但不合格 | 864 | 0 | +40.83% | -9.22% | -2.74% | -21.68% |

排程敏感度測試另掃描 `42` 個固定 `4h` 相位；將交易與組合資金再平衡都固定到 UTC 後，合格數為 `0`。最接近的相位雖有正超額，但最小值僅 `+1.23` 個百分點，未達要求的 `+5`；預先指定的每日、每半日及全相位分散籃子也全部未通過。這表示曾出現的高報酬依賴不可接受的排程相位選擇，不能視為實戰 alpha。若將「大盤」定義為四幣等權買入持有而非 BTC，第三年基準為 `+288.18%`，門檻只會更嚴格。

結論：牛市進攻衛星目前拒絕啟用；資金部署仍僅能考慮已通過歷史成熟度檢查的低風險 SMI 三袖組合。大盤超額配置候選已可做影子通知與紙上追蹤，但尚不可視為實盤策略。

## 大盤超額配置研究

在取消必須使用 SMI 的限制後，主要比較基準定義為 `BTCUSDT / ETHUSDT / DOGEUSDT / SOLUSDT` 靜態等權買入持有；另以 `BTCUSDT` 買入持有作更嚴格的次要比較。基準僅計入一次進場及一次出場的標準成本，策略壓力測試則計入永續資金費率與三倍交易成本。

新影子候選為無槓桿 long/cash 趨勢與相對強弱配置：

- 將資金等分為 `42` 個 sleeve，覆蓋每週 `42` 根 `4h` K 線的全部固定 UTC 換倉相位；各 sleeve 每週僅在自身相位調整，避免挑選有利換倉時點。
- `BTCUSDT` 收盤必須高於 `EMA(100)` 才可持有風險資產；候選幣須高於自身 `EMA(42)` 且 `180` 根動能為正。
- 通過閘門時持有動能排名第一的幣；否則持有現金。總風險資產曝險上限 `35%`，不使用槓桿。
- 本次固定網格選型只使用前 `60%` calibration 及接續 `20%` validation；最後 `20%` holdout 在此輪選定後檢驗。但前期迭代已查看過同一段五年歷史，因此 holdout 不可當作完全未觀測的外樣本。

選定候選 `mom180_assetema42_btcema100_top1_reb42_exp0.35` 在三倍交易成本壓力情境下的結果如下：

| 區段 | 新策略 | 四幣等權大盤 | BTC 買入持有 | 策略最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| Calibration | +118.39% | +114.67% | +79.74% | -26.77% |
| Validation | +35.89% | +15.87% | +58.07% | -19.35% |
| Holdout | -13.49% | -39.80% | -31.18% | -20.94% |
| Evaluation（後 40%） | +17.97% | -31.90% | +9.11% | -25.47% |
| 五年全期 | **+157.44%** | **+37.45%** | **+96.70%** | **-26.77%** |

此候選在所有分段及全期均超越主要的四幣等權大盤，並在全期與 holdout 超越 BTC；但 validation 區段未超越 BTC。它目前標示為 `shadow_tracking`：通知器與前瞻紙上追蹤已接通，但仍不能視為已可資金部署的策略。

### 實盤缺口與門檻

加密配置策略目前仍不是實盤策略，主要缺口已轉成程式化檢查：

- 前瞻紙上追蹤至少 `30` 天。
- 紙上追蹤需勝過四幣等權大盤。
- 紙上回撤不得低於 `-10%` 的 live-readiness guardrail。
- 最新市場資料延遲不得超過 `12` 小時。
- 至少累積 `3` 次前瞻再平衡事件，確認換倉與通知行為。
- 尚未接交易所下單、倉位 reconciliation、API key 權限隔離、最小下單量與失敗重試，因此即使通過紙上門檻，也只能進入極小資金試運行評估。

目前前瞻追蹤會同時輸出策略、四幣等權大盤與 BTC 基準，並在通知文字中列出 `Live ready` 與阻擋原因。

## 台美股選股策略

台股與美股新增共用的 long-only 選股架構。這不是實盤通知策略，目前定位為研究與紙上追蹤候選。

- 市場閘門：台股使用 `0050.TW`，美股使用 `SPY`；大盤高於趨勢線才允許持股。
- 選股分數：長動能、短動能、是否高於趨勢線、年化波動懲罰。
- 台股預設：top 3、每 `40` 交易日再平衡、`63/60` 日動能、`EMA100` 趨勢、基準 `0050.TW`。
- 美股預設：top 3、每 `20` 交易日再平衡、`40/126` 日動能、`EMA200` 趨勢、基準 `SPY`。
- 預設觀察清單：台股 `2330.TW / 2317.TW / 2454.TW / 2308.TW / 2603.TW`；美股 `AAPL / MSFT / NVDA / TSLA / AMZN / META / GOOGL`。

使用兩年 Yahoo chart 資料的目前檢查結果：

| 市場 | 選股策略 | 基準 | 策略最大回撤 | 基準最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| 台股 | +300.69% | +149.14% | -13.86% | -28.47% |
| 美股 | +56.32% | +44.11% | -14.09% | -19.00% |

這些結果仍有明顯樣本限制：觀察清單很小，且尚未納入完整市場成分股、流動性、除權息、財報事件與交易稅細節。下一階段應先擴大股票池並做前瞻紙上追蹤。

## 訊號規則

每個袖都使用 SMI 動能、EMA 趨勢與 ADX 強度過濾，並且每筆訊號皆含三段止盈與一個停損。

| 袖 | 方向 | SMI 設定 | 趨勢/強度 | 停損 | TP1 / TP2 / TP3 |
| --- | --- | --- | --- | --- | --- |
| `trend_core` | 多空 | `SMI(28,3,5)`, signal `7`, 回撤交叉 | `EMA(150)`, `ADX >= 18` | `2.2 ATR` | `1R / 2.2R / 3.5R` |
| `defensive_short` | 只空 | `SMI(36,7,3)`, signal `5`, 回撤交叉 | `EMA(100)`, `ADX >= 18` | `2.6 ATR` | `1.2R / 2.4R / 3.6R` |
| `riskon_ranked_breakout_long` | 只多 | `SMI(20,3,5)`, signal `7`, `40` 根突破確認 | `EMA(200)`, `ADX >= 18`; BTC/240 根相對動能閘門 | `3 ATR` | `1.5R / 3R / 6R` |

- TP1 出場 `40%`，TP2 出場 `35%`，TP3 出場 `25%`。
- 觸及 TP1 後停損移至進場價；觸及 TP2 後對剩餘部位使用 ATR 追蹤停損。
- 每袖以 `2%` 內部停損風險設定並按配額混合；四市場與全部袖同時觸發時，組合初始停損風險目標約 `2%`。此風控仍需要以實際交易所槓桿與保證金規則另行核對。

## 執行

儀表板目前以部署組合進行通知掃描：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

跨平台手機版使用同一個 Streamlit web app。區網手機開啟 Streamlit 顯示的 Network URL；部署到 VPS 或 Streamlit Cloud 後即可從外網手機使用。完整說明見 `APP_GUIDE.md`。

重跑永續合約、資金費率及 Cboe 狀態候選搜尋：

```powershell
.\.venv\Scripts\python.exe run_research.py --years 5 --interval 4h --market perpetual --cboe-regime-search --candidates 320 --shortlist 40 --refresh --baseline-config outputs\baseline_strategy.json --output-dir outputs\futures_regime
.\.venv\Scripts\python.exe refine_portfolio.py --research-dir outputs\futures_regime --baseline-config outputs\baseline_strategy.json
.\.venv\Scripts\python.exe research_practical.py
.\.venv\Scripts\python.exe research_maturity.py
.\.venv\Scripts\python.exe research_bull_offense.py
.\.venv\Scripts\python.exe research_bull_rotation.py
.\.venv\Scripts\python.exe research_bull_combined.py
.\.venv\Scripts\python.exe research_bull_filtered.py
.\.venv\Scripts\python.exe research_bull_portfolio_brake.py
.\.venv\Scripts\python.exe research_bull_schedule.py
.\.venv\Scripts\python.exe research_bull_basket.py
.\.venv\Scripts\python.exe research_bull_ensemble.py
.\.venv\Scripts\python.exe research_bull_event.py
.\.venv\Scripts\python.exe research_market_alpha.py
.\.venv\Scripts\python.exe research_market_alpha_staggered.py
.\.venv\Scripts\python.exe research_equity_selection.py --market both --range 2y --refresh
.\.venv\Scripts\python.exe track_paper.py --refresh
```

掃描最新已收盤 K 線訊號：

```powershell
.\.venv\Scripts\python.exe scan_signals.py --interval 4h --bars 500 --refresh
.\.venv\Scripts\python.exe scan_signals.py --strategy allocation --interval 4h --bars 1200 --refresh
```

Discord 通知：

```powershell
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
.\.venv\Scripts\python.exe scan_signals.py --refresh --channel discord
```

Telegram 通知：

```powershell
$env:TELEGRAM_BOT_TOKEN = "..."
$env:TELEGRAM_CHAT_ID = "..."
.\.venv\Scripts\python.exe scan_signals.py --refresh --channel telegram
```

通知訊息會標示 `trend_core`、`defensive_short` 或 `riskon_ranked_breakout_long` 以及風險配額。同一袖、同一根 K 線、同一幣種及方向的訊號只會發送一次。

## 主要輸出

- `outputs/maturity_candidate/paper_portfolio.json`：目前供前瞻訊號通知使用、歷史成熟度通過的永續三袖候選。
- `outputs/maturity_candidate/early_period_screen.csv`：僅使用前 `80%` 兩階段資料的 54 組候選排序。
- `outputs/maturity_candidate/phase_metrics.csv`：標準、雙倍及三倍成本下的階段績效。
- `outputs/maturity_candidate/walk_forward_aggregate.csv`：逐段前推式重選的彙總績效。
- `outputs/maturity_candidate/leave_one_symbol_out.csv`：排除單一幣種壓力測試。
- `outputs/maturity_candidate/holdout_parameter_sensitivity.csv`：參數鄰域的最後一年敏感度。
- `outputs/maturity_candidate/metadata.json`：成熟度門檻判斷與未進行資金部署的原因。
- `outputs/bull_portfolio_brake/candidate_screen.csv`：固定 UTC 排程與嚴格風控下的進攻候選篩選，合格數為零。
- `outputs/bull_schedule/single_offset_screen.csv`：全部固定換倉相位的敏感度檢查。
- `outputs/bull_schedule/staggered_summary.csv`：預定分散換倉相位的失敗結果。
- `outputs/bull_ensemble/selected_annual_metrics.csv`：校正後兩模型分散策略的年度壓力測試。
- `outputs/bull_basket/basket_summary.csv`：校正後預定等權參數籃子檢查。
- `outputs/bull_event/selected_annual_metrics.csv`：無交易相位依賴的事件驅動版本年度結果。
- `outputs/market_alpha_staggered/candidate_screen.csv`：全 `42` 相位配置在三倍交易成本下的選型篩選。
- `outputs/market_alpha_staggered/selected_metrics.csv`：影子候選於標準、雙倍及三倍成本下的各區段績效。
- `outputs/market_alpha_staggered/selected_benchmark_comparison.csv`：候選對四幣等權大盤及 BTC 的比較。
- `outputs/market_alpha_staggered/metadata.json`：基準定義、選型流程、影子通知與部署限制。
- `outputs/forward_tracking/market_alpha_staggered_status.json`：大盤超額配置候選的前瞻紙上追蹤狀態。
- `outputs/forward_tracking/market_alpha_staggered_equity.csv`：紙上帳戶權益曲線。
- `outputs/forward_tracking/market_alpha_staggered_events.csv`：紙上再平衡事件紀錄。
- `outputs/forward_tracking/market_alpha_staggered_forward_benchmarks.csv`：前瞻追蹤相對四幣等權與 BTC 的基準比較。
- `outputs/equity_selection/tw_ranking.csv`、`outputs/equity_selection/us_ranking.csv`：台美股目前選股排名。
- `outputs/equity_selection/tw_metrics.csv`、`outputs/equity_selection/us_metrics.csv`：台美股選股策略與基準比較。

## 資料/API 文件

- [Binance USD-M Futures Kline API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data)
- [Binance USD-M Futures Funding Rate API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
- [Cboe Index Data](https://www.cboe.com/us/indices/market_statistics/historical_data/)
- [Discord Incoming Webhooks](https://docs.discord.com/developers/platform/webhooks)
- [Telegram Bot API](https://core.telegram.org/bots/api#sendmessage)
