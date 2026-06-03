# 掃盤作業手冊

更新日期：2026-06-02

## 原則

掃盤的角色是產生「策略候選與再平衡意圖」，不是直接下單。

候選池來源與策略推薦必須分開：

- 候選池：由市值、流動性、資料可得性建立。
- 策略推薦：由 market gate、動能、趨勢、波動與排名決定。

因此 TopN 推薦不是市值排行，也不是固定 watchlist。

## 加密掃盤

### Universe

來源是 CoinGecko 市值前 100：

1. 抓取 USD market-cap 排名。
2. 排除穩定幣、包裝幣、RWA/美元收益型代幣。
3. 將剩餘 symbol 映射成 `USDT` 交易對。
4. 實際載入 Binance USD-M、Bybit linear 或 Binance spot fallback 的 K 線。
5. 載入失敗只記錄 failure，不中斷整次掃盤。

目前實測：過濾後可掃 70 檔，失敗 0 檔。

### 頻率

每 4 小時一次，配合加密策略的 4h candle。

### 手動命令

```powershell
.\.venv\Scripts\python.exe scan_signals.py --strategy allocation --crypto-universe top100 --crypto-limit 100 --interval 4h --bars 1200 --refresh
```

強制舊核心四幣：

```powershell
.\.venv\Scripts\python.exe scan_signals.py --strategy allocation --crypto-universe core --interval 4h --bars 1200 --refresh
```

自訂幣種：

```powershell
.\.venv\Scripts\python.exe scan_signals.py --strategy allocation --crypto-universe symbols --symbols BTCUSDT ETHUSDT SOLUSDT --interval 4h --bars 1200 --refresh
```

### 主要輸出

- `outputs/forward_tracking/market_alpha_staggered_status.json`
- `outputs/forward_tracking/market_alpha_staggered_equity.csv`
- `outputs/forward_tracking/market_alpha_staggered_events.csv`
- `data/crypto_universe/coingecko_top_markets.json`

`data/` 與 `outputs/` 不進 Git。

## 台股掃盤

### Universe

目前 137 檔大型與中大型流動台股，定義在 `smi_lab/equity_universe.py`。

Gate / Benchmark：`0050.TW`。

### 策略設定

- Top N：3
- 再平衡：40 日
- 短動能：40 日
- 長動能：60 日
- 趨勢線：EMA200
- 最大年化波動：80%
- 成本：14.25 bps commission proxy + 5 bps slippage

### 頻率

台灣時間週一到週五 13:35，收盤後掃描。

不使用每小時掃盤作為正式策略輸出，因為目前策略是 daily bar 回測。

### 手動命令

```powershell
.\.venv\Scripts\python.exe scan_equity_signals.py --market tw --interval 1d --range 2y --refresh --channel discord
```

### 主要輸出

- `outputs/equity_scan/tw_recommendations.csv`
- `outputs/equity_scan/tw_scan_ranking.csv`
- `outputs/equity_scan/tw_scan_metrics.csv`
- `outputs/equity_scan/tw_scan_failures.csv`
- `outputs/equity_scan/tw_scan_summary.json`

## 美股掃盤

### Universe

目前 155 檔美股大型、半導體、軟體、金融、醫療、工業與消費龍頭，定義在 `smi_lab/equity_universe.py`。

Gate / Benchmark：`SPY`。

### 策略設定

- Top N：3
- 再平衡：40 日
- 短動能：63 日
- 長動能：126 日
- 趨勢線：EMA200
- 最大年化波動：80%
- 成本：1 bps commission proxy + 3 bps slippage

### 頻率

台灣時間週二到週六 07:30，美股收盤後掃描。

### 手動命令

```powershell
.\.venv\Scripts\python.exe scan_equity_signals.py --market us --interval 1d --range 2y --refresh
```

### 主要輸出

- `outputs/equity_scan/us_recommendations.csv`
- `outputs/equity_scan/us_scan_ranking.csv`
- `outputs/equity_scan/us_scan_metrics.csv`
- `outputs/equity_scan/us_scan_failures.csv`
- `outputs/equity_scan/us_scan_summary.json`

## 全市場股市掃盤

```powershell
.\.venv\Scripts\python.exe scan_equity_signals.py --market both --interval 1d --range 2y --refresh
```

合併輸出：

- `outputs/equity_scan/latest_recommendations.csv`
- `outputs/equity_scan/latest_scan_summary.json`

Dashboard 與 Live Desk 會讀取這些最新輸出。

## 新聞資料

新聞只提供參考，不改變策略分數、權重或下單意圖。

目前包含：

- 市場新聞：加密、台股、美股分頁。
- 策略選股相關新聞：依 TopN 選股抓取個股新聞。

主要快取：

- `outputs/news/crypto_news.json`
- `outputs/news/tw_news.json`
- `outputs/news/us_news.json`
- `outputs/news/equity_symbols/`

## 通知規則

已暫停：

- 舊版 entry / stop / TP 價格層級提醒。

應改成：

- 新標的進入 TopN。
- 原持倉掉出 TopN。
- 目標權重變化超過門檻。
- market gate 轉 risk-off。
- 掃盤資料異常，例如失敗數突然升高。

## 資料問題處理

加密：

- Binance futures HTTP 451 時，改走 Bybit linear。
- Bybit 不可用時，最後 fallback 到 Binance spot。
- top100 裡沒有可用 K 線的標的會記錄為 failures。

股票：

- Yahoo 429 時優先用 cache。
- daily bar 無 cache 時 fallback 到 Stooq。
- Yahoo/Stooq 都不可用的標的會記錄到 scan failures，不中斷掃盤。

## 驗證清單

每次改掃盤或策略後至少跑：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

策略有變更時也要：

1. 重跑對應市場掃盤。
2. 更新 `APP_GUIDE.md` 與 `docs/equity_strategy_review.md`。
3. 若 Live Desk 策略含義改變，更新 `LIVE_STRATEGY_VERSION`。
4. Commit 並 push。
