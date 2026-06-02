# AI Strategy Command Center Project

更新日期：2026-06-02

## 專案定位

這個專案把目前聊天中討論過的加密貨幣、台股、美股策略，整理成一個可持續維護的量化策略與通知系統。

核心目標：

- 每日或固定週期掃描加密、台股、美股市場。
- 用經過回測的策略產生 TopN 推薦與再平衡意圖。
- 前端用 Streamlit 提供跨平台 Dashboard、策略頁、帳戶追蹤與研究紀錄。
- Discord 用於策略通知與未來再平衡事件提醒。
- 實盤頁目前只產生 order intents，不自動下單。

## 目前結論

1. 選股不是單純市值排行。市值或流動性只用來建立候選池，最終推薦由策略排名決定。
2. 股市策略目前是動態輪動/再平衡模型，不使用未回測的 entry、TP、stop、RR。
3. 加密策略目前是 4h 趨勢/動能配置模型，支援 cash/no-trade 狀態。
4. 實盤策略需要記憶，記憶存在 `outputs/equity_live/live_strategy_memory.json`，不進 Git。策略版本改變時才重置。
5. 掃盤範圍已擴大：加密以市值前百為來源，台股 137 檔，美股 155 檔。

## 最新策略狀態

### 加密

- Universe：CoinGecko 市值前 100，排除穩定幣、包裝幣、RWA/美元收益型代幣，再映射成可用 USDT 交易對。
- 實測：市值前百過濾後可掃 70 檔，失敗 0 檔。
- 頻率：每 4 小時掃描一次。
- 策略：趨勢與相對動能輪動配置；若無合格標的，持有現金。

### 台股

- Universe：137 檔大型與中大型流動台股。
- Gate / Benchmark：0050.TW。
- 策略：Top3、40 日再平衡、40/60 日動能、EMA200 趨勢。
- 最新 2 年回測：策略 +472.23%，0050.TW +149.73%，最大回撤 -19.35%。
- 最新掃盤 Top3：2327.TW、2356.TW、2383.TW。

### 美股

- Universe：155 檔大型、半導體、軟體、金融、醫療、工業、消費龍頭。
- Gate / Benchmark：SPY。
- 策略：Top3、40 日再平衡、63/126 日動能、EMA200 趨勢。
- 最新 2 年回測：策略 +209.40%，SPY +43.27%，最大回撤 -20.54%。
- 最新掃盤 Top3：MRVL、HPE、PANW。

## 主要文件

- `APP_GUIDE.md`：Streamlit app、部署、資料來源、帳戶追蹤、通知與操作指南。
- `docs/equity_strategy_review.md`：台美股策略、回測、實盤邊界與為什麼移除 entry/TP/stop/RR。
- `docs/scan_operations.md`：加密、台股、美股掃盤規則、排程、命令與輸出檔。

## 主要程式模組

- `app.py`：Streamlit 前端。
- `scan_signals.py`：加密訊號與 allocation 掃描。
- `scan_equity_signals.py`：台股/美股策略掃盤。
- `smi_lab/crypto_universe.py`：加密市值前百 universe 建立與容錯載入。
- `smi_lab/equity_universe.py`：台股/美股掃盤 universe。
- `smi_lab/equity_strategy.py`：台股/美股 ranking、回測、權重與預設策略。
- `smi_lab/equity_live.py`：Live Desk 實盤意圖與策略記憶版本。
- `smi_lab/equity_scanner.py`：股票掃盤、ranking、recommendations、metrics 輸出。
- `smi_lab/market_info.py`：市場新聞與個股新聞快取。

## 現行自動化

- 台股：台灣時間週一到週五 13:35，收盤後掃描。
- 美股：台灣時間週二到週六 07:30，美股收盤後掃描。
- 加密：每 4 小時掃描一次。
- 舊版 entry/stop/TP 價格提醒已暫停，未來應改成再平衡事件提醒。

## 實盤邊界

- Firstrade 與國泰證券：保留手動下單，只透過前端追蹤持倉與策略意圖。
- Pionex：目前只做帳戶與訂單追蹤，不自動下單。
- 自動下單需要額外完成 API key 管理、canary 資金、單筆上限、日損上限、kill switch、錯單 reconciliation。

## 下一步

1. 把 Discord 通知從舊的價格層級提醒，改成策略再平衡事件提醒。
2. 對擴大後 universe 做更長週期與 walk-forward 驗證，降低 2 年最佳化過度貼合近期行情的風險。
3. 建立台美股持倉自動同步的固定流程，減少手動輸入。
4. 若要恢復 entry/TP/stop/RR，需要重新設計執行層並完整回測。
5. 加密實盤前需要完成 Pionex API 風控層。

## 常用命令

```powershell
.\.venv\Scripts\python.exe scan_signals.py --strategy allocation --crypto-universe top100 --crypto-limit 100 --interval 4h --bars 1200 --refresh
.\.venv\Scripts\python.exe scan_equity_signals.py --market both --interval 1d --range 2y --refresh
.\.venv\Scripts\python.exe research_equity_selection.py --market both --range 2y --weighting all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe start_streamlit.py
```
