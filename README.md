# Ancser Alpaca Lab

Cross-sectional factor-based quantitative trading system with web dashboard.

横截面因子量化交易系统，含网页仪表板。

---

## Quick Start / 快速啟動

```bash
# 1. Install dependencies / 安裝依賴
pip install -r requirements.txt

# 2. Create .env file with Alpaca API keys / 建立 .env 填入 Alpaca API 金鑰
#    APCA_API_KEY_ID=your_key
#    APCA_API_SECRET_KEY=your_secret

# 3. Launch dashboard / 啟動儀表板
python web_server.py

# 4. Open browser / 開啟瀏覽器
#    http://localhost:5000
```

---

## File Structure / 檔案結構

```
ancserAlpacaLab/
  config.yaml             Global config / 全域設定
  .env                    Alpaca API keys / API 金鑰
  requirements.txt        Dependencies / 依賴清單

  web_server.py           Web dashboard server / 網頁儀表板伺服器
  backtest_engine.py      Backtest engine / 回測引擎
  factor_library.py       Factor library / 因子庫
  data_manager.py         Data download & cache / 資料下載與快取
  alpaca_execute.py       Live trading executor / 實盤執行器

  static/
    index.html            Dashboard UI / 儀表板介面
    app.js                Frontend logic / 前端邏輯
    style.css             Styles / 樣式

  data_cache/             Cached market data / 市場資料快取 (auto-created)
  logs/                   Execution & backtest logs / 執行與回測日誌 (auto-created)
    backtest/             Structured JSON logs per run / 每次回測的 JSON 日誌
  saved_data/             Saved configurations / 已儲存設定
```

---

## Factors / 因子說明

Each factor computes a daily cross-sectional score for all stocks.

每個因子每日為所有股票計算橫截面分數。

Higher composite score = more likely to be selected into the portfolio.

綜合分數越高，越可能被選入投資組合。

### Built-in Factors / 內建因子

| Factor | Default | Description |
|--------|---------|-------------|
| **Momentum 12-1** | lookback=252, skip=21 | 12-month return minus most recent 1-month return. Classic cross-sectional momentum. |
|  |  | 12 個月報酬扣除最近 1 個月報酬，經典橫截面動量。 |
| **Pullback 5D** | lookback=5 | 5-day reversal factor. Stocks that dropped more rank higher (mean reversion). |
|  |  | 5 日反轉因子，跌幅越大排名越前（均值回歸）。 |
| **RSI** | period=14 | Relative Strength Index (0-100). Can be used for oversold screening. |
|  |  | 相對強弱指標（0-100），可用於超賣篩選。 |
| **Volatility** | period=20 | Annualized return volatility. Lower volatility = more stable. |
|  |  | 年化波動率，波動越低越穩定。 |
| **Volume Surge** | period=20 | Current volume / 20-day average volume. Detects unusual volume. |
|  |  | 當前成交量 / 20 日均量，偵測異常放量。 |
| **Trend Strength** | short=20, long=50 | Short SMA / Long SMA - 1. Positive = uptrend. |
|  |  | 短期均線 / 長期均線 - 1，正值代表上升趨勢。 |
| **KDJ (J-Line)** | period=9, signal=3 | Stochastic J-line (inverted). Oversold stocks score higher. Derived from TradingView KDJMA. |
|  |  | 隨機指標 J 線（反轉），超賣股票分數較高。源自 TradingView KDJMA。 |
| **PMO** | first=100, second=50 | Price Momentum Oscillator. Double-smoothed ROC. Higher = stronger momentum. Derived from TradingView EMAPMO. |
|  |  | 價格動量振盪器，雙重平滑 ROC，數值越高動量越強。源自 TradingView EMAPMO。 |
| **Graham Value** | cache_days=7 | Benjamin Graham composite: Earnings Yield (1/PE) + Book-to-Market (1/PB) + Dividend Yield. Fundamental data via yfinance, cached to disk. |
|  |  | 班傑明·葛拉漢價值因子：盈餘殖利率 (1/PE) + 帳面市值比 (1/PB) + 股息殖利率。基本面資料透過 yfinance 取得並快取。 |
| **Beta** | period=60 | Rolling beta vs SPY. Lower beta = less market risk. |
|  |  | 相對 SPY 的滾動 Beta，Beta 越低市場風險越小。 |

---

## Dashboard Stats / 儀表板指標

| Metric | Formula | Meaning |
|--------|---------|---------|
| Calmar | CAGR / abs(MaxDD) | Reward per unit of maximum pain / 每單位最大痛苦的報酬 |
| CAGR | Annualized compound return | Compound annual growth rate / 年化複合成長率 |
| MaxDD | Peak-to-trough decline | Largest drawdown / 最大回撤 |
| Sharpe | Mean daily return / Std * sqrt(252) | Risk-adjusted return / 風險調整報酬 |
| Win Rate | % of positive return days | Percentage of profitable days / 正報酬天數比例 |

---

## Live Trading / 實盤交易

```bash
# Dry run (no orders) / 模擬（不下單）
python alpaca_execute.py --paper --dry-run

# Paper trading / 模擬賬戶交易
python alpaca_execute.py --paper

# Force immediate execution / 強制立即執行
python alpaca_execute.py --paper --force
```

---

## Config Reference / 設定參考

All settings are in `config.yaml`.

所有設定在 `config.yaml` 中。

Key sections: `data`, `universe`, `factors`, `portfolio`, `regime`, `costs`, `backtest`, `execution`.

主要區塊：`data` 資料、`universe` 股票池、`factors` 因子、`portfolio` 組合、`regime` 趨勢開關、`costs` 成本、`backtest` 回測、`execution` 執行。

---

## Adding a New Factor / 新增因子

```python
# 1. Add function in factor_library.py / 在 factor_library.py 加入函式
def my_factor(close: pd.DataFrame, param: int = 10) -> pd.DataFrame:
    return close.rolling(param).std()

# 2. Register in FACTOR_FUNCTIONS dict in web_server.py
#    在 web_server.py 的 FACTOR_FUNCTIONS 字典中註冊

# 3. Add FACTOR_INFO entry in web_server.py for dashboard UI
#    在 web_server.py 的 FACTOR_INFO 加入描述以顯示在儀表板
```