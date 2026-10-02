# 開明體系 Quant：回測引擎＋策略模組

在原有監控系統（`scripts/`）旁新增，不影響既有 workflow。

## 結構
```
quant/
  config.py          台灣50名單、交易成本（手續費6折、證交稅0.3%/ETF 0.1%、滑價0.05%）
  data.py            日K/籌碼載入、yfinance 與 FinMind 抓取、分割與異常股利修正
  engine.py          日頻回測：收盤決策→隔日開盤成交、漲停買不到、除息加回
  validation.py      前視偏差檢查、walk-forward 樣本外評估
  strategies/
    swing_momentum.py     波段動能（120日動能＋MA60＋大盤濾網＋ATR停損，每4週調整）
    chip_flow.py          法人籌碼動能（外資＋投信買超佔量，需 FinMind 資料）
    dividend_yield.py     存股殖利率（近12月殖利率＋價值陷阱濾網，每季調整）
    legacy_confidence.py  現行 Phase 4 信心度系統（比較基準）
tools/
  update_data.py     yfinance 更新日K（建議 2015 起）
  fetch_chip.py      FinMind 三大法人
  run_backtest.py    全策略比較 → output/backtest_report.md
  export_signals.py  最新訊號 → output/signals.json
  build_dashboard.py 看盤網頁 → output/dashboard.html
  notify_signals.py  Telegram 推播 BUY/SELL
.github/workflows/quant-daily.yml  每個交易日 17:30 全自動執行並 commit 結果
```

## 第一次使用
1. GitHub Secrets 可加 `FINMIND_TOKEN`（選用，提高額度；沿用既有 Telegram secrets）。
2. Actions → 「Quant 每日訊號與回測」→ Run workflow。會抓 2015 年起資料重跑回測，結果在 `output/`。
3. 本機：`pip install -r requirements.txt pytest && python -m pytest -q && python tools/run_backtest.py`

## 新增策略
繼承 `quant.strategies.base.Strategy`，實作 `target_weights()`，在 `strategies/__init__.py` 登記，
並在測試中呼叫 `assert_no_lookahead()`。param_grid 請 ≤ 8 組，以樣本外結果判斷。

## 目前結論（2024-02 ~ 2026-04 真實資料，樣本外 2025-05 ~ 2026-04）
見 `output/backtest_report.md`。資料只有約 2 年、樣本外是強多頭，**沒有任何策略在風險調整後勝過 0050 買進持有**；
用 10 年資料重跑前請不要實盤。
