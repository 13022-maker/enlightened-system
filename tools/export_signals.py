"""產出看盤介面用的 output/signals.json（最新行情 + 各策略訊號 + 回測摘要）。

用法：python tools/export_signals.py
"""
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.config import ETFS, OUTPUT_DIR, TW50  # noqa: E402
from quant.data import load_panel  # noqa: E402
from quant.strategies import REGISTRY  # noqa: E402

TW_TZ = timezone(timedelta(hours=8))


def clean(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), nd)


def main():
    names = {**TW50, **ETFS}
    panel = load_panel(list(TW50) + list(ETFS), chip=True)
    report_path = OUTPUT_DIR / "backtest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    c, o, h, l, v = panel.close, panel.open, panel.high, panel.low, panel.volume
    last, prev = c.iloc[-1], c.iloc[-2]
    stocks = []
    for code in panel.codes:
        if math.isnan(last[code]):
            continue
        stocks.append({
            "code": code, "name": names.get(code, code), "etf": panel.is_etf(code),
            "close": clean(last[code]), "change": clean(last[code] - prev[code]),
            "change_pct": clean((last[code] / prev[code] - 1) * 100),
            "open": clean(o[code].iloc[-1]), "high": clean(h[code].iloc[-1]), "low": clean(l[code].iloc[-1]),
            "volume_lots": int(v[code].iloc[-1] // 1000),
            "spark": [clean(x) for x in c[code].iloc[-60:].tolist()],
            "signals": {},
        })
    by_code = {s["code"]: s for s in stocks}

    strategies = []
    for name, cls in REGISTRY.items():
        strat = cls()
        rep = report.get(name, {})
        params = rep.get("best_params", strat.default_params)
        synthetic = bool(strat.needs_chip and panel.chip_is_synthetic)
        try:
            sigs = strat.latest_signals(panel, **params)
        except Exception as e:  # 單一策略失敗不影響其他
            print(f"⚠️ {name} 產生訊號失敗：{e}")
            continue
        for s in sigs:
            if s["code"] in by_code:
                by_code[s["code"]]["signals"][name] = {
                    "action": s["action"], "weight": s["weight"], "score": s["score"], "reason": s["reason"]}
        test = rep.get("test", {})
        strategies.append({
            "name": name, "title": strat.title, "description": strat.description,
            "synthetic": synthetic,
            "test": {k: clean(test.get(k), 4) for k in ["cagr", "sharpe", "max_drawdown", "annual_turnover", "cost_drag", "win_rate"]} if test else None,
            "test_period": rep.get("test_period"),
        })

    bench = next(iter(report.values()), {}).get("benchmarks_test", {})
    out = {
        "as_of": str(panel.dates[-1].date()),
        "generated_at": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M"),
        "chip_synthetic": panel.chip_is_synthetic,
        "strategies": strategies,
        "benchmarks": {k: {m: clean(b.get(m), 4) for m in ["cagr", "sharpe", "max_drawdown"]} for k, b in bench.items()},
        "stocks": stocks,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "signals.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n_buy = sum(1 for s in stocks for x in s["signals"].values() if x["action"] == "BUY")
    print(f"✅ signals.json：{out['as_of']}，{len(stocks)} 檔，{len(strategies)} 個策略，BUY 訊號 {n_buy} 個")


if __name__ == "__main__":
    main()
