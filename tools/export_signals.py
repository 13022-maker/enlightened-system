"""產出看盤介面用的 output/signals.json（最新行情 + 各策略訊號 + 回測摘要）。

用法：python tools/export_signals.py

去重：若既有 signals.json 的 as_of 與本次相同，generated_at 沿用舊值；
      除 generated_at 外內容也完全相同時不改檔（假日重跑不產生無意義 commit）。
陳舊：output/data_status.json 的 stale_codes 會帶入 signals.json（看盤頁顯示提示）。
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


def load_stale_codes(path: Path) -> list:
    """讀 update_data.py 產生的 data_status.json；不存在或損毀時回傳空清單。"""
    try:
        st = json.loads(Path(path).read_text(encoding="utf-8"))
        return sorted(str(c) for c in st.get("stale_codes", []) or [])
    except Exception:  # noqa: BLE001
        return []


def write_if_changed(out: dict, path: Path) -> bool:
    """依 as_of 去重後寫檔。回傳是否真的寫入。"""
    path = Path(path)
    old = None
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = None
    if old and old.get("as_of") == out.get("as_of"):
        out = {**out, "generated_at": old.get("generated_at", out.get("generated_at"))}
        if old == out:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return True


def main():
    import pandas as pd
    from quant.config import ROOT
    names = {**TW50, **ETFS}
    mem_path = ROOT / "data" / "universe" / "tw50_membership.csv"
    if mem_path.exists():   # 歷史成分股檔的名稱（含新納入的股票）
        names.update(dict(pd.read_csv(mem_path, dtype=str)[["code", "name"]].values))
    panel = load_panel(list(TW50) + list(ETFS), chip=True, point_in_time=True)
    current = panel.universe_mask().iloc[-1]   # 最新一日的成分股
    report_path = OUTPUT_DIR / "backtest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    c, o, h, l, v = panel.close, panel.open, panel.high, panel.low, panel.volume
    last, prev = c.iloc[-1], c.iloc[-2]
    stocks = []
    for code in panel.codes:
        if math.isnan(last[code]):
            continue
        if not panel.is_etf(code) and not bool(current.get(code, True)):
            continue   # 已被調出成分股（若仍有持倉訊號，下方會補回）
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
            if s["code"] not in by_code and s["action"] in ("HOLD", "SELL") and not math.isnan(last[s["code"]]):
                code = s["code"]   # 已調出但策略仍持有／要賣出：列入清單提醒
                row = {"code": code, "name": names.get(code, code) + "（已調出）", "etf": False,
                       "close": clean(last[code]), "change": clean(last[code] - prev[code]),
                       "change_pct": clean((last[code] / prev[code] - 1) * 100),
                       "open": clean(o[code].iloc[-1]), "high": clean(h[code].iloc[-1]), "low": clean(l[code].iloc[-1]),
                       "volume_lots": int(v[code].iloc[-1] // 1000),
                       "spark": [clean(x) for x in c[code].iloc[-60:].tolist()], "signals": {}}
                stocks.append(row)
                by_code[code] = row
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
        "stale_codes": load_stale_codes(OUTPUT_DIR / "data_status.json"),
        "stocks": stocks,
    }
    changed = write_if_changed(out, OUTPUT_DIR / "signals.json")
    n_buy = sum(1 for s in stocks for x in s["signals"].values() if x["action"] == "BUY")
    print(f"✅ signals.json：{out['as_of']}，{len(stocks)} 檔，{len(strategies)} 個策略，BUY 訊號 {n_buy} 個"
          + ("" if changed else "（內容與既有檔相同，未改檔）"))
    if out["stale_codes"]:
        print(f"::warning::{len(out['stale_codes'])} 檔日K 未更新到最新：{','.join(out['stale_codes'])}")


if __name__ == "__main__":
    main()
