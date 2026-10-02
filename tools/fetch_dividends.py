"""從 FinMind 抓除權息結果，存成 data/dividends/<code>.csv（需網路）。

用法：
    python tools/fetch_dividends.py                       # TW50 + ETF + EXTRA_CODES + 歷史成分股
    python tools/fetch_dividends.py --codes 2327,0050 --start 2015-01-01
    python tools/fetch_dividends.py --merge               # 保留現有檔中 FinMind 沒有的列（例如手動整理的分割）

資料集（依 FinMind 文件 https://finmind.github.io/tutor/TaiwanMarket/Fundamental/ 查證）：
    TaiwanStockDividendResult（除權除息結果表，2003-05 起）：
        date, stock_id, before_price（除權息前收盤價）, after_price（除權息後收盤價）,
        stock_and_cache_dividend（權息值）, stock_or_cache_dividend（「權」/「息」/「權息」）,
        max_price, min_price, open_price, reference_price（減除股利參考價）
    TaiwanStockDividend（股利政策表）：
        CashEarningsDistribution + CashStatutorySurplus = 每股現金股利（元）
        StockEarningsDistribution + StockStatutorySurplus = 每股股票股利（元，面額 10 元 → 每股配 x/10 股）
        CashExDividendTradingDate / StockExDividendTradingDate = 除息 / 除權交易日
    減資、分割、面額變更另有 TaiwanStockCapitalReductionReferencePrice / TaiwanStockSplitPrice /
    TaiwanStockParValueChange，本工具目前不抓（分割請手動補在 CSV，見 data/dividends/SOURCES.md）。

輸出格式（quant.data.load_dividend_events 讀取）：
    date,cash_dividend,stock_dividend,share_ratio,note
    以 DividendResult 為主（實際除權息日）：
      - 「息」：cash = 權息值。
      - 「權」/「權息」：用股利政策表拆出現金與股票股利；share_ratio = 1 + 股票股利/10（面額 10 元），
        若政策表查無，以 share_ratio = (before_price − cash) / reference_price 推算。
    ETF 的收益分配同樣以「息」出現在 DividendResult。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from quant.config import DIVIDEND_DIR, ETFS, EXTRA_CODES, TW50  # noqa: E402
from quant.universe import load_universe  # noqa: E402

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
OUT_COLS = ["date", "cash_dividend", "stock_dividend", "share_ratio", "note"]


def finmind_get(dataset: str, code: str, start: str, session=None) -> pd.DataFrame:
    import requests

    s = session or requests
    params = {"dataset": dataset, "data_id": code, "start_date": start}
    token = os.environ.get("FINMIND_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = s.get(FINMIND_URL, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") not in (None, 200):
        raise RuntimeError(f"FinMind 回應錯誤 {js.get('status')}: {js.get('msg')}")
    return pd.DataFrame(js.get("data", []))


def _num(x) -> float:
    try:
        v = float(x)
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_events(result: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    """把 DividendResult（+ 股利政策表）轉成事件表。"""
    rows: List[Dict] = []
    if result is None or result.empty:
        return pd.DataFrame(columns=OUT_COLS)
    pol_cash: Dict[str, float] = {}
    pol_stock: Dict[str, float] = {}
    if policy is not None and not policy.empty:
        for _, p in policy.iterrows():
            cash = _num(p.get("CashEarningsDistribution")) + _num(p.get("CashStatutorySurplus"))
            stock = _num(p.get("StockEarningsDistribution")) + _num(p.get("StockStatutorySurplus"))
            cd, sd = str(p.get("CashExDividendTradingDate") or ""), str(p.get("StockExDividendTradingDate") or "")
            if cd and cash > 0:
                pol_cash[cd[:10]] = pol_cash.get(cd[:10], 0.0) + cash
            if sd and stock > 0:
                pol_stock[sd[:10]] = pol_stock.get(sd[:10], 0.0) + stock
    for _, r in result.sort_values("date").iterrows():
        day = str(r["date"])[:10]
        kind = str(r.get("stock_or_cache_dividend", "")).strip()
        value = _num(r.get("stock_and_cache_dividend"))
        before, ref = _num(r.get("before_price")), _num(r.get("reference_price"))
        if kind == "息":
            rows.append({"date": day, "cash_dividend": value, "stock_dividend": 0.0, "share_ratio": 1.0,
                         "note": "FinMind DividendResult 息"})
            continue
        cash = pol_cash.get(day, 0.0)
        stock = pol_stock.get(day, 0.0)
        if stock > 0:
            ratio = 1 + stock / 10.0
        elif ref > 0 and before > 0:
            ratio = (before - cash) / ref
        else:
            ratio = 1.0
        rows.append({"date": day, "cash_dividend": cash, "stock_dividend": stock, "share_ratio": round(ratio, 6),
                     "note": f"FinMind DividendResult {kind}（前收 {before}、參考價 {ref}）"})
    return pd.DataFrame(rows, columns=OUT_COLS)


def merge_existing(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """保留舊檔中新資料沒有的日期（例：手動整理的分割、面額變更）。"""
    if not path.exists():
        return new
    old = pd.read_csv(path, comment="#", dtype={"date": str})
    keep = old[~old["date"].str[:10].isin(set(new["date"]))]
    return pd.concat([new, keep[[c for c in OUT_COLS if c in keep.columns]]]).sort_values("date")


def run(codes: List[str], start: str, directory: Path = DIVIDEND_DIR, merge: bool = True,
        fetch: Optional[Callable[[str, str, str], pd.DataFrame]] = None, pause: float = 0.5) -> Dict[str, str]:
    fetch = fetch or finmind_get
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    status = {}
    for k, code in enumerate(codes):
        try:
            res = fetch("TaiwanStockDividendResult", code, start)
            pol = fetch("TaiwanStockDividend", code, start) if not code.startswith("00") else pd.DataFrame()
            ev = build_events(res, pol)
            path = directory / f"{code}.csv"
            if merge:
                ev = merge_existing(ev, path)
            if ev.empty:
                status[code] = "empty"
                continue
            ev.to_csv(path, index=False)
            status[code] = f"ok {len(ev)}"
            print(code, len(ev), "筆")
        except Exception as e:  # noqa: BLE001
            status[code] = f"failed: {e}"
            print(f"❌ {code}：{e}")
        if k < len(codes) - 1 and pause:
            time.sleep(pause)
    return status


def default_codes() -> List[str]:
    codes = list(TW50) + list(ETFS) + list(EXTRA_CODES)
    u = load_universe()
    if u is not None:
        codes += u.codes
    return list(dict.fromkeys(codes))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FinMind 除權息結果 → data/dividends/")
    ap.add_argument("--codes", default="")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--out", default=str(DIVIDEND_DIR))
    ap.add_argument("--no-merge", action="store_true", help="覆寫現有檔（不保留手動整理的列）")
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args(argv)
    codes = a.codes.split(",") if a.codes else default_codes()
    st = run(codes, a.start, Path(a.out), merge=not a.no_merge, pause=a.sleep)
    failed = [c for c, v in st.items() if v.startswith("failed")]
    return 2 if failed and len(failed) == len(codes) else 0


if __name__ == "__main__":
    sys.exit(main())
