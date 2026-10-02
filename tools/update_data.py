"""在有網路的環境（本機 / GitHub Actions）用 yfinance 更新 data/daily/ 日K（增量更新）。

用法：
    python tools/update_data.py                       # 增量：從各檔最後日期 −10 天抓取並合併
    python tools/update_data.py --full                # 強制全量重抓（從 --start 起）
    python tools/update_data.py --codes 2330,2317     # 只更新部分股票

增量規則：
    1. 讀現有 CSV 最後日期，從「最後日期 − overlap_days（預設 10）天」開始抓。
    2. 新舊合併，同一天以新資料為準。
    3. 若重疊區間收盤差 > 0.5%，或窗口內出現股利 / 分割跡象（未還原價格與舊檔對不上），
       代表歷史需要重新調整 → 該檔改為全量重抓。
    4. 全量重抓結果筆數少於舊檔 95% 時不覆蓋（避免資料源暫時殘缺把歷史洗掉）。

結果寫到 output/data_status.json：
    {"last_date", "updated_codes", "stale_codes", "failed_codes", "full_refetch_codes", "generated_at"}
    stale_codes：本次未成功更新、且檔案最後日期落後於全體最新日期的股票。

離開碼：0 = 至少部分成功；2 = 全部失敗（workflow 設 continue-on-error，沿用舊資料繼續）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from quant.config import DAILY_DIR, ETFS, OUTPUT_DIR, TW50  # noqa: E402

TW_TZ = timezone(timedelta(hours=8))
PRICE_COLS = ["open", "high", "low", "close", "volume", "dividend"]
OVERLAP_DAYS = 10          # 增量抓取往前重疊的日曆天數
CLOSE_TOL = 0.005          # 重疊段收盤價容許誤差（0.5%）
MIN_KEEP_RATIO = 0.95      # 新資料筆數 < 舊檔 95% 時不覆蓋


def _default_fetch(code: str, start: str) -> pd.DataFrame:
    """包裝 quant.data.fetch_yfinance_daily（延遲 import，測試時可注入假的 fetch）。"""
    from quant.data import fetch_yfinance_daily
    return fetch_yfinance_daily(code, start=start)


def read_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    except Exception as e:  # noqa: BLE001  — 檔案損毀就當作沒有，改全量
        print(f"⚠️ 讀取 {path.name} 失敗，改全量：{e}")
        return None
    return df.sort_index() if len(df) else None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out.index.name = "date"
    out = out.reindex(columns=PRICE_COLS)
    out["dividend"] = out["dividend"].fillna(0.0)
    return out[~out.index.duplicated(keep="last")].sort_index()


def needs_full_refetch(old: pd.DataFrame, new: pd.DataFrame, tol: float = CLOSE_TOL) -> Optional[str]:
    """比對重疊區間，回傳需要全量重抓的原因（None = 可安全增量合併）。

    - 收盤差 > tol：資料源已回溯調整歷史（分割 / 合併 / 修正）。
    - 窗口內出現新股利，或股利數值與舊檔不同：未還原價格的股利欄需整段重算。
    """
    overlap = old.index.intersection(new.index)
    if len(overlap) == 0:
        return "與舊檔無重疊日期"
    oc, nc = old.loc[overlap, "close"].astype(float), new.loc[overlap, "close"].astype(float)
    diff = ((nc - oc).abs() / oc.abs()).max()
    if pd.notna(diff) and diff > tol:
        return f"重疊段收盤差 {diff:.2%}"
    od = old.loc[overlap, "dividend"].fillna(0.0).astype(float)
    nd = new.loc[overlap, "dividend"].fillna(0.0).astype(float)
    if (od - nd).abs().max() > 1e-6:
        return "重疊段股利不一致"
    tail = new.loc[new.index > old.index[-1]]
    if len(tail) and (tail["dividend"].fillna(0.0) > 0).any():
        return "新資料含除權息"
    return None


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """合併新舊資料，同一天以新資料為準。"""
    both = pd.concat([_normalize(old), _normalize(new)])
    return both[~both.index.duplicated(keep="last")].sort_index()


def fetch_retry(fetch: Callable, code: str, start: str, retries: int = 3,
                sleep: Callable[[float], None] = time.sleep) -> pd.DataFrame:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = fetch(code, start)
            if df is None or len(df) == 0:
                raise ValueError("回傳空資料")
            return _normalize(df)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"⚠️ {code} 第 {attempt + 1}/{retries} 次失敗：{e}")
            if attempt < retries - 1:
                sleep(2 * (attempt + 1))
    raise RuntimeError(f"{code} 抓取失敗：{last}")


def update_one(code: str, directory: Path, start: str, full: bool, fetch: Callable,
               retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> Dict:
    """更新單一股票，回傳 {"status": updated|failed|kept, "mode": ..., "reason": ...}。"""
    path = directory / f"{code}.csv"
    old = None if full else read_existing(path)
    mode, reason = "full", "強制全量" if full else "無舊檔"
    try:
        if old is not None:
            inc_start = (old.index[-1] - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
            new = fetch_retry(fetch, code, inc_start, retries, sleep)
            why = needs_full_refetch(old, new)
            if why is None:
                merged = merge(old, new)
                merged.round(4).to_csv(path)
                print(f"{code} 增量 +{len(merged) - len(old)} 筆 → {merged.index[-1].date()}")
                return {"status": "updated", "mode": "incremental"}
            mode, reason = "full", why
            print(f"{code} {why} → 全量重抓")
        data = fetch_retry(fetch, code, start, retries, sleep)
        existing = read_existing(path)
        if existing is not None and len(data) < MIN_KEEP_RATIO * len(existing):
            msg = f"全量結果 {len(data)} 筆 < 舊檔 {len(existing)} 筆的 95%，不覆蓋"
            print(f"⚠️ {code} {msg}")
            return {"status": "kept", "mode": mode, "reason": msg}
        data.round(4).to_csv(path)
        print(f"{code} 全量 {len(data)} 筆 {data.index[0].date()} ~ {data.index[-1].date()}（{reason}）")
        return {"status": "updated", "mode": mode, "reason": reason}
    except Exception as e:  # noqa: BLE001
        print(f"❌ {code}：{e}")
        return {"status": "failed", "mode": mode, "reason": str(e)}


def last_dates(codes: Iterable[str], directory: Path) -> Dict[str, Optional[str]]:
    out = {}
    for c in codes:
        df = read_existing(directory / f"{c}.csv")
        out[c] = None if df is None else str(df.index[-1].date())
    return out


def run(codes: List[str], directory: Path = DAILY_DIR, start: str = "2015-01-01", full: bool = False,
        fetch: Optional[Callable] = None, pause: float = 0.5, retries: int = 3,
        sleep: Optional[Callable[[float], None]] = None, status_path: Optional[Path] = None) -> Dict:
    # 執行時才取預設值（方便測試 monkeypatch 模組層的 _default_fetch / time.sleep）
    fetch = fetch or _default_fetch
    sleep = sleep or time.sleep
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for k, code in enumerate(codes):
        results[code] = update_one(code, directory, start, full, fetch, retries, sleep)
        if k < len(codes) - 1:
            sleep(pause)
    dates = last_dates(codes, directory)
    valid = [d for d in dates.values() if d]
    last_date = max(valid) if valid else None
    failed = [c for c, r in results.items() if r["status"] == "failed"]
    # 陳舊：檔案最後日期落後全體最新日期（含失敗與「不覆蓋」而沒更新的股票）
    stale = [c for c in codes if dates[c] is None or (last_date and dates[c] < last_date)]
    status = {
        "generated_at": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M"),
        "last_date": last_date,
        "updated_codes": [c for c, r in results.items() if r["status"] == "updated"],
        "full_refetch_codes": [c for c, r in results.items()
                               if r["status"] == "updated" and r["mode"] == "full"],
        "kept_codes": [c for c, r in results.items() if r["status"] == "kept"],
        "failed_codes": failed,
        "stale_codes": stale,
        "all_failed": bool(codes) and len(failed) == len(codes),
    }
    status_path = Path(status_path) if status_path else OUTPUT_DIR / "data_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="增量更新 data/daily 日K（yfinance）")
    ap.add_argument("--start", default="2015-01-01", help="全量抓取的起始日")
    ap.add_argument("--codes", default="", help="逗號分隔代號（預設 TW50 + ETF）")
    ap.add_argument("--full", action="store_true", help="強制全量重抓")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--out", default=str(DAILY_DIR))
    ap.add_argument("--status", default=str(OUTPUT_DIR / "data_status.json"))
    a = ap.parse_args(argv)
    codes = a.codes.split(",") if a.codes else list(TW50) + list(ETFS)
    st = run(codes, Path(a.out), a.start, a.full, pause=a.sleep, retries=a.retries,
             status_path=Path(a.status))
    print(f"資料最新日 {st['last_date']}｜更新 {len(st['updated_codes'])}｜失敗 {len(st['failed_codes'])}"
          f"｜陳舊 {len(st['stale_codes'])}")
    if st["failed_codes"]:
        print(f"::warning::日K 更新失敗 {len(st['failed_codes'])} 檔：{','.join(st['failed_codes'])}")
    if st["all_failed"]:
        print("::error::日K 全部更新失敗，後續步驟將沿用舊資料")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
