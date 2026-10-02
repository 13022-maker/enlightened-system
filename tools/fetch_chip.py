"""抓取台灣50成分股的三大法人買賣超（FinMind），存成 data/chip/<code>.csv。

需要網路（本機或 GitHub Actions）。可設定環境變數 FINMIND_TOKEN 提高額度。

用法：
    python3 tools/fetch_chip.py                       # 增量：各檔從現有檔最後日期 −10 天抓起（無檔則從 --start）
    python3 tools/fetch_chip.py --full                # 強制從 --start（預設 2015-01-01）全量抓
    python3 tools/fetch_chip.py --start 2024-01-01    # 指定（無舊檔或 --full 時的）起始日
    python3 tools/fetch_chip.py --codes 2330 2317     # 只抓部分股票
    python3 tools/fetch_chip.py --sleep 2 --retries 5

輸出欄位：date,foreign_net,trust_net,dealer_net（單位：股）
若檔案已存在，會與舊資料合併（同日期以新資料為準），避免覆蓋掉較早的歷史。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from quant.config import CHIP_DIR, TW50  # noqa: E402
from quant.data import fetch_finmind_chip  # noqa: E402

COLS = ["foreign_net", "trust_net", "dealer_net"]
OVERLAP_DAYS = 10          # 增量抓取往前重疊的日曆天數（FinMind 偶有回補修正）


def incremental_start(code: str, directory: Path, default: str, overlap: int = OVERLAP_DAYS) -> str:
    """現有檔最後日期 − overlap 天；無檔或讀取失敗則回傳 default。不早於 default。"""
    path = Path(directory) / f"{code}.csv"
    if not path.exists():
        return default
    try:
        last = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"].max()
    except Exception:  # noqa: BLE001  — 檔案損毀就從頭抓
        return default
    if pd.isna(last):
        return default
    return max(pd.Timestamp(default), last - pd.Timedelta(days=overlap)).strftime("%Y-%m-%d")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """整理成標準格式：index 名稱 date、三個欄位、依日期排序、去重。"""
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.reindex(columns=COLS).fillna(0.0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def fetch_with_retry(code: str, start: str, fetch: Callable = fetch_finmind_chip,
                     retries: int = 3, backoff: float = 5.0,
                     sleep: Callable[[float], None] = time.sleep) -> pd.DataFrame:
    """失敗時以遞增間隔重試（限流、暫時性網路錯誤）。全部失敗則拋出最後的例外。"""
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return fetch(code, start=start)
        except Exception as e:  # noqa: BLE001  — 任何錯誤都重試
            last = e
            print(f"  {code} 第 {attempt}/{retries} 次失敗：{e}", file=sys.stderr)
            if attempt < retries:
                sleep(backoff * attempt)
    raise RuntimeError(f"{code} 抓取失敗（已重試 {retries} 次）：{last}")


def save(code: str, df: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{code}.csv"
    new = normalize(df)
    if path.exists():
        old = normalize(pd.read_csv(path, parse_dates=["date"]))
        new = normalize(pd.concat([old, new]))       # 同日期以新資料為準
    new.to_csv(path, float_format="%.0f")
    return path


def run(codes: Iterable[str], start: str, directory: Path = CHIP_DIR, pause: float = 1.0,
        retries: int = 3, fetch: Callable = fetch_finmind_chip,
        sleep: Callable[[float], None] = time.sleep, incremental: bool = False) -> List[str]:
    """逐檔抓取並儲存，回傳失敗代號清單。

    incremental=True 時每檔從「現有檔最後日期 − 10 天」開始抓（無檔則從 start）。
    """
    failed = []
    codes = list(codes)
    for k, code in enumerate(codes):
        try:
            s0 = incremental_start(code, Path(directory), start) if incremental else start
            df = fetch_with_retry(code, s0, fetch=fetch, retries=retries, sleep=sleep)
            if df is None or len(df) == 0:
                print(f"{code}: 無資料，略過")
                failed.append(code)
            else:
                p = save(code, df, Path(directory))
                print(f"{code}: {len(df)} 筆 → {p}")
        except Exception as e:  # noqa: BLE001
            print(f"{code}: 放棄（{e}）", file=sys.stderr)
            failed.append(code)
        if k < len(codes) - 1:
            sleep(pause)                                 # 請求間隔，避免 FinMind 限流
    return failed


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="抓取 TW50 三大法人買賣超（FinMind）")
    ap.add_argument("--start", default="2015-01-01", help="起始日 YYYY-MM-DD")
    ap.add_argument("--codes", nargs="*", default=None, help="股票代號（預設 TW50 全部）")
    ap.add_argument("--out", default=str(CHIP_DIR), help="輸出目錄")
    ap.add_argument("--sleep", type=float, default=1.0, help="每檔請求間隔秒數")
    ap.add_argument("--retries", type=int, default=3, help="失敗重試次數")
    ap.add_argument("--full", action="store_true", help="強制從 --start 全量抓取（預設為增量）")
    a = ap.parse_args(argv)
    codes = a.codes or list(TW50)
    failed = run(codes, a.start, Path(a.out), pause=a.sleep, retries=a.retries,
                 incremental=not a.full)
    print(f"完成：成功 {len(codes) - len(failed)} 檔，失敗 {len(failed)} 檔"
          + (f"：{' '.join(failed)}" if failed else ""))
    if failed:
        # GitHub Actions 註記：在 run 摘要顯示黃色警告，不中斷流程
        print(f"::warning::法人籌碼抓取失敗 {len(failed)}/{len(codes)} 檔：{' '.join(failed)}")
    return 1 if codes and len(failed) == len(codes) else 0   # 全部失敗才回傳錯誤碼


if __name__ == "__main__":
    sys.exit(main())
