"""在有網路的環境（本機 / GitHub Actions）用 yfinance 更新 data/daily/ 日K。

用法：python tools/update_data.py [--start 2015-01-01] [--codes 2330,2317]
首次執行建議 --start 2015-01-01 取得 10 年以上歷史，回測才有統計意義。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.config import DAILY_DIR, ETFS, TW50  # noqa: E402
from quant.data import fetch_yfinance_daily  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--codes", default="")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    codes = args.codes.split(",") if args.codes else list(TW50) + list(ETFS)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for code in codes:
        for attempt in range(3):
            try:
                df = fetch_yfinance_daily(code, start=args.start)
                df.round(4).to_csv(DAILY_DIR / f"{code}.csv")
                print(f"{code} {len(df)} 筆 {df.index[0].date()} ~ {df.index[-1].date()}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ {code} 第 {attempt + 1} 次失敗：{e}")
                time.sleep(2 * (attempt + 1))
        else:
            failed.append(code)
        time.sleep(args.sleep)
    if failed:
        print("❌ 失敗：", ",".join(failed))
    if len(failed) > len(codes) // 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
