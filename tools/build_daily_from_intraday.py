"""把 voidful/tw_stocker 的 5 分K CSV 轉成 data/daily/<code>.csv 日K。

用法：python tools/build_daily_from_intraday.py <tw_stocker/data 目錄> [--codes 2330,2317]

轉檔股票 = TW50 + ETFS + EXTRA_CODES + data/universe/tw50_membership.csv 中所有曾為成分股者
（被調出/調入的股票也要有價格，回測才能做 point-in-time 股票池，見 quant/universe.py）。
股數變動：data/dividends/<code>.csv 的 share_ratio > 5 分K 的 Stock Splits 欄 > 跳空推測（見 quant/data.py）。
缺 5 分K 的股票會列出（例：7769 鴻勁、6919 康霈* 不在 tw_stocker 中）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.config import DAILY_DIR, ETFS, EXTRA_CODES, TW50  # noqa: E402
from quant.data import intraday_to_daily  # noqa: E402
from quant.universe import load_universe  # noqa: E402


def all_codes():
    codes = list(TW50) + list(ETFS) + list(EXTRA_CODES)
    u = load_universe()
    if u is not None:
        codes += u.codes
    return list(dict.fromkeys(codes))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="tw_stocker/data 目錄")
    ap.add_argument("--codes", default="", help="逗號分隔；預設全部")
    a = ap.parse_args(argv)
    src = Path(a.src)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for code in (a.codes.split(",") if a.codes else all_codes()):
        p = src / f"{code}.csv"
        if not p.exists():
            missing.append(code)
            continue
        df = intraday_to_daily(str(p), code=code).round(4)
        df["volume"] = df["volume"].round().astype("int64")       # 量保持整數，避免無謂的 diff
        df.to_csv(DAILY_DIR / f"{code}.csv")
        print(code, len(df), df.index[0].date(), df.index[-1].date())
    if missing:
        print("缺少 5 分K：", ",".join(missing))


if __name__ == "__main__":
    main()
