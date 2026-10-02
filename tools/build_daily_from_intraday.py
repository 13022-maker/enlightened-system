"""把 voidful/tw_stocker 的 5 分K CSV 轉成 data/daily/<code>.csv 日K。

用法：python tools/build_daily_from_intraday.py <tw_stocker/data 目錄>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.config import DAILY_DIR, ETFS, TW50  # noqa: E402
from quant.data import intraday_to_daily  # noqa: E402

src = Path(sys.argv[1])
DAILY_DIR.mkdir(parents=True, exist_ok=True)
for code in list(TW50) + list(ETFS):
    p = src / f"{code}.csv"
    if not p.exists():
        print("缺少", code)
        continue
    df = intraday_to_daily(str(p))
    df.round(4).to_csv(DAILY_DIR / f"{code}.csv")
    print(code, len(df), df.index[0].date(), df.index[-1].date())
