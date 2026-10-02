"""維護台灣50歷史成分股檔 data/universe/tw50_membership.csv。

台灣50（FTSE TWSE Taiwan 50）沒有可程式化取得的免費歷史成分股 API，名單來自臺灣指數公司
每季定期審核新聞稿（每年 3/6/9/12 月第一個週五左右公告、第三個週五收盤後生效）。
本工具把「調整事件表」展開成成分股區間檔，並驗證每個時點恰好 50 檔。

用法：
    python tools/fetch_universe.py --check                      # 驗證現有檔每日成分數 = 50
    python tools/fetch_universe.py --add 2026-12-21 --in 1234,5678 --out 2345,3456 --last 2026-12-18 \\
        --names 1234=甲,5678=乙
    （新增一次定期審核：--add = 生效後第一個交易日、--last = 生效日（調出股最後成分日））

查證來源請記在 data/universe/SOURCES.md。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from quant.config import UNIVERSE_FILE  # noqa: E402
from quant.universe import Universe  # noqa: E402


def check(path=UNIVERSE_FILE, start="2024-01-02", end=None, expected=50) -> list:
    """回傳成分數 ≠ expected 的日期清單（以週一～週五曆日檢查）。"""
    u = Universe.from_csv(path)
    end = end or pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start, end)
    n = u.membership_mask(idx).sum(axis=1)
    bad = n[n != expected]
    return [(str(d.date()), int(v)) for d, v in bad.items()]


def add_review(path, first_day: str, last_day: str, ins, outs, names=None) -> pd.DataFrame:
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    names = names or {}
    for code in outs:
        open_rows = t.index[(t["code"] == code) & (t["end"] == "")]
        if len(open_rows) != 1:
            raise ValueError(f"{code} 目前不是成分股，無法調出")
        t.loc[open_rows[0], "end"] = last_day
    for code in ins:
        if ((t["code"] == code) & (t["end"] == "")).any():
            raise ValueError(f"{code} 已是成分股")
        name = names.get(code) or (t.loc[t["code"] == code, "name"].iloc[0] if (t["code"] == code).any() else "")
        t = pd.concat([t, pd.DataFrame([{"code": code, "name": name, "start": first_day, "end": ""}])])
    t = t.sort_values(["code", "start"])
    t.to_csv(path, index=False)
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(UNIVERSE_FILE))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--add")
    ap.add_argument("--last")
    ap.add_argument("--in", dest="ins", default="")
    ap.add_argument("--out", dest="outs", default="")
    ap.add_argument("--names", default="")
    a = ap.parse_args(argv)
    if a.add:
        names = dict(kv.split("=", 1) for kv in a.names.split(",") if "=" in kv)
        add_review(a.file, a.add, a.last, [c for c in a.ins.split(",") if c], [c for c in a.outs.split(",") if c],
                   names)
    bad = check(a.file)
    if bad:
        print("成分數不是 50 的日期：", bad[:10], "..." if len(bad) > 10 else "")
        return 1
    print("OK：每個交易日恰好 50 檔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
