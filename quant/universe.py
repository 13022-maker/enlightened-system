"""Point-in-time 股票池（歷史成分股），用來消除存活者偏差。

成分股檔 data/universe/tw50_membership.csv：
    code,name,start,end
    - start：該股「開始」為成分股的第一個交易日（含）。空白 = 早於資料起點即為成分股。
    - end：該股「最後」為成分股的交易日（含）。空白 = 至今仍為成分股。
    - 同一檔股票可有多列（調出後又調入，例：3037 欣興 2025-06 調出、2026-03 再調入）。
    台灣50 定期審核於「生效日（週五）收盤後」生效 → 調出股最後成分日 = 該週五，調入股 start = 下週一。

時序：成分股名單在審核公告日（生效日前約兩週）即已公開，因此在 start 當天使用該名單沒有前視；
本模組不推估「公告日 ~ 生效日」之間的預期效應。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


class Universe:
    def __init__(self, table: pd.DataFrame):
        t = table.copy()
        t["code"] = t["code"].astype(str).str.strip()
        t["start"] = pd.to_datetime(t["start"], errors="coerce")
        t["end"] = pd.to_datetime(t["end"], errors="coerce")
        self.table = t.reset_index(drop=True)

    @classmethod
    def from_csv(cls, path) -> "Universe":
        return cls(pd.read_csv(Path(path), dtype={"code": str}, comment="#"))

    @property
    def codes(self) -> List[str]:
        """所有曾經出現在名單中的股票（依檔案順序、去重）。"""
        return list(dict.fromkeys(self.table["code"]))

    def _active(self, date: pd.Timestamp) -> pd.Series:
        t = self.table
        return (t["start"].isna() | (t["start"] <= date)) & (t["end"].isna() | (t["end"] >= date))

    def members_on(self, date) -> List[str]:
        """某日的成分股代號清單。"""
        d = pd.Timestamp(date)
        return sorted(set(self.table.loc[self._active(d), "code"]))

    def membership_mask(self, dates: Iterable, codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """回傳 date×code 布林表（True = 當日為成分股）。codes 預設為名單中所有股票。"""
        idx = pd.DatetimeIndex(dates)
        cols = list(codes) if codes is not None else self.codes
        out = pd.DataFrame(False, index=idx, columns=cols)
        for _, r in self.table.iterrows():
            if r["code"] not in out.columns:
                continue
            m = pd.Series(True, index=idx)
            if pd.notna(r["start"]):
                m &= idx >= r["start"]
            if pd.notna(r["end"]):
                m &= idx <= r["end"]
            out[r["code"]] |= m.values
        return out


def load_universe(path=None) -> Optional[Universe]:
    from .config import UNIVERSE_FILE
    p = Path(path or UNIVERSE_FILE)
    return Universe.from_csv(p) if p.exists() else None


def members_on(date, path=None) -> List[str]:
    u = load_universe(path)
    if u is None:
        raise FileNotFoundError("找不到成分股檔 data/universe/tw50_membership.csv")
    return u.members_on(date)


def membership_mask(dates, codes, path=None) -> pd.DataFrame:
    """無成分股檔時回傳全 True（= 不做 point-in-time 篩選）。"""
    u = load_universe(path)
    if u is None:
        return pd.DataFrame(True, index=pd.DatetimeIndex(dates), columns=list(codes))
    return u.membership_mask(dates, codes)
