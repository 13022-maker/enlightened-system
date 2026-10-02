"""存股／殖利率策略（低週轉）。

邏輯：
    1. 股利清理：單筆現金股利 > 前收 15% 視為資料異常（例：2603 2025-06-19 記為 962 元，
       實際開盤跳空僅約 31 元），改以「前收 − 除息日開盤」跳空估算；跳空不明顯時沿用該股上一筆正常股利。
    2. 年化股利估計：以「最近一次除息日」為錨，加總錨點往前 330 天內的除息（季配 4 次、半年配 2 次、
       年配 1 次皆可涵蓋，且不受年度除息日漂移影響）；最近一次除息若已超過 365+grace 天，視為停發（0）。
       資料不足一年且尚未見到任何除息 → 未知（NaN），不參與排名。
    3. 殖利率 = 年化股利 / 收盤價。品質/風險濾網（價值陷阱）：收盤 > MA(trend_n)、
       含息 mom_n 日報酬 > mom_floor、殖利率 ≤ max_yield（過高多為一次性或景氣循環高峰）。
    4. Point-in-time 股票池：只選再平衡日當天的指數成分股（panel.universe_mask()）；被調出的持股
       不再合格，於下一個再平衡日賣出。
    5. 只在再平衡日（每季或每半年第一個交易日）調整，其餘整列 NaN；持股在名次 ≤ top_n+buffer
       且仍通過濾網則續抱（降低週轉），其餘依殖利率名次補滿，等權或殖利率加權。只做多個股，不含 ETF。

所有計算只使用 t 日收盤（含）以前資料。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data import Panel
from .base import Strategy

REBAL_MONTHS = {"Q": (1, 4, 7, 10), "H": (4, 10), "Y": (10,)}


# ---------------------------------------------------------------- 股利輔助函式
def clean_dividends(panel: Panel, max_single: float = 0.15) -> pd.DataFrame:
    """回傳清理後的股利寬表（只用除息日當天開盤與前一日收盤，無前視）。"""
    div = panel.dividend.reindex_like(panel.close).fillna(0.0)
    prev_c = panel.close.shift(1)
    ratio = div / prev_c
    bad = (div > 0) & (ratio > max_single)
    if not bad.values.any():
        return div
    gap_est = (prev_c - panel.open).clip(lower=0.0)
    gap_est = gap_est.where(gap_est <= prev_c * max_single, prev_c * max_single)
    # 上一筆「正常」股利（往前填）作為跳空不明顯時的備援
    normal = div.where((div > 0) & ~bad).ffill().shift(1)
    fallback = normal.fillna(0.0)
    est = gap_est.where(gap_est >= prev_c * 0.002, fallback)
    return div.where(~bad, est.fillna(0.0))


def cleaned_panel(panel: Panel, max_single: float = 0.15) -> Panel:
    """回傳股利已清理的 Panel 複本（回測時讓引擎不要把異常股利當成報酬）。"""
    return Panel(open=panel.open, high=panel.high, low=panel.low, close=panel.close,
                 volume=panel.volume, dividend=clean_dividends(panel, max_single),
                 chip=panel.chip, chip_is_synthetic=panel.chip_is_synthetic)


def annual_dividend(div: pd.DataFrame, close: pd.DataFrame, window_days: int = 330,
                    grace_days: int = 90) -> pd.DataFrame:
    """年化每股現金股利估計（以最近一次除息日為錨，詳見模組說明）。

    回傳 NaN 代表「資料不足一年且尚未觀察到除息」→ 未知。
    """
    dates = div.index
    out = np.full(div.shape, np.nan)
    day_num = (dates - dates[0]).days.values           # 距資料起點的天數
    for j, code in enumerate(div.columns):
        d = div[code].values
        first_valid = np.argmax(~np.isnan(close[code].values)) if close[code].notna().any() else len(d)
        ex_idx: List[int] = []
        for t in range(len(dates)):
            if t < first_valid:
                continue
            if d[t] > 0:
                ex_idx.append(t)
            hist_days = day_num[t] - day_num[first_valid]
            if not ex_idx:
                out[t, j] = 0.0 if hist_days >= 365 else np.nan
                continue
            last = ex_idx[-1]
            if day_num[t] - day_num[last] > 365 + grace_days:
                out[t, j] = 0.0                           # 久未配息 → 視為停發
                continue
            anchor = day_num[last]
            out[t, j] = sum(d[k] for k in ex_idx if anchor - day_num[k] < window_days)
    return pd.DataFrame(out, index=dates, columns=div.columns)


def rebalance_mask(dates: pd.DatetimeIndex, freq: str = "Q") -> pd.Series:
    """每個再平衡月份的第一個交易日（只需比較前一日月份，無前視）。"""
    months = REBAL_MONTHS[freq]
    m = pd.Series(dates.month, index=dates)
    first = m.ne(m.shift())
    first.iloc[0] = False                                 # 資料第一天不視為再平衡日
    return first & m.isin(months)


# ---------------------------------------------------------------- 策略
class DividendYield(Strategy):
    name = "dividend_yield"
    title = "存股殖利率策略"
    description = ("近12月現金殖利率排名取前 N 檔，排除跌破半年線／半年報酬過差／殖利率異常高的價值陷阱，"
                   "每季（或半年）再平衡、等權持有")
    default_params = {
        "top_n": 8, "freq": "Q", "buffer": 3, "weighting": "equal",
        "trend_n": 120, "mom_n": 120, "mom_floor": -0.10, "max_yield": 0.12,
        "min_hist_days": 200, "window_days": 330, "grace_days": 90, "max_single": 0.15,
    }
    # 6 組：持股數 × 再平衡頻率；濾網參數事先固定，不在格點中調整
    param_grid = [
        {"top_n": n, "freq": f} for n in (5, 8, 12) for f in ("Q", "H")
    ]

    # -------------------------------------------------- 內部計算
    def _features(self, panel: Panel, trend_n=120, mom_n=120, min_hist_days=200,
                  window_days=330, grace_days=90, max_single=0.15, **_) -> Dict[str, pd.DataFrame]:
        close = panel.close
        div = clean_dividends(panel, max_single)
        ann = annual_dividend(div, close, window_days, grace_days)
        yld = ann / close
        # 全域暖身：資料起點後至少 min_hist_days 天才開始計算殖利率
        start = panel.dates[0] if len(panel.dates) else None
        if start is not None:
            too_early = (panel.dates - start).days < min_hist_days
            yld.loc[too_early] = np.nan
        tri = ind.total_return_index(close, div)
        ma = ind.sma(close, trend_n)
        mom = tri / tri.shift(mom_n) - 1
        return {"yield": yld, "ma": ma, "mom": mom, "close": close}

    def _eligible(self, f, mom_floor=-0.10, max_yield=0.12, **_) -> pd.DataFrame:
        return ((f["yield"] > 0) & (f["yield"] <= max_yield)
                & (f["close"] > f["ma"]) & (f["mom"] > mom_floor))

    def _stock_cols(self, panel: Panel) -> List[str]:
        return [c for c in panel.codes if not panel.is_etf(c)]

    # -------------------------------------------------- 介面
    def scores(self, panel: Panel, **params) -> pd.DataFrame:
        p = {**self.default_params, **params}
        f = self._features(panel, **p)
        out = f["yield"].copy()
        for c in panel.codes:
            if panel.is_etf(c):
                out[c] = np.nan
        return out

    def target_weights(self, panel: Panel, **params) -> pd.DataFrame:
        p = {**self.default_params, **params}
        top_n, buffer, weighting = int(p["top_n"]), int(p["buffer"]), p["weighting"]
        f = self._features(panel, **p)
        elig = self._eligible(f, **p) & panel.universe_mask()
        stocks = self._stock_cols(panel)
        yld = f["yield"][stocks]
        reb = rebalance_mask(panel.dates, p["freq"])

        w = pd.DataFrame(np.nan, index=panel.dates, columns=panel.codes)
        held: List[str] = []
        for t in panel.dates[reb.values]:
            y = yld.loc[t]
            if y.notna().sum() == 0:                     # 暖身期：無任何殖利率可用 → 不調整
                continue
            e = elig.loc[t, stocks]
            cand = y[e.fillna(False)].sort_values(ascending=False)
            rank = {c: i + 1 for i, c in enumerate(cand.index)}
            keep = [c for c in held if c in rank and rank[c] <= top_n + buffer]
            keep = sorted(keep, key=lambda c: rank[c])[:top_n]
            picks = keep + [c for c in cand.index if c not in keep][: top_n - len(keep)]
            row = pd.Series(0.0, index=panel.codes)
            if picks:
                if weighting == "yield":
                    yy = cand[picks]
                    row[picks] = yy / yy.sum() * len(picks) / top_n
                else:
                    row[picks] = 1.0 / top_n            # 不足 N 檔的部分留現金
            w.loc[t] = row.values
            held = picks
        return w

    def explain(self, panel: Panel, code: str, **params) -> str:
        p = {**self.default_params, **params}
        if panel.is_etf(code):
            return "ETF 不在選股範圍"
        f = self._features(panel, **p)
        t = panel.dates[-1]
        y = f["yield"].loc[t, code]
        if pd.isna(y):
            return "股利資料不足一年，無法估算殖利率"
        stocks = self._stock_cols(panel)
        ranks = f["yield"].loc[t, stocks].rank(ascending=False, method="min")
        c, ma, mom = f["close"].loc[t, code], f["ma"].loc[t, code], f["mom"].loc[t, code]
        trend = "站上" if c > ma else "跌破"
        line = "年線" if p["trend_n"] >= 200 else "半年線" if p["trend_n"] >= 100 else f"MA{p['trend_n']}"
        s = f"近12月殖利率 {y:.1%}，排名第{int(ranks[code])}，{trend}{line}"
        if not pd.isna(mom):
            s += f"，近{p['mom_n']}日含息報酬 {mom:+.1%}"
        reasons = []
        if y <= 0:
            reasons.append("無配息")
        if y > p["max_yield"]:
            reasons.append("殖利率異常偏高")
        if not (c > ma):
            reasons.append("趨勢轉弱")
        if not pd.isna(mom) and mom <= p["mom_floor"]:
            reasons.append("股價長期下跌")
        if reasons:
            s += "（價值陷阱濾網剔除：" + "、".join(reasons) + "）"
        return s
