"""波段動能策略（橫截面動能 + 趨勢濾網 + 大盤濾網 + ATR 移動停損 + 持有緩衝區）。

邏輯：
    1. 分數 = 含息報酬指數的 lookback 日動能（跳過最近 skip 日，避開短期反轉雜訊）。
    2. 只在再平衡日（每 rebal_weeks 週的第一個交易日）調整持股：
       - 候選股：非 ETF、收盤 > MA(ma_trend)、動能 > 0
       - 大盤濾網：0050 收盤 < MA(regime_ma) → 全數出場、空手
       - 新進：排名前 top_n；已持有：排名仍在前 top_n + buffer 且仍站上均線就續抱（緩衝區降低週轉）
       - 持股組合不變時整列 NaN（不交易，不做再平衡微調）
       - 續抱者不修剪回等權；賣出釋放的現金分給新進者（每檔上限 1/top_n）
    3. 每日檢查 ATR 移動停損：收盤 < 持有期最高收盤 − atr_mult × ATR(atr_n) → 隔日開盤出場，
       其餘持股以「估計的漂移權重」送出，避免為了把其他部位拉回等權而產生額外週轉；
       停損釋放的現金留到下個再平衡日才再投入。
    4. 股利欄異常值（單次 > 前收 15%）視為資料錯誤，見 clean_dividend()。
    5. Point-in-time 股票池：候選股只限再平衡日當天的指數成分股（panel.universe_mask()）；
       已持有但被調出的股票不再符合續抱條件，於下一個再平衡日賣出。無成分股資料時全部可選（舊行為）。

時序：t 日權重只用到 ≤ t 日的收盤資料，於 t+1 開盤成交（由引擎處理）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..config import BENCHMARK
from ..data import Panel
from .base import Strategy


def _rebalance_mask(dates: pd.DatetimeIndex, every_weeks: int) -> np.ndarray:
    """每 every_weeks 週的第一個交易日為再平衡日。

    只比較「當日與前一交易日」是否跨週，不需知道下一個交易日 → 無前視。
    週次以固定錨點（1970-01-05 週一）起算，截斷資料不會改變哪幾天是再平衡日。
    """
    if len(dates) == 0:
        return np.zeros(0, dtype=bool)
    anchor = pd.Timestamp("1970-01-05")
    week_no = ((dates - anchor).days // 7).values
    new_week = np.r_[True, week_no[1:] != week_no[:-1]]
    return new_week & (week_no % every_weeks == 0) if every_weeks > 1 else new_week


MAX_DIV_YIELD = 0.15


def clean_dividend(panel: Panel, max_yield: float = MAX_DIV_YIELD) -> pd.DataFrame:
    """防禦性清理：單次現金股利 > 前收 15% 視為資料錯誤，當作 0。

    資料中 2025-06~07 有數筆股利欄異常（例如 2603 於 2025-06-19 記為 962 元，股價僅約 247 元），
    若直接用於含息動能，會讓該股動能虛增數倍並長期霸佔排名。只用當日與前一日資料，無前視。
    """
    d = panel.dividend.reindex(columns=panel.codes).fillna(0.0)
    y = d / panel.close.shift(1)
    return d.where(~(y > max_yield), 0.0)


class SwingMomentum(Strategy):
    name = "swing_momentum"
    title = "波段動能（趨勢＋ATR停損）"
    description = ("台灣50個股 lookback 日動能排名，站上 MA 且大盤 0050 多頭才持股；"
                   "每 4 週再平衡、持有緩衝區、ATR 移動停損")
    default_params = {
        "lookback": 120, "skip": 5, "top_n": 8, "buffer": 4, "rebal_weeks": 4,
        "ma_trend": 60, "regime_ma": 60, "atr_n": 14, "atr_mult": 8.0,
    }
    # 8 組：動能長度 × 持股數 × 停損寬度（4×ATR 一般波段停損 / 8×ATR 僅防災難性下跌）
    param_grid = [
        {"lookback": lb, "top_n": n, "atr_mult": m}
        for lb in (60, 120) for n in (5, 8) for m in (4.0, 8.0)
    ]

    # ------------------------------------------------------------ 指標
    def _features(self, panel: Panel, lookback=60, skip=5, ma_trend=60, regime_ma=60, atr_n=14, **_):
        c = panel.close
        tri = ind.total_return_index(c, clean_dividend(panel))
        mom = tri.shift(skip) / tri.shift(lookback) - 1
        ma = ind.sma(c, ma_trend)
        atr = ind.atr(panel.high, panel.low, c, atr_n)
        tradable = pd.Series([not panel.is_etf(x) for x in panel.codes], index=panel.codes)
        mom = mom.loc[:, tradable.values].reindex(columns=panel.codes)   # ETF 分數為 NaN
        if BENCHMARK in panel.codes and regime_ma and regime_ma > 1:
            b = c[BENCHMARK]
            regime = (b > ind.sma(b.to_frame(), regime_ma)[BENCHMARK])
            regime = regime.where(b.notna(), np.nan).ffill().fillna(False).astype(bool)
        else:
            regime = pd.Series(True, index=panel.dates)   # 無 0050 或 regime_ma=0 時不啟用大盤濾網
        return {"close": c, "tri": tri, "mom": mom, "ma": ma, "atr": atr,
                "tradable": tradable, "regime": regime}

    def scores(self, panel: Panel, **params) -> pd.DataFrame:
        """分數 = 動能；未站上趨勢均線者分數 −1 以下（排到後面但保留相對順序）。"""
        params = {**self.default_params, **params}
        f = self._features(panel, **params)
        ok = f["close"] > f["ma"]
        return f["mom"].where(ok, f["mom"] - 1.0)

    # ------------------------------------------------------------ 主邏輯
    def _simulate(self, panel: Panel, lookback=120, skip=5, top_n=8, buffer=4, rebal_weeks=4,
                  ma_trend=60, regime_ma=60, atr_n=14, atr_mult=8.0, **_) -> Tuple[pd.DataFrame, Dict]:
        f = self._features(panel, lookback=lookback, skip=skip, ma_trend=ma_trend,
                           regime_ma=regime_ma, atr_n=atr_n)
        c = f["close"].values
        d = clean_dividend(panel).values
        mom = f["mom"].values
        ma = f["ma"].values
        atr = f["atr"].values
        regime = f["regime"].values
        member = panel.universe_mask().values
        rebal = _rebalance_mask(panel.dates, rebal_weeks)
        n_days, n = c.shape

        out = np.full((n_days, n), np.nan)
        held: Dict[int, Dict[str, float]] = {}   # i -> {"peak": 最高收盤, "v": 估計部位價值}
        cash = 1.0
        info = {"stops": 0, "regime_off_days": 0}

        for t in range(n_days):
            # 1) 依當日收盤更新估計部位價值（含除息）與最高價
            if t > 0:
                for i, h in held.items():
                    if not np.isnan(c[t, i]) and not np.isnan(c[t - 1, i]):
                        h["v"] *= (c[t, i] + d[t, i]) / c[t - 1, i]
                    if not np.isnan(c[t, i]):
                        h["peak"] = max(h["peak"], c[t, i])

            changed = False
            # 2) ATR 移動停損（每日）
            for i in list(held):
                p, a = c[t, i], atr[t, i]
                if np.isnan(p) or np.isnan(a):
                    continue
                if p < held[i]["peak"] - atr_mult * a:
                    cash += held.pop(i)["v"]
                    info["stops"] += 1
                    changed = True

            # 3) 再平衡日：重新排名
            if rebal[t]:
                if not regime[t]:
                    info["regime_off_days"] += 1
                    target_set: List[int] = []
                else:
                    elig = (~np.isnan(mom[t]) & ~np.isnan(ma[t]) & (c[t] > ma[t]) & (mom[t] > 0)
                            & member[t])
                    cand = np.nonzero(elig)[0]
                    ranked = cand[np.argsort(-mom[t, cand], kind="stable")]
                    rank = {i: r for r, i in enumerate(ranked)}
                    keep = [i for i in held if i in rank and rank[i] < top_n + buffer]
                    new = [i for i in ranked if i not in keep][: max(top_n - len(keep), 0)]
                    target_set = keep + new
                if set(target_set) != set(held):
                    # 續抱者維持漂移後權重（不修剪），賣出釋放的現金平均分給新進者，
                    # 每檔新進上限 1/top_n，避免為了「拉回等權」產生額外週轉
                    total = cash + sum(h["v"] for h in held.values())
                    for i in [i for i in held if i not in target_set]:
                        cash += held.pop(i)["v"]
                    new = [i for i in target_set if i not in held]
                    if new:
                        alloc = min(total / top_n, cash / len(new))
                        for i in new:
                            held[i] = {"peak": c[t, i], "v": alloc}
                            cash -= alloc
                    changed = True

            # 4) 有變動才送出整列權重，否則整列 NaN（不調整）
            if changed:
                total = cash + sum(h["v"] for h in held.values())
                row = np.zeros(n)
                for i, h in held.items():
                    row[i] = h["v"] / total
                out[t] = row
                # 正規化估計值，避免數值漂移
                cash, held = cash / total, {i: {**h, "v": h["v"] / total} for i, h in held.items()}

        w = pd.DataFrame(out, index=panel.dates, columns=panel.codes)
        info["held"] = sorted(panel.codes[i] for i in held)
        return w, info

    def target_weights(self, panel: Panel, **params) -> pd.DataFrame:
        params = {**self.default_params, **params}
        return self._simulate(panel, **params)[0]

    # ------------------------------------------------------------ 說明
    def explain(self, panel: Panel, code: str, **params) -> str:
        params = {**self.default_params, **params}
        if panel.is_etf(code):
            return "ETF 不交易（0050 僅作大盤濾網）"
        f = self._features(panel, **params)
        mom = f["mom"].iloc[-1]
        m = mom.get(code, np.nan)
        if pd.isna(m):
            return "資料不足"
        rank = int((mom > m).sum()) + 1
        c, ma = f["close"][code].iloc[-1], f["ma"][code].iloc[-1]
        trend = f"站上MA{params['ma_trend']}" if c > ma else f"跌破MA{params['ma_trend']}"
        regime = "大盤多頭" if f["regime"].iloc[-1] else f"大盤跌破MA{params['regime_ma']}（空手）"
        _, info = self._simulate(panel, **params)
        hold = "持有中" if code in info["held"] else "未持有"
        return (f"{params['lookback']}日動能 {m:+.1%}（排名第{rank}）、{trend}、{regime}、{hold}")
