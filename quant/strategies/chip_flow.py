"""籌碼面策略：三大法人（外資＋投信）買賣超強度 + 價格趨勢濾網。

邏輯：
    1. 標準化：法人 N 日累積買賣超 ÷ 同期 N 日成交量（皆為股數）→「買超佔量比」，
       可跨個股比較（大型權值股與中型股的張數不能直接比）。
    2. 分數 = 外資佔量比 + trust_weight × 投信佔量比 + streak_bonus × min(投信連買日數, N)/N
       （投信規模小、連續買進的訊號較明確，給予較高權重與連買加分）。
    3. 新進場濾網（全部成立才「合格」）：收盤 > MA(ma)（避免接刀）、分數 > 0（法人淨買）、
       非 ETF、資料完整（成交量 > 0、指標暖身完成）。
    4. 再平衡：每 rebalance_weeks 週的第一個交易日（以固定曆法錨點計算，與資料起點無關）。
       已持有者只要「全體排名 < top_n + buffer」、「分數 > 0（法人仍淨買）」且
       「收盤 > MA(exit_ma)」即續抱（緩衝區，
       用較寬鬆的條件出場以降低週轉）；其餘名額由合格股中排名最高者補滿；
       每檔等權 1/top_n，不足 top_n 檔的部分為現金。
    5. 非再平衡日輸出整列 NaN（= 不調整）。
    6. 籌碼缺資料的防護（FinMind 法人資料常晚於收盤價公布）：
       - 「籌碼可用日」＝ 當天至少一檔有法人資料（只看當日，無前視）。
       - 再平衡日若籌碼尚未公布 → 該列整列 NaN（不調整），再平衡順延到下一個籌碼可用日
         （最多順延 _MAX_DEFER 個交易日），避免 rolling 整段 NaN 導致全部持股被判 SELL。
       - 已持有者若分數因籌碼缺漏而為 NaN，只要價格仍在 MA(exit_ma) 之上就續抱
         （缺資料 ≠ 賣出訊號）。
       - scores()/explain() 對籌碼缺漏的日子沿用最後一個籌碼可用日的數值（ffill，只用過去）。

時序：t 日權重只用 ≤ t 日的收盤與籌碼（法人資料於收盤後公布），隔日開盤成交，無前視。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data import Panel
from .base import Strategy

_CHIP_KEYS = ("foreign_net", "trust_net", "dealer_net")
# 固定曆法錨點（星期一），讓「每 k 週再平衡」的排程不因資料起點不同而改變
_WEEK_ANCHOR = pd.Timestamp("2000-01-03")
# 再平衡日遇到籌碼未公布時，最多順延幾個交易日（超過則放棄本次再平衡）
_MAX_DEFER = 5


def _require_chip(panel: Panel) -> None:
    missing = [k for k in ("foreign_net", "trust_net") if k not in (panel.chip or {})]
    if not panel.chip or missing:
        raise ValueError(
            "chip_flow 策略需要三大法人籌碼資料（panel.chip 至少需含 foreign_net、trust_net），"
            "但目前為空或缺欄位"
            + (f"：缺少 {', '.join(missing)}" if panel.chip else "")
            + "。請以 load_panel(codes, chip=True) 載入，或先執行 "
              "`python3 tools/fetch_chip.py` 抓取 data/chip/<code>.csv。")


def _streak(flag: pd.DataFrame) -> pd.DataFrame:
    """逐欄計算「截至當日」的連續 True 天數（只用過去資料）。"""
    a = flag.fillna(False).values.astype(bool)
    out = np.zeros(a.shape, dtype=float)
    for t in range(a.shape[0]):
        out[t] = np.where(a[t], (out[t - 1] if t > 0 else 0) + 1, 0)
    return pd.DataFrame(out, index=flag.index, columns=flag.columns)


def _chip_available(panel: Panel) -> pd.Series:
    """逐日判斷籌碼是否已公布：當天 foreign_net 與 trust_net 至少各有一檔非 NaN。

    只看當日那一列，與未來資料無關（不會造成前視）。
    """
    idx, cols = panel.dates, panel.codes
    ok = pd.Series(True, index=idx)
    for k in ("foreign_net", "trust_net"):
        ok &= panel.chip[k].reindex(index=idx, columns=cols).notna().any(axis=1)
    return ok


def _rebalance_mask(dates: pd.DatetimeIndex, weeks: int) -> np.ndarray:
    """每 weeks 週的第一個交易日為 True。週序號以固定錨點計算。"""
    wk = ((dates.normalize() - _WEEK_ANCHOR).days // 7).values
    first_of_week = np.r_[True, wk[1:] != wk[:-1]]
    return first_of_week & (wk % max(int(weeks), 1) == 0)


class ChipFlow(Strategy):
    name = "chip_flow"
    title = "法人籌碼動能"
    description = ("外資＋投信 N 日買超佔成交量比排名（投信加權、連買加分），"
                   "收盤站上 MA20 才進場，每 k 週再平衡＋緩衝區續抱，只做多個股")
    needs_chip = True
    default_params = {
        "lookback": 20,          # 累積買超天數 N
        "top_n": 10,             # 持股檔數上限（每檔 1/top_n）
        "buffer": 20,            # 已持有者排名 < top_n + buffer 即可續抱
        "rebalance_weeks": 4,    # 每幾週再平衡一次
        "ma": 20,                # 新進場趨勢濾網均線
        "exit_ma": 60,           # 持有中跌破此均線才出場（較寬鬆，降低週轉）
        "trust_weight": 2.0,     # 投信佔量比的權重
        "streak_bonus": 0.02,    # 投信連買滿 N 日的加分（約等於 2% 佔量比）
    }
    param_grid = [
        {"lookback": lb, "top_n": n, "rebalance_weeks": k}
        for lb in (20, 40) for n in (8, 12) for k in (4, 6)
    ]   # 共 8 組

    # ------------------------------------------------------------ 核心計算
    def _features(self, panel: Panel, lookback=20, ma=20, exit_ma=60, trust_weight=2.0,
                  streak_bonus=0.02, **_) -> Dict[str, pd.DataFrame]:
        _require_chip(panel)
        idx, cols = panel.dates, panel.codes
        chip = {k: panel.chip[k].reindex(index=idx, columns=cols).astype(float)
                for k in _CHIP_KEYS if k in panel.chip}
        n = int(lookback)
        vol = panel.volume.reindex(index=idx, columns=cols).astype(float)
        vol_n = vol.rolling(n, min_periods=n).sum().where(lambda x: x > 0)
        f_n = chip["foreign_net"].rolling(n, min_periods=n).sum()
        t_n = chip["trust_net"].rolling(n, min_periods=n).sum()
        f_ratio = f_n / vol_n
        t_ratio = t_n / vol_n
        t_streak = _streak(chip["trust_net"] > 0)
        score = (f_ratio + trust_weight * t_ratio
                 + streak_bonus * t_streak.clip(upper=n) / n)
        ma_s = ind.sma(panel.close, int(ma))
        trend = panel.close > ma_s
        not_etf = pd.Series([not panel.is_etf(c) for c in cols], index=cols)
        score = score.where(ma_s.notna())
        score.loc[:, ~not_etf.values] = np.nan          # ETF 不評分
        eligible = trend & (score > 0) & score.notna()
        exit_ok = panel.close > ind.sma(panel.close, int(exit_ma))
        return {"score": score, "eligible": eligible, "hold_ok": exit_ok, "f_ratio": f_ratio, "t_ratio": t_ratio,
                "chip_ok": _chip_available(panel),
                "f_n": f_n, "t_n": t_n, "t_streak": t_streak, "ma": ma_s,
                "f_streak": _streak(chip["foreign_net"] > 0),
                "f_prev": chip["foreign_net"].rolling(n, min_periods=n).sum().shift(n)}

    def scores(self, panel: Panel, **params) -> pd.DataFrame:
        """顯示用分數：籌碼未公布的日子沿用最後一個籌碼可用日的分數（ffill，只用過去資料）。"""
        p = {**self.default_params, **params}
        feat = self._features(panel, **p)
        return _carry_over_missing_chip(feat["score"], feat["chip_ok"])

    def target_weights(self, panel: Panel, lookback=20, top_n=10, buffer=20, rebalance_weeks=4,
                       ma=20, exit_ma=60, trust_weight=2.0, streak_bonus=0.02, **_) -> pd.DataFrame:
        feat = self._features(panel, lookback=lookback, ma=ma, exit_ma=exit_ma,
                              trust_weight=trust_weight, streak_bonus=streak_bonus)
        score = feat["score"].values
        elig = feat["eligible"].values
        hold_ok = feat["hold_ok"].values
        reb = _rebalance_mask(panel.dates, rebalance_weeks)
        chip_ok = feat["chip_ok"].values
        univ = panel.universe_mask().values      # 歷史成分股遮罩：只能新選入當日成分股，被調出者於再平衡日出場
        n_days, n = score.shape
        top_n, buffer = int(top_n), int(buffer)
        size = 1.0 / top_n
        w = np.full((n_days, n), np.nan)
        held: set = set()
        pending = -1                     # 被順延的再平衡日索引（-1 = 無）
        for t in range(n_days):
            if reb[t]:
                pending = t
            if pending < 0:
                continue
            if not chip_ok[t]:
                # 籌碼尚未公布：本日不調整（整列 NaN），再平衡順延；超過上限就放棄
                if t - pending >= _MAX_DEFER:
                    pending = -1
                continue
            pending = -1
            scored = [i for i in range(n) if not np.isnan(score[t, i])]
            ranked = sorted(scored, key=lambda i: (-score[t, i], i))  # 同分以欄位順序穩定排序
            rank = {i: r for r, i in enumerate(ranked)}
            # 續抱條件：排名在緩衝區內且法人仍淨買；或分數因籌碼缺漏為 NaN 但價格趨勢仍在
            keep = [i for i in held if univ[t, i] and hold_ok[t, i] and (
                (i in rank and rank[i] < top_n + buffer and score[t, i] > 0)
                or np.isnan(score[t, i]))]
            keep = sorted(keep, key=lambda i: (rank.get(i, len(ranked)), i))[:top_n]  # 無分數者排最後
            new = [i for i in ranked if elig[t, i] and univ[t, i] and i not in keep][: top_n - len(keep)]
            held = set(keep) | set(new)
            w[t] = 0.0
            for i in held:
                w[t, i] = size
        return pd.DataFrame(w, index=panel.dates, columns=panel.codes)

    # ------------------------------------------------------------ 說明
    def explain(self, panel: Panel, code: str, **params) -> str:
        p = {**self.default_params, **params}
        if panel.is_etf(code):
            return "ETF 不納入籌碼策略"
        feat = self._features(panel, **p)
        n = int(p["lookback"])
        ok = feat["chip_ok"]
        if not ok.any():
            return "尚無籌碼資料"
        # 籌碼未公布時，以最後一個籌碼可用日說明，並註明資料日期
        at = ok[ok].index[-1]
        g = lambda k: feat[k][code].loc[at]
        if pd.isna(g("f_ratio")) or pd.isna(g("ma")):
            return "資料不足（暖身期、成交量為 0 或籌碼缺漏）"
        parts = []
        if at != panel.dates[-1]:
            parts.append(f"籌碼資料至{at.date()}")
        ts, fs = int(g("t_streak")), int(g("f_streak"))
        if ts >= 2:
            parts.append(f"投信連買{ts}日")
        if fs >= 2:
            parts.append(f"外資連買{fs}日")
        fp = g("f_prev")
        if not pd.isna(fp) and fp < 0 < g("f_n"):
            parts.append("外資轉買")
        elif not pd.isna(fp) and fp > 0 > g("f_n"):
            parts.append("外資轉賣")
        parts.append(f"外資{n}日{'買' if g('f_n') >= 0 else '賣'}超{abs(g('f_n')) / 1000:,.0f}張"
                     f"（佔量{g('f_ratio'):+.1%}）")
        parts.append(f"投信{n}日{'買' if g('t_n') >= 0 else '賣'}超{abs(g('t_n')) / 1000:,.0f}張"
                     f"（佔量{g('t_ratio'):+.1%}）")
        above = panel.close[code].loc[at] > g("ma")
        parts.append(f"收盤{'站上' if above else '跌破'}MA{int(p['ma'])}")
        sc = g("score")
        parts.append(f"分數{sc:+.3f}" + ("（合格）" if bool(feat["eligible"][code].iloc[-1]) else ""))
        return "、".join(parts)


def _carry_over_missing_chip(df: pd.DataFrame, chip_ok: pd.Series) -> pd.DataFrame:
    """籌碼未公布日的數值改用最後一個籌碼可用日的值（向前填補，不使用未來資料）。

    籌碼可用日的數值維持原樣（包含暖身期 / ETF 的 NaN）。
    """
    bad = ~chip_ok.reindex(df.index).fillna(False).astype(bool)
    out = df.copy()
    out.loc[bad] = np.nan
    filled = out.ffill()
    out.loc[bad] = filled.loc[bad]
    return out
