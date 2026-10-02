"""日頻回測引擎（台股規則）。

時序約定（避免偷看未來）：
    weights.loc[t] = 以 t 日收盤（含）之前資料決定的「目標權重」
    → 於 t+1 日開盤價成交
    → 持有期間報酬 = 開盤→收盤（新權重）＋ 隔夜跳空與除息（舊權重）

台股規則：
    - 買進：手續費（含折扣）＋滑價；賣出：手續費＋證交稅（股票 0.3%、ETF 0.1%）＋滑價
    - 開盤漲停（≥ 前收 +9.5%）買不到；開盤跌停（≤ -9.5%）賣不掉
    - 停牌或尚未上市（價格 NaN）維持原部位
    - 除息日將現金股利加回（價格為未還原價）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Costs
from .data import Panel

LIMIT_BLOCK = 0.095


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    dividends: float = 0.0

    @property
    def ret(self) -> float:
        if self.exit_price is None:
            return np.nan
        return (self.exit_price + self.dividends) / self.entry_price - 1


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame          # 實際收盤持有權重
    turnover: pd.Series
    costs: pd.Series
    trades: List[Trade] = field(default_factory=list)

    def metrics(self) -> Dict[str, float]:
        from .metrics import compute_metrics
        return compute_metrics(self)


def backtest(panel: Panel, target: pd.DataFrame, costs: type = Costs) -> BacktestResult:
    """執行回測。target：index=date, columns=code 的目標權重（列總和 ≤ 1，其餘為現金）。"""
    dates = panel.dates
    codes = panel.codes
    target = target.reindex(index=dates, columns=codes)

    o = panel.open.reindex(columns=codes).values
    c = panel.close.reindex(columns=codes).values
    d = panel.dividend.reindex(columns=codes).fillna(0.0).values
    tgt = target.values
    n_days, n = c.shape

    is_etf = np.array([panel.is_etf(x) for x in codes])
    buy_cost = costs.buy_cost()
    sell_cost = np.where(is_etf, costs.sell_cost(True), costs.sell_cost(False))

    equity = np.ones(n_days)
    held = np.zeros((n_days, n))
    turnover = np.zeros(n_days)
    cost_paid = np.zeros(n_days)
    w = np.zeros(n)                       # 目前持有權重（收盤後）
    eq = 1.0
    open_trades: Dict[int, Trade] = {}
    trades: List[Trade] = []

    for t in range(n_days):
        prev_c = c[t - 1] if t > 0 else c[t]
        valid = ~np.isnan(o[t]) & ~np.isnan(c[t]) & ~np.isnan(prev_c)

        # 1) 隔夜：跳空 + 除息（舊部位）
        gap = np.zeros(n)
        gap[valid] = (o[t][valid] + d[t][valid]) / prev_c[valid] - 1
        for i in np.nonzero((d[t] > 0) & (w > 0))[0]:
            if i in open_trades:
                open_trades[i].dividends += d[t][i]
        vals = w * (1 + gap)
        eq_open = eq * (1 - w.sum() + vals.sum())
        w_open = vals * eq / eq_open if eq_open > 0 else w

        # 2) 開盤調整至前一日決定的目標權重
        w_new = w_open.copy()
        if t > 0:
            row = tgt[t - 1]
            if not np.all(np.isnan(row)):
                desired = np.nan_to_num(row, nan=0.0)
                desired = np.clip(desired, 0, None)
                if desired.sum() > 1:
                    desired = desired / desired.sum()
                chg = gap - d[t] / np.where(valid, prev_c, 1)          # 純價格跳空
                can_buy = valid & (chg < LIMIT_BLOCK)
                can_sell = valid & (chg > -LIMIT_BLOCK)
                inc = desired > w_open
                dec = desired < w_open
                upd = (inc & can_buy) | (dec & can_sell)
                w_new[upd] = desired[upd]
        delta = w_new - w_open
        cost = buy_cost * delta[delta > 0].sum() + (sell_cost * np.clip(-delta, 0, None)).sum()
        turnover[t] = np.abs(delta).sum()
        cost_paid[t] = cost
        eq_open *= (1 - cost)

        # 交易紀錄
        for i in range(n):
            if w_open[i] <= 1e-9 < w_new[i] and i not in open_trades:
                open_trades[i] = Trade(codes[i], dates[t], o[t][i])
            elif w_new[i] <= 1e-9 and i in open_trades:
                tr = open_trades.pop(i)
                tr.exit_date, tr.exit_price = dates[t], o[t][i]
                trades.append(tr)

        # 3) 盤中：開盤 → 收盤（新部位）
        intra = np.zeros(n)
        intra[valid] = c[t][valid] / o[t][valid] - 1
        vals = w_new * (1 + intra)
        eq = eq_open * (1 - w_new.sum() + vals.sum())
        w = vals * eq_open / eq if eq > 0 else w_new
        equity[t] = eq
        held[t] = w

    for i, tr in open_trades.items():   # 期末未平倉以最後收盤價結算
        last = c[-1][i] if not np.isnan(c[-1][i]) else tr.entry_price
        tr.exit_date, tr.exit_price = dates[-1], last
        trades.append(tr)

    eq_s = pd.Series(equity, index=dates, name="equity")
    return BacktestResult(
        equity=eq_s,
        returns=eq_s.pct_change().fillna(eq_s.iloc[0] - 1),
        weights=pd.DataFrame(held, index=dates, columns=codes),
        turnover=pd.Series(turnover, index=dates),
        costs=pd.Series(cost_paid, index=dates),
        trades=trades,
    )


def buy_and_hold(panel: Panel, code: str) -> BacktestResult:
    tgt = pd.DataFrame(0.0, index=panel.dates, columns=panel.codes)
    tgt[code] = 1.0
    return backtest(panel, tgt)


def equal_weight(panel: Panel, codes: Optional[List[str]] = None) -> BacktestResult:
    codes = codes or [x for x in panel.codes if not panel.is_etf(x)]
    tgt = pd.DataFrame(0.0, index=panel.dates, columns=panel.codes)
    avail = panel.close[codes].notna()
    tgt[codes] = avail.div(avail.sum(axis=1), axis=0)
    # 只在每月第一個交易日再平衡，避免每日微調的成本
    months = tgt.index.to_series().dt.to_period("M")
    month_start = months.ne(months.shift())          # 第一列也算月初
    tgt.loc[~month_start.values] = np.nan
    return backtest(panel, tgt)
