"""日頻回測引擎（台股規則）。

時序約定（避免偷看未來）：
    weights.loc[t] = 以 t 日收盤（含）之前資料決定的「目標權重」
    → 於 t+1 日開盤價成交
    → 持有期間報酬 = 開盤→收盤（新權重）＋ 隔夜跳空與除息（舊權重）

台股規則：
    - 買進：手續費（含折扣）＋滑價；賣出：手續費＋證交稅（股票 0.3%、ETF 0.1%）＋滑價
    - 開盤漲停（≥ 參考價 +9.5%）買不到；開盤跌停（≤ −9.5%）賣不掉；參考價 = 前收 − 當日股利
    - 停牌或尚未上市（價格 NaN）：部位維持、無法交易；復牌日以「停牌前最後收盤」為前收計算跳空，
      停牌期間的除息也累積到復牌日一起計入（舊版要求前一日收盤非 NaN，復牌日報酬被吞掉）
    - 除息日將現金股利加回（價格為未還原價；股利留在該部位中，相當於以開盤價再投入、不計成本）

未完成目標（pending）：
    目標權重因漲跌停或停牌無法成交時，記住該檔的目標，之後每天開盤重試，
    直到成交或被新的目標列（非整列 NaN）覆蓋。低頻策略（多數日子整列 NaN）也不會永遠卡住。
    已成交的股票不會因為之後價格漂移而被「拉回」目標權重。

現金限制：賣出受阻而買進照常時，總權重可能 > 1（變成融資）；此時等比例縮減當日買進量，
    未買足的部分列為 pending，隔日重試。

交易紀錄（份額帳）：
    每次買進（含加碼）視為一個新 lot（股數、成本價、日期）；賣出（含部分減碼）依 FIFO 扣減 lot，
    每個被平倉的 lot（或其部分）記為一筆 Trade。Trade.ret 已扣除買賣成本並含持有期間現金股利。
    期末未平倉 lot 以最後有效收盤價評價（open_at_end=True，不扣賣出成本）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Costs
from .data import Panel

LIMIT_BLOCK = 0.095
_EPS = 1e-9


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    dividends: float = 0.0          # 持有期間每股累積現金股利
    shares: float = 0.0             # 此筆（lot 或其部分）股數（以初始權益 1 為單位的相對股數）
    buy_cost: float = 0.0           # 買進成本率
    sell_cost: float = 0.0          # 賣出成本率（期末未平倉為 0）
    open_at_end: bool = False

    @property
    def ret(self) -> float:
        """扣成本、含息的單筆報酬。"""
        if self.exit_price is None or not self.entry_price:
            return np.nan
        return ((self.exit_price * (1 - self.sell_cost) + self.dividends)
                / (self.entry_price * (1 + self.buy_cost)) - 1)

    @property
    def pnl(self) -> float:
        """此筆損益（以初始權益 1 為單位）。"""
        if self.exit_price is None:
            return np.nan
        return self.shares * (self.exit_price * (1 - self.sell_cost) + self.dividends
                              - self.entry_price * (1 + self.buy_cost))


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame          # 實際收盤持有權重
    turnover: pd.Series
    costs: pd.Series
    trades: List[Trade] = field(default_factory=list)
    pending: Optional[pd.DataFrame] = None   # 每日收盤後仍未完成的目標（NaN = 無）

    def metrics(self) -> Dict[str, float]:
        from .metrics import compute_metrics
        return compute_metrics(self)


class _Ledger:
    """FIFO 份額帳。"""

    def __init__(self, codes, buy_cost, sell_cost):
        self.codes = codes
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.lots: Dict[int, List[Trade]] = {}
        self.closed: List[Trade] = []

    def dividend(self, i: int, d: float) -> None:
        for lot in self.lots.get(i, []):
            lot.dividends += d

    def buy(self, i: int, date, price: float, shares: float) -> None:
        if shares <= _EPS or not np.isfinite(price):
            return
        self.lots.setdefault(i, []).append(Trade(self.codes[i], date, price, shares=shares,
                                                 buy_cost=self.buy_cost))

    def sell_fraction(self, i: int, date, price: float, frac: float) -> None:
        """賣出目前持股的 frac 比例（1 = 全部），FIFO。"""
        lots = self.lots.get(i, [])
        if not lots:
            return
        total = sum(l.shares for l in lots)
        qty = total if frac >= 1 - 1e-9 else total * frac
        while lots and qty > _EPS * max(total, 1):
            lot = lots[0]
            take = min(lot.shares, qty)
            self.closed.append(Trade(lot.code, lot.entry_date, lot.entry_price, date, price, lot.dividends,
                                     take, lot.buy_cost, float(self.sell_cost[i])))
            lot.shares -= take
            qty -= take
            if lot.shares <= _EPS * max(total, 1):
                lots.pop(0)
        if not lots:
            self.lots.pop(i, None)

    def close_all(self, date, last_price: np.ndarray) -> List[Trade]:
        out = list(self.closed)
        for i, lots in self.lots.items():
            px = last_price[i]
            for lot in lots:
                p = px if np.isfinite(px) else lot.entry_price
                out.append(Trade(lot.code, lot.entry_date, lot.entry_price, date, p, lot.dividends,
                                 lot.shares, lot.buy_cost, 0.0, open_at_end=True))
        return out


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
    # 前一個「有效」收盤（停牌期間沿用停牌前收盤）
    last_c = pd.DataFrame(c).ffill().values

    is_etf = np.array([panel.is_etf(x) for x in codes])
    buy_cost = costs.buy_cost()
    sell_cost = np.where(is_etf, costs.sell_cost(True), costs.sell_cost(False))

    equity = np.ones(n_days)
    held = np.zeros((n_days, n))
    pend_hist = np.full((n_days, n), np.nan)
    turnover = np.zeros(n_days)
    cost_paid = np.zeros(n_days)
    w = np.zeros(n)                       # 目前持有權重（收盤後）
    eq = 1.0
    acc_div = np.zeros(n)                 # 停牌（無有效價格）期間累積、尚未計入的股利
    pending = np.full(n, np.nan)          # 未完成目標（NaN = 無）
    ledger = _Ledger(codes, buy_cost, sell_cost)

    for t in range(n_days):
        prev_c = last_c[t - 1] if t > 0 else np.full(n, np.nan)
        has_px = ~np.isnan(o[t]) & ~np.isnan(c[t])
        valid = has_px & ~np.isnan(prev_c)
        div_t = d[t] + acc_div                  # 今日可計入的股利（含停牌期間累積）
        div_now = np.where(valid, div_t, 0.0)
        acc_div = np.where(valid | np.isnan(prev_c), 0.0, div_t)   # 無價格日的股利延到復牌日

        # 1) 隔夜：跳空 + 除息（舊部位）
        gap = np.zeros(n)
        gap[valid] = (o[t][valid] + div_now[valid]) / prev_c[valid] - 1
        for i in np.nonzero((div_now > 0) & (w > 0))[0]:
            ledger.dividend(i, div_now[i])
        vals = w * (1 + gap)
        eq_open = eq * (1 - w.sum() + vals.sum())
        w_open = vals * eq / eq_open if eq_open > 0 else w

        # 2) 開盤調整：新目標列覆蓋 pending；否則重試 pending
        if t > 0:
            row = tgt[t - 1]
            if not np.all(np.isnan(row)):
                desired = np.clip(np.nan_to_num(row, nan=0.0), 0, None)
                if desired.sum() > 1:
                    desired = desired / desired.sum()
                pending = desired.copy()
        w_new = w_open.copy()
        has_pend = ~np.isnan(pending)
        if has_pend.any():
            ref = np.where(valid, prev_c - div_now, np.nan)
            chg = np.zeros(n)
            chg[valid] = o[t][valid] / np.where(ref[valid] > 0, ref[valid], prev_c[valid]) - 1
            can_buy = valid & (chg < LIMIT_BLOCK)
            can_sell = valid & (chg > -LIMIT_BLOCK)
            p = np.where(has_pend, pending, w_open)
            inc = has_pend & (p > w_open + _EPS)
            dec = has_pend & (p < w_open - _EPS)
            sell_ok = dec & can_sell
            buy_ok = inc & can_buy
            w_new[sell_ok] = p[sell_ok]
            w_new[buy_ok] = p[buy_ok]
            # 現金限制：總權重 > 1 時等比例縮減買進
            excess = w_new.sum() - 1
            if excess > _EPS and buy_ok.any():
                add = (w_new - w_open)[buy_ok]
                scale = max(0.0, 1 - excess / add.sum())
                w_new[buy_ok] = w_open[buy_ok] + add * scale
            done = np.abs(w_new - p) <= 1e-7
            pending = np.where(has_pend & ~done, pending, np.nan)
        delta = w_new - w_open
        cost = buy_cost * delta[delta > 0].sum() + (sell_cost * np.clip(-delta, 0, None)).sum()
        turnover[t] = np.abs(delta).sum()
        cost_paid[t] = cost

        # 份額帳（以成本前的開盤權益換算股數）
        for i in np.nonzero(np.abs(delta) > _EPS)[0]:
            if delta[i] < 0:
                ledger.sell_fraction(i, dates[t], o[t][i], -delta[i] / w_open[i] if w_open[i] > _EPS else 1.0)
            else:
                ledger.buy(i, dates[t], o[t][i], delta[i] * eq_open / o[t][i])
        eq_open *= (1 - cost)

        # 3) 盤中：開盤 → 收盤（新部位）
        intra = np.zeros(n)
        intra[valid] = c[t][valid] / o[t][valid] - 1
        # 首個交易日（無前收）：有價格者以開盤→收盤計
        first = has_px & np.isnan(prev_c)
        intra[first] = c[t][first] / o[t][first] - 1
        vals = w_new * (1 + intra)
        eq = eq_open * (1 - w_new.sum() + vals.sum())
        w = vals * eq_open / eq if eq > 0 else w_new
        equity[t] = eq
        held[t] = w
        pend_hist[t] = pending

    trades = ledger.close_all(dates[-1], last_c[-1]) if n_days else []
    eq_s = pd.Series(equity, index=dates, name="equity")
    return BacktestResult(
        equity=eq_s,
        returns=eq_s.pct_change().fillna(eq_s.iloc[0] - 1),
        weights=pd.DataFrame(held, index=dates, columns=codes),
        turnover=pd.Series(turnover, index=dates),
        costs=pd.Series(cost_paid, index=dates),
        trades=trades,
        pending=pd.DataFrame(pend_hist, index=dates, columns=codes),
    )


def buy_and_hold(panel: Panel, code: str) -> BacktestResult:
    tgt = pd.DataFrame(0.0, index=panel.dates, columns=panel.codes)
    tgt[code] = 1.0
    return backtest(panel, tgt)


def equal_weight(panel: Panel, codes: Optional[List[str]] = None) -> BacktestResult:
    """等權基準（每月第一個交易日再平衡）。

    point-in-time：只持有「當日為成分股」（panel.universe_mask）且有價格的股票；
    被調出的股票在下一個月初再平衡時賣出。panel.universe 為 None 時等同舊行為（全部股票）。
    """
    codes = codes or [x for x in panel.codes if not panel.is_etf(x)]
    tgt = pd.DataFrame(0.0, index=panel.dates, columns=panel.codes)
    avail = panel.close[codes].notna() & panel.universe_mask()[codes]
    tgt[codes] = avail.div(avail.sum(axis=1), axis=0)
    # 只在每月第一個交易日再平衡，避免每日微調的成本
    months = tgt.index.to_series().dt.to_period("M")
    month_start = months.ne(months.shift())          # 第一列也算月初
    tgt.loc[~month_start.values] = np.nan
    return backtest(panel, tgt)
