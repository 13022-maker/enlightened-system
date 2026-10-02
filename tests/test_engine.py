"""回測引擎單元測試：用人工資料驗證成本、時序、漲停限制、除息與分割調整。"""
import numpy as np
import pandas as pd
import pytest

from quant.config import Costs
from quant.data import Panel, adjust_splits
from quant.engine import backtest


def make_panel(opens, closes, divs=None, code="9999"):
    idx = pd.bdate_range("2025-01-01", periods=len(closes))
    df = lambda v: pd.DataFrame({code: v}, index=idx, dtype=float)
    return Panel(open=df(opens), high=df(np.maximum(opens, closes)), low=df(np.minimum(opens, closes)),
                 close=df(closes), volume=df([1000] * len(closes)),
                 dividend=df(divs if divs is not None else [0] * len(closes)))


def test_flat_price_round_trip_costs_only():
    p = make_panel([100] * 4, [100] * 4)
    w = pd.DataFrame({"9999": [1.0, 1.0, 0.0, 0.0]}, index=p.dates)
    res = backtest(p, w)
    expected = (1 - Costs.buy_cost()) * (1 - Costs.sell_cost(False))
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert len(res.trades) == 1


def test_execution_next_open_no_lookahead():
    # 第 0 天收盤決定買，第 1 天開盤 100 成交，收盤 110
    p = make_panel([100, 100, 110], [100, 110, 110])
    w = pd.DataFrame({"9999": [1.0, 1.0, 1.0]}, index=p.dates)
    res = backtest(p, w)
    assert res.equity.iloc[0] == pytest.approx(1.0)
    assert res.equity.iloc[1] == pytest.approx((1 - Costs.buy_cost()) * 1.10)


def test_limit_up_open_blocks_buy():
    p = make_panel([100, 110, 110], [100, 110, 110])
    w = pd.DataFrame({"9999": [1.0, 1.0, 1.0]}, index=p.dates)
    res = backtest(p, w)
    assert res.weights.iloc[1, 0] == 0.0          # 開盤漲停買不到
    assert res.weights.iloc[2, 0] > 0.99          # 隔天開盤平盤才買到


def test_dividend_is_credited():
    # 除息 5 元：開盤從 100 跌到 95，但持有者拿到 5 元股利 → 總值不變
    p = make_panel([100, 100, 95], [100, 100, 95], divs=[0, 0, 5])
    w = pd.DataFrame({"9999": [1.0, 1.0, 1.0]}, index=p.dates)
    res = backtest(p, w)
    assert res.equity.iloc[2] == pytest.approx(res.equity.iloc[1], rel=1e-9)


def test_nan_row_means_no_rebalance():
    p = make_panel([100, 100, 120, 120], [100, 100, 120, 120])
    w = pd.DataFrame({"9999": [0.5, np.nan, np.nan, np.nan]}, index=p.dates)
    res = backtest(p, w)
    # 0.5 部位漲 20% 後權重漂移，NaN 列不應被拉回 0.5
    assert res.weights.iloc[-1, 0] > 0.5
    assert res.turnover.iloc[2:].sum() == 0


def test_split_adjustment():
    idx = pd.bdate_range("2025-01-01", periods=3)
    df = pd.DataFrame({"open": [200, 200, 50], "high": [200, 200, 50], "low": [200, 200, 50],
                       "close": [200, 200, 50], "volume": [10, 10, 40], "dividend": [0, 0, 0]},
                      index=idx, dtype=float)
    out = adjust_splits(df)
    assert out["close"].tolist() == [50, 50, 50]
    assert out["volume"].tolist() == [40, 40, 40]


# ---------------------------------------------------------------- 停牌／缺資料日
def test_halt_resumption_keeps_return():
    # 第 2 天停牌（NaN），第 3 天復牌：收盤 100 → 120，持有者權益應反映 +20%
    nan = np.nan
    p = make_panel([100, 100, nan, 120, 120], [100, 100, nan, 120, 120])
    w = pd.DataFrame({"9999": [1.0] * 5}, index=p.dates)
    res = backtest(p, w)
    assert res.equity.iloc[-1] == pytest.approx((1 - Costs.buy_cost()) * 1.20, rel=1e-9)
    assert res.weights.iloc[2, 0] > 0.99                  # 停牌期間部位維持


def test_dividend_during_halt_credited_on_resume():
    # 停牌日除息 5 元（價格 NaN）→ 復牌日一起計入：開盤 95 + 5 股利 = 100，權益不變
    nan = np.nan
    p = make_panel([100, 100, nan, 95], [100, 100, nan, 95], divs=[0, 0, 5, 0])
    w = pd.DataFrame({"9999": [1.0] * 4}, index=p.dates)
    res = backtest(p, w)
    assert res.equity.iloc[3] == pytest.approx(res.equity.iloc[1], rel=1e-9)
    assert res.trades[0].dividends == pytest.approx(5.0)


def test_cannot_trade_while_halted_then_retries():
    nan = np.nan
    p = make_panel([100, nan, nan, 100, 100], [100, nan, nan, 100, 100])
    w = pd.DataFrame({"9999": [1.0, nan, nan, nan, nan]}, index=p.dates)
    res = backtest(p, w)
    assert res.weights.iloc[1:3, 0].sum() == 0            # 停牌買不到
    assert res.weights.iloc[3, 0] > 0.99                  # 復牌後自動重試


# ---------------------------------------------------------------- 未完成目標重試
def test_blocked_sell_is_retried_on_nan_rows():
    # t1 決定賣出，t2 開盤跌停賣不掉；之後整列 NaN（低頻策略常態）→ 隔天要自動重試賣出
    o = [100, 100, 90, 85, 80, 75]
    p = make_panel(o, o)
    w = pd.DataFrame({"9999": [1.0, 0.0, np.nan, np.nan, np.nan, np.nan]}, index=p.dates)
    res = backtest(p, w)
    assert res.weights["9999"].iloc[2] > 0.9              # 跌停日賣不掉
    assert res.weights["9999"].iloc[3] == 0.0             # 次日（85 開盤，跌幅 5.6%）賣出
    assert res.pending["9999"].iloc[3:].isna().all()


def test_new_target_overrides_pending():
    o = [100, 100, 90, 85, 85, 85]
    p = make_panel(o, o)
    # t1 要賣（t2 跌停失敗），t2 收盤後又改回 1.0 → 不應再賣
    w = pd.DataFrame({"9999": [1.0, 0.0, 1.0, np.nan, np.nan, np.nan]}, index=p.dates)
    res = backtest(p, w)
    assert (res.weights["9999"].iloc[2:] > 0.9).all()


def test_limit_up_buy_retried_without_new_row():
    p = make_panel([100, 110, 110, 110], [100, 110, 110, 110])
    w = pd.DataFrame({"9999": [1.0, np.nan, np.nan, np.nan]}, index=p.dates)
    res = backtest(p, w)
    assert res.weights.iloc[1, 0] == 0.0                  # 漲停買不到
    assert res.weights.iloc[2, 0] > 0.99                  # 隔天平盤自動買進


def test_cash_constraint_when_sell_blocked():
    # A 跌停賣不掉、B 要買滿 → 不可超過 100% 權重（不融資），B 待 A 賣出後補足
    idx = pd.bdate_range("2025-01-01", periods=4)
    o = pd.DataFrame({"A": [100, 100, 90, 90], "B": [50, 50, 50, 50]}, index=idx, dtype=float)
    p = Panel(open=o, high=o, low=o, close=o, volume=o * 0 + 1, dividend=o * 0)
    w = pd.DataFrame({"A": [1.0, 0.0, np.nan, np.nan], "B": [0.0, 1.0, np.nan, np.nan]}, index=idx)
    res = backtest(p, w)
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all()
    assert res.weights["B"].iloc[3] > 0.99 and res.weights["A"].iloc[3] == 0.0


# ---------------------------------------------------------------- 份額帳
def test_partial_adds_recorded_as_lots_fifo():
    o = [100] * 3 + [50] * 3 + [100] * 3
    p = make_panel(o, o)
    # 100 買 0.1，50 加碼到 1.0，100 全賣（整列 NaN 不再平衡）→ 兩筆 lot：100→100 與 50→100
    w = pd.DataFrame({"9999": [0.1, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, 0, np.nan]}, index=p.dates)
    res = backtest(p, w)
    closed = [t for t in res.trades if not t.open_at_end]
    assert len(closed) == 2
    rets = sorted(t.ret for t in closed)
    assert rets[0] == pytest.approx((1 - Costs.sell_cost(False)) / (1 + Costs.buy_cost()) - 1, rel=1e-6)
    assert rets[1] == pytest.approx(2 * (1 - Costs.sell_cost(False)) / (1 + Costs.buy_cost()) - 1, rel=1e-6)
    assert [t.entry_price for t in sorted(closed, key=lambda t: t.entry_date)] == [100, 50]
    # 份額帳損益合計 ≈ 權益變化（差異僅為成本的複利近似）
    assert sum(t.pnl for t in closed) == pytest.approx(res.equity.iloc[-1] - 1, abs=2e-3)


def test_partial_reduce_fifo_and_cost_in_trade_return():
    p = make_panel([100, 100, 120, 120, 120], [100, 100, 120, 120, 120])
    w = pd.DataFrame({"9999": [1.0, 1.0, 0.5, 0.5, 0.5]}, index=p.dates)
    res = backtest(p, w)
    closed = [t for t in res.trades if not t.open_at_end]
    still = [t for t in res.trades if t.open_at_end]
    assert len(closed) == 1 and len(still) == 1
    assert closed[0].ret == pytest.approx(1.2 * (1 - Costs.sell_cost(False)) / (1 + Costs.buy_cost()) - 1)
    assert closed[0].shares + still[0].shares == pytest.approx(still[0].shares * 2, rel=0.2)
