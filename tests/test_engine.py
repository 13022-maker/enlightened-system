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
