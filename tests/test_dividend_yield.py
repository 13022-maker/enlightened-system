"""存股殖利率策略測試：人工合成 Panel 驗證核心邏輯，真實資料驗證無前視偏差與權重合法性。"""
import numpy as np
import pandas as pd
import pytest

from quant.config import ETFS, TW50
from quant.data import Panel, load_panel
from quant.strategies.dividend_yield import (DividendYield, annual_dividend, clean_dividends,
                                             rebalance_mask)
from quant.validation import assert_no_lookahead

N_DAYS = 420


def synth_panel():
    """五檔合成股票（約 1.7 年營業日）：
    A：高殖利率（每年 8 元 / 價 100）、價格緩漲 → 應被選
    B：中殖利率（4 元）、緩漲 → 應被選（top_n=2 時第二名）
    C：殖利率最高（10 元）但股價長期下跌 → 價值陷阱，應被排除
    D：低殖利率（1 元）、緩漲 → 不應被選
    E：不配息 → 不應被選
    F：殖利率 30%（異常偏高）→ 應被 max_yield 濾網排除
    """
    idx = pd.bdate_range("2023-01-02", periods=N_DAYS)
    t = np.arange(N_DAYS)
    up = 100 * (1 + 0.0004 * t)
    down = 100 * (1 - 0.0012 * t)
    closes = {"A": up, "B": up, "C": down, "D": up, "E": up, "F": up}
    divs = {k: np.zeros(N_DAYS) for k in closes}
    for k, amt in {"A": 8.0, "B": 4.0, "C": 10.0, "D": 1.0, "F": 30.0}.items():
        for day in (60, 320):                  # 一年配一次（相隔 260 營業日 ≈ 1 年）
            divs[k][day] = amt
    df = lambda d: pd.DataFrame(d, index=idx, dtype=float)
    c = df(closes)
    return Panel(open=c.copy(), high=c.copy(), low=c.copy(), close=c,
                 volume=df({k: np.full(N_DAYS, 1e6) for k in closes}), dividend=df(divs))


PARAMS = {"top_n": 2, "buffer": 0, "freq": "Q", "max_yield": 0.5, "max_single": 0.5}


def test_selects_high_yield_uptrend_and_excludes_value_trap():
    p = synth_panel()
    s = DividendYield()
    w = s.target_weights(p, **{**PARAMS, "max_yield": 0.12})
    rows = w.dropna(how="all")
    assert len(rows) > 0
    last = rows.iloc[-1]
    assert last["A"] == pytest.approx(0.5) and last["B"] == pytest.approx(0.5)
    assert last["C"] == 0.0                     # 殖利率最高但趨勢向下 → 排除
    assert last["D"] == 0.0 and last["E"] == 0.0 and last["F"] == 0.0


def test_abnormal_yield_filtered():
    p = synth_panel()
    s = DividendYield()
    w = s.target_weights(p, **{**PARAMS, "max_yield": 0.12})
    assert (w["F"].fillna(0.0) == 0.0).all()   # 30% 殖利率 → 剔除
    w2 = s.target_weights(p, **PARAMS)           # max_yield=0.5 時 F 會進榜
    assert w2["F"].fillna(0.0).max() > 0


def test_only_rebalance_days_have_values():
    p = synth_panel()
    w = DividendYield().target_weights(p, **PARAMS)
    has = w.notna().any(axis=1)
    mask = rebalance_mask(p.dates, "Q")
    assert not has[~mask].any()                 # 非再平衡日整列 NaN
    assert has.sum() >= 2
    for d in has[has].index:                    # 都在 1/4/7/10 月的第一個交易日
        assert d.month in (1, 4, 7, 10)


def test_annual_dividend_window_and_unknown():
    p = synth_panel()
    ann = annual_dividend(p.dividend, p.close)
    assert np.isnan(ann["E"].iloc[100])         # 未滿一年且無除息 → 未知
    assert ann["E"].iloc[-1] == 0.0             # 滿一年仍無除息 → 0
    assert ann["A"].iloc[100] == 8.0
    assert ann["A"].iloc[-1] == 8.0             # 只算最近一次除息前 330 天內，不重複累計


def test_clean_dividends_fixes_corrupt_value():
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = lambda v: pd.DataFrame({"X": v}, index=idx, dtype=float)
    p = Panel(open=df([100, 100, 96]), high=df([100, 100, 96]), low=df([100, 100, 96]),
              close=df([100, 100, 96]), volume=df([1, 1, 1]), dividend=df([0, 0, 900]))
    cd = clean_dividends(p)
    assert cd["X"].iloc[2] == pytest.approx(4.0)  # 以跳空 100→96 估算


@pytest.fixture(scope="module")
def real_panel():
    return load_panel(list(TW50) + list(ETFS))


@pytest.mark.realdata
def test_real_data_no_lookahead(real_panel):
    assert_no_lookahead(DividendYield(), real_panel)
    assert_no_lookahead(DividendYield(), real_panel, params={"freq": "H", "top_n": 5})


@pytest.mark.realdata
def test_real_weights_valid(real_panel):
    s = DividendYield()
    for params in s.param_grid:
        w = s.target_weights(real_panel, **params)
        rows = w.dropna(how="all")
        assert len(rows) >= 1
        assert (rows.fillna(0.0) >= 0).all().all()
        assert (rows.fillna(0.0).sum(axis=1) <= 1 + 1e-9).all()
        etf_cols = [c for c in real_panel.codes if real_panel.is_etf(c)]
        assert (rows[etf_cols].fillna(0.0) == 0).all().all()
    assert s.explain(real_panel, "2412")
