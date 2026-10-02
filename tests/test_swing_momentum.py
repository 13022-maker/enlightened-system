"""波段動能策略測試：前視偏差、權重約束、ETF 排除、合成資料驗證核心邏輯。"""
import numpy as np
import pandas as pd
import pytest

from quant.config import ETFS, TW50
from quant.data import Panel, load_panel
from quant.engine import backtest
from quant.strategies.swing_momentum import SwingMomentum, _rebalance_mask
from quant.validation import assert_no_lookahead

# 合成資料用的小參數，讓暖身期短
SMALL = {"lookback": 20, "skip": 0, "top_n": 1, "buffer": 0, "rebal_weeks": 1,
         "ma_trend": 10, "regime_ma": 10, "atr_n": 5, "atr_mult": 3.0}


@pytest.fixture(scope="module")
def real_panel():
    return load_panel(list(TW50) + list(ETFS))


def make_panel(closes: dict, n: int) -> Panel:
    """由收盤序列建立 Panel；高低價 = 收盤 ±0.5%，開盤 = 前日收盤。"""
    idx = pd.bdate_range("2025-01-06", periods=n)
    c = pd.DataFrame(closes, index=idx, dtype=float)
    o = c.shift(1).fillna(c)
    return Panel(open=o, high=c * 1.005, low=c * 0.995, close=c,
                 volume=pd.DataFrame(1000.0, index=idx, columns=c.columns),
                 dividend=pd.DataFrame(0.0, index=idx, columns=c.columns))


# ---------------------------------------------------------------- 真實資料
def test_no_lookahead_real_data(real_panel):
    s = SwingMomentum()
    assert_no_lookahead(s, real_panel)
    for g in s.param_grid[:2]:
        assert_no_lookahead(s, real_panel, g)


def test_weights_constraints_and_no_etf(real_panel):
    s = SwingMomentum()
    for g in s.param_grid:
        w = s.target_weights(real_panel, **{**s.default_params, **g})
        valid = w.dropna(how="all")
        assert len(valid) > 0
        assert (valid.fillna(0) >= -1e-12).all().all()
        assert (valid.fillna(0).sum(axis=1) <= 1 + 1e-9).all()
        for etf in ETFS:
            assert (valid[etf].fillna(0) == 0).all()
        # 列要嘛全 NaN 要嘛完整（不可只有部分欄 NaN）
        assert not w.isna().any(axis=1).where(~w.isna().all(axis=1), False).any()


def test_low_turnover_and_runs(real_panel):
    s = SwingMomentum()
    res = backtest(real_panel, s.target_weights(real_panel))
    m = res.metrics()
    assert m["annual_turnover"] < 12
    assert not any(real_panel.is_etf(t.code) for t in res.trades)


def test_scores_and_explain(real_panel):
    s = SwingMomentum()
    sc = s.scores(real_panel)
    assert sc.shape == real_panel.close.shape
    assert sc["0050"].isna().all()
    txt = s.explain(real_panel, "2330")
    assert "動能" in txt and "排名第" in txt
    assert "ETF" in s.explain(real_panel, "0050")
    sig = s.latest_signals(real_panel)
    assert len(sig) == len(real_panel.codes)


# ---------------------------------------------------------------- 合成資料
def test_rebalance_mask_weekly_and_stable():
    d = pd.bdate_range("2025-01-06", periods=30)
    m1 = _rebalance_mask(d, 1)
    assert m1.sum() == 6 and all(d[m1].dayofweek == 0)
    m4 = _rebalance_mask(d, 4)
    # 截斷後同一天的判斷不變
    assert (m4[:12] == _rebalance_mask(d[:12], 4)).all()


def test_strong_uptrend_selected_and_etf_ignored():
    n = 60
    t = np.arange(n)
    p = make_panel({
        "1111": 100 * 1.01 ** t,          # 穩定上漲
        "2222": 100 * 1.002 ** t,         # 緩漲
        "3333": 100 * 0.995 ** t,         # 下跌
        "0056": 100 * 1.02 ** t,          # ETF 漲最多但不可買
        "0050": 100 * 1.003 ** t,         # 大盤多頭
    }, n)
    w = SwingMomentum().target_weights(p, **SMALL)
    held = w.ffill().fillna(0).iloc[-1]
    assert held["1111"] > 0.99
    assert held[["2222", "3333", "0056", "0050"]].sum() == 0


def test_atr_trailing_stop_exits():
    n = 60
    t = np.arange(n)
    up = 100 * 1.01 ** t
    crash = up.copy()
    crash[47:] = up[46] * 0.85            # 第 47 天（週三，非再平衡日）急跌 15%，遠大於 3×ATR
    p = make_panel({"1111": crash, "2222": 100 * 1.001 ** t, "0050": 100 * 1.003 ** t}, n)
    s = SwingMomentum()
    w, info = s._simulate(p, **SMALL)
    assert info["stops"] >= 1
    day = p.dates[47]
    assert day.dayofweek == 2
    assert w.loc[day, "1111"] == 0           # 急跌當日收盤決定出場（隔日開盤成交）
    assert w.loc[:p.dates[46], "1111"].ffill().iloc[-1] > 0.99
    # 非再平衡日停損：只出場、不立刻換股（2222 仍為 0，資金留現金）
    assert w.loc[day, "2222"] == 0
    # 下一個再平衡日（週一）才把現金換到符合條件的 2222
    nxt = p.dates[50]
    assert nxt.dayofweek == 0 and w.loc[nxt, "2222"] > 0.8


def test_regime_filter_goes_to_cash():
    n = 60
    t = np.arange(n)
    bench = np.r_[100 * 1.003 ** t[:40], 100 * 1.003 ** 39 * 0.97 ** (t[40:] - 39)]
    p = make_panel({"1111": 100 * 1.01 ** t, "0050": bench}, n)
    p_big = {**SMALL, "atr_mult": 99.0}       # 關閉停損，單純測大盤濾網
    w = SwingMomentum().target_weights(p, **p_big)
    filled = w.ffill().fillna(0)
    assert filled["1111"].iloc[35] > 0.99      # 大盤多頭時持有
    assert filled["1111"].iloc[-1] == 0        # 大盤跌破均線後的再平衡日出場


def test_buffer_keeps_holding_reduces_trades():
    """排名在前 top_n+buffer 內的持股不賣：buffer 大時交易次數不增加。"""
    rng = np.random.default_rng(0)
    n = 120
    closes = {f"{1000 + k}": 100 * np.exp(np.cumsum(0.002 + 0.015 * rng.standard_normal(n))) for k in range(10)}
    closes["0050"] = 100 * 1.002 ** np.arange(n)
    p = make_panel(closes, n)
    s = SwingMomentum()
    base = {**SMALL, "top_n": 3, "atr_mult": 99.0}
    t0 = backtest(p, s.target_weights(p, **{**base, "buffer": 0})).turnover.sum()
    t1 = backtest(p, s.target_weights(p, **{**base, "buffer": 4})).turnover.sum()
    assert t1 <= t0


def test_clean_dividend_drops_absurd_values():
    from quant.strategies.swing_momentum import clean_dividend
    p = make_panel({"1111": [100.0] * 5}, 5)
    p.dividend.iloc[2, 0] = 3.0       # 3% 合理
    p.dividend.iloc[4, 0] = 500.0     # 500% 異常
    d = clean_dividend(p)["1111"].tolist()
    assert d == [0.0, 0.0, 3.0, 0.0, 0.0]
