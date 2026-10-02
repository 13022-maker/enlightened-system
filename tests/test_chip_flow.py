"""chip_flow 籌碼策略與 tools/fetch_chip.py 的單元測試。

注意：本環境無外網，真實價格搭配的是「模擬籌碼」，只驗證程式正確性，不驗證績效。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.config import DAILY_DIR, ETFS, TW50
from quant.data import Panel, load_chip, load_panel
from quant.engine import backtest
from quant.validation import assert_no_lookahead
from quant.strategies.chip_flow import ChipFlow, _rebalance_mask, _streak

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 人工資料
def make_panel(n_days=80):
    """4 檔股票 + 1 檔 ETF：
        A 投信持續大買、價格穩定上漲（應入選）
        B 法人持續大賣、價格上漲（不應入選：分數 < 0）
        C 外資小買、價格下跌在均線下（不應入選：趨勢濾網）
        D 法人零買賣、價格上漲（不應入選：分數 = 0）
        0050 ETF 法人大買、上漲（不應入選：ETF）
    """
    idx = pd.bdate_range("2025-01-06", periods=n_days)   # 2025-01-06 是星期一
    codes = ["A", "B", "C", "D", "0050"]
    up = 100 * 1.003 ** np.arange(n_days)
    down = 100 * 0.997 ** np.arange(n_days)
    close = pd.DataFrame({"A": up, "B": up, "C": down, "D": up, "0050": up}, index=idx)
    vol = pd.DataFrame(1_000_000.0, index=idx, columns=codes)
    z = pd.DataFrame(0.0, index=idx, columns=codes)
    foreign, trust = z.copy(), z.copy()
    trust["A"] = 50_000          # 每日投信買 50 張（佔量 5%）
    foreign["B"] = -200_000
    trust["B"] = -50_000
    foreign["C"] = 100_000
    foreign["0050"] = 300_000
    trust["0050"] = 100_000
    p = Panel(open=close.copy(), high=close * 1.01, low=close * 0.99, close=close,
              volume=vol, dividend=z.copy(),
              chip={"foreign_net": foreign, "trust_net": trust, "dealer_net": z.copy()})
    return p


def test_core_selection():
    p = make_panel()
    s = ChipFlow()
    w = s.target_weights(p, top_n=2, buffer=0, rebalance_weeks=1, lookback=5, ma=20, exit_ma=20)
    reb = w.dropna(how="all")
    assert len(reb) > 0
    late = reb.loc[reb.index >= p.dates[25]]     # 均線暖身之後
    assert (late["A"] == 0.5).all()              # A 被選入，每檔 1/top_n
    for c in ["B", "C", "D", "0050"]:
        assert (late[c] == 0).all(), c
    # 暖身期（MA20 尚未算出）不應持有任何股票
    early = reb.loc[reb.index < p.dates[19]]
    assert (early.fillna(0).sum(axis=1) == 0).all()


def test_exit_when_institutions_sell():
    """A 先被投信大買，後半段轉為大賣 → 下次再平衡應出場。"""
    p = make_panel()
    half = p.dates[50]
    p.chip["trust_net"].loc[half:, "A"] = -80_000
    w = ChipFlow().target_weights(p, top_n=2, buffer=5, rebalance_weeks=1, lookback=5, exit_ma=20)
    reb = w.dropna(how="all")
    assert reb.loc[:half, "A"].iloc[-1] == 0.5
    assert reb.loc[p.dates[60]:, "A"].eq(0).all()


def test_buffer_keeps_holding():
    """緩衝區：已持有者排名掉到 top_n 之外但仍在 top_n+buffer 內 → 續抱。"""
    p = make_panel()
    # 新增 E：前半無籌碼、後半投信更大買，排名超越 A
    e_close = p.close["A"].copy()
    for f in ["open", "close"]:
        getattr(p, f)["E"] = e_close
    p.high["E"], p.low["E"] = e_close * 1.01, e_close * 0.99
    p.volume["E"], p.dividend["E"] = 1_000_000.0, 0.0
    for k in p.chip:
        p.chip[k]["E"] = 0.0
    p.chip["trust_net"].loc[p.dates[40]:, "E"] = 90_000
    kw = dict(top_n=1, rebalance_weeks=1, lookback=5, exit_ma=20)
    with_buf = ChipFlow().target_weights(p, buffer=3, **kw).dropna(how="all")
    no_buf = ChipFlow().target_weights(p, buffer=0, **kw).dropna(how="all")
    late = with_buf.index >= p.dates[50]
    assert (with_buf.loc[late, "A"] == 1.0).all()     # 有緩衝：續抱 A
    assert (no_buf.loc[late, "E"] == 1.0).all()       # 無緩衝：換成排名第一的 E
    assert (no_buf.loc[late, "A"] == 0.0).all()


def test_rebalance_schedule_nan_rows():
    p = make_panel()
    w = ChipFlow().target_weights(p, rebalance_weeks=2)
    mask = w.notna().any(axis=1)
    # 非再平衡日整列 NaN；再平衡日整列有值
    assert (w[mask].notna().all(axis=1)).all()
    assert (w[~mask].isna().all(axis=1)).all()
    # 每兩週一次：再平衡日之間至少間隔 8 個交易日（無假日的人工資料為 10）
    gaps = np.diff(np.nonzero(mask.values)[0])
    assert (gaps >= 8).all()
    # 排程以固定錨點計算 → 截掉前段資料後，排程日期不變
    #（截斷後的第一天無法得知是否為該週首日，排除不比）
    sub = p.dates[7:]
    m2 = _rebalance_mask(sub, 2)
    assert set(sub[1:][m2[1:]]) == set(p.dates[mask.values]) & set(sub[1:])


def test_streak():
    f = pd.DataFrame({"x": [True, True, False, True, True, True]})
    assert _streak(f)["x"].tolist() == [1, 2, 0, 1, 2, 3]


def test_missing_chip_raises():
    p = make_panel()
    p.chip = {}
    with pytest.raises(ValueError, match="籌碼"):
        ChipFlow().target_weights(p)
    with pytest.raises(ValueError, match="籌碼"):
        ChipFlow().scores(p)
    p2 = make_panel()
    del p2.chip["trust_net"]
    with pytest.raises(ValueError, match="trust_net"):
        ChipFlow().target_weights(p2)


def test_explain_and_scores():
    p = make_panel()
    s = ChipFlow()
    msg = s.explain(p, "A", lookback=5)
    assert "投信連買" in msg and "投信5日買超250張" in msg and "站上MA20" in msg
    assert "賣超" in s.explain(p, "B", lookback=5)
    assert "ETF" in s.explain(p, "0050")
    sc = s.scores(p)
    assert sc["A"].iloc[-1] > 0 > sc["B"].iloc[-1]
    assert sc["0050"].isna().all()
    sig = {x["code"]: x for x in s.latest_signals(p)}
    assert sig["A"]["action"] in ("BUY", "HOLD") and sig["A"]["reason"]


# ---------------------------------------------------------------- 真實價格 + 模擬籌碼
@pytest.fixture(scope="module")
def real_panel():
    codes = [c for c in list(TW50) + list(ETFS) if (DAILY_DIR / f"{c}.csv").exists()]
    if len(codes) < 10:
        pytest.skip("缺少 data/daily 日K")
    p = load_panel(codes, chip=True)
    return p


def test_real_no_lookahead(real_panel):
    s = ChipFlow()
    assert_no_lookahead(s, real_panel)
    assert_no_lookahead(s, real_panel, params={"lookback": 20, "top_n": 8, "rebalance_weeks": 1})


def test_real_weight_sanity(real_panel):
    s = ChipFlow()
    for params in [{}] + s.param_grid:
        w = s.target_weights(real_panel, **{**s.default_params, **params})
        assert list(w.columns) == real_panel.codes and w.index.equals(real_panel.dates)
        rows = w.dropna(how="all")
        assert len(rows) > 10
        assert (rows.fillna(0) >= 0).all().all()
        assert (rows.sum(axis=1) <= 1 + 1e-9).all()
        etfs = [c for c in real_panel.codes if real_panel.is_etf(c)]
        assert (rows[etfs].fillna(0) == 0).all().all()
        assert rows.notna().all(axis=1).all()        # 有值的列不能部分 NaN


def test_real_turnover_target(real_panel):
    s = ChipFlow()
    m = backtest(real_panel, s.target_weights(real_panel, **s.default_params)).metrics()
    assert m["annual_turnover"] < 12, m["annual_turnover"]


# ---------------------------------------------------------------- tools/fetch_chip.py
def _load_fetch_chip():
    spec = importlib.util.spec_from_file_location("fetch_chip", ROOT / "tools" / "fetch_chip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_finmind_rows(code):
    rows = []
    for d, fb, tb in [("2025-01-02", 1000, 500), ("2025-01-03", 0, 2000)]:
        rows += [
            {"date": d, "stock_id": code, "name": "Foreign_Investor", "buy": fb, "sell": 300},
            {"date": d, "stock_id": code, "name": "Foreign_Dealer_Self", "buy": 10, "sell": 0},
            {"date": d, "stock_id": code, "name": "Investment_Trust", "buy": tb, "sell": 100},
            {"date": d, "stock_id": code, "name": "Dealer_self", "buy": 50, "sell": 70},
            {"date": d, "stock_id": code, "name": "Dealer_Hedging", "buy": 5, "sell": 0},
        ]
    return rows


def test_fetch_chip_writes_csv(tmp_path, monkeypatch):
    """mock requests.get → 經真正的 fetch_finmind_chip 解析 → 寫出 CSV → load_chip 讀得回來。"""
    import requests

    calls = []

    class Resp:
        def __init__(self, code):
            self.code = code

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": _fake_finmind_rows(self.code)}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        if params["data_id"] == "2317" and sum(c["data_id"] == "2317" for c in calls) == 1:
            raise requests.ConnectionError("暫時性錯誤")      # 第一次失敗 → 應重試成功
        return Resp(params["data_id"])

    monkeypatch.setattr(requests, "get", fake_get)
    mod = _load_fetch_chip()
    sleeps = []
    failed = mod.run(["2330", "2317"], "2025-01-01", tmp_path, pause=1.5, retries=3,
                     sleep=sleeps.append)
    assert failed == []
    assert all(c["start_date"] == "2025-01-01" for c in calls)
    assert len(calls) == 3                      # 2317 重試一次
    assert 1.5 in sleeps                         # 請求間隔有 sleep

    df = pd.read_csv(tmp_path / "2330.csv")
    assert list(df.columns) == ["date", "foreign_net", "trust_net", "dealer_net"]
    assert df["date"].tolist() == ["2025-01-02", "2025-01-03"]
    # 外資 = Foreign_Investor + Foreign_Dealer_Self；自營 = Dealer_self + Dealer_Hedging
    assert df["foreign_net"].tolist() == [710, -290]
    assert df["trust_net"].tolist() == [400, 1900]
    assert df["dealer_net"].tolist() == [-15, -15]

    chip = load_chip(["2330", "2317"], pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"])),
                     directory=tmp_path)
    assert chip["trust_net"].loc["2025-01-03", "2317"] == 1900


def test_fetch_chip_retry_exhausted_and_merge(tmp_path):
    mod = _load_fetch_chip()

    def boom(code, start):
        raise RuntimeError("429 Too Many Requests")

    assert mod.run(["2330"], "2025-01-01", tmp_path, retries=2, fetch=boom, sleep=lambda s: None) == ["2330"]
    assert not (tmp_path / "2330.csv").exists()

    # 合併：舊檔有較早日期，新抓的覆蓋重疊日期
    idx = lambda ds: pd.DatetimeIndex(pd.to_datetime(ds), name="date")
    old = pd.DataFrame({"foreign_net": [1, 2], "trust_net": [0, 0], "dealer_net": [0, 0]},
                       index=idx(["2025-01-01", "2025-01-02"]))
    new = pd.DataFrame({"foreign_net": [9, 3], "trust_net": [0, 0], "dealer_net": [0, 0]},
                       index=idx(["2025-01-02", "2025-01-03"]))
    mod.save("2330", old, tmp_path)
    mod.save("2330", new, tmp_path)
    df = pd.read_csv(tmp_path / "2330.csv")
    assert df["foreign_net"].tolist() == [1, 9, 3]


def test_fetch_chip_cli_parses_args(monkeypatch, tmp_path):
    mod = _load_fetch_chip()
    got = {}

    def fake_run(codes, start, directory, pause, retries):
        got.update(codes=codes, start=start, directory=directory, pause=pause, retries=retries)
        return []

    monkeypatch.setattr(mod, "run", fake_run)
    rc = mod.main(["--start", "2023-06-01", "--codes", "2330", "--out", str(tmp_path), "--sleep", "0.5"])
    assert rc == 0
    assert got["start"] == "2023-06-01" and got["codes"] == ["2330"] and got["pause"] == 0.5
    rc = mod.main(["--out", str(tmp_path)])
    assert got["codes"] == list(TW50)
