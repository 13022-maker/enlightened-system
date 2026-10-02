"""資料層測試：股數變動調整（分割／配股／減資）、股利事件檔、yfinance Stock Splits、
point-in-time 成分股、資料品質檢查。全部用人工資料，不依賴 data/ 真實檔案。"""
import numpy as np
import pandas as pd
import pytest

from quant.data import (Panel, adjust_splits, apply_dividend_file, check_data_quality, detect_share_events,
                        intraday_to_daily, load_dividend_events, load_panel, yfinance_frame)
from quant.universe import Universe


def daily(opens, closes, divs=None, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                         "volume": [1000.0] * len(c), "dividend": divs or [0.0] * len(c)}, index=idx)


# ---------------------------------------------------------------- 跳空推測股數變動
def test_integer_split_adjusted():
    df = daily([200, 200, 50, 51], [200, 200, 50, 51])
    out = adjust_splits(df)
    assert out["close"].tolist() == pytest.approx([50, 50, 50, 51])
    assert out["volume"].iloc[0] == pytest.approx(4000)


def test_integer_split_snaps_even_with_open_move():
    # 2327 2025-08-25：前收 545、1 拆 4 後開盤 142（相對參考價 136.25 +4%）→ 比率仍應為 4
    df = daily([545, 545, 142, 140], [545, 545, 140, 141])
    ev = detect_share_events(df)
    assert ev.iloc[0] == pytest.approx(4.0)


def test_stock_dividend_non_integer_ratio():
    # 2327 2024-08-15 配股 1.95 元：前收 745 → 參考價 623.51，比率 1.19485（舊版 round() 會跳過）
    df = daily([745, 745, 637, 630], [745, 745, 625, 630])
    out = adjust_splits(df)
    ratio = 745 / 637
    assert out["close"].iloc[1] == pytest.approx(745 / ratio)
    gap = out["open"] / out["close"].shift() - 1
    assert abs(gap.iloc[2]) < 1e-9


def test_capital_reduction_ratio_below_one():
    # 減資 4 成：1 股變 0.6 股，參考價 = 前收 / 0.6
    df = daily([30, 30, 50, 51], [30, 30, 50, 51])
    out = adjust_splits(df)
    assert out["close"].iloc[0] == pytest.approx(50)
    assert out["volume"].iloc[0] == pytest.approx(600)


def test_new_listing_first_week_not_adjusted():
    # 新股上市前 5 日無漲跌幅：開盤 +30%、+25% 是真實漲幅，不是公司行動
    df = daily([100, 169, 210, 215, 220, 222], [130, 168, 212, 218, 221, 222])
    out = adjust_splits(df, listing_date=df.index[0])
    pd.testing.assert_frame_equal(out, df)
    # 若不知道上市日，同樣資料就會被誤判 → 證明需要 listing_date
    assert len(detect_share_events(df)) > 0


def test_old_7pct_limit_before_2015_06_01():
    # 2015-06-01 前漲跌幅 7%：開盤跳空 −15% 必為公司行動；同幅度在新制下也會判斷
    df = daily([100, 100, 85, 85], [100, 100, 85, 85], start="2014-03-03")
    assert len(detect_share_events(df)) == 1
    # 舊制 −9%（超過 7%+2%）也算；新制下 −9% 不算
    old = daily([100, 100, 90.5, 90], [100, 100, 90, 90], start="2014-03-03")
    new = daily([100, 100, 90.5, 90], [100, 100, 90, 90], start="2024-03-01")
    assert len(detect_share_events(old)) == 1 and len(detect_share_events(new)) == 0


def test_cash_dividend_gap_not_treated_as_split():
    # 除息 12 元（前收 100 → 開盤 88）：扣掉股利後跳空 0%，不是股數變動
    df = daily([100, 100, 88, 88], [100, 100, 88, 88], divs=[0, 0, 12, 0])
    assert detect_share_events(df).empty


def test_halt_then_split_uses_last_valid_close():
    nan = np.nan
    df = daily([400, 400, nan, nan, 100, 101], [400, 400, nan, nan, 100, 101])
    out = adjust_splits(df)
    assert out["close"].iloc[0] == pytest.approx(100)


# ---------------------------------------------------------------- 5 分K 轉日K
def _write_intraday(path, rows):
    pd.DataFrame(rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "Dividends",
                                "Stock Splits"]).to_csv(path, index=False)


def test_intraday_duplicate_snapshots_keep_dividend(tmp_path):
    # 同一根 K 棒有多個快照，只有第二個帶股利（2303 2025-06-24 的實際情形）
    rows = [["2025-06-23 09:00:00+08:00", 46.5, 46.6, 46.4, 46.5, 100, 0.0, 0.0],
            ["2025-06-24 01:00:00+00:00", 42.5, 43.8, 42.5, 43.5, 0, 0.0, 0.0],
            ["2025-06-24 01:00:00+00:00", 42.5, 43.8, 42.5, 43.5, 100, 2.85, 0.0],
            ["2025-06-24 01:00:00+00:00", 42.5, 43.8, 42.5, 43.5, 100, 2.85, 0.0]]
    p = tmp_path / "2303.csv"
    _write_intraday(p, rows)
    out = intraday_to_daily(str(p), dividend_dir=tmp_path / "none")
    assert out.loc["2025-06-24", "dividend"] == pytest.approx(2.85)


def test_intraday_uses_yahoo_split_column(tmp_path):
    # 金融股配股 1.05（跳空小於漲跌停，推測不到）→ 用 Stock Splits 欄調整
    rows = [["2024-09-06 09:00:00+08:00", 92.0, 92.6, 92.0, 92.6, 100, 0.0, 0.0],
            ["2024-09-09 09:00:00+08:00", 86.5, 86.5, 86.0, 86.1, 100, 0.0, 1.05]]
    p = tmp_path / "2881.csv"
    _write_intraday(p, rows)
    out = intraday_to_daily(str(p), dividend_dir=tmp_path / "none")
    assert out["close"].iloc[0] == pytest.approx(92.6 / 1.05)


def test_intraday_event_file_overrides(tmp_path):
    rows = [["2024-08-14 09:00:00+08:00", 745, 745, 745, 745, 100, 0.0, 1.1948],
            ["2024-08-15 09:00:00+08:00", 637, 640, 620, 625, 100, 0.0, 0.0]]
    p = tmp_path / "2327.csv"
    _write_intraday(p, rows)
    dd = tmp_path / "div"
    dd.mkdir()
    (dd / "2327.csv").write_text("date,cash_dividend,stock_dividend,share_ratio,note\n"
                                 "2024-08-15,0,1.95,1.19485,配股\n")
    out = intraday_to_daily(str(p), dividend_dir=dd)
    assert out["close"].iloc[0] == pytest.approx(745 / 1.19485)


# ---------------------------------------------------------------- 股利事件檔
def _events(tmp_path, text, code="X"):
    (tmp_path / f"{code}.csv").write_text(text)
    return load_dividend_events(code, tmp_path)


def test_dividend_file_fills_missing_and_moves_misdated(tmp_path):
    df = daily([50] * 8, [50] * 8, divs=[0, 0, 0, 0, 0, 0, 1.0, 0])     # 來源把 1.0 記錯在第 7 天
    ev = _events(tmp_path, "date,cash_dividend,stock_dividend,share_ratio\n"
                           f"{df.index[1].date()},0.8,0,1\n{df.index[5].date()},1.0,0,1\n")
    out = apply_dividend_file(df, ev)
    assert out["dividend"].iloc[1] == pytest.approx(0.8)                # 補上缺漏
    assert out["dividend"].iloc[5] == pytest.approx(1.0)                # 正確日期
    assert out["dividend"].iloc[6] == 0.0                               # 錯置的那筆移除，不重複計入


def test_dividend_file_event_on_holiday_rolls_forward(tmp_path):
    df = daily([30] * 5, [30] * 5, start="2024-07-22")                  # 週一～週五
    df = df.drop(pd.Timestamp("2024-07-25"))                            # 颱風休市
    ev = _events(tmp_path, "date,cash_dividend,stock_dividend,share_ratio\n2024-07-25,0.35,0,1\n")
    out = apply_dividend_file(df, ev)
    assert out.loc["2024-07-26", "dividend"] == pytest.approx(0.35)


def test_dividend_file_cash_scaled_by_later_split_and_missing_split_applied(tmp_path):
    # 已調整日K 漏了 1.195 配股（舊版 round 跳過）＋之後 1 拆 4（已套用）；
    # 配股前的 20 元現金股利要換算成拆分後基礎 = 20 / (1.195 × 4)
    idx = pd.bdate_range("2024-06-26", periods=6)
    c = [186.25, 181.25, 186.25, 155.75, 157.5, 136.5]
    df = pd.DataFrame({"open": [186.25, 186.25, 186.25, 159.25, 157.75, 136.5], "high": c, "low": c,
                       "close": c, "volume": 1.0, "dividend": 0.0}, index=idx)
    ev = _events(tmp_path, "date,cash_dividend,stock_dividend,share_ratio\n"
                           f"{idx[1].date()},20,0,1\n{idx[3].date()},0,1.95,1.195\n{idx[5].date()},0,0,4\n",
                 code="2327")
    out = apply_dividend_file(df, ev)
    # 價格基礎：拆分（4）已在日K 中、配股（1.195）由本函式補做 → 股利 ÷ (1.195 × 4)
    assert out["dividend"].iloc[1] == pytest.approx(20 / (1.195 * 4), rel=1e-6)
    assert out["close"].iloc[2] == pytest.approx(186.25 / 1.195)        # 補做配股調整
    gap = out["open"].iloc[3] / out["close"].iloc[2] - 1
    assert abs(gap) < 0.03                                             # 剩下的是真實開盤漲跌（+2.2%）
    # 已套用的 1 拆 4 不可重複調整
    assert out["close"].iloc[4] == pytest.approx(157.5)


# ---------------------------------------------------------------- yfinance Stock Splits
def _hist(opens, closes, splits, divs=None, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": opens, "High": closes, "Low": closes, "Close": closes,
                         "Volume": [1.0] * len(closes), "Dividends": divs or [0.0] * len(closes),
                         "Stock Splits": splits}, index=idx, dtype=float)


def test_yfinance_already_adjusted_split_not_doubled():
    # Yahoo 正常情況：分割已套用到舊價格（無跳空），Stock Splits=4 只是事件標記 → 不再調整
    out = yfinance_frame(_hist([50, 50, 51], [50, 50, 51], [0, 0, 4]))
    assert out["close"].tolist() == [50, 50, 51]


def test_yfinance_unapplied_split_is_fixed():
    # Yahoo 偶發漏套（issue #2660 所述）：舊價格仍是 200 → 依 Stock Splits 欄補調整
    out = yfinance_frame(_hist([200, 200, 50], [200, 200, 50], [0, 0, 4]))
    assert out["close"].tolist() == pytest.approx([50, 50, 50])


def test_yfinance_no_gap_guessing():
    # yfinance 路徑不再用跳空推測：沒有 Stock Splits 的大跌（例如真實崩跌）保留原樣
    out = yfinance_frame(_hist([100, 100, 70], [100, 100, 70], [0, 0, 0]))
    assert out["close"].tolist() == [100, 100, 70]


# ---------------------------------------------------------------- 成分股
@pytest.fixture
def uni():
    return Universe(pd.DataFrame({
        "code": ["A", "B", "C", "C"], "name": ["a", "b", "c", "c"],
        "start": [None, "2025-01-06", None, "2025-01-10"],
        "end": [None, None, "2025-01-03", None]}))


def test_members_on(uni):
    assert uni.members_on("2025-01-02") == ["A", "C"]
    assert uni.members_on("2025-01-06") == ["A", "B"]
    assert uni.members_on("2025-01-10") == ["A", "B", "C"]          # C 調出後再調入


def test_membership_mask(uni):
    idx = pd.bdate_range("2025-01-02", "2025-01-10")
    m = uni.membership_mask(idx, ["A", "B", "C", "Z"])
    assert m["A"].all() and not m["Z"].any()
    assert m.loc["2025-01-03", "C"] and not m.loc["2025-01-06", "C"] and m.loc["2025-01-10", "C"]
    assert not m.loc["2025-01-03", "B"] and m.loc["2025-01-06", "B"]


def test_load_panel_point_in_time(tmp_path):
    dd = tmp_path / "daily"
    dd.mkdir()
    for code in ["A", "B", "0050"]:
        daily([10] * 5, [10] * 5, start="2025-01-01").rename_axis("date").to_csv(dd / f"{code}.csv")
    uf = tmp_path / "m.csv"
    uf.write_text("code,name,start,end\nA,a,,2025-01-02\nB,b,2025-01-03,\nNODATA,x,2025-01-01,\n")
    p = load_panel(["A", "0050"], directory=dd, point_in_time=True, universe_file=uf, dividend_dir=None)
    assert set(p.codes) == {"A", "B", "0050"}                        # 自動加入 B；無資料的略過
    m = p.universe_mask()
    assert m["0050"].all()                                            # ETF 一律可用
    assert m.loc["2025-01-02", "A"] and not m.loc["2025-01-03", "A"]
    p2 = load_panel(["A", "0050"], directory=dd, dividend_dir=None)
    assert p2.universe is None and p2.universe_mask().all().all()    # 預設行為不變
    assert p.slice("2025-01-03", None).universe.shape[0] == 3


# ---------------------------------------------------------------- 資料品質
def test_check_data_quality_flags():
    idx = pd.bdate_range("2025-01-01", periods=6)
    c = pd.DataFrame({"X": [100, 100, np.nan, 95, 120, 120], "Y": [50, 50, 50, 50, 50, 50]}, index=idx, dtype=float)
    o = c.copy()
    o.loc[idx[3], "X"] = 95
    d = pd.DataFrame(0.0, index=idx, columns=["X", "Y"])
    d.loc[idx[5], "Y"] = 20                                            # 殖利率 40% → 異常
    p = Panel(open=o, high=c, low=c, close=c, volume=c * 0 + 1, dividend=d)
    kinds = {(i["code"], i["type"]) for i in check_data_quality(p, market_relative=False)}
    assert ("X", "halt") in kinds
    assert ("X", "gap_no_dividend") in kinds                          # 100 → 95 開盤 −5%、無股利
    assert ("X", "big_move") in kinds                                  # 95 → 120 +26%
    assert ("Y", "dividend_yield") in kinds


def test_check_data_quality_market_wide_gap_not_flagged():
    # 全市場同步跳空 −6%（例：關稅、春節後）不是個股資料問題；只有比大盤再低 3% 以上才標記
    idx = pd.bdate_range("2025-01-01", periods=3)
    c = pd.DataFrame({k: [100, 100, 94] for k in "ABCD"}, index=idx, dtype=float)
    c["E"] = [100, 100, 85]
    p = Panel(open=c, high=c, low=c, close=c, volume=c * 0 + 1, dividend=c * 0)
    flagged = {i["code"] for i in check_data_quality(p) if i["type"] == "gap_no_dividend"}
    assert flagged == {"E"}


# ---------------------------------------------------------------- tools/fetch_dividends.py（mock FinMind）
def _load_tool(name):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent.parent / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_dividends_builds_events_with_mock(tmp_path):
    mod = _load_tool("fetch_dividends")
    result = pd.DataFrame([
        {"date": "2024-06-27", "stock_id": "2327", "before_price": 765, "after_price": 760,
         "stock_and_cache_dividend": 19.97, "stock_or_cache_dividend": "息", "reference_price": 745.03},
        {"date": "2024-08-15", "stock_id": "2327", "before_price": 745, "after_price": 625,
         "stock_and_cache_dividend": 121.49, "stock_or_cache_dividend": "權", "reference_price": 623.51},
        {"date": "2024-07-26", "stock_id": "2884", "before_price": 29.8, "after_price": 27.05,
         "stock_and_cache_dividend": 1.76, "stock_or_cache_dividend": "權息", "reference_price": 28.04}])
    policy = pd.DataFrame([
        {"StockEarningsDistribution": 0.2, "StockStatutorySurplus": 0, "CashEarningsDistribution": 1.2,
         "CashStatutorySurplus": 0, "CashExDividendTradingDate": "2024-07-26",
         "StockExDividendTradingDate": "2024-07-26"}])
    calls = []

    def fake(dataset, code, start):
        calls.append((dataset, code))
        r = result[result["stock_id"] == code]
        return r if dataset == "TaiwanStockDividendResult" else (policy if code == "2884" else pd.DataFrame())

    (tmp_path / "2327.csv").write_text("date,cash_dividend,stock_dividend,share_ratio,note\n2025-08-25,0,0,4,面額變更\n")
    st = mod.run(["2327", "2884"], "2024-01-01", tmp_path, merge=True, fetch=fake, pause=0)
    assert st["2327"].startswith("ok") and ("TaiwanStockDividendResult", "2327") in calls
    ev = load_dividend_events("2327", tmp_path)
    assert ev.loc["2024-06-27", "cash_dividend"] == pytest.approx(19.97)
    assert ev.loc["2024-08-15", "share_ratio"] == pytest.approx(745 / 623.51, rel=1e-5)   # 政策表查無 → 參考價推算
    assert ev.loc["2025-08-25", "share_ratio"] == 4                                       # merge 保留手動列
    ev2 = load_dividend_events("2884", tmp_path)
    assert ev2.loc["2024-07-26", "cash_dividend"] == pytest.approx(1.2)
    assert ev2.loc["2024-07-26", "share_ratio"] == pytest.approx(1.02)                    # 政策表 0.2 元 → 1.02


def test_fetch_universe_add_and_check(tmp_path):
    mod = _load_tool("fetch_universe")
    f = tmp_path / "m.csv"
    f.write_text("code,name,start,end\nA,a,,\nB,b,,\n")
    mod.add_review(f, "2025-01-06", "2025-01-03", ["C"], ["B"], {"C": "c"})
    u = Universe.from_csv(f)
    assert u.members_on("2025-01-03") == ["A", "B"] and u.members_on("2025-01-06") == ["A", "C"]
    assert mod.check(f, "2025-01-01", "2025-01-10", expected=2) == []
    with pytest.raises(ValueError):
        mod.add_review(f, "2025-02-03", "2025-01-31", [], ["B"])                         # B 已不是成分股
