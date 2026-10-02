"""tools/ 自動化腳本的單元測試（全部用 mock，不需網路）：
    update_data 增量合併 / 全量重抓 / 不覆蓋 / data_status.json
    fetch_chip 增量起始日
    export_signals 依 as_of 去重
    notify_signals 去重、訊息切割、推播失敗不中斷
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ud():
    return _load("update_data")


def _prices(start, n, base=100.0, div_at=None):
    idx = pd.bdate_range(start, periods=n, name="date")
    c = base + np.arange(n, dtype=float)
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c,
                       "volume": 1000.0, "dividend": 0.0}, index=idx)
    if div_at is not None:
        df.iloc[div_at, df.columns.get_loc("dividend")] = 2.5
    return df


class FakeFetch:
    """依 start 參數回傳「真實」歷史的切片；記錄每次呼叫。"""

    def __init__(self, truth: pd.DataFrame, fail=False):
        self.truth, self.fail, self.calls = truth, fail, []

    def __call__(self, code, start):
        self.calls.append(start)
        if self.fail:
            raise ConnectionError("無網路")
        return self.truth.loc[pd.Timestamp(start):].copy()


NOSLEEP = dict(sleep=lambda s: None, pause=0, retries=1)


# ---------------------------------------------------------------- update_data
def test_incremental_merge_appends_and_new_wins(ud, tmp_path):
    truth = _prices("2025-01-01", 60)
    old = truth.iloc[:50].copy()
    old.to_csv(tmp_path / "2330.csv")
    truth.iloc[45, truth.columns.get_loc("volume")] = 9999.0   # 重疊段量修正 → 以新資料為準
    f = FakeFetch(truth)
    st = ud.run(["2330"], tmp_path, fetch=f, status_path=tmp_path / "st.json", **NOSLEEP)
    last_old = old.index[-1]
    assert f.calls == [(last_old - pd.Timedelta(days=10)).strftime("%Y-%m-%d")]   # 只抓最後日期 −10 天
    got = pd.read_csv(tmp_path / "2330.csv", parse_dates=["date"], index_col="date")
    assert len(got) == 60 and got.index.is_unique
    assert got["volume"].iloc[45] == 9999.0
    assert st["updated_codes"] == ["2330"] and st["full_refetch_codes"] == []
    assert st["last_date"] == str(truth.index[-1].date()) and st["stale_codes"] == []


def test_close_mismatch_triggers_full_refetch(ud, tmp_path):
    truth = _prices("2025-01-01", 60)
    old = truth.iloc[:50].copy()
    old[["open", "high", "low", "close"]] *= 2          # 舊檔是分割前價格 → 重疊段差很多
    old.to_csv(tmp_path / "2330.csv")
    f = FakeFetch(truth)
    st = ud.run(["2330"], tmp_path, start="2025-01-01", fetch=f,
                status_path=tmp_path / "st.json", **NOSLEEP)
    assert f.calls[-1] == "2025-01-01" and len(f.calls) == 2
    got = pd.read_csv(tmp_path / "2330.csv", parse_dates=["date"], index_col="date")
    assert got["close"].iloc[0] == pytest.approx(truth["close"].iloc[0])
    assert st["full_refetch_codes"] == ["2330"]


def test_dividend_in_window_triggers_full_refetch(ud, tmp_path):
    truth = _prices("2025-01-01", 60, div_at=55)
    truth.iloc[:50].to_csv(tmp_path / "2330.csv")
    f = FakeFetch(truth)
    ud.run(["2330"], tmp_path, start="2025-01-01", fetch=f, status_path=tmp_path / "st.json", **NOSLEEP)
    assert len(f.calls) == 2 and f.calls[-1] == "2025-01-01"


def test_full_flag_and_shrink_guard(ud, tmp_path):
    long = _prices("2024-01-01", 200)
    long.to_csv(tmp_path / "2330.csv")
    short = long.iloc[-100:]                            # 資料源只回傳一半 → 不覆蓋
    f = FakeFetch(short)
    st = ud.run(["2330"], tmp_path, full=True, start="2024-01-01", fetch=f,
                status_path=tmp_path / "st.json", **NOSLEEP)
    assert f.calls == ["2024-01-01"]
    assert len(pd.read_csv(tmp_path / "2330.csv")) == 200
    assert st["kept_codes"] == ["2330"] and st["updated_codes"] == []


def test_all_failed_exit_2_and_status(ud, tmp_path, monkeypatch):
    _prices("2025-01-01", 30).to_csv(tmp_path / "2330.csv")
    _prices("2025-01-01", 25).to_csv(tmp_path / "2317.csv")
    monkeypatch.setattr(ud, "_default_fetch", FakeFetch(None, fail=True))
    monkeypatch.setattr(ud.time, "sleep", lambda s: None)
    status = tmp_path / "data_status.json"
    rc = ud.main(["--codes", "2330,2317", "--out", str(tmp_path), "--status", str(status),
                  "--sleep", "0", "--retries", "1"])
    assert rc == 2
    st = json.loads(status.read_text(encoding="utf-8"))
    assert st["all_failed"] and sorted(st["failed_codes"]) == ["2317", "2330"]
    assert st["stale_codes"] == ["2317"]                # 2317 檔案日期落後 2330
    assert st["last_date"] == str(pd.bdate_range("2025-01-01", periods=30)[-1].date())
    assert len(pd.read_csv(tmp_path / "2330.csv")) == 30   # 舊資料完好


def test_partial_failure_exit_0(ud, tmp_path, monkeypatch):
    truth = _prices("2025-01-01", 30)
    calls = {"n": 0}

    def fetch(code, start):
        calls["n"] += 1
        if code == "2317":
            raise ConnectionError("x")
        return truth.loc[start:]

    monkeypatch.setattr(ud, "_default_fetch", fetch)
    monkeypatch.setattr(ud.time, "sleep", lambda s: None)
    rc = ud.main(["--codes", "2330,2317", "--out", str(tmp_path), "--status", str(tmp_path / "s.json"),
                  "--sleep", "0", "--retries", "1"])
    assert rc == 0
    st = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert st["failed_codes"] == ["2317"] and st["stale_codes"] == ["2317"]


# ---------------------------------------------------------------- fetch_chip 增量
def test_fetch_chip_incremental_start(tmp_path):
    fc = _load("fetch_chip")
    assert fc.incremental_start("2330", tmp_path, "2015-01-01") == "2015-01-01"   # 無檔
    pd.DataFrame({"date": ["2025-03-20", "2025-03-21"], "foreign_net": [1, 2],
                  "trust_net": [0, 0], "dealer_net": [0, 0]}).to_csv(tmp_path / "2330.csv", index=False)
    assert fc.incremental_start("2330", tmp_path, "2015-01-01") == "2025-03-11"
    seen = []
    fc.run(["2330"], "2015-01-01", tmp_path, pause=0, retries=1,
           fetch=lambda c, start: seen.append(start) or pd.DataFrame(), sleep=lambda s: None,
           incremental=True)
    assert seen == ["2025-03-11"]


# ---------------------------------------------------------------- export_signals 去重
def test_export_write_if_changed(tmp_path):
    ex = _load("export_signals")
    path = tmp_path / "signals.json"
    base = {"as_of": "2026-09-30", "generated_at": "2026-09-30 20:55", "stocks": [1]}
    assert ex.write_if_changed(dict(base), path)
    # 同 as_of、同內容、不同 generated_at → 不改檔
    assert not ex.write_if_changed({**base, "generated_at": "2026-10-01 20:55"}, path)
    assert json.loads(path.read_text())["generated_at"] == "2026-09-30 20:55"
    # 同 as_of、內容變了 → 改檔但沿用舊 generated_at
    assert ex.write_if_changed({**base, "generated_at": "2026-10-01 21:00", "stocks": [2]}, path)
    assert json.loads(path.read_text())["generated_at"] == "2026-09-30 20:55"
    # as_of 改變 → 更新 generated_at
    assert ex.write_if_changed({**base, "as_of": "2026-10-01", "generated_at": "2026-10-01 21:00"}, path)
    assert json.loads(path.read_text())["generated_at"] == "2026-10-01 21:00"


def test_export_load_stale_codes(tmp_path):
    ex = _load("export_signals")
    assert ex.load_stale_codes(tmp_path / "nope.json") == []
    (tmp_path / "s.json").write_text(json.dumps({"stale_codes": ["2317", "1101"]}))
    assert ex.load_stale_codes(tmp_path / "s.json") == ["1101", "2317"]


# ---------------------------------------------------------------- notify_signals
def _signals(as_of="2026-10-01", n=3, reason="理由"):
    return {
        "as_of": as_of, "stale_codes": ["2317"],
        "strategies": [{"name": "s", "title": "策略", "synthetic": False}],
        "stocks": [{"code": f"{1000 + i}", "name": "某股", "close": 10.0,
                    "signals": {"s": {"action": "BUY", "reason": reason}}} for i in range(n)],
    }


class Post:
    def __init__(self, ok=True, exc=None):
        self.ok, self.exc, self.calls = ok, exc, []

    def __call__(self, url, data=None, timeout=None):
        self.calls.append(data)
        if self.exc:
            raise self.exc
        return type("R", (), {"ok": self.ok, "text": "err"})()


def test_notify_dedup_by_as_of(tmp_path, monkeypatch):
    ns = _load("notify_signals")
    (tmp_path / "signals.json").write_text(json.dumps(_signals()), encoding="utf-8")
    post = Post()
    monkeypatch.setattr(ns.requests, "post", post)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    assert ns.main([], output_dir=tmp_path) == 0
    assert len(post.calls) == 1 and "日K 未更新：2317" in post.calls[0]["text"]
    assert (tmp_path / "last_notified.txt").read_text().strip() == "2026-10-01"
    assert ns.main([], output_dir=tmp_path) == 0          # 同 as_of 第二次 → 不推
    assert len(post.calls) == 1
    assert ns.main(["--force"], output_dir=tmp_path) == 0
    assert len(post.calls) == 2


def test_notify_failure_is_warning_not_exit(tmp_path, monkeypatch, capsys):
    import requests
    ns = _load("notify_signals")
    (tmp_path / "signals.json").write_text(json.dumps(_signals()), encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    for post in [Post(ok=False), Post(exc=requests.ConnectionError("down"))]:
        monkeypatch.setattr(ns.requests, "post", post)
        assert ns.main([], output_dir=tmp_path) == 0
        assert "::warning::" in capsys.readouterr().out
        assert not (tmp_path / "last_notified.txt").exists()   # 失敗不記錄 → 下次會重推


def test_split_message_by_lines_and_tags():
    ns = _load("notify_signals")
    msg = ns.build_message(_signals(n=200, reason="外資連買" * 10))
    chunks = ns.split_message(msg, 3500)
    assert len(chunks) > 1 and all(len(c) < 3500 for c in chunks)
    assert "\n".join(chunks).replace("\n", "") == msg.replace("\n", "")
    for c in chunks:
        assert c.count("<b>") == c.count("</b>")
    long = "<b>x</b>" + "字" * 8000
    parts = ns.split_message(long, 3500)
    assert all(len(p) < 3500 for p in parts) and all("<" not in p for p in parts)
