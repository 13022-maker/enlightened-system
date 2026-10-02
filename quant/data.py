"""資料層：日K 載入、盤中資料轉日K、yfinance / FinMind 抓取、籌碼資料。

日K CSV 格式（data/daily/<code>.csv）:
    date,open,high,low,close,volume,dividend
價格為「未還原」價格；除權息以 dividend 欄（每股現金股利）在回測中加回，
股票分割則在轉檔時自動向前調整（台股漲跌幅 10%，單日 >11% 的跳空視為公司行動）。

籌碼 CSV 格式（data/chip/<code>.csv）:
    date,foreign_net,trust_net,dealer_net     （單位：股）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import CHIP_DIR, DAILY_DIR, ETFS, LIMIT_PCT

PRICE_COLS = ["open", "high", "low", "close", "volume", "dividend"]


# ---------------------------------------------------------------- 轉檔工具
def adjust_splits(df: pd.DataFrame, threshold: float = LIMIT_PCT + 0.01) -> pd.DataFrame:
    """偵測超過漲跌停幅度的跳空（分割/合併），將之前的價格向前調整。"""
    df = df.copy()
    prev_close = df["close"].shift(1)
    gap = df["open"] / prev_close - 1
    for day in df.index[gap.abs() > threshold]:
        ratio = prev_close.loc[day] / df.loc[day, "open"]
        factor = round(ratio) if ratio > 1 else 1 / round(1 / ratio)
        if factor in (0, 1):
            continue
        before = df.index < day
        for c in ["open", "high", "low", "close", "dividend"]:
            df.loc[before, c] = df.loc[before, c] / factor
        df.loc[before, "volume"] = df.loc[before, "volume"] * factor
    return df


def sanitize_dividends(df: pd.DataFrame, max_single: float = 0.15) -> pd.DataFrame:
    """修正異常股利：單筆股利 > 前收 15% 視為資料錯誤（來源資料偶有單位錯誤，例如 2603 記成 962 元）。

    以「前收 − 除息日開盤」的跳空估算；跳空不明顯時沿用該股上一筆正常股利。
    只用除息日當天開盤與前一日收盤，沒有前視。
    """
    df = df.copy()
    prev_c = df["close"].shift(1)
    div = df["dividend"].fillna(0.0)
    bad = (div > 0) & (div / prev_c > max_single)
    if bad.any():
        gap = (prev_c - df["open"]).clip(lower=0.0).clip(upper=prev_c * max_single)
        last_ok = div.where((div > 0) & ~bad).ffill().shift(1).fillna(0.0)
        est = gap.where(gap >= prev_c * 0.002, last_ok)
        df.loc[bad, "dividend"] = est[bad].fillna(0.0)
    df["dividend"] = df["dividend"].fillna(0.0)
    return df


def intraday_to_daily(path: str) -> pd.DataFrame:
    """將 5 分K CSV（Datetime,Open,High,Low,Close,Volume,Dividends,...）轉成日K。"""
    raw = pd.read_csv(path)
    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["ts"] = pd.to_datetime(raw[ts_col], utc=True).dt.tz_convert("Asia/Taipei")
    raw = raw.drop_duplicates(subset="ts").sort_values("ts")
    raw = raw[raw["Close"] > 0]
    raw["date"] = raw["ts"].dt.tz_localize(None).dt.normalize()
    div = raw.get("Dividends", pd.Series(0.0, index=raw.index)).fillna(0.0)
    raw["div"] = div
    g = raw.groupby("date")
    daily = pd.DataFrame({
        "open": g["Open"].first(),
        "high": g["High"].max(),
        "low": g["Low"].min(),
        "close": g["Close"].last(),
        "volume": g["Volume"].sum(),
        "dividend": g["div"].max(),
    })
    daily.index.name = "date"
    return sanitize_dividends(adjust_splits(daily))


def fetch_yfinance_daily(code: str, start: str = "2015-01-01") -> pd.DataFrame:
    """在有網路的環境（本機 / GitHub Actions）用 yfinance 抓日K（未還原 + 股利）。"""
    import yfinance as yf

    suffixes = [".TW", ".TWO"]
    for suf in suffixes:
        hist = yf.Ticker(code + suf).history(start=start, auto_adjust=False, actions=True)
        if not hist.empty:
            break
    else:
        raise ValueError(f"yfinance 查無資料: {code}")
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    df = pd.DataFrame({
        "open": hist["Open"], "high": hist["High"], "low": hist["Low"],
        "close": hist["Close"], "volume": hist["Volume"],
        "dividend": hist.get("Dividends", 0.0),
    })
    df.index.name = "date"
    df = df[df["close"] > 0]
    # yfinance 未還原價格在分割日會跳空，同樣做向前調整
    return sanitize_dividends(adjust_splits(df))


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def fetch_finmind_chip(code: str, start: str = "2015-01-01") -> pd.DataFrame:
    """FinMind 三大法人買賣超（需網路；FINMIND_TOKEN 環境變數可選）。"""
    import requests

    params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell",
              "data_id": code, "start_date": start}
    token = os.environ.get("FINMIND_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") not in (None, 200):
        raise RuntimeError(f"FinMind 回應錯誤 {js.get('status')}: {js.get('msg')}")
    rows = js.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["foreign_net", "trust_net", "dealer_net"])
    df = pd.DataFrame(rows)
    df["net"] = df["buy"] - df["sell"]
    df["date"] = pd.to_datetime(df["date"])
    group = df["name"].map(lambda n: "foreign" if n.startswith("Foreign")
                           else "trust" if n == "Investment_Trust" else "dealer")
    out = df.assign(g=group).pivot_table(index="date", columns="g", values="net", aggfunc="sum")
    out = out.reindex(columns=["foreign", "trust", "dealer"]).fillna(0.0)
    out.columns = ["foreign_net", "trust_net", "dealer_net"]
    return out


# ---------------------------------------------------------------- Panel
@dataclass
class Panel:
    """寬表資料：每個欄位都是 index=date、columns=code 的 DataFrame。"""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dividend: pd.DataFrame
    chip: Dict[str, pd.DataFrame] = field(default_factory=dict)  # foreign_net / trust_net / dealer_net
    chip_is_synthetic: bool = False

    @property
    def codes(self) -> List[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def slice(self, start=None, end=None) -> "Panel":
        sl = slice(start, end)
        return Panel(
            *(getattr(self, c).loc[sl] for c in PRICE_COLS),
            chip={k: v.loc[sl] for k, v in self.chip.items()},
            chip_is_synthetic=self.chip_is_synthetic,
        )

    def is_etf(self, code: str) -> bool:
        return code in ETFS or code.startswith("00")


def load_daily(code: str, directory=DAILY_DIR) -> pd.DataFrame:
    df = pd.read_csv(f"{directory}/{code}.csv", parse_dates=["date"], index_col="date")
    return df[PRICE_COLS]


def load_panel(codes: Iterable[str], directory=DAILY_DIR, chip: bool = False,
               synthetic_chip_if_missing: bool = True) -> Panel:
    frames = {c: load_daily(c, directory) for c in codes}
    fields = {f: pd.DataFrame({c: d[f] for c, d in frames.items()}).sort_index() for f in PRICE_COLS}
    fields["dividend"] = fields["dividend"].fillna(0.0)
    panel = Panel(**fields)
    if chip:
        loaded = load_chip(panel.codes, panel.dates)
        if loaded:
            panel.chip = loaded
        elif synthetic_chip_if_missing:
            panel.chip = synthetic_chip(panel)
            panel.chip_is_synthetic = True
    return panel


def load_chip(codes: Iterable[str], dates: pd.DatetimeIndex, directory=CHIP_DIR) -> Dict[str, pd.DataFrame]:
    out = {k: {} for k in ["foreign_net", "trust_net", "dealer_net"]}
    found = False
    for c in codes:
        p = f"{directory}/{c}.csv"
        if not os.path.exists(p):
            continue
        found = True
        df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        for k in out:
            out[k][c] = df[k]
    if not found:
        return {}
    # 有籌碼檔的股票：檔案期間內缺的日子留 NaN；沒有檔案的股票整欄 NaN（不偽裝成「法人沒買賣」）
    return {k: pd.DataFrame(v).reindex(index=dates, columns=list(codes)) for k, v in out.items()}


def synthetic_chip(panel: Panel, seed: int = 7) -> Dict[str, pd.DataFrame]:
    """模擬法人買賣超，僅供在無網路環境測試程式流程。

    買賣超與「當日」報酬、成交量相關（真實市場常見），但刻意不含任何未來資訊，
    因此用它回測出的績效沒有參考價值。
    """
    rng = np.random.default_rng(seed)
    ret = panel.close.pct_change().fillna(0.0)
    vol = panel.volume.fillna(0.0)
    out = {}
    for name, beta, share in [("foreign_net", 3.0, 0.25), ("trust_net", 1.5, 0.05), ("dealer_net", 1.0, 0.05)]:
        noise = rng.standard_normal(ret.shape)
        signal = np.tanh(beta * ret.values * 10 + 0.8 * noise)
        out[name] = pd.DataFrame(signal * vol.values * share, index=ret.index, columns=ret.columns).round()
    return out
