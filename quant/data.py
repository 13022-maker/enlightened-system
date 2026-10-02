"""資料層：日K 載入、盤中資料轉日K、yfinance / FinMind 抓取、籌碼資料、公司行動與資料品質。

日K CSV 格式（data/daily/<code>.csv）:
    date,open,high,low,close,volume,dividend
價格為「未還原股利、但已還原股數變動」的價格：
    - 現金股利以 dividend 欄（每股現金股利，與當日價格同一股數基礎）在回測中加回；
    - 股數變動（分割、配股、減資、面額變更）在轉檔時以比率向前調整（之前的價格 ÷ 比率、量 × 比率）。

除權息事件檔（data/dividends/<code>.csv，選用；格式見 load_dividend_events）:
    date,cash_dividend,stock_dividend,share_ratio,note
    share_ratio = 除權後 1 股變成幾股（配股 1.95 元 → 1.195；1 拆 4 → 4；減資 4 成 → 0.6）。
    金額皆為「除權息當時」的每股金額（未經之後分割調整）。
    有此檔時：轉檔時用它的 share_ratio 做股數調整；載入時用它覆寫／補齊 dividend 欄。

籌碼 CSV 格式（data/chip/<code>.csv）:
    date,foreign_net,trust_net,dealer_net     （單位：股）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import (CHIP_DIR, DAILY_DIR, DIVIDEND_DIR, ETFS, LIMIT_PCT, NEW_LISTING_FREE_DAYS,
                     UNIVERSE_FILE, limit_pct_on)

PRICE_COLS = ["open", "high", "low", "close", "volume", "dividend"]
EVENT_COLS = ["cash_dividend", "stock_dividend", "share_ratio"]

# 跳空推測公司行動時，超出漲跌停幅度多少才算（Yahoo 5 分K 的首筆開盤偶有超出漲跌停 ~1.7% 的錯誤值，
# 例：6669 2025-04-10 開盤 +11.7%，因此留 2% 緩衝，避免把真實大漲大跌誤判為公司行動）
GAP_MARGIN = 0.02


# ---------------------------------------------------------------- 公司行動（股數變動）
def load_dividend_events(code: str, directory=DIVIDEND_DIR) -> Optional[pd.DataFrame]:
    """讀取 data/dividends/<code>.csv；不存在回傳 None。

    回傳 index=date，欄位 cash_dividend（元/股）、stock_dividend（元/股，僅供參考）、share_ratio（股數倍數）。
    同一天多筆（例如分開列除權與除息）會合併：現金相加、股數倍數相乘。
    """
    path = Path(directory) / f"{code}.csv"
    if not path.exists():
        return None
    ev = pd.read_csv(path, comment="#")
    if ev.empty:
        return None
    ev["date"] = pd.to_datetime(ev["date"])
    for c in EVENT_COLS:
        if c not in ev.columns:
            ev[c] = 1.0 if c == "share_ratio" else 0.0
    ev["cash_dividend"] = pd.to_numeric(ev["cash_dividend"], errors="coerce").fillna(0.0)
    ev["stock_dividend"] = pd.to_numeric(ev["stock_dividend"], errors="coerce").fillna(0.0)
    ev["share_ratio"] = pd.to_numeric(ev["share_ratio"], errors="coerce").fillna(1.0)
    g = ev.groupby("date")
    out = pd.DataFrame({"cash_dividend": g["cash_dividend"].sum(),
                        "stock_dividend": g["stock_dividend"].sum(),
                        "share_ratio": g["share_ratio"].prod()})
    return out.sort_index()


def _snap_ratio(r: float, lo: float, hi: float) -> float:
    """推測比率時，若 [lo, hi]（漲跌停容許範圍）內有整數 n 或 1/n，取最接近 r 者（分割多為整數倍）。

    例：2327 2025-08-25 1 拆 4，前收 545、復牌開盤 142 → r = 3.84（含開盤漲 4%），
    容許範圍 [3.45, 4.22] 含 4 → 用 4。配股（例 1.195）附近沒有整數 → 用 r。
    """
    cands = [float(n) for n in range(2, 51)] + [1.0 / n for n in range(2, 51)]
    ok = [c for c in cands if lo <= c <= hi]
    return min(ok, key=lambda c: abs(np.log(c / r))) if ok else r


def detect_share_events(df: pd.DataFrame, listing_date=None, skip_dates=(),
                        margin: float = GAP_MARGIN) -> pd.Series:
    """以「超過漲跌停的開盤跳空」推測未記錄的股數變動，回傳 Series(date → share_ratio)。

    參考價 = (前收 − 當日現金股利) / 比率；台股開盤只能落在參考價 ±漲跌停幅度內，
    因此開盤相對「除息後前收」的跳空超過漲跌停（+margin）時，必有股數變動：
        比率 r = (前收 − 當日現金股利) / 開盤，若容許範圍內有整數倍則取整數（見 _snap_ratio）。
    不適用的情境：
        - 新股上市前 NEW_LISTING_FREE_DAYS 個交易日無漲跌幅限制 → 不判斷（需傳入 listing_date；
          資料起點不一定是上市日，所以預設不套用）。
        - 漲跌幅以日期判斷：2015-06-01 前 7%、之後 10%（config.limit_pct_on）。
    skip_dates：已有明確事件（事件檔或 Yahoo Stock Splits 欄）的日期，不重複推測。
    前收用前一個有效收盤（停牌期間跨越多日仍以停牌前收盤為參考，與交易所規則一致）。
    """
    prev = df["close"].ffill().shift(1)
    cash = df["dividend"].fillna(0.0)
    base = prev - cash
    gap = df["open"] / base - 1
    lim = limit_pct_on(df.index)
    free = pd.Series(False, index=df.index)
    if listing_date is not None:
        first = df.index[df.index >= pd.Timestamp(listing_date)][:NEW_LISTING_FREE_DAYS]
        free.loc[first] = True
    skip = set(pd.to_datetime(list(skip_dates)))
    hit = (gap.abs() > lim + margin) & ~free & base.gt(0) & df["open"].gt(0)
    out = {}
    for day in df.index[hit.fillna(False).values]:
        if day in skip:
            continue
        r = base.loc[day] / df.loc[day, "open"]
        L = lim.loc[day]
        out[day] = _snap_ratio(r, r * (1 - L), r * (1 + L))
    return pd.Series(out, dtype=float)


def apply_share_events(df: pd.DataFrame, ratios: pd.Series) -> pd.DataFrame:
    """依股數變動比率向前調整：事件日之前的價格 ÷ 比率、成交量 × 比率；股利則「含事件日」÷ 比率。

    同日除權息（例：2884 現金 1.2 元＋配股 1.02）時，現金股利是按「除權前每股」發放，
    換成除權後股數基礎為 cash / ratio；參考價 = (前收 − cash) / ratio，兩者一致才不會多算報酬。
    """
    df = df.astype({c: float for c in PRICE_COLS if c in df.columns})
    for day, ratio in ratios.items():
        if not np.isfinite(ratio) or ratio <= 0 or abs(ratio - 1) < 1e-9:
            continue
        before = df.index < day
        for c in ["open", "high", "low", "close"]:
            df.loc[before, c] = df.loc[before, c] / ratio
        df.loc[df.index <= day, "dividend"] = df.loc[df.index <= day, "dividend"] / ratio
        df.loc[before, "volume"] = df.loc[before, "volume"] * ratio
    return df


def adjust_splits(df: pd.DataFrame, threshold: Optional[float] = None, events: Optional[pd.Series] = None,
                  listing_date=None) -> pd.DataFrame:
    """未還原價格 → 股數調整後價格（分割、配股、減資、面額變更）。

    events：已知事件 Series(date → share_ratio)（事件檔、Yahoo「Stock Splits」欄），優先採用；
    其餘以 detect_share_events 由跳空推測（非整數比率也會調整，例如 2327 2024-08-15 配股 1.195）。
    threshold 參數保留相容性（舊版的固定門檻），新版改用依日期的漲跌停幅度。
    """
    known = pd.Series(dtype=float) if events is None else events[events.index.isin(df.index)]
    known = known[(known - 1).abs() > 1e-9]
    margin = GAP_MARGIN if threshold is None else max(threshold - LIMIT_PCT, 0.0)
    guessed = detect_share_events(df, listing_date=listing_date, skip_dates=known.index, margin=margin)
    return apply_share_events(df, pd.concat([known, guessed]).sort_index())


def sanitize_dividends(df: pd.DataFrame, max_single: float = 0.15) -> pd.DataFrame:
    """修正異常股利：單筆股利 > 前收 15% 視為資料錯誤（來源資料偶有單位錯誤，例如 2603 記成 962 元）。

    以「前收 − 除息日開盤」的跳空估算；跳空不明顯時沿用該股上一筆正常股利。
    只用除息日當天開盤與前一日收盤，沒有前視。
    """
    df = df.copy()
    prev_c = df["close"].ffill().shift(1)
    div = df["dividend"].fillna(0.0)
    bad = (div > 0) & (div / prev_c > max_single)
    if bad.any():
        gap = (prev_c - df["open"]).clip(lower=0.0).clip(upper=prev_c * max_single)
        last_ok = div.where((div > 0) & ~bad).ffill().shift(1).fillna(0.0)
        est = gap.where(gap >= prev_c * 0.002, last_ok)
        df.loc[bad, "dividend"] = est[bad].fillna(0.0)
    df["dividend"] = df["dividend"].fillna(0.0)
    return df


def _mode_nonzero(x: pd.Series) -> float:
    """同一時間戳多筆快照時挑選股利/分割值：取非零值的眾數，平手取較小者（大值多為單位錯誤）。"""
    v = x[x.fillna(0) != 0]
    if v.empty:
        return 0.0
    counts = v.round(6).value_counts()
    top = counts[counts == counts.max()].index
    return float(min(top))


def _read_intraday(path: str) -> pd.DataFrame:
    """讀 5 分K，處理重複快照。

    voidful/tw_stocker 的檔案把同一根 K 棒重複存了多次（不同日期的抓取快照），
    其中股利欄只出現在部分快照（例：2303 2025-06-24 十筆中只有三筆有 2.85 元股利）。
    舊版 drop_duplicates(keep="first") 因此漏掉 2303、2308 的 2025 年股利。
    現在：OHLCV 取最後一筆快照，Dividends / Stock Splits 取所有快照的非零眾數。
    """
    raw = pd.read_csv(path)
    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["ts"] = pd.to_datetime(raw[ts_col], utc=True).dt.tz_convert("Asia/Taipei")
    for c in ["Dividends", "Stock Splits"]:
        if c not in raw.columns:
            raw[c] = 0.0
    acts = raw.groupby("ts")[["Dividends", "Stock Splits"]].agg(_mode_nonzero)
    bars = raw.drop_duplicates(subset="ts", keep="last").set_index("ts").sort_index()
    bars[["Dividends", "Stock Splits"]] = acts.reindex(bars.index).values
    return bars[bars["Close"] > 0]


def intraday_to_daily(path: str, code: Optional[str] = None, dividend_dir=DIVIDEND_DIR,
                      listing_date=None) -> pd.DataFrame:
    """將 5 分K CSV（Datetime,Open,High,Low,Close,Volume,Dividends,Stock Splits）轉成日K。

    股數變動來源（優先序）：data/dividends/<code>.csv 的 share_ratio > Yahoo「Stock Splits」欄 > 跳空推測。
    5 分K 是逐日存下的原始報價，Yahoo 事後的分割調整不會回溯到這些檔案（實測 2881 2024-09-09
    配股 1.05：前收 92.6、開盤 86.5，價格未調整），所以 Stock Splits 欄的比率必須自行套用。
    """
    raw = _read_intraday(path)
    raw["date"] = raw.index.tz_localize(None).normalize()
    g = raw.groupby("date")
    daily = pd.DataFrame({
        "open": g["Open"].first(),
        "high": g["High"].max(),
        "low": g["Low"].min(),
        "close": g["Close"].last(),
        "volume": g["Volume"].sum(),
        "dividend": g["Dividends"].max(),
    })
    daily.index.name = "date"
    splits = g["Stock Splits"].max()
    events = splits[splits > 0]
    code = code or Path(path).stem
    ev = load_dividend_events(code, dividend_dir)
    if ev is not None:
        # 事件檔優先：覆寫現金股利與股數比率（事件日若非交易日，移到下一個交易日）
        daily = _overlay_cash(daily, ev)
        er = _align_events(ev["share_ratio"], daily.index)
        events = pd.concat([events[~events.index.isin(er.index)], er]).sort_index()
    return sanitize_dividends(adjust_splits(daily, events=events, listing_date=listing_date))


def _align_events(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """事件日若不是該股交易日（停牌、颱風休市順延），移到之後第一個交易日；超出資料範圍者捨棄。"""
    out = {}
    for day, v in s.items():
        pos = index.searchsorted(day)
        if pos < len(index) and (pos > 0 or index[0] == day):
            d = index[pos]
            out[d] = out.get(d, 1.0 if s.name == "share_ratio" else 0.0)
            out[d] = out[d] * v if s.name == "share_ratio" else out[d] + v
    return pd.Series(out, dtype=float, name=s.name)


def _overlay_cash(df: pd.DataFrame, ev: pd.DataFrame, later_factor: Optional[pd.Series] = None,
                  window: int = 5) -> pd.DataFrame:
    """以事件檔覆寫／補齊現金股利。

    - 事件日（對齊到交易日）dividend = 事件現金股利 ÷ later_factor（之後股數變動的累積倍數，
      載入已調整的日K 時用；轉檔時為 1）。
    - 事件日前後 window 個交易日內原有的股利若與事件金額相近（±25%），視為同一筆、日期錯置 → 移除，避免重複計入。
    """
    df = df.copy()
    cash = _align_events(ev.loc[ev["cash_dividend"] > 0, "cash_dividend"].rename("cash_dividend"), df.index)
    if cash.empty:
        return df
    div = df["dividend"].fillna(0.0).copy()
    pos_of = {d: i for i, d in enumerate(df.index)}
    for day, amt in cash.items():
        f = 1.0 if later_factor is None else float(later_factor.get(day, 1.0))
        amt_adj = amt / f
        i = pos_of[day]
        for j in range(max(0, i - window), min(len(df), i + window + 1)):
            dj = df.index[j]
            if dj != day and dj not in cash.index and div.iloc[j] > 0 and abs(div.iloc[j] / amt_adj - 1) < 0.25:
                div.iloc[j] = 0.0
        div.loc[day] = amt_adj
    df["dividend"] = div
    return df


def later_share_factor(ratios: pd.Series, index: pd.DatetimeIndex, inclusive: bool = False) -> pd.Series:
    """每個日期之後所有股數變動比率的乘積：已調整價格 = 原始價格 ÷ 此值。

    inclusive=True 時含當日事件（用於現金股利：同日除權息的股利以除權後股數為基礎，見 apply_share_events）。
    """
    f = pd.Series(1.0, index=index)
    for day, r in ratios.items():
        f[(index <= day) if inclusive else (index < day)] *= r
    return f


def share_event_applied(df: pd.DataFrame, day, ratio: float, cash: float = 0.0,
                        min_detectable: float = 0.04) -> bool:
    """判斷已存在的日K 是否已套用某次股數變動（避免重複調整）。

    比較事件日開盤相對前收的跳空：已套用 → 跳空接近 1；未套用 → 接近 1/比率。
    比率太接近 1（< min_detectable，例如金融股配股 1%~3%）時價格雜訊會蓋過訊號，無法判斷，
    依慣例視為「已套用」（轉檔與 yfinance 兩條路徑都會套用股數變動）。
    """
    if abs(np.log(ratio)) < min_detectable or day not in df.index:
        return True
    pos = df.index.get_loc(day)
    if pos == 0:
        return True
    prev = df["close"].iloc[:pos].dropna()
    if prev.empty:
        return True
    g = df["open"].iloc[pos] / (prev.iloc[-1] - cash / ratio)
    return abs(np.log(g)) <= abs(np.log(g * ratio))


def apply_dividend_file(df: pd.DataFrame, ev: Optional[pd.DataFrame]) -> pd.DataFrame:
    """載入時套用事件檔（df 為已做股數調整的日K）。

    1. 事件檔中「尚未被套用」的大幅股數變動（例：舊版轉檔因 round() 略過的 2327 配股 1.195）→ 向前調整。
    2. 現金股利換算到已調整基礎（÷ 之後所有股數變動的累積倍數）後覆寫/補齊 dividend 欄。
    """
    if ev is None or ev.empty or df.empty:
        return df
    ratios = _align_events(ev["share_ratio"].rename("share_ratio"), df.index)
    ratios = ratios[(ratios - 1).abs() > 1e-9]
    cash_on = _align_events(ev["cash_dividend"].rename("cash_dividend"), df.index)
    cash_f = later_share_factor(ratios, df.index, inclusive=True)
    # share_event_applied 的 cash 參數是「除權前每股」金額：= 事件檔金額 ÷ 之後（不含當日）的累積倍數
    pre_f = later_share_factor(ratios, df.index)
    missing = pd.Series({d: r for d, r in ratios.items()
                         if not share_event_applied(df, d, r, float(cash_on.get(d, 0.0)) / pre_f.get(d, 1.0))},
                        dtype=float)
    if len(missing):
        df = apply_share_events(df, missing)
    return _overlay_cash(df, ev, later_factor=cash_f)


def fetch_yfinance_daily(code: str, start: str = "2015-01-01") -> pd.DataFrame:
    """在有網路的環境（本機 / GitHub Actions）用 yfinance 抓日K（未還原股利 + 股利欄）。

    股數變動處理（依 yfinance 1.x 行為，查證依據如下）：
      - Yahoo 回傳的日K OHLC（auto_adjust=False 的 Open/High/Low/Close）本身「已做分割調整」，
        auto_adjust 只控制股利調整（Adj Close）。依據：
          * yfinance 維護者 ValueRaider 於 issue #2660（Price repair split false-positive）：
            "Yahoo returns split-adjusted prices and sometimes doesn't apply split fully"；
          * yfinance 1.7.0 原始碼 scrapers/history.py `_fix_bad_stock_splits` 註解：
            大幅跳空若與「最近一次未來分割比率」吻合，表示 Yahoo 沒有把新分割套用到舊價格（修復邏輯的前提
            就是正常情況下舊價格已調整）；utils.auto_adjust 只用 Adj Close / Close 比率縮放 OHLC（只含股利）；
          * yfinance-cache 文件："yfinance only allows control of dividends adjustment via auto_adjust"。
        台股配股在 Yahoo 也記為 Stock Splits（例：2881 2024-09-09 = 1.05），同樣已套用在日K。
      - 因此：不再用跳空推測；改以 hist["Stock Splits"] 逐筆檢查是否「已套用」（share_event_applied），
        只對 Yahoo 偶發漏套的大比率分割補做調整，避免重複調整。
    5 分K（voidful/tw_stocker）則是逐日保存的原始報價，分割「未」回溯調整，見 intraday_to_daily。
    """
    import yfinance as yf

    suffixes = [".TW", ".TWO"]
    for suf in suffixes:
        hist = yf.Ticker(code + suf).history(start=start, auto_adjust=False, actions=True)
        if not hist.empty:
            break
    else:
        raise ValueError(f"yfinance 查無資料: {code}")
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    return yfinance_frame(hist)


def yfinance_frame(hist: pd.DataFrame) -> pd.DataFrame:
    """把 yfinance history() 的結果轉成日K 格式並處理 Stock Splits（拆出來方便以假資料測試）。"""
    df = pd.DataFrame({
        "open": hist["Open"], "high": hist["High"], "low": hist["Low"],
        "close": hist["Close"], "volume": hist["Volume"],
        "dividend": hist["Dividends"] if "Dividends" in hist else 0.0,
    })
    df.index.name = "date"
    df = df[df["close"] > 0]
    splits = hist["Stock Splits"] if "Stock Splits" in hist else pd.Series(dtype=float)
    splits = splits[splits.reindex(df.index).fillna(0) > 0] if len(splits) else splits
    splits = splits[splits.index.isin(df.index)]
    todo = pd.Series({d: r for d, r in splits.items()
                      if not share_event_applied(df, d, float(r), float(df.loc[d, "dividend"] or 0.0))},
                     dtype=float)
    return sanitize_dividends(apply_share_events(df, todo))


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
    """寬表資料：每個欄位都是 index=date、columns=code 的 DataFrame。

    universe：point-in-time 股票池遮罩（date×code 布林表，True = 當日為指數成分股，可被策略新選入）。
        沒有成分股資料時為 None，此時 universe_mask() 回傳全 True（= 舊行為）。
        ETF（0050/0056）不是成分股，但作為基準與大盤濾網，遮罩中一律 True。
    """
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dividend: pd.DataFrame
    chip: Dict[str, pd.DataFrame] = field(default_factory=dict)  # foreign_net / trust_net / dealer_net
    chip_is_synthetic: bool = False
    universe: Optional[pd.DataFrame] = None

    @property
    def codes(self) -> List[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def universe_mask(self) -> pd.DataFrame:
        """回傳與 close 對齊的布林遮罩；無成分股資料時全 True。"""
        if self.universe is None:
            return pd.DataFrame(True, index=self.dates, columns=self.codes)
        return self.universe.reindex(index=self.dates, columns=self.codes).fillna(False).astype(bool)

    def slice(self, start=None, end=None) -> "Panel":
        sl = slice(start, end)
        return Panel(
            *(getattr(self, c).loc[sl] for c in PRICE_COLS),
            chip={k: v.loc[sl] for k, v in self.chip.items()},
            chip_is_synthetic=self.chip_is_synthetic,
            universe=None if self.universe is None else self.universe.loc[sl],
        )

    def is_etf(self, code: str) -> bool:
        return code in ETFS or code.startswith("00")


def load_daily(code: str, directory=DAILY_DIR, dividend_dir=DIVIDEND_DIR) -> pd.DataFrame:
    """讀日K；data/dividends/<code>.csv 存在時以其覆寫/補齊 dividend 欄（並補做尚未套用的大幅股數變動）。"""
    df = pd.read_csv(f"{directory}/{code}.csv", parse_dates=["date"], index_col="date")[PRICE_COLS]
    if dividend_dir is not None:
        df = apply_dividend_file(df, load_dividend_events(code, dividend_dir))
    return df


def load_panel(codes: Iterable[str], directory=DAILY_DIR, chip: bool = False,
               synthetic_chip_if_missing: bool = True, point_in_time: bool = False,
               universe_file=UNIVERSE_FILE, dividend_dir=DIVIDEND_DIR) -> Panel:
    """載入多檔日K 成 Panel。

    point_in_time=True：另外載入 universe_file 中所有曾為成分股、且 data/daily 有檔案的股票
        （被調出的股票價格仍保留，讓已持有部位能正確計價與賣出），並建立 panel.universe 遮罩。
        成分股檔不存在時 universe 為 None（全 True）。
    缺檔的股票會略過並印出警告（不再直接拋錯），方便 membership 檔含無資料的股票（例：7769 鴻勁）。
    """
    codes = list(dict.fromkeys(codes))
    membership = None
    if point_in_time and universe_file is not None and Path(universe_file).exists():
        from .universe import Universe
        membership = Universe.from_csv(universe_file)
        codes += [c for c in membership.codes if c not in codes]
    frames = {}
    for c in codes:
        if not Path(f"{directory}/{c}.csv").exists():
            print(f"⚠️ 缺少日K：{c}（略過）")
            continue
        frames[c] = load_daily(c, directory, dividend_dir)
    fields = {f: pd.DataFrame({c: d[f] for c, d in frames.items()}).sort_index() for f in PRICE_COLS}
    fields["dividend"] = fields["dividend"].fillna(0.0)
    panel = Panel(**fields)
    if membership is not None:
        mask = membership.membership_mask(panel.dates, panel.codes)
        for c in panel.codes:
            if panel.is_etf(c):
                mask[c] = True
        panel.universe = mask
    if chip:
        loaded = load_chip(panel.codes, panel.dates)
        if loaded:
            panel.chip = loaded
        elif synthetic_chip_if_missing:
            panel.chip = synthetic_chip(panel)
            panel.chip_is_synthetic = True
    return panel


# ---------------------------------------------------------------- 資料品質
def check_data_quality(panel: Panel, gap_down: float = -0.03, big_move: float = 0.105,
                       yield_lo: float = 0.0, yield_hi: float = 0.15, min_halt_days: int = 1,
                       market_relative: bool = True) -> List[Dict]:
    """檢查 Panel 資料品質，回傳問題清單 [{"code","date","type","detail"}, ...]。

    type：
      gap_no_dividend   開盤相對前收跳空 < gap_down（−3%）且當日無股利 → 可能漏記除權息（或真實利空，需人工確認）。
                        market_relative=True（預設）時，跳空須「同時」比當日全體個股跳空中位數再低 3% 以上，
                        排除大盤整體跳空日（例：2025-04-07 關稅、春節後開盤，全體 −5%~−10%），
                        否則 2 年資料會有約 800 筆、無法人工檢查。
      big_move          單日含息報酬（收盤＋股利）/ 前收 超過 ±big_move（10.5%，超過漲跌停）→ 多為未調整的公司行動
      halt              上市後、最後交易日前出現無價格的交易日（停牌或資料缺漏）；連續日合併為一筆
      dividend_yield    單筆股利／前收 不在 (yield_lo, yield_hi] → 單位錯誤或異常
      no_dividend_year  （非 ETF）資料滿一年以上的某個曆年完全沒有除息紀錄 → 可能漏抓（虧損公司則屬正常）
    """
    issues: List[Dict] = []
    o, c, d = panel.open, panel.close, panel.dividend.fillna(0.0)
    prev = c.ffill().shift(1)
    gap = o / prev - 1
    move = (c + d) / prev - 1
    stocks = [x for x in panel.codes if not panel.is_etf(x)]
    mkt_gap = gap[stocks].median(axis=1) if stocks else pd.Series(0.0, index=panel.dates)
    rel_gap = gap.sub(mkt_gap, axis=0)
    for code in panel.codes:
        cc = c[code]
        if cc.notna().sum() == 0:
            continue
        first, last = cc.first_valid_index(), cc.last_valid_index()
        live = (panel.dates >= first) & (panel.dates <= last)
        g = gap[code][live]
        flag = (g < gap_down) & (d[code][live] <= 0)
        if market_relative:
            flag &= rel_gap[code][live] < gap_down
        for day in g.index[flag.fillna(False)]:
            issues.append({"code": code, "date": str(day.date()), "type": "gap_no_dividend",
                           "detail": f"開盤跳空 {g[day]:+.1%}（大盤中位數 {mkt_gap[day]:+.1%}），當日無股利"})
        m = move[code][live]
        for day in m.index[m.abs() > big_move]:
            issues.append({"code": code, "date": str(day.date()), "type": "big_move",
                           "detail": f"含息單日報酬 {m[day]:+.1%}"})
        miss = cc[live].isna()
        if miss.any():
            runs = (miss != miss.shift()).cumsum()[miss]
            for _, grp in runs.groupby(runs):
                if len(grp) >= min_halt_days:
                    issues.append({"code": code, "date": str(grp.index[0].date()), "type": "halt",
                                   "detail": f"{len(grp)} 個交易日無價格（至 {grp.index[-1].date()}）"})
        y = (d[code] / prev[code])[d[code] > 0]
        for day in y.index[(y <= yield_lo) | (y > yield_hi)]:
            issues.append({"code": code, "date": str(day.date()), "type": "dividend_yield",
                           "detail": f"股利 {d[code][day]:.4g} 元 = 前收 {y[day]:.1%}"})
        if not panel.is_etf(code):
            years = pd.Series(cc[live].index.year).value_counts()
            for yr in sorted(years.index):
                yr_days = (cc.index.year == yr) & live
                full = cc.index[yr_days].min() <= pd.Timestamp(f"{yr}-01-15") and \
                    cc.index[yr_days].max() >= pd.Timestamp(f"{yr}-12-15")
                if full and d[code][cc.index.year == yr].sum() <= 0:
                    issues.append({"code": code, "date": f"{yr}", "type": "no_dividend_year",
                                   "detail": f"{yr} 年全年無除息紀錄"})
    return issues


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
