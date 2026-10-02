"""常用技術指標（寬表運算：index=date, columns=code）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """與 scripts/stock_monitor.py 相同的簡單平均版 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    prev = close.shift(1)
    tr = pd.concat({"a": high - low, "b": (high - prev).abs(), "c": (low - prev).abs()}).groupby(level=1).max()
    tr = tr.reindex(close.index)
    return tr.rolling(n, min_periods=n).mean()


def rolling_high(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).max()


def total_return_index(close: pd.DataFrame, dividend: pd.DataFrame) -> pd.DataFrame:
    """含息報酬指數（用未還原價 + 現金股利計算），適合算動能。"""
    r = (close + dividend.fillna(0.0)) / close.shift(1) - 1
    return (1 + r.fillna(0.0)).cumprod().where(close.notna())
