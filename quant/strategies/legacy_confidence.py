"""現行「開明體系 Phase 4」多指標信心度系統的回測版本（作為比較基準）。

信心度（一般股，純技術面 100 分；與 scripts/stock_monitor.py calculate_confidence_phase4 一致）:
    價格 > MA20 20 分、MA20 > MA50 20 分、RSI 40~65 20 分、MACD 當日金叉 25 分、量 > 20 日均量 1.2 倍 15 分
進場：信心度 ≥ threshold
出場：收盤跌破 MA20、RSI > 80、停損 -5%、停利 +15%（README Phase 4 規則）
部位：每檔固定 1/max_positions，最多 max_positions 檔
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data import Panel
from .base import Strategy


class LegacyConfidence(Strategy):
    name = "legacy_confidence"
    title = "現行信心度系統"
    description = "開明體系 Phase 4 五指標信心度，≥門檻買進，跌破 MA20 / RSI>80 / -5% / +15% 出場"
    default_params = {"threshold": 0.5, "max_positions": 10, "stop": 0.05, "target": 0.15}
    param_grid = [
        {"threshold": 0.5}, {"threshold": 0.6}, {"threshold": 0.7}, {"threshold": 0.8},
    ]

    def _confidence(self, panel: Panel) -> pd.DataFrame:
        c = panel.close
        ma20, ma50 = ind.sma(c, 20), ind.sma(c, 50)
        r = ind.rsi(c, 14)
        line, sig, _ = ind.macd(c)
        golden = (line > sig) & (line.shift(1) <= sig.shift(1))
        vr = panel.volume / ind.sma(panel.volume, 20)
        score = ((c > ma20) * 20 + (ma20 > ma50) * 20 + ((r >= 40) & (r <= 65)) * 20
                 + golden * 25 + (vr > 1.2) * 15).astype(float)
        return (score / 100).where(ma50.notna())

    def scores(self, panel: Panel, **params) -> pd.DataFrame:
        return self._confidence(panel)

    def target_weights(self, panel: Panel, threshold=0.5, max_positions=10, stop=0.05, target=0.15, **_) -> pd.DataFrame:
        conf = self._confidence(panel).values
        c = panel.close.values
        ma20 = ind.sma(panel.close, 20).values
        r = ind.rsi(panel.close, 14).values
        tradable = np.array([not panel.is_etf(x) for x in panel.codes])
        n_days, n = c.shape
        w = np.zeros((n_days, n))
        held = {}                                     # i -> entry close
        size = 1.0 / max_positions
        for t in range(n_days):
            for i in list(held):
                p = c[t, i]
                if np.isnan(p):
                    continue
                e = held[i]
                if p < ma20[t, i] or r[t, i] > 80 or p <= e * (1 - stop) or p >= e * (1 + target):
                    del held[i]
            slots = max_positions - len(held)
            if slots > 0:
                cand = [(conf[t, i], i) for i in range(n)
                        if tradable[i] and i not in held and conf[t, i] >= threshold]
                for _, i in sorted(cand, reverse=True)[:slots]:
                    held[i] = c[t, i]
            for i in held:
                w[t, i] = size
        return pd.DataFrame(w, index=panel.dates, columns=panel.codes)

    def explain(self, panel: Panel, code: str, **params) -> str:
        conf = self._confidence(panel)[code].iloc[-1]
        return "" if pd.isna(conf) else f"信心度 {conf:.0%}"
