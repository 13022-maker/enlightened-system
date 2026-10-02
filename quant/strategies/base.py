"""策略介面。

每個策略繼承 Strategy，實作 target_weights(panel, **params) 回傳目標權重表：
    index = panel.dates, columns = panel.codes
    值 = 以「當日收盤（含）之前」資料決定的目標權重，隔日開盤成交
    整列 NaN = 當天不調整（例如只在月初再平衡）
    列總和 ≤ 1（不足的部分為現金）；不可做空（負值會被截為 0）

嚴禁使用未來資料：t 日的權重只能依賴 ≤ t 的資料。
請用 quant.validation.assert_no_lookahead 自我檢查。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..data import Panel


class Strategy:
    name: str = "base"
    title: str = "基礎策略"          # 中文名稱（顯示於看盤介面）
    description: str = ""
    needs_chip: bool = False
    default_params: Dict = {}
    param_grid: List[Dict] = []      # walk-forward 時在訓練期挑選；請保持 ≤ 12 組以防過擬合

    def target_weights(self, panel: Panel, **params) -> pd.DataFrame:
        raise NotImplementedError

    def scores(self, panel: Panel, **params) -> pd.DataFrame:
        """選用：回傳每檔每日的分數（越高越看好），看盤介面用來排序。"""
        return pd.DataFrame(np.nan, index=panel.dates, columns=panel.codes)

    def explain(self, panel: Panel, code: str, **params) -> str:
        """選用：最新一日對某檔股票的判斷理由（繁體中文，一行）。"""
        return ""

    def latest_signals(self, panel: Panel, **params) -> List[Dict]:
        """依最後兩個有效權重列推導最新訊號。"""
        params = {**self.default_params, **params}
        w = self.target_weights(panel, **params).ffill().fillna(0.0)
        sc = self.scores(panel, **params)
        last = w.iloc[-1]
        prev = w.iloc[-2] if len(w) > 1 else last * 0
        out = []
        for code in panel.codes:
            a, b = prev.get(code, 0.0), last.get(code, 0.0)
            if b > 1e-9 and a <= 1e-9:
                action = "BUY"
            elif b <= 1e-9 and a > 1e-9:
                action = "SELL"
            elif b > 1e-9:
                action = "HOLD"
            else:
                action = "WATCH"
            s = sc[code].iloc[-1] if code in sc.columns else np.nan
            out.append({
                "code": code, "action": action, "weight": round(float(b), 4),
                "score": None if pd.isna(s) else round(float(s), 4),
                "reason": self.explain(panel, code, **params),
            })
        return out
