"""策略自我驗證工具：前視偏差檢查、walk-forward 樣本外評估。"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BENCHMARK
from .data import Panel
from .engine import backtest, buy_and_hold, equal_weight
from .metrics import compute_metrics


def assert_no_lookahead(strategy, panel: Panel, params: Optional[Dict] = None, cut_points=(0.5, 0.75)) -> None:
    """截斷資料後重算，截斷點之前的權重必須完全相同，否則代表偷看了未來。"""
    params = {**strategy.default_params, **(params or {})}
    full = strategy.target_weights(panel, **params)
    for frac in cut_points:
        cut = panel.dates[int(len(panel.dates) * frac)]
        part = strategy.target_weights(panel.slice(None, cut), **params)
        a = full.loc[:cut].fillna(-1.0)
        b = part.reindex_like(a).fillna(-1.0)
        diff = (a - b).abs().max().max()
        if diff > 1e-9:
            bad = (a - b).abs().max(axis=1)
            raise AssertionError(
                f"{strategy.name}: 偵測到前視偏差，截斷於 {cut.date()} 時權重不同 "
                f"(首個差異日 {bad[bad > 1e-9].index[0].date()}, max diff={diff:.4f})")


def _carry_in(w: pd.DataFrame, split) -> pd.DataFrame:
    """測試期第一天沿用切分日（含）之前最後一筆有效目標權重，避免低頻策略在測試期初空手。"""
    w_test = w.loc[split:].copy()
    if w_test.iloc[0].isna().all():
        valid = w.loc[:split].dropna(how="all")
        if len(valid):
            w_test.iloc[0] = valid.iloc[-1].values
    return w_test


def walk_forward(strategy, panel: Panel, train_frac: float = 0.6,
                 grid: Optional[List[Dict]] = None, select_by: str = "sharpe") -> Dict:
    """訓練期挑參數 → 樣本外（測試期）評估。指標只在各自區間內計算。

    權重在全期資料上計算（指標暖身用到訓練期資料是允許的，因為不含未來資訊），
    再分別截取訓練期與測試期回測。
    """
    grid = grid or strategy.param_grid or [strategy.default_params]
    split = panel.dates[int(len(panel.dates) * train_frac)]
    train_p, test_p = panel.slice(None, split), panel.slice(split, None)

    rows = []
    for params in grid:
        p = {**strategy.default_params, **params}
        w = strategy.target_weights(panel, **p)
        tr = compute_metrics(backtest(train_p, w.loc[:split]))
        te = compute_metrics(backtest(test_p, _carry_in(w, split)))
        rows.append({"params": p, "train": tr, "test": te})

    best = max(rows, key=lambda r: r["train"][select_by])
    bench = {}
    if BENCHMARK in panel.codes:
        bench["0050買進持有"] = compute_metrics(buy_and_hold(test_p, BENCHMARK))
    bench["台灣50等權(月調)"] = compute_metrics(equal_weight(test_p))
    test_sharpes = [r["test"]["sharpe"] for r in rows]
    return {
        "strategy": strategy.name,
        "split_date": str(split.date()),
        "train_period": f"{panel.dates[0].date()} ~ {split.date()}",
        "test_period": f"{split.date()} ~ {panel.dates[-1].date()}",
        "best_params": best["params"],
        "train": best["train"],
        "test": best["test"],
        "grid": rows,
        "robustness": {   # 所有參數組在樣本外的表現分布：差距越小越穩健
            "test_sharpe_median": float(np.median(test_sharpes)),
            "test_sharpe_min": float(np.min(test_sharpes)),
            "test_sharpe_max": float(np.max(test_sharpes)),
        },
        "benchmarks_test": bench,
    }
