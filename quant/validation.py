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


def _carry_in(w: pd.DataFrame, split, test_start=None) -> pd.DataFrame:
    """測試期權重：從 test_start（預設 = split）起；第一列沿用切分日（含）之前最後一筆有效目標權重，
    避免低頻策略在測試期初空手。

    時序：w.loc[split] 是切分日收盤決定的目標 → 於 test_start（切分日次一交易日）開盤成交。
    引擎在第 0 天不交易（只有第 t−1 列才會在第 t 天成交），因此把 carry-in 目標放在測試期第 0 列，
    會在測試期第 1 天開盤成交（第 0 天空手、權益 = 1，作為測試期基準點）。
    """
    test_start = split if test_start is None else test_start
    w_test = w.loc[test_start:].copy()
    if len(w_test) and w_test.iloc[0].isna().all():
        valid = w.loc[:split].dropna(how="all")
        if len(valid):
            w_test.iloc[0] = valid.iloc[-1].values
    return w_test


def split_panel(panel: Panel, train_frac: float):
    """切分訓練／測試期，兩段不重疊。

    舊版 train = [起點, split]、test = [split, 終點] 共用 split 這一天 → 該日報酬同時算進兩段。
    新版：train = [起點, split]，test = [split 次一交易日, 終點]。
    """
    k = int(len(panel.dates) * train_frac)
    k = min(max(k, 1), len(panel.dates) - 2)
    split, test_start = panel.dates[k], panel.dates[k + 1]
    return split, test_start, panel.slice(None, split), panel.slice(test_start, None)


def walk_forward(strategy, panel: Panel, train_frac: float = 0.6,
                 grid: Optional[List[Dict]] = None, select_by: str = "sharpe") -> Dict:
    """訓練期挑參數 → 樣本外（測試期）評估。指標只在各自區間內計算。

    權重在全期資料上計算（指標暖身用到訓練期資料是允許的，因為不含未來資訊），
    再分別截取訓練期與測試期回測。
    """
    grid = grid or strategy.param_grid or [strategy.default_params]
    split, test_start, train_p, test_p = split_panel(panel, train_frac)

    # 0050 基準：測試期與策略同樣「第 0 天決定、第 1 天開盤買進」，日報酬可直接相減
    bench_ret = {}
    if BENCHMARK in panel.codes:
        bench_ret["train"] = buy_and_hold(train_p, BENCHMARK).returns
        bench_ret["test"] = buy_and_hold(test_p, BENCHMARK).returns

    rows = []
    for params in grid:
        p = {**strategy.default_params, **params}
        w = strategy.target_weights(panel, **p)
        tr = compute_metrics(backtest(train_p, w.loc[:split]), bench_ret.get("train"))
        te = compute_metrics(backtest(test_p, _carry_in(w, split, test_start)), bench_ret.get("test"))
        rows.append({"params": p, "train": tr, "test": te})

    best = max(rows, key=lambda r: r["train"][select_by])
    bench = {}
    if BENCHMARK in panel.codes:
        bench["0050買進持有"] = compute_metrics(buy_and_hold(test_p, BENCHMARK), bench_ret["test"])
    ew_name = "台灣50等權(月調，歷史成分股)" if panel.universe is not None else "台灣50等權(月調)"
    bench[ew_name] = compute_metrics(equal_weight(test_p), bench_ret.get("test"))
    test_sharpes = [r["test"]["sharpe"] for r in rows]
    return {
        "strategy": strategy.name,
        "split_date": str(split.date()),
        "train_period": f"{panel.dates[0].date()} ~ {split.date()}",
        "test_period": f"{test_start.date()} ~ {panel.dates[-1].date()}",
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
