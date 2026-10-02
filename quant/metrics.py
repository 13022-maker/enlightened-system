"""績效指標。"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 245  # 台股一年約 245 個交易日


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def compute_metrics(res) -> Dict[str, float]:
    eq = res.equity
    r = res.returns
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
    mdd = max_drawdown(eq)
    trade_rets = np.array([t.ret for t in res.trades if not np.isnan(t.ret)])
    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets <= 0]
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "exposure": float(res.weights.sum(axis=1).mean()),
        "annual_turnover": float(res.turnover.sum() / years),
        "cost_drag": float(res.costs.sum() / years),
        "n_trades": int(len(trade_rets)),
        "win_rate": float(len(wins) / len(trade_rets)) if len(trade_rets) else 0.0,
        "avg_trade": float(trade_rets.mean()) if len(trade_rets) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
    }


def format_metrics(m: Dict[str, float]) -> str:
    pct = ["total_return", "cagr", "volatility", "max_drawdown", "exposure", "win_rate", "avg_trade", "cost_drag"]
    parts = []
    for k, v in m.items():
        if k in pct:
            parts.append(f"{k}={v:+.1%}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
