"""績效指標。"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 245  # 台股一年約 245 個交易日


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def sharpe_se(sharpe_annual: float, n_obs: int, periods: int = TRADING_DAYS) -> float:
    """Sharpe 標準誤（Lo 2002，iid 簡化）：SE(日 SR) = sqrt((1 + 0.5·SR_d²) / N)，年化 × sqrt(periods)。"""
    if n_obs <= 1:
        return float("nan")
    sr_d = sharpe_annual / np.sqrt(periods)
    return float(np.sqrt((1 + 0.5 * sr_d ** 2) / n_obs) * np.sqrt(periods))


def excess_tstat(r: pd.Series, bench: pd.Series) -> Dict[str, float]:
    """日超額報酬（策略 − 基準）的 t 值與年化超額。第一天（建倉）不計。"""
    d = (r - bench.reindex(r.index)).dropna().iloc[1:]
    if len(d) < 3 or d.std() == 0:
        return {"excess_t": float("nan"), "excess_ann": float("nan")}
    return {"excess_t": float(d.mean() / d.std() * np.sqrt(len(d))),
            "excess_ann": float(d.mean() * TRADING_DAYS)}


def compute_metrics(res, bench_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """bench_returns：給定時另算相對基準的日超額報酬 t 值（excess_t）與年化超額（excess_ann）。

    期間 < 1 年時 CAGR 不年化（= 總報酬），並以 annualized=False 標註。
    交易統計（n_trades、win_rate、avg_trade、profit_factor）只計已平倉的 lot（份額帳，已扣成本）。
    """
    eq = res.equity
    r = res.returns
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    total = float(eq.iloc[-1] - 1)
    annualized = years >= 1.0
    if not annualized:
        cagr = total
    else:
        cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
    mdd = max_drawdown(eq)
    closed = [t for t in res.trades if not getattr(t, "open_at_end", False)]
    trade_rets = np.array([t.ret for t in closed if not np.isnan(t.ret)])
    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets <= 0]
    out = {
        "total_return": total,
        "cagr": cagr,
        "annualized": annualized,
        "n_days": int(len(r)),
        "volatility": vol,
        "sharpe": sharpe,
        "sharpe_se": sharpe_se(sharpe, len(r)),
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
        "n_open_lots": int(len(res.trades) - len(closed)),
    }
    if bench_returns is not None:
        out.update(excess_tstat(r, bench_returns))
    return out


def format_metrics(m: Dict[str, float]) -> str:
    pct = ["total_return", "cagr", "volatility", "max_drawdown", "exposure", "win_rate", "avg_trade", "cost_drag"]
    parts = []
    for k, v in m.items():
        if k in pct:
            parts.append(f"{k}={v:+.1%}")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
