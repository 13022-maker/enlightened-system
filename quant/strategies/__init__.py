"""策略註冊表。新增策略後在 REGISTRY 登記即可被 tools/run_backtest.py、tools/export_signals.py 使用。"""
from .chip_flow import ChipFlow
from .dividend_yield import DividendYield
from .legacy_confidence import LegacyConfidence
from .swing_momentum import SwingMomentum

REGISTRY = {s.name: s for s in [SwingMomentum, ChipFlow, DividendYield, LegacyConfidence]}


def get(name):
    return REGISTRY[name]()
