"""
策略模块 —— 提供策略基类和 4 个经典策略实现

用法:
    from src.strategy import MACrossStrategy, MACDStrategy, RSIStrategy, BollingerStrategy

    data = load_csv("000001", processed=True)
    strategy = MACrossStrategy(fast=5, slow=20)
    signals = strategy.run(data)
    print(strategy.signal_stats())
"""

from .base import Strategy
from .ma_cross import MACrossStrategy
from .macd import MACDStrategy
from .rsi import RSIStrategy
from .bollinger import BollingerStrategy

# 实时指标（增量递推版本，用于实盘/模拟交易）
from .realtime_indicators import (
    OnlineSMA, OnlineEMA, OnlineMACD, OnlineRSI, OnlineBollinger,
    RealtimeIndicatorCalculator,
)

# 策略注册表（CLI 通过名称查找策略）
STRATEGY_REGISTRY = {
    "ma_cross": MACrossStrategy,
    "macd": MACDStrategy,
    "rsi": RSIStrategy,
    "bollinger": BollingerStrategy,
}
