"""
双均线交叉策略 (SMA Crossover)

最经典的量化策略 —— "金叉买入，死叉卖出"

原理：
- 短期均线（快线）反映近期价格趋势，变化灵敏
- 长期均线（慢线）反映中长期趋势，变化平缓
- 金叉 (Golden Cross): 快线从下方上穿慢线 → 短期趋势转强 → 买入信号
- 死叉 (Death Cross): 快线从上方下穿慢线 → 短期趋势转弱 → 卖出信号

优点: 简单直观，在趋势行情中表现好
缺点: 震荡行情中频繁假信号（反复穿越），产生高额手续费
"""

import pandas as pd
from .base import Strategy
from .indicators import sma


class MACrossStrategy(Strategy):
    """
    双均线交叉策略

    参数:
        fast: 快线周期（默认 5 日）
        slow: 慢线周期（默认 20 日）
    """

    def __init__(self, fast: int = 5, slow: int = 20):
        super().__init__(fast=fast, slow=slow)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast = self.params["fast"]
        slow = self.params["slow"]
        close = data["close"]

        # 计算均线
        ma_fast = sma(close, fast)
        ma_slow = sma(close, slow)

        # 信号逻辑：
        # 快线 > 慢线 → 持有多头（1）
        # 快线 < 慢线 → 空仓（-1）
        signals = pd.Series(0, index=data.index, dtype=int)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1

        # NaN 区域（均线未计算出来的前 N-1 天）保持 0
        signals = signals.fillna(0)

        self._signals = signals
        return signals
