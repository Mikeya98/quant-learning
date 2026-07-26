"""
RSI 均值回归策略 (Mean Reversion)

原理：
- RSI 衡量价格变动的速度和幅度，范围 0-100
- RSI < 30: 超卖区域，价格可能被过度打压 → 反弹概率大 → 买入
- RSI > 70: 超买区域，价格可能被过度追捧 → 回调概率大 → 卖出
- 均值回归的核心信念：价格总是围绕均值波动，极端值会回归

与趋势跟踪策略的区别：
- 双均线、MACD 是趋势跟踪（追涨杀跌）
- RSI 是均值回归（抄底逃顶）
- 两种思路在数学上是矛盾的 —— 但在不同市场环境下各有优势
"""

import pandas as pd
from .base import Strategy
from .indicators import rsi


class RSIStrategy(Strategy):
    """
    RSI 超买超卖策略

    参数:
        period:       RSI 计算周期（默认 14）
        oversold:     超卖阈值（默认 30，低于此值买入）
        overbought:   超买阈值（默认 70，高于此值卖出）
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(period=period, oversold=oversold, overbought=overbought)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        period = self.params["period"]
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        rsi_values = rsi(data["close"], period)

        # 信号逻辑：
        # RSI 进入超卖区 → 买入信号（1）
        # RSI 进入超买区 → 卖出信号（-1）
        # 其他区域 → 保持当前状态（0）
        signals = pd.Series(0, index=data.index, dtype=int)
        signals[rsi_values < oversold] = 1
        signals[rsi_values > overbought] = -1

        signals = signals.fillna(0)
        self._signals = signals
        return signals
