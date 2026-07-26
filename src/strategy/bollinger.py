"""
布林带策略 (Bollinger Bands Breakout/Reversal)

原理：
- 布林带由中轨（SMA）、上轨（+2σ）、下轨（-2σ）构成
- 统计学上，价格在布林带内的概率约 95%（正态分布假设）
- 价格触碰下轨 → 处于统计极端低位 → 均值回归买入
- 价格触碰上轨 → 处于统计极端高位 → 均值回归卖出
- 带宽收窄 → 波动率降低 → 往往预示即将出现大行情（盘整突破）

策略变体：
1. 均值回归型（本策略）：价格碰下轨买、碰上轨卖
2. 突破型：价格突破上轨追买、跌破下轨追卖（适合强趋势）
"""

import pandas as pd
from .base import Strategy
from .indicators import bollinger_bands


class BollingerStrategy(Strategy):
    """
    布林带均值回归策略

    参数:
        period:  SMA 周期（默认 20）
        num_std: 标准差倍数（默认 2.0）
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(period=period, num_std=num_std)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        period = self.params["period"]
        num_std = self.params["num_std"]
        close = data["close"]

        bb = bollinger_bands(close, period, num_std)

        # 信号逻辑：
        # 收盘价 <= 下轨 → 买入（均值回归：超卖反弹）
        # 收盘价 >= 上轨 → 卖出（均值回归：超买回调）
        # 在轨道内 → 持有
        signals = pd.Series(0, index=data.index, dtype=int)
        signals[close <= bb["lower"]] = 1
        signals[close >= bb["upper"]] = -1

        signals = signals.fillna(0)
        self._signals = signals
        return signals
