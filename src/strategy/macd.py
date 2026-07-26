"""
MACD 策略 —— DIF 与 DEA 的金叉死叉

原理：
- DIF（快慢 EMA 差值）围绕 DEA（DIF 的 EMA）上下波动
- DIF 上穿 DEA → MACD 金叉 → 买入
- DIF 下穿 DEA → MACD 死叉 → 卖出
- 零轴上方金叉（DIF > 0）视为强买入，零轴下方死叉视为强卖出

与双均线策略的对比：
- 双均线直接比较两条均线的位置
- MACD 比较的是差值线与差值线的均线 —— 对价格的微小波动更敏感
"""

import pandas as pd
from .base import Strategy
from .indicators import macd


class MACDStrategy(Strategy):
    """
    MACD 金叉死叉策略

    参数:
        fast:   快线周期（默认 12）
        slow:   慢线周期（默认 26）
        signal: 信号线周期（默认 9）
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast = self.params["fast"]
        slow = self.params["slow"]
        signal = self.params["signal"]

        m = macd(data["close"], fast=fast, slow=slow, signal=signal)
        dif = m["dif"]
        dea = m["dea"]

        # 信号逻辑：
        # DIF > DEA → 多头
        # DIF < DEA → 空头
        signals = pd.Series(0, index=data.index, dtype=int)
        signals[dif > dea] = 1
        signals[dif < dea] = -1

        signals = signals.fillna(0)
        self._signals = signals
        return signals
