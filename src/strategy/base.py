"""
策略基类 —— 所有交易策略的抽象父类。

设计思路（类比 C 语言）：
- 类似 C++ 的纯虚基类：定义接口，子类必须实现 generate_signals()
- 类似嵌入式的中断服务函数：每个 bar 调用一次 next()，返回买/卖/持有信号

信号约定：
    1  = 买入（或持有多仓）
   -1  = 卖出（或平仓）
    0  = 不操作（继续持有现金或当前仓位）

对于 A 股（不能做空），-1 表示平掉已有仓位，而不是开空仓。
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Strategy(ABC):
    """
    策略抽象基类。

    每个具体策略继承此类，实现 generate_signals() 方法。

    用法:
        class MyStrategy(Strategy):
            def generate_signals(self, data):
                signals = pd.Series(0, index=data.index)
                # ... 计算逻辑 ...
                return signals

        strategy = MyStrategy(fast=5, slow=20)
        signals = strategy.generate_signals(data)
    """

    def __init__(self, **params):
        """
        Args:
            **params: 策略参数，例如 fast=5, slow=20, period=14 等
                      参数名和默认值由子类定义
        """
        self.params = params
        self._signals: Optional[pd.Series] = None  # 缓存最近一次生成的信号
        self._data: Optional[pd.DataFrame] = None  # 缓存最近一次使用的数据

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        核心方法。根据 OHLCV 数据生成交易信号。

        Args:
            data: 包含至少 date, open, high, low, close, volume 列的 DataFrame

        Returns:
            pd.Series，长度与 data 相同，每个元素为 1（买入）、-1（卖出）、0（持有）
        """
        ...

    def run(self, data: pd.DataFrame) -> pd.Series:
        """
        运行策略（调用 generate_signals 并缓存结果）。

        Args:
            data: OHLCV DataFrame

        Returns:
            信号 Series
        """
        self._data = data
        self._signals = self.generate_signals(data)
        return self._signals

    @property
    def signals(self) -> Optional[pd.Series]:
        """获取最近一次 run() 生成的信号"""
        return self._signals

    @property
    def name(self) -> str:
        """策略名称（用于显示和报告）"""
        return self.__class__.__name__

    def describe(self) -> str:
        """策略描述字符串，包含参数"""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"

    def signal_stats(self) -> dict:
        """统计信号分布"""
        if self._signals is None:
            return {}
        total = len(self._signals)
        buys = (self._signals == 1).sum()
        sells = (self._signals == -1).sum()
        holds = (self._signals == 0).sum()
        return {
            "total_bars": total,
            "buy_signals": buys,
            "sell_signals": sells,
            "hold_signals": holds,
            "buy_ratio": f"{buys / total * 100:.1f}%",
            "trade_count": min(buys, sells),  # 完整交易对数量
        }
