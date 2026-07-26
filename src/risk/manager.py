"""
风险管理 —— 仓位控制和止损止盈。

仓位管理 (Position Sizing):
- FixedFraction: 每次交易使用固定比例的资金（如 20%）
- KellyCriterion: 基于胜率和盈亏比的凯利公式

止损止盈:
- StopLoss: 亏损超过 N% 强制平仓
- TakeProfit: 盈利超过 N% 锁定利润
- TrailingStop: 从最高点回撤 N% 平仓

这些规则嵌入回测主循环中，在每次 bar 更新后检查。
"""

import numpy as np
import pandas as pd


class PositionSizer:
    """仓位计算器"""

    @staticmethod
    def fixed_fraction(cash: float, price: float, fraction: float = 0.2) -> float:
        """
        固定比例仓位。

        Args:
            cash:     可用资金
            price:    当前价格
            fraction: 资金使用比例（0-1）

        Returns:
            应买入的金额
        """
        return cash * fraction

    @staticmethod
    def kelly(win_rate: float, avg_win: float, avg_loss: float,
              half: bool = True) -> float:
        """
        凯利公式。

        公式: f* = win_rate - (1 - win_rate) / (avg_win / avg_loss)

        Args:
            win_rate: 胜率（0-1）
            avg_win:  平均盈利金额（取绝对值）
            avg_loss: 平均亏损金额（取绝对值）
            half:     是否使用半凯利（更保守，默认 True）

        Returns:
            建议仓位比例（0-1），上限 0.5
        """
        if avg_loss == 0:
            return 0.0
        win_loss_ratio = avg_win / avg_loss
        f = win_rate - (1 - win_rate) / win_loss_ratio
        if half:
            f = f / 2
        return max(0.0, min(f, 0.5))  # 限制最大 50% 仓位


class StopManager:
    """
    止损止盈管理器。

    在每个 bar 检查是否需要触发止损/止盈。

    用法:
        sm = StopManager(stop_loss_pct=0.05, take_profit_pct=0.15)
        sm.enter(entry_price, date)
        ...
        exit_signal = sm.check(current_price, current_date)
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.05,       # 5% 止损
        take_profit_pct: float = 0.15,     # 15% 止盈
        trailing_stop_pct: float = 0.0,    # 移动止损（0 = 禁用）
        max_holding_days: int = 0,          # 最大持仓天数（0 = 不限）
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_days = max_holding_days

        self._entry_price: float = 0.0
        self._entry_date = None
        self._highest_price: float = 0.0   # 持仓期间最高价（移动止损用）
        self._active: bool = False

    def enter(self, price: float, date: pd.Timestamp):
        """记录入场"""
        self._entry_price = price
        self._entry_date = date
        self._highest_price = price
        self._active = True

    def check(self, current_price: float, current_date: pd.Timestamp) -> tuple:
        """
        检查是否需要退出。

        Returns:
            (should_exit: bool, reason: str)
        """
        if not self._active:
            return False, ""

        self._highest_price = max(self._highest_price, current_price)
        pnl_pct = (current_price / self._entry_price - 1)

        # 止损
        if self.stop_loss_pct > 0 and pnl_pct <= -self.stop_loss_pct:
            self._active = False
            return True, f"stop_loss({pnl_pct:.1%})"

        # 止盈
        if self.take_profit_pct > 0 and pnl_pct >= self.take_profit_pct:
            self._active = False
            return True, f"take_profit({pnl_pct:.1%})"

        # 移动止损
        if self.trailing_stop_pct > 0:
            drawdown_from_high = (self._highest_price - current_price) / self._highest_price
            if drawdown_from_high >= self.trailing_stop_pct:
                self._active = False
                return True, f"trailing_stop({drawdown_from_high:.1%})"

        # 时间止损
        if self.max_holding_days > 0 and self._entry_date is not None:
            days = (current_date - self._entry_date).days
            if days >= self.max_holding_days:
                self._active = False
                return True, f"time_stop({days}d)"

        return False, ""

    def reset(self):
        """重置状态"""
        self._active = False
        self._entry_price = 0.0
        self._entry_date = None
        self._highest_price = 0.0
