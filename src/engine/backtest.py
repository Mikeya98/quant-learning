"""
回测引擎 —— 连接策略信号与模拟交易的核心循环。

主循环逻辑（逐 bar 迭代）:

    for i in range(len(data)):
        signal = signals[i]

        if signal changed and signal == 1 and no position:
            → 买入
        elif signal changed and signal == -1 and has position:
            → 卖出

        portfolio.update(close_price)   # 每日净值快照

执行模型说明：
- 信号基于当日收盘价计算，在当日收盘执行（简化模型）
- 更严格的做法是次日开盘执行（避免 look-ahead bias），但复杂度高
- 对日线级别的学习性回测，当日收盘执行足够说明问题
- T+1 限制：今日买入的仓位不会在当日被卖出（信号变化触发卖出，但仓位是刚买的）
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .broker import Broker, Fill, OrderSide
from .portfolio import Portfolio


@dataclass
class BacktestResult:
    """回测结果"""
    equity_df: pd.DataFrame       # 每日净值
    trades_df: pd.DataFrame       # 交易记录
    summary: dict                 # 总结指标
    symbol: str = ""
    strategy_name: str = ""


class BacktestEngine:
    """
    回测引擎。

    用法:
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(data, signals, symbol="000001")
        print(result.summary)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.00025,
        slippage: float = 0.001,
        position_pct: float = 1.0,    # 每次买入使用的资金比例
    ):
        self.initial_capital = initial_capital
        self.broker = Broker(
            commission_rate=commission_rate,
            slippage=slippage,
        )
        self.position_pct = position_pct

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "",
        strategy_name: str = "",
    ) -> BacktestResult:
        """
        执行回测。

        Args:
            data:    OHLCV DataFrame，必须含 date, close 列
            signals: 交易信号 Series，索引与 data 对齐
            symbol:  股票代码（用于报告）
            strategy_name: 策略名称

        Returns:
            BacktestResult
        """
        n = len(data)
        if n == 0:
            raise ValueError("数据为空")

        pf = Portfolio(initial_capital=self.initial_capital)
        prev_signal = 0

        for i in range(n):
            date = data["date"].iloc[i]
            close = data["close"].iloc[i]
            signal = int(signals.iloc[i])

            # 买入逻辑：信号从非1变为1，且当前无持仓
            if signal == 1 and prev_signal != 1:
                if pf.position.is_empty:
                    shares = self.broker.calc_shares(
                        pf.cash, close, self.position_pct
                    )
                    if shares > 0:
                        fill = self.broker.execute_buy(close, shares)
                        pf.buy(date, fill.price, fill.shares, fill.commission)

            # 卖出逻辑：信号从1变为非1，且当前有持仓
            elif signal != 1 and prev_signal == 1:
                if not pf.position.is_empty:
                    shares = pf.position.shares
                    fill = self.broker.execute_sell(close, shares)
                    pf.sell(date, fill.price, fill.shares,
                            fill.commission, fill.stamp_duty)

            # 每日净值快照
            pf.update(date, close)
            prev_signal = signal

        # 如果回测结束时仍有持仓，按最后收盘价强制平仓
        if not pf.position.is_empty:
            last_date = data["date"].iloc[-1]
            last_close = data["close"].iloc[-1]
            fill = self.broker.execute_sell(last_close, pf.position.shares)
            pf.sell(last_date, fill.price, fill.shares,
                    fill.commission, fill.stamp_duty)
            pf.update(last_date, last_close)

        return BacktestResult(
            equity_df=pf.equity_df(),
            trades_df=pf.trades_df(),
            summary=pf.summary(),
            symbol=symbol,
            strategy_name=strategy_name,
        )
